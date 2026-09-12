"""Stage 6 -- karaoke captions, one word highlighted at a time.

Captions are drawn with Pillow and composited by ffmpeg rather than burned in
with libass, because the installed ffmpeg has no libass (and no drawtext).
Doing the drawing ourselves removes that dependency entirely and gives exact
control over the per-word pop.

The efficient part is what gets rendered: the caption only changes when the
active word changes, so we render one image per *word state* -- a few hundred
for a typical video -- and let ffmpeg's concat demuxer hold each one for the
right length of time.  That is a small fraction of the frame count, and the
whole track composites in a single overlay pass.

Everything here works in output time.  Word timings arrive in source time and
must be mapped through the TimeMap first, which ``build`` does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..config import CaptionConfig
from ..models import TimeMap, Word

log = logging.getLogger(__name__)

#: Fonts shipped with the package, so a named style draws the same glyphs on
#: every machine.  Searched before the system directories: a bundled
#: ``Montserrat-Black`` must win over any copy a user happens to have installed,
#: or a style would look different from one machine to the next.
_BUNDLED_FONTS = Path(__file__).parent / "fonts"

_FONT_DIRECTORIES = (
    _BUNDLED_FONTS,
    Path("/System/Library/Fonts/Supplemental"),
    Path("/System/Library/Fonts"),
    Path("/Library/Fonts"),
    Path.home() / "Library/Fonts",
)

#: Tried in order when the configured font cannot be found.  The bundled face
#: comes first so the fallback is the same everywhere; the rest are what macOS
#: ships, for a checkout with the fonts directory missing.
_FALLBACK_FONTS = (
    "Montserrat-ExtraBold.ttf", "Arial Black.ttf", "Arial Bold.ttf", "Helvetica.ttc", "Arial.ttf",
)


class CaptionError(RuntimeError):
    pass


def find_font(name: str) -> Path:
    """Resolve a font name to a file on disk.

    ``name`` is a file stem (``Anton-Regular``) or a family name as macOS
    spells it (``Arial Black``); both forms are tried, with and without the
    space, so either spelling in a config finds the same file.  Collections
    (``.ttc``) are opened at face 0, which is the regular face for every
    collection this has been tried on.
    """
    stems = [name, name.replace(" ", ""), name.replace(" ", "-")]
    candidates = [f"{stem}{ext}" for stem in stems for ext in (".ttf", ".otf", ".ttc")]
    candidates += _FALLBACK_FONTS
    for candidate in candidates:
        for directory in _FONT_DIRECTORIES:
            path = directory / candidate
            if path.exists():
                return path
    raise CaptionError(f"no usable font found for {name!r}")


@dataclass
class CaptionWord:
    """A word placed on the output timeline."""

    text: str
    start: float
    end: float


@dataclass
class Line:
    """A group of words shown together, one of which is highlighted at a time."""

    words: list[CaptionWord]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


def to_output_words(words: list[Word], time_map: TimeMap, config: CaptionConfig) -> list[CaptionWord]:
    """Map word timings through the cut, dropping anything that was removed."""
    placed: list[CaptionWord] = []
    for word in words:
        if time_map.kept_fraction(word.start, word.end) < config.min_visible_fraction:
            continue
        span = time_map.map_span(word.start, word.end)
        if span is None:
            continue
        start, end = span
        start += config.offset
        end += config.offset
        if end <= start:
            end = start + 0.05
        if placed and start < placed[-1].end:
            # Two words either side of a cut can collide once mapped; the later
            # one yields so the highlight never runs backwards.
            start = placed[-1].end
            end = max(end, start + 0.05)
        placed.append(CaptionWord(text=word.text.strip(), start=start, end=end))
    return placed


def group_lines(words: list[CaptionWord], config: CaptionConfig) -> list[Line]:
    """Break the word stream into caption lines.

    Breaks on sentence-ending punctuation and on pauses first, and only then on
    the length limits -- so lines follow the speech rather than chopping it into
    equal-sized blocks.
    """
    lines: list[Line] = []
    current: list[CaptionWord] = []

    def flush() -> None:
        nonlocal current
        if current:
            lines.append(Line(words=current))
            current = []

    for index, word in enumerate(words):
        current.append(word)
        if word.text.endswith((".", "!", "?")):
            flush()
            continue

        is_last = index == len(words) - 1
        if is_last:
            break
        gap = words[index + 1].start - word.end
        length = sum(len(w.text) + 1 for w in current) - 1
        next_length = length + 1 + len(words[index + 1].text)
        if (
            gap >= config.line_break_gap
            or len(current) >= config.max_words_per_line
            or next_length > config.max_chars_per_line
        ):
            flush()
    flush()
    return lines


class Renderer:
    """Draws one caption band image per word state."""

    def __init__(self, config: CaptionConfig, width: int) -> None:
        self.config = config
        self.width = width
        font_path = find_font(config.font)
        self.font = ImageFont.truetype(str(font_path), config.font_size)
        self.pop_font = ImageFont.truetype(
            str(font_path), int(config.font_size * config.pop_scale)
        )
        # Glyph extent of the popped face, for sizing boxes: ascent above the
        # baseline plus descent below it.
        ascent, descent = self.pop_font.getmetrics()
        self.glyph_height = ascent + descent
        # A band tall enough for the popped glyphs plus outline and shadow --
        # and for a box's padding and a glow's blur, both of which reach past
        # the glyphs.  A band that clips a pill's bottom corners is the first
        # thing a preview shows.
        self.height = int(
            config.font_size * config.pop_scale * 1.6
            + config.outline * 2 + config.shadow_offset
            + config.box_padding * 2 + config.glow * 2
        )
        # The outline grows every word by `outline` pixels on each side, so a
        # plain space advance leaves the strokes of adjacent words touching.
        # A pill reaches `box_padding` further still.
        self.space = self.font.getlength(" ") + config.outline
        if config.active_box_colour:
            self.space += config.box_padding

    def _text(self, word: CaptionWord) -> str:
        """What is drawn for a word.

        The one place case is changed.  Widths are measured on the same
        string that is drawn -- measuring "how" and drawing "HOW" would
        centre every line slightly off.
        """
        return word.text.upper() if self.config.uppercase else word.text

    def _widths(self, line: Line, popped_index: int | None = None) -> list[float]:
        """Slot width for each word.

        The popped word claims its enlarged width, otherwise it grows over the
        top of its neighbours.  The line stays centred, so the other words slide
        outwards very slightly as the highlight passes -- which reads as part of
        the animation rather than as a glitch.
        """
        return [
            (self.pop_font if index == popped_index else self.font).getlength(self._text(word))
            for index, word in enumerate(line.words)
        ]

    def layout(self, line: Line, popped_index: int | None = None) -> tuple[float, list[float]]:
        """Left edge of the centred line, and each word's slot width."""
        widths = self._widths(line, popped_index)
        total = sum(widths) + self.space * (len(widths) - 1)
        return (self.width - total) / 2, widths

    def _box(self, draw: ImageDraw.ImageDraw, left: float, right: float, baseline: float, colour: str) -> None:
        """A rounded box spanning ``left``..``right`` around the glyph band."""
        pad = self.config.box_padding + self.config.outline
        half = self.glyph_height / 2
        draw.rounded_rectangle(
            (left - pad, baseline - half - self.config.box_padding,
             right + pad, baseline + half + self.config.box_padding),
            radius=self.config.box_radius, fill=colour,
        )

    def render(self, line: Line, active: int, popped: bool) -> Image.Image:
        """One frame of the caption: ``line`` with word ``active`` highlighted."""
        image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        config = self.config

        x, widths = self.layout(line, active if popped else None)
        total = sum(widths) + self.space * (len(widths) - 1)
        baseline = self.height / 2

        # Boxes go down first, under everything.  The line block spans the
        # whole line; the pill only the active word's slot, whose width is
        # already the popped width when popped, so the pill grows with it.
        if config.line_box_colour:
            self._box(draw, x, x + total, baseline, config.line_box_colour)
        if config.active_box_colour and 0 <= active < len(widths):
            left = x + sum(widths[:active]) + self.space * active
            self._box(draw, left, left + widths[active], baseline, config.active_box_colour)

        # The glow is a wide stroke of the whole line, blurred, composited
        # under the crisp text.  One blur of the band per frame.
        if config.glow:
            halo = Image.new("RGBA", image.size, (0, 0, 0, 0))
            halo_draw = ImageDraw.Draw(halo)
            hx = x
            for index, (word, width) in enumerate(zip(line.words, widths)):
                font = self.pop_font if (index == active and popped) else self.font
                halo_draw.text(
                    (hx + width / 2, baseline), self._text(word), font=font,
                    fill=config.outline_colour, anchor="mm",
                    stroke_width=config.outline + config.glow // 2,
                    stroke_fill=config.outline_colour,
                )
                hx += width + self.space
            image.alpha_composite(halo.filter(ImageFilter.GaussianBlur(config.glow)))
            draw = ImageDraw.Draw(image)

        for index, (word, width) in enumerate(zip(line.words, widths)):
            is_active = index == active
            font = self.pop_font if (is_active and popped) else self.font
            if is_active:
                colour = config.active_text_colour or config.highlight_colour
            else:
                colour = config.text_colour
            text = self._text(word)
            # Popped glyphs are drawn from the centre of the word's slot so the
            # line's layout does not shift as the highlight moves along it.
            # ``anchor="mm"`` centres on the ascender/descender midpoint, so
            # all-caps text sits a shade high; it reads fine.
            centre = x + width / 2
            if config.shadow_offset:
                draw.text(
                    (centre + config.shadow_offset, baseline + config.shadow_offset),
                    text, font=font, fill=config.shadow_colour,
                    anchor="mm", stroke_width=config.outline,
                    stroke_fill=config.shadow_colour,
                )
            draw.text(
                (centre, baseline), text, font=font, fill=colour, anchor="mm",
                stroke_width=config.outline, stroke_fill=config.outline_colour,
            )
            x += width + self.space
        return image


def build(
    words: list[Word],
    time_map: TimeMap,
    config: CaptionConfig,
    width: int,
    work_dir: Path,
) -> tuple[Path, int] | None:
    """Render the caption track.

    Returns the concat list file ffmpeg should read and the band height, or
    ``None`` if there is nothing to caption.
    """
    if not config.enabled:
        return None

    placed = to_output_words(words, time_map, config)
    lines = group_lines(placed, config)
    if not lines:
        log.warning("no words survived the cut; skipping captions")
        return None

    frames_dir = work_dir / "captions"
    frames_dir.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(config, width)

    blank = frames_dir / "blank.png"
    Image.new("RGBA", (width, renderer.height), (0, 0, 0, 0)).save(blank)

    entries: list[tuple[Path, float]] = []
    cursor = 0.0
    counter = 0

    for line in lines:
        if line.start > cursor + 0.01:
            entries.append((blank, line.start - cursor))
            cursor = line.start

        for index, word in enumerate(line.words):
            # A word's slot runs until the next word starts, so the highlight
            # never flickers off during the gap between two words in a line.
            slot_end = line.words[index + 1].start if index + 1 < len(line.words) else line.end
            slot_end = max(slot_end, word.start + 0.04)
            duration = slot_end - cursor
            if duration <= 0:
                continue

            pop = min(config.pop_duration, duration / 2) if config.pop_duration else 0.0
            for popped, length in ((True, pop), (False, duration - pop)):
                if length <= 0.001:
                    continue
                frame = frames_dir / f"{counter:05d}.png"
                renderer.render(line, index, popped).save(frame)
                entries.append((frame, length))
                counter += 1
            cursor = slot_end

    listing = work_dir / "captions.txt"
    lines_out = []
    for path, duration in entries:
        lines_out.append(f"file '{path.resolve()}'\nduration {duration:.4f}\n")
    # The concat demuxer ignores the final entry's duration unless the last
    # file is repeated.
    if entries:
        lines_out.append(f"file '{entries[-1][0].resolve()}'\n")
    listing.write_text("".join(lines_out))

    log.info("rendered %d caption states across %d lines", counter, len(lines))
    return listing, renderer.height
