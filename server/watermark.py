"""The version of a finished video a free user gets to see.

The download is what Pro buys, so the copy that plays in the page must not
*be* the download: the network tab shows the URL, and a URL to the real file
is a download.  So a free user's player is fed a separate file -- 540x960,
heavily compressed, with a watermark drawn across it -- made once from the
render and kept beside it.  Pro plays the real thing.

The watermark is a Pillow image laid over the video by ffmpeg, the same way
the captions are: the project already draws with Pillow and never leans on
which filters an ffmpeg build happens to include.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from autocut.ffmpeg import ffmpeg
from autocut.render.captions import _BUNDLED_FONTS as FONT_DIR

log = logging.getLogger(__name__)

WIDTH, HEIGHT = 540, 960
SUFFIX = "-preview.mp4"
TEXT = "autocut  ·  upgrade to download"

#: One preview is built at a time per render; a second viewer waits for the
#: first rather than starting the same ffmpeg over again.
_locks: dict[Path, threading.Lock] = {}
_locks_guard = threading.Lock()


def preview_path(output: Path) -> Path:
    return output.with_name(output.stem + SUFFIX)


def _lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(path, threading.Lock())


def _stamp(path: Path) -> None:
    """A diagonal band of faint text, tiled so no crop escapes it."""
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    font = ImageFont.truetype(str(FONT_DIR / "Montserrat-SemiBold.ttf"), 26)
    layer = Image.new("RGBA", (WIDTH * 2, HEIGHT * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    step_y = 140
    for row, y in enumerate(range(0, HEIGHT * 2, step_y)):
        x = -300 + (row % 2) * 220
        while x < WIDTH * 2:
            draw.text((x, y), TEXT, font=font, fill=(255, 255, 255, 110))
            x += 560
    layer = layer.rotate(-28, resample=Image.BICUBIC)
    image.alpha_composite(layer, (-(WIDTH // 2), -(HEIGHT // 2)))
    # A solid line at the bottom that reads even where the band is between rows.
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, HEIGHT - 44, WIDTH, HEIGHT], fill=(0, 0, 0, 150))
    draw.text((16, HEIGHT - 36), "PREVIEW  ·  autocut", font=font, fill=(255, 255, 255, 230))
    image.save(path)


def ensure_preview(output: Path) -> Path:
    """The watermarked copy of ``output``, built if it is not there yet."""
    target = preview_path(output)
    with _lock_for(target):
        if target.exists() and target.stat().st_mtime >= output.stat().st_mtime:
            return target
        stamp = output.with_name(output.stem + "-stamp.png")
        partial = target.with_suffix(".part.mp4")
        _stamp(stamp)
        try:
            ffmpeg(
                [
                    "-i", str(output), "-i", str(stamp),
                    "-filter_complex",
                    f"[0:v]scale={WIDTH}:{HEIGHT}:flags=bicubic[v0];[v0][1:v]overlay=0:0:format=auto[v]",
                    "-map", "[v]", "-map", "0:a?",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "32", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "64k",
                    "-movflags", "+faststart",
                    str(partial),
                ],
                description=f"preview of {output.name}",
            )
            partial.replace(target)
        finally:
            stamp.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)
        log.info("built preview %s", target.name)
        return target


def drop_preview(output: Path) -> None:
    preview_path(output).unlink(missing_ok=True)
