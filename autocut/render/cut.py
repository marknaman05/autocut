"""Stage 4 -- apply the edit decision list.

One decode pass, frame accurate, no per-segment temporary files: ``select`` and
``aselect`` keep the wanted frames and ``setpts``/``asetpts`` restamp them into
a continuous timeline.

The filter graph is written to a script file rather than passed on the command
line, so a heavily chopped video cannot blow the argument length limit.  Very
large edit lists are additionally split into chunks that are rendered with a
seek and concatenated, which keeps any single expression small and lets each
pass decode only the part of the source it needs.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import EncodeConfig
from ..ffmpeg import ffmpeg, filter_script_args
from ..models import KeepSegment

log = logging.getLogger(__name__)

#: Above this many segments in one pass, switch to the chunked path.
MAX_SEGMENTS_PER_PASS = 250

#: Segments are matched with a small tolerance so that floating point noise in
#: a boundary cannot drop the frame sitting exactly on it.
_EPSILON = 1e-4


def select_expression(segments: list[KeepSegment], offset: float = 0.0) -> str:
    """An ffmpeg ``select`` expression matching every kept segment.

    ``offset`` is subtracted from each boundary, for chunked passes where the
    input has already been seeked.
    """
    terms = [
        f"between(t\\,{seg.start - offset - _EPSILON:.6f}\\,{seg.end - offset + _EPSILON:.6f})"
        for seg in segments
    ]
    return "+".join(terms) if terms else "0"


def _filter_script(segments: list[KeepSegment], fps: float, offset: float) -> str:
    expression = select_expression(segments, offset)
    return (
        f"[0:v]select='{expression}',setpts=N/FRAME_RATE/TB[v];"
        f"[0:a]aselect='{expression}',asetpts=N/SR/TB[a]"
    )


def _render_pass(
    source: Path,
    destination: Path,
    segments: list[KeepSegment],
    fps: float,
    config: EncodeConfig,
    work_dir: Path,
    *,
    seek: float = 0.0,
    until: float | None = None,
) -> Path:
    script = work_dir / f"cut-{destination.stem}.filter"
    script.write_text(_filter_script(segments, fps, seek))

    args: list[str] = []
    if seek > 0.0:
        # Input seeking, so the decoder skips straight to the chunk.  The
        # expression is offset to match.
        args += ["-ss", f"{seek:.6f}"]
    if until is not None:
        args += ["-to", f"{until:.6f}"]
    args += [
        "-i", str(source),
        *filter_script_args(script),
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(config.intermediate_crf),
        "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr", "-r", f"{fps:.6f}",
        "-c:a", "pcm_s16le",
        str(destination),
    ]
    ffmpeg(args, description=f"cut {len(segments)} segments -> {destination.name}")
    return destination


def _chunk(segments: list[KeepSegment], size: int) -> list[list[KeepSegment]]:
    return [segments[i : i + size] for i in range(0, len(segments), size)]


def apply_cuts(
    source: Path,
    destination: Path,
    segments: list[KeepSegment],
    fps: float,
    config: EncodeConfig,
    work_dir: Path,
) -> Path:
    """Render ``source`` down to ``segments``.

    The intermediate is written with a high quality setting and uncompressed
    audio, because two more filter passes still have to run over it.
    """
    if not segments:
        raise ValueError("cannot render an empty edit decision list")

    work_dir.mkdir(parents=True, exist_ok=True)

    if len(segments) <= MAX_SEGMENTS_PER_PASS:
        return _render_pass(source, destination, segments, fps, config, work_dir)

    log.info("%d segments; rendering in chunks", len(segments))
    chunks = _chunk(segments, MAX_SEGMENTS_PER_PASS)
    parts: list[Path] = []
    for index, chunk in enumerate(chunks):
        part = work_dir / f"cut-part{index:03d}.mkv"
        _render_pass(
            source, part, chunk, fps, config, work_dir,
            seek=chunk[0].start,
            until=chunk[-1].end,
        )
        parts.append(part)

    listing = work_dir / "cut-parts.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(destination)],
        description=f"concatenate {len(parts)} cut chunks",
    )
    return destination
