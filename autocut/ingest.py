"""Stage 1 -- probe the upload and extract analysis audio.

The subtle part is rotation.  Phone portrait footage is commonly stored as a
landscape 1920x1080 stream plus a 90 degree display matrix, so the coded
dimensions lie about how the video actually looks.  Every later stage reasons
about framing, so we resolve display dimensions here, once.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

from .ffmpeg import FFmpegError, ffmpeg, ffprobe_json, require_binaries
from .models import Timeline

log = logging.getLogger(__name__)

#: Analysis audio format expected by every supported ASR backend.
ASR_SAMPLE_RATE = 16_000

MAX_DURATION_SECONDS = 60 * 60


class UnsupportedMediaError(ValueError):
    """The upload is not something the pipeline can process."""


def _rotation_degrees(stream: dict) -> int:
    """Display rotation in degrees, from either the modern display matrix side
    data or the legacy ``rotate`` tag."""
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            return int(round(float(side_data["rotation"]))) % 360
    tag = (stream.get("tags") or {}).get("rotate")
    if tag is not None:
        return int(round(float(tag))) % 360
    return 0


def _frame_rate(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(key)
        if value and value != "0/0":
            rate = float(Fraction(value))
            if rate > 0:
                return rate
    return 30.0


def _duration(probe: dict, stream: dict) -> float:
    for source in (stream, probe.get("format", {})):
        value = source.get("duration")
        if value:
            try:
                return float(value)
            except ValueError:
                continue
    raise UnsupportedMediaError("could not determine the video's duration")


def probe(path: Path) -> Timeline:
    """Inspect ``path`` and return a Timeline with its metadata filled in."""
    require_binaries()
    if not path.exists():
        raise FileNotFoundError(path)

    data = ffprobe_json(path, "-show_streams", "-show_format")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise UnsupportedMediaError("no video stream found -- is this an audio file?")
    if not any(s.get("codec_type") == "audio" for s in streams):
        raise UnsupportedMediaError("no audio stream found; there is nothing to transcribe")

    width, height = int(video["width"]), int(video["height"])
    if _rotation_degrees(video) in (90, 270):
        width, height = height, width

    duration = _duration(data, video)
    if duration > MAX_DURATION_SECONDS:
        raise UnsupportedMediaError(
            f"video is {duration / 60:.0f} minutes long; the limit is "
            f"{MAX_DURATION_SECONDS // 60} minutes"
        )

    return Timeline(
        source=path,
        duration=duration,
        fps=_frame_rate(video),
        width=width,
        height=height,
    )


def extract_audio(source: Path, destination: Path) -> Path:
    """Write 16 kHz mono PCM audio for transcription and energy analysis."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(
        [
            "-i", str(source),
            "-vn",
            "-ac", "1",
            "-ar", str(ASR_SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(destination),
        ],
        description="extract analysis audio",
    )
    return destination


def ingest(source: Path, work_dir: Path) -> Timeline:
    """Probe ``source`` and extract its analysis audio into ``work_dir``."""
    timeline = probe(source)
    try:
        timeline.audio = extract_audio(source, work_dir / "audio.wav")
    except FFmpegError as error:
        raise UnsupportedMediaError(f"could not decode the audio track: {error}") from error
    log.info(
        "ingested %s: %.1fs, %dx%d @ %.2f fps",
        source.name, timeline.duration, timeline.width, timeline.height, timeline.fps,
    )
    return timeline


def make_preview(source: Path, destination: Path) -> Path:
    """A small, browser-friendly copy of the source, for reviewing cuts.

    The original may be in a container or codec no browser will play, and
    reviewing a proposed cut means hearing it -- so the review UI gets its own
    H.264/AAC proxy rather than the source file.  Deliberately cheap: it is
    scrubbed through a few seconds at a time and never seen at full size.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(
        [
            "-i", str(source),
            "-vf", "scale=-2:360",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
            "-c:a", "aac", "-b:a", "96k",
            "-movflags", "+faststart",
            str(destination),
        ],
        description="preview proxy",
    )
    return destination
