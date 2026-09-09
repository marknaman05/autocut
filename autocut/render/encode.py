"""Stage 7 -- the final composite and encode.

Reframing, caption compositing and loudness normalisation all happen in a
single ffmpeg pass over the cut intermediate.  Doing them separately would mean
two extra decode/encode round trips and the generation loss that comes with
them.

Loudness is measured first and then applied with those measurements, rather
than normalised on the fly.  The one-pass version reacts to the audio as it
goes, which pumps the level on speech with long gaps -- exactly the material
this pipeline produces.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..config import CaptionConfig, EncodeConfig, ReframeConfig
from ..ffmpeg import FFmpegError, ffmpeg, filter_script_args, run
from .reframe import ReframePlan

log = logging.getLogger(__name__)

_TARGET_TRUE_PEAK = -1.5
_TARGET_LRA = 11.0


def measure_loudness(source: Path, config: EncodeConfig) -> dict[str, str] | None:
    """First loudnorm pass: measure the programme loudness of ``source``."""
    filter_spec = (
        f"loudnorm=I={config.loudness_lufs}:TP={_TARGET_TRUE_PEAK}:LRA={_TARGET_LRA}:"
        "print_format=json"
    )
    try:
        result = run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", str(source),
             "-af", filter_spec, "-f", "null", "-"],
            description="measure loudness",
            # loudnorm prints its JSON report to stderr, not stdout.
            include_stderr=True,
        )
    except FFmpegError as error:
        # ffmpeg writes the JSON to stderr, so a non-zero exit is not the only
        # way this can fail to give us numbers.  Normalising is a nicety; never
        # fail a render over it.
        log.warning("loudness measurement failed (%s); using single-pass normalisation", error)
        return None

    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", result, re.DOTALL)
    if not match:
        log.warning("could not parse loudnorm measurements; using single-pass normalisation")
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _loudnorm_filter(config: EncodeConfig, measured: dict[str, str] | None) -> str:
    base = f"loudnorm=I={config.loudness_lufs}:TP={_TARGET_TRUE_PEAK}:LRA={_TARGET_LRA}"
    if not measured:
        return base
    return (
        f"{base}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
        f"offset={measured.get('target_offset', '0.0')}:linear=true"
    )


def compose(
    source: Path,
    destination: Path,
    plan: ReframePlan,
    fps: float,
    reframe_config: ReframeConfig,
    caption_config: CaptionConfig,
    encode_config: EncodeConfig,
    work_dir: Path,
    captions: tuple[Path, int] | None = None,
) -> Path:
    """Reframe, caption, normalise and encode ``source`` into ``destination``."""
    width, height = reframe_config.width, reframe_config.height
    measured = measure_loudness(source, encode_config)

    inputs = ["-i", str(source)]
    video_chain = f"[0:v]{plan.filter_string(width, height, fps)}"

    if captions is not None:
        listing, band_height = captions
        overlay_y = max(height - caption_config.margin_v - band_height, 0)
        inputs += ["-f", "concat", "-safe", "0", "-i", str(listing)]
        graph = (
            f"{video_chain}[framed];"
            f"[1:v]fps={fps:.6f},format=rgba[caps];"
            # eof_action=pass so the video keeps playing past the last caption.
            f"[framed][caps]overlay=x=0:y={overlay_y}:eof_action=pass:format=auto[v];"
            f"[0:a]{_loudnorm_filter(encode_config, measured)}[a]"
        )
    else:
        graph = f"{video_chain}[v];[0:a]{_loudnorm_filter(encode_config, measured)}[a]"

    script = work_dir / "compose.filter"
    script.write_text(graph)

    ffmpeg(
        [
            *inputs,
            *filter_script_args(script),
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264",
            "-preset", encode_config.preset,
            "-crf", str(encode_config.crf),
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-r", f"{fps:.6f}",
            "-c:a", "aac", "-b:a", encode_config.audio_bitrate, "-ar", "48000",
            "-movflags", "+faststart",
            str(destination),
        ],
        description="compose final video",
    )
    return destination
