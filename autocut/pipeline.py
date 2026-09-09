"""Stage orchestration.

The pipeline is a straight line, and every stage writes its output into the job
directory before the next one starts.  That is what makes a bad result
diagnosable: the intermediate files say exactly which stage introduced the
problem, and a run can be resumed from any of them.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .analyze import fillers, merge, punchins, retakes, silence
from .asr import get_transcriber, load_words, refine_timings, save_words
from .audio import Envelope
from .config import DEFAULT, Preset
from .ingest import ingest
from .models import RemovalSpan, Timeline, ZoomSpan
from .render import captions as captions_module
from .render import cut, encode, reframe

log = logging.getLogger(__name__)


@dataclass
class Progress:
    stage: str
    percent: int
    message: str


ProgressCallback = Callable[[Progress], None]

#: Stage weights, used to turn stage completion into an overall percentage.
#: Transcription and the final encode dominate; the rest is nearly instant.
_WEIGHTS = {
    "ingest": 4,
    "transcribe": 38,
    "analyze": 6,
    "cut": 16,
    "track": 12,
    "captions": 6,
    "compose": 18,
}


@dataclass
class Result:
    output: Path
    timeline: Timeline
    zooms: list[ZoomSpan] = field(default_factory=list)
    tracked: bool = False
    elapsed: float = 0.0

    @property
    def removed_fraction(self) -> float:
        if not self.timeline.duration:
            return 0.0
        return 1.0 - self.timeline.kept_duration / self.timeline.duration


class _Reporter:
    """Turns per-stage completion into a monotonic overall percentage."""

    def __init__(self, callback: ProgressCallback | None) -> None:
        self.callback = callback
        self.completed = 0

    def __call__(self, stage: str, message: str) -> None:
        percent = min(int(self.completed * 100 / sum(_WEIGHTS.values())), 99)
        log.info("[%3d%%] %s: %s", percent, stage, message)
        if self.callback:
            self.callback(Progress(stage=stage, percent=percent, message=message))

    def finish(self, stage: str) -> None:
        self.completed += _WEIGHTS.get(stage, 0)


def analyze(timeline: Timeline, preset: Preset, envelope: Envelope | None) -> list[RemovalSpan]:
    """Run every detector over the transcript."""
    proposals: list[RemovalSpan] = []
    proposals += silence.detect(timeline.words, timeline.duration, preset.silence, envelope)
    proposals += fillers.detect(timeline.words, preset.filler)
    proposals += retakes.detect(timeline.words, preset.retake)
    return proposals


def run(
    source: Path,
    work_dir: Path,
    preset: Preset = DEFAULT,
    on_progress: ProgressCallback | None = None,
    output_name: str = "final.mp4",
) -> Result:
    """Take a raw video all the way to a finished vertical cut."""
    started = time.monotonic()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report = _Reporter(on_progress)

    report("ingest", "reading the video")
    timeline = ingest(Path(source), work_dir)
    envelope = Envelope.from_wav(timeline.audio) if timeline.audio else None
    report.finish("ingest")

    transcript_path = work_dir / "transcript.json"
    if transcript_path.exists():
        # Transcription is the most expensive stage by far; never redo it for a
        # job directory that already has one.
        report("transcribe", "reusing the existing transcript")
        timeline.words = load_words(transcript_path)
    else:
        report("transcribe", "transcribing speech")
        transcriber = get_transcriber(preset.asr_backend, preset.asr_model)
        words = transcriber.transcribe(timeline.audio, language=preset.language)
        if envelope is not None:
            words = refine_timings(words, envelope)
        timeline.words = words
        save_words(words, transcript_path)
    report.finish("transcribe")
    report("transcribe", f"{len(timeline.words)} words")

    report("analyze", "finding cuts")
    proposals = analyze(timeline, preset, envelope)
    segments, applied = merge.build(proposals, timeline.duration, preset, envelope)
    timeline.keep_segments = segments
    timeline.removals = applied
    (work_dir / "timeline.json").write_text(timeline.model_dump_json(indent=1))
    report.finish("analyze")
    report(
        "analyze",
        f"{len(applied)} cuts, {timeline.duration - timeline.kept_duration:.1f}s removed",
    )

    report("cut", "applying the edit")
    cut_path = cut.apply_cuts(
        Path(timeline.source), work_dir / "cut.mkv", segments,
        timeline.fps, preset.encode, work_dir,
    )
    report.finish("cut")

    time_map = timeline.time_map()
    zooms = punchins.schedule(segments, time_map, preset.punchin)

    report("track", "framing the shot")
    track = reframe.detect_faces(cut_path, preset.reframe) if preset.reframe.enabled else None
    plan = reframe.plan(
        timeline.width, timeline.height, track, zooms, preset.reframe, timeline.fps
    )
    report.finish("track")
    report("track", "face tracked" if track else "centre crop")

    report("captions", "drawing captions")
    caption_track = captions_module.build(
        timeline.words, time_map, preset.caption, preset.reframe.width, work_dir
    )
    report.finish("captions")

    report("compose", "rendering the final video")
    output = encode.compose(
        cut_path, work_dir / output_name, plan, timeline.fps,
        preset.reframe, preset.caption, preset.encode, work_dir,
        captions=caption_track,
    )
    report.finish("compose")

    elapsed = time.monotonic() - started
    report("done", f"finished in {elapsed:.0f}s")
    return Result(
        output=output, timeline=timeline, zooms=zooms,
        tracked=track is not None, elapsed=elapsed,
    )
