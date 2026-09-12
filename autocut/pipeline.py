"""Stage orchestration.

The pipeline is a straight line, and every stage writes its output into the job
directory before the next one starts.  That is what makes a bad result
diagnosable: the intermediate files say exactly which stage introduced the
problem, and a run can be resumed from any of them.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .analyze import fillers, merge, punchins, retakes, silence
from .asr import get_transcriber, load_words, refine_timings, save_words, trim_overlong
from .asr import holes, vad
from .audio import Envelope
from .config import DEFAULT, Preset
from .ffmpeg import FFmpegError
from .ingest import ingest, make_preview
from .models import KeepSegment, RemovalSpan, Timeline, Word, ZoomSpan
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
#:
#: Split by phase, because the two halves are driven separately: proposing an
#: edit runs on upload, and rendering one runs when a person has approved it.
#: Each phase reports 0-100 over its own stages, so a progress bar is
#: meaningful without knowing which phase it belongs to.
_PROPOSE_WEIGHTS = {
    "ingest": 5,
    "transcribe": 78,
    "analyze": 5,
    "preview": 12,
}
_RENDER_WEIGHTS = {
    "cut": 31,
    "track": 23,
    "captions": 11,
    "compose": 35,
}
_WEIGHTS = {**_PROPOSE_WEIGHTS, **_RENDER_WEIGHTS}


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

    def __init__(
        self, callback: ProgressCallback | None, weights: dict[str, int] | None = None
    ) -> None:
        self.callback = callback
        self.weights = weights or _WEIGHTS
        self.total = sum(self.weights.values()) or 1
        self.completed = 0

    def __call__(self, stage: str, message: str) -> None:
        percent = min(int(self.completed * 100 / self.total), 99)
        log.info("[%3d%%] %s: %s", percent, stage, message)
        if self.callback:
            self.callback(Progress(stage=stage, percent=percent, message=message))

    def finish(self, stage: str) -> None:
        self.completed += self.weights.get(stage, 0)


def analyze(timeline: Timeline, preset: Preset, envelope: Envelope | None) -> list[RemovalSpan]:
    """Run every detector over the transcript."""
    proposals: list[RemovalSpan] = []
    proposals += silence.detect(timeline.words, timeline.duration, preset.silence, envelope)
    proposals += fillers.detect(timeline.words, preset.filler)
    proposals += retakes.detect(timeline.words, preset.retake)
    return proposals


EDIT_FILE = "edit.json"


def part_envelope(timeline: Timeline) -> Envelope | None:
    """The analysis envelope for a timeline, or ``None`` if it is not on disk.

    Review and render both split the timeline into parts, and both must place
    the boundaries identically -- what a person listened to has to be what gets
    rendered -- so both go through here rather than loading the waveform their
    own way.  Missing audio is not an error: the boundaries simply stay where
    the word timestamps put them, which is where they were before.
    """
    if timeline.audio is None:
        return None
    path = Path(timeline.audio)
    if not path.exists():
        log.warning("no analysis audio at %s; part boundaries stay unrelaxed", path)
        return None
    try:
        return Envelope.from_wav(path)
    except (OSError, ValueError) as error:
        log.warning("could not read %s (%s); part boundaries stay unrelaxed", path, error)
        return None


def _prepare(
    source: Path, work_dir: Path, preset: Preset, report: _Reporter
) -> tuple[Timeline, Envelope | None]:
    """Ingest and transcribe -- everything needed before cuts can be proposed."""
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
        words = _repair_timings(words, timeline.audio, preset, envelope)
        # Only once the timings are honest can a hole be seen: before the
        # repair, a swallowed retake hides inside the stretched last word of
        # the sentence before it.
        if preset.retranscribe_holes > 0 and envelope is not None:
            report("transcribe", "re-reading stretches with no words")
            words = holes.fill_holes(
                words, timeline.audio, timeline.duration, envelope, transcriber, work_dir,
                language=preset.language,
                min_gap=preset.retranscribe_holes,
                sustained=preset.silence.sustained_speech,
            )
        timeline.words = words
        save_words(words, transcript_path)

    # Applied to a reused transcript too, not just a fresh one: this repairs
    # ASR timestamps, and a cached transcript has exactly the same broken ones.
    # Every repair here only ever shortens a word, so running it twice is a
    # no-op.
    timeline.words = _repair_timings(timeline.words, timeline.audio, preset, envelope)
    report.finish("transcribe")
    report("transcribe", f"{len(timeline.words)} words")
    return timeline, envelope


def _repair_timings(
    words: list[Word],
    audio: Path | None,
    preset: Preset,
    envelope: Envelope | None,
) -> list[Word]:
    """Pull each word back to the part of it that is actually spoken.

    A voice detector does this far better than the waveform can -- measured
    across three recordings, word onsets ran 0.39-0.90s early, well beyond the
    0.12s the envelope-based nudge is able to search -- so it is tried first.
    The envelope remains the fallback, because the detector is optional and a
    render must never fail for want of it.
    """
    if preset.vad.enabled and audio is not None:
        try:
            track = vad.analyse(audio, threshold=preset.vad.threshold)
            return vad.align_to_speech(
                words,
                track,
                max_shift=preset.vad.max_shift,
                trim_longer_than=preset.silence.max_word_duration,
            )
        except vad.VADUnavailable as error:
            log.warning("voice detector unusable (%s); falling back to the waveform", error)

    if envelope is not None:
        return trim_overlong(
            words,
            envelope,
            max_duration=preset.silence.max_word_duration,
            min_gap=preset.silence.min_gap,
        )
    return words


def propose(
    source: Path,
    work_dir: Path,
    preset: Preset = DEFAULT,
    on_progress: ProgressCallback | None = None,
    preview: bool = True,
) -> Timeline:
    """Work out which cuts to offer, and stop there.

    This is the half of the pipeline that runs before anyone has decided
    anything.  It ends with a set of proposed removals written to
    ``timeline.json``, which is the record of what the detectors suggested --
    never overwritten afterwards, so an edit can be reviewed again and revised.

    The expensive stages deliberately sit on the other side of that decision:
    nothing is rendered until a person has said what to cut.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report = _Reporter(on_progress, _PROPOSE_WEIGHTS)

    timeline, envelope = _prepare(Path(source), work_dir, preset, report)

    report("analyze", "finding cuts")
    proposals = analyze(timeline, preset, envelope)
    segments, applied = merge.build(
        proposals, timeline.duration, preset, envelope, timeline.words
    )
    timeline.keep_segments = segments
    timeline.removals = applied
    (work_dir / "timeline.json").write_text(timeline.model_dump_json(indent=1))
    report.finish("analyze")
    report(
        "analyze",
        f"{len(applied)} cuts proposed, {timeline.duration - timeline.kept_duration:.1f}s",
    )

    if preview:
        report("preview", "building the review copy")
        try:
            make_preview(Path(timeline.source), work_dir / "preview.mp4")
        except FFmpegError as error:
            # The review UI degrades to text without it; not worth failing for.
            log.warning("could not build the preview proxy: %s", error)
    report.finish("preview")
    return timeline


def _finish(
    timeline: Timeline,
    work_dir: Path,
    preset: Preset,
    report: _Reporter,
    output_name: str,
    started: float,
) -> Result:
    """Cut, reframe, caption and encode an edit that has been decided."""
    segments = timeline.keep_segments

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


def render_edit(
    work_dir: Path,
    keep: list[int],
    preset: Preset = DEFAULT,
    on_progress: ProgressCallback | None = None,
    output_name: str = "final.mp4",
) -> Result:
    """Render the parts a person chose to keep, indexed into ``merge.parts``.

    ``keep`` names pieces of the timeline to stitch, in order -- what the final
    video is made of, rather than what was taken out of it.  Adjacent pieces
    are joined, so a stretch nobody split stays one segment and the punch-in
    scheduler does not see an edit where there is none.

    The proposal record in ``timeline.json`` is left untouched, so the same job
    can be reviewed again with different answers.
    """
    started = time.monotonic()
    work_dir = Path(work_dir)
    report = _Reporter(on_progress, _RENDER_WEIGHTS)

    timeline = Timeline(**json.loads((work_dir / "timeline.json").read_text()))
    envelope = part_envelope(timeline)
    pieces = merge.parts(
        timeline.removals, timeline.duration, timeline.words,
        min_pause=preset.review_min_pause,
        part_pause=preset.part_pause,
        envelope=envelope,
        relax=preset.boundary_relax,
    )
    chosen = set(keep)
    segments = merge.coalesce(
        [
            KeepSegment(start=start, end=end)
            for index, (start, end, _) in enumerate(pieces)
            if index in chosen
        ]
    )
    if not segments:
        log.error("no parts were kept; falling back to the uncut timeline")
        segments = [KeepSegment(start=0.0, end=timeline.duration)]
    # A person's choice of parts is final; this only ever adds room tone at a
    # join they made, and only where a sentence would otherwise have none.
    segments = merge.coalesce(
        merge.breathe(segments, timeline.words, envelope, preset.silence.sentence_pause)
    )

    timeline.keep_segments = segments
    (work_dir / EDIT_FILE).write_text(
        json.dumps(
            {
                "keep": sorted(chosen),
                "keep_segments": [s.model_dump() for s in segments],
            },
            indent=1,
        )
    )
    log.info(
        "stitching %d of %d parts into %d segments: %.1fs of %.1fs kept",
        len(chosen), len(pieces), len(segments),
        timeline.kept_duration, timeline.duration,
    )
    return _finish(timeline, work_dir, preset, report, output_name, started)


def run(
    source: Path,
    work_dir: Path,
    preset: Preset = DEFAULT,
    on_progress: ProgressCallback | None = None,
    output_name: str = "final.mp4",
) -> Result:
    """Take a raw video all the way to a finished vertical cut, unattended.

    Every proposed cut is applied.  The web app splits this in two so a person
    can decide; the command line renders straight through.
    """
    started = time.monotonic()
    work_dir = Path(work_dir)
    report = _Reporter(on_progress)

    timeline = propose(source, work_dir, preset, on_progress, preview=False)
    report.completed = sum(_PROPOSE_WEIGHTS.values()) - _PROPOSE_WEIGHTS["preview"]
    return _finish(timeline, work_dir, preset, report, output_name, started)
