"""End-to-end run over the synthetic sample.

``samples/talking_head.mp4`` has three kinds of defect planted in it on
purpose -- long pauses, filler words, and a restarted sentence -- and this
asserts the pipeline actually removes each of them, rather than merely
producing a file.

Transcription is seeded from a committed transcript so the run needs no ASR
model and stays deterministic.  ``test_transcription_matches_the_sample``
covers the real recogniser and is opt-in, because it downloads ~1.5 GB.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from autocut.asr.base import normalize
from autocut.config import DEFAULT
from autocut.pipeline import run

SAMPLES = Path(__file__).parent.parent / "samples"
SAMPLE = SAMPLES / "talking_head.mp4"
TRANSCRIPT = SAMPLES / "talking_head.transcript.json"

pytestmark = pytest.mark.skipif(
    not SAMPLE.exists() or shutil.which("ffmpeg") is None,
    reason="needs ffmpeg and samples/talking_head.mp4 (uv run python samples/make_sample.py)",
)


def probe(path: Path) -> dict:
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(output)
    stream = data["streams"][0]
    return {
        "width": stream["width"],
        "height": stream["height"],
        "duration": float(data["format"]["duration"]),
    }


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> tuple[Path, object]:
    work = tmp_path_factory.mktemp("job")
    # Seed the transcript so the run does not need the ASR model.
    shutil.copy(TRANSCRIPT, work / "transcript.json")
    result = run(SAMPLE, work, DEFAULT)
    return work, result


def kept_text(result) -> str:
    """The transcript of what survived the edit, normalised for matching.

    Uses the same "mostly survived" rule as the caption renderer: snapping cut
    points to quiet samples can leave a few milliseconds of a removed word
    behind, and that is not the word surviving.
    """
    time_map = result.timeline.time_map()
    return " ".join(
        normalize(word.text)
        for word in result.timeline.words
        if time_map.kept_fraction(word.start, word.end) >= 0.5
    )


class TestOutput:
    def test_the_file_exists_and_plays(self, rendered) -> None:
        _, result = rendered
        assert result.output.exists()
        assert result.output.stat().st_size > 10_000

    def test_it_is_vertical_1080x1920(self, rendered) -> None:
        _, result = rendered
        info = probe(result.output)
        assert (info["width"], info["height"]) == (1080, 1920)

    def test_it_is_meaningfully_shorter_than_the_source(self, rendered) -> None:
        _, result = rendered
        source = probe(SAMPLE)["duration"]
        output = probe(result.output)["duration"]
        assert output < source * 0.95
        # ...but the edit has not run away with it.
        assert output > source * 0.5

    def test_the_output_duration_matches_the_edit_decision_list(self, rendered) -> None:
        _, result = rendered
        assert probe(result.output)["duration"] == pytest.approx(
            result.timeline.kept_duration, abs=0.4
        )

    def test_intermediates_are_kept_for_debugging(self, rendered) -> None:
        work, _ = rendered
        for name in ("audio.wav", "transcript.json", "timeline.json", "cut.mkv"):
            assert (work / name).exists(), f"{name} should survive the run"


class TestPlantedDefects:
    """Each defect deliberately recorded into the sample must be gone."""

    def test_the_filler_words_are_removed(self, rendered) -> None:
        _, result = rendered
        surviving = kept_text(result).split()
        assert "um" not in surviving
        assert "ah" not in surviving and "uh" not in surviving

    def test_the_restarted_sentence_is_removed(self, rendered) -> None:
        _, result = rendered
        # "The first thing you should..." was said twice; only the complete
        # take should survive.
        assert kept_text(result).count("the first thing you") == 1

    def test_the_long_pauses_are_tightened(self, rendered) -> None:
        _, result = rendered
        marks = json.loads((SAMPLES / "talking_head.marks.json").read_text())
        removed = [(span.start, span.end) for span in result.timeline.removals]
        for defect in marks["defects"]:
            if defect["kind"] != "silence":
                continue
            # The pause sits at the end of the marked line.
            pause_at = defect["end"] - 0.5
            assert any(
                start <= pause_at <= end for start, end in removed
            ), f"the pause after {defect['text']!r} was not cut"

    def test_the_content_survives(self, rendered) -> None:
        _, result = rendered
        surviving = kept_text(result)
        for phrase in (
            "welcome back to the channel",
            "write everything down",
            "batch similar tasks together",
            "thanks for watching",
        ):
            assert phrase in surviving, f"the edit removed real content: {phrase!r}"


class TestFraming:
    def test_the_sample_has_no_face_so_it_falls_back_to_a_centre_crop(self, rendered) -> None:
        _, result = rendered
        assert result.tracked is False

    def test_punch_ins_were_scheduled(self, rendered) -> None:
        _, result = rendered
        assert result.zooms
        assert result.zooms[0].start == pytest.approx(0.0)
        for earlier, later in zip(result.zooms, result.zooms[1:]):
            assert later.start >= earlier.start


@pytest.mark.slow
def test_transcription_matches_the_sample(tmp_path) -> None:
    """The real recogniser, opt-in: `uv run pytest -m slow`.

    Downloads the ASR model on first run.
    """
    from autocut.asr import get_transcriber
    from autocut.ingest import ingest

    timeline = ingest(SAMPLE, tmp_path)
    transcriber = get_transcriber(DEFAULT.asr_backend, DEFAULT.asr_model)
    words = transcriber.transcribe(timeline.audio, language=DEFAULT.language)

    text = " ".join(normalize(word.text) for word in words)
    assert "welcome back to the channel" in text
    assert "write everything down" in text
    assert all(word.end > word.start for word in words)
    assert all(a.start <= b.start for a, b in zip(words, words[1:]))
