"""Caption timing and layout.

The visual result has to be judged by eye, but the timing must not be: a
caption that drifts, runs backwards, or references a word that was cut is a bug
these tests can catch.
"""

from __future__ import annotations

import pytest

from autocut.config import CaptionConfig
from autocut.models import KeepSegment, TimeMap, Word
from autocut.render import captions


def words(*specs: tuple[str, float, float]) -> list[Word]:
    return [Word(text=text, start=start, end=end) for text, start, end in specs]


class TestToOutputWords:
    def test_timings_are_mapped_through_the_cut(self) -> None:
        time_map = TimeMap([KeepSegment(start=0, end=2), KeepSegment(start=5, end=7)])
        placed = captions.to_output_words(
            words(("hello", 0.5, 1.0), ("world", 5.5, 6.0)), time_map, CaptionConfig()
        )
        assert [(w.text, w.start, w.end) for w in placed] == [
            ("hello", pytest.approx(0.5), pytest.approx(1.0)),
            ("world", pytest.approx(2.5), pytest.approx(3.0)),
        ]

    def test_words_that_were_cut_are_dropped(self) -> None:
        time_map = TimeMap([KeepSegment(start=0, end=2), KeepSegment(start=5, end=7)])
        placed = captions.to_output_words(
            words(("keep", 0.5, 1.0), ("um", 3.0, 3.4), ("keep", 5.5, 6.0)),
            time_map, CaptionConfig(),
        )
        assert [w.text for w in placed] == ["keep", "keep"]

    def test_the_highlight_never_runs_backwards(self) -> None:
        # Two words either side of a cut land on the same output instant.
        time_map = TimeMap([KeepSegment(start=0, end=2), KeepSegment(start=5, end=7)])
        placed = captions.to_output_words(
            words(("before", 1.8, 2.0), ("after", 5.0, 5.2)), time_map, CaptionConfig()
        )
        for earlier, later in zip(placed, placed[1:]):
            assert later.start >= earlier.end
        assert all(w.end > w.start for w in placed)

    def test_the_offset_shifts_every_caption(self) -> None:
        time_map = TimeMap.identity(10.0)
        placed = captions.to_output_words(
            words(("hi", 1.0, 1.5)), time_map, CaptionConfig(offset=0.2)
        )
        assert placed[0].start == pytest.approx(1.2)
        assert placed[0].end == pytest.approx(1.7)


class TestGroupLines:
    def config(self, **overrides) -> CaptionConfig:
        return CaptionConfig(**overrides)

    def test_breaks_at_the_end_of_a_sentence(self) -> None:
        placed = captions.to_output_words(
            words(("Hi.", 0, 0.3), ("Next", 0.35, 0.6)), TimeMap.identity(5), self.config()
        )
        lines = captions.group_lines(placed, self.config())
        assert [[w.text for w in line.words] for line in lines] == [["Hi."], ["Next"]]

    def test_breaks_on_a_long_pause(self) -> None:
        placed = captions.to_output_words(
            words(("one", 0, 0.3), ("two", 1.5, 1.8)), TimeMap.identity(5), self.config()
        )
        lines = captions.group_lines(placed, self.config(line_break_gap=0.4))
        assert len(lines) == 2

    def test_breaks_at_the_word_limit(self) -> None:
        specs = [(f"w{i}", i * 0.3, i * 0.3 + 0.25) for i in range(9)]
        placed = captions.to_output_words(words(*specs), TimeMap.identity(5), self.config())
        lines = captions.group_lines(placed, self.config(max_words_per_line=3))
        assert all(len(line.words) <= 3 for line in lines)

    def test_breaks_at_the_character_limit(self) -> None:
        specs = [("elephantine", 0, 0.4), ("magnificent", 0.4, 0.8)]
        placed = captions.to_output_words(words(*specs), TimeMap.identity(5), self.config())
        lines = captions.group_lines(placed, self.config(max_chars_per_line=14))
        assert len(lines) == 2

    def test_every_word_appears_exactly_once(self) -> None:
        specs = [(f"w{i}", i * 0.3, i * 0.3 + 0.25) for i in range(20)]
        placed = captions.to_output_words(words(*specs), TimeMap.identity(10), self.config())
        lines = captions.group_lines(placed, self.config())
        flattened = [w.text for line in lines for w in line.words]
        assert flattened == [f"w{i}" for i in range(20)]

    def test_no_empty_lines(self) -> None:
        placed = captions.to_output_words(words(("a.", 0, 0.2)), TimeMap.identity(5), self.config())
        assert all(line.words for line in captions.group_lines(placed, self.config()))


class TestBuild:
    def test_renders_a_track_whose_states_run_in_order(self, tmp_path) -> None:
        specs = [("Hello,", 0.2, 0.6), ("this", 0.7, 0.9), ("works.", 0.95, 1.4)]
        time_map = TimeMap.identity(2.0)
        result = captions.build(words(*specs), time_map, CaptionConfig(), 1080, tmp_path)
        assert result is not None
        listing, band_height = result
        assert band_height > 0

        text = listing.read_text()
        durations = [
            float(line.split()[1]) for line in text.splitlines() if line.startswith("duration")
        ]
        assert durations, "expected at least one caption state"
        assert all(d > 0 for d in durations), "a caption state must have positive duration"
        # The track starts at t=0 and runs to the last word without gaps in the
        # timeline, so the overlay stays in sync with the video.
        assert sum(durations) == pytest.approx(1.4, abs=0.05)

        files = [line.split("'")[1] for line in text.splitlines() if line.startswith("file")]
        assert all(f.endswith(".png") for f in files)

    def test_a_leading_gap_is_filled_with_a_blank(self, tmp_path) -> None:
        result = captions.build(
            words(("late", 1.0, 1.4)), TimeMap.identity(2.0), CaptionConfig(), 1080, tmp_path
        )
        assert result is not None
        text = result[0].read_text()
        assert "blank.png" in text

    def test_disabled_captions_render_nothing(self, tmp_path) -> None:
        result = captions.build(
            words(("hi", 0, 0.4)), TimeMap.identity(2), CaptionConfig(enabled=False), 1080, tmp_path
        )
        assert result is None

    def test_a_fully_cut_transcript_renders_nothing(self, tmp_path) -> None:
        time_map = TimeMap([KeepSegment(start=5, end=6)])
        result = captions.build(
            words(("gone", 0.0, 0.4)), time_map, CaptionConfig(), 1080, tmp_path
        )
        assert result is None


def test_a_font_is_always_found() -> None:
    assert captions.find_font("Arial Black").exists()
    # An unknown family falls back rather than failing the render.
    assert captions.find_font("No Such Font At All").exists()
