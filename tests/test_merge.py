"""Reconciling detector output into an edit decision list."""

from __future__ import annotations

import pytest

import numpy as np

from autocut.analyze import merge
from autocut.audio import FRAME_SECONDS, Envelope
from autocut.config import Preset
from autocut.models import KeepSegment, Reason, RemovalSpan, Word


def span(start: float, end: float, reason: Reason = Reason.SILENCE) -> RemovalSpan:
    return RemovalSpan(start=start, end=end, reason=reason)


class TestUnion:
    def test_disjoint_spans_are_left_alone(self) -> None:
        merged = merge.union([span(0, 1), span(3, 4)])
        assert [(s.start, s.end) for s in merged] == [(0, 1), (3, 4)]

    def test_overlapping_spans_collapse(self) -> None:
        merged = merge.union([span(0, 2), span(1, 3)])
        assert [(s.start, s.end) for s in merged] == [(0, 3)]

    def test_touching_spans_collapse(self) -> None:
        merged = merge.union([span(0, 2), span(2, 4)])
        assert [(s.start, s.end) for s in merged] == [(0, 4)]

    def test_nested_span_does_not_shrink_its_container(self) -> None:
        merged = merge.union([span(0, 10), span(3, 4)])
        assert [(s.start, s.end) for s in merged] == [(0, 10)]

    def test_unsorted_input_is_handled(self) -> None:
        merged = merge.union([span(5, 6), span(0, 1), span(5.5, 7)])
        assert [(s.start, s.end) for s in merged] == [(0, 1), (5, 7)]

    def test_merged_span_reports_its_longest_contributor(self) -> None:
        merged = merge.union([span(0, 1, Reason.FILLER), span(0.5, 5, Reason.RETAKE)])
        assert merged[0].reason is Reason.RETAKE

    def test_empty_input(self) -> None:
        assert merge.union([]) == []


class TestInvert:
    def test_gaps_between_removals_become_keep_segments(self) -> None:
        segments = merge.invert([span(2, 4), span(6, 8)], duration=10)
        assert [(s.start, s.end) for s in segments] == [(0, 2), (4, 6), (8, 10)]

    def test_removal_at_the_head_and_tail(self) -> None:
        segments = merge.invert([span(0, 2), span(8, 10)], duration=10)
        assert [(s.start, s.end) for s in segments] == [(2, 8)]

    def test_no_removals_keeps_everything(self) -> None:
        segments = merge.invert([], duration=10)
        assert [(s.start, s.end) for s in segments] == [(0, 10)]

    def test_removals_are_clamped_to_the_duration(self) -> None:
        segments = merge.invert([span(8, 99)], duration=10)
        assert [(s.start, s.end) for s in segments] == [(0, 8)]

    def test_total_duration_is_conserved(self) -> None:
        removals = merge.union([span(1, 2), span(4, 7)])
        segments = merge.invert(removals, duration=10)
        removed = sum(s.duration for s in removals)
        kept = sum(s.duration for s in segments)
        assert kept + removed == pytest.approx(10)


class TestBuild:
    def test_glitch_length_segments_are_dropped(self) -> None:
        preset = Preset()
        # The 0.1s island between the two removals is too short to read.
        segments, _ = merge.build([span(1, 2), span(2.1, 4)], duration=10, preset=preset)
        assert all(s.duration >= preset.min_segment for s in segments)
        assert [(s.start, s.end) for s in segments] == [(0, 1), (4, 10)]

    def test_an_over_budget_edit_drops_the_least_trusted_detector(self) -> None:
        preset = Preset()
        removals = [
            span(0, 1, Reason.SILENCE),
            span(2, 8, Reason.RETAKE),   # 60% of the video on its own
        ]
        segments, applied = merge.build(removals, duration=10, preset=preset)
        assert all(s.reason is not Reason.RETAKE for s in applied)
        kept = sum(s.duration for s in segments)
        assert kept / 10 >= 1 - preset.max_total_removal_ratio

    def test_a_runaway_edit_falls_back_to_the_uncut_timeline(self) -> None:
        # Silence is never sacrificed, so an absurd silence span survives the
        # budget loop and must be caught by the final guard.
        segments, applied = merge.build(
            [span(0, 9.99, Reason.SILENCE)], duration=10, preset=Preset()
        )
        assert [(s.start, s.end) for s in segments] == [(0, 10)]
        assert applied == []

    def test_ordinary_edits_pass_through_untouched(self) -> None:
        segments, applied = merge.build(
            [span(2, 3, Reason.FILLER)], duration=10, preset=Preset()
        )
        assert [(s.start, s.end) for s in segments] == [(0, 2), (3, 10)]
        assert len(applied) == 1


SPEECH_DB = -20.0
SILENT_DB = -60.0


def envelope(duration: float, loud: list[tuple[float, float]]) -> Envelope:
    """An envelope that is silent everywhere except ``loud``."""
    db = np.full(int(duration / FRAME_SECONDS), SILENT_DB, dtype=np.float32)
    for start, end in loud:
        db[int(start / FRAME_SECONDS) : int(end / FRAME_SECONDS)] = SPEECH_DB
    return Envelope(db)


def speech_with_dip(dip_db: float = -35.0) -> Envelope:
    """Ten seconds of continuous speech with one quieter instant in it, over a
    realistic noise floor.

    The floor matters: with speech alone, the dip is the quietest thing in the
    file and every derived level collapses onto it.
    """
    speech = np.full(1000, SPEECH_DB, dtype=np.float32)
    speech[500:503] = dip_db
    room = np.full(400, SILENT_DB, dtype=np.float32)
    return Envelope(np.concatenate([speech, room]))


class TestBoundaryRelaxation:
    """Cut boundaries must land on silence, not on the edge of a word.

    A word is audible slightly before the timestamp it starts at and rings on
    after the one it ends at.  A cut placed exactly on a timestamp therefore
    truncates a decay or clips an onset, and that is what an abrupt edit
    sounds like.
    """

    preset = Preset()

    def test_a_boundary_on_a_ringing_tail_is_pushed_past_it(self) -> None:
        # Speech to 2.1s, but the removal starts at 2.0 -- mid-decay.
        env = envelope(10.0, [(0.0, 2.1)])
        relaxed = merge._snap([span(2.0, 5.0)], env, self.preset.boundary_relax)
        assert relaxed[0].start >= 2.1 - 0.02, "the tail should survive the cut"
        assert relaxed[0].end == pytest.approx(5.0, abs=0.08)

    def test_a_boundary_on_an_onset_is_pulled_back_off_it(self) -> None:
        # The next word is audible from 4.9s, but the removal ends at 5.0.
        env = envelope(10.0, [(4.9, 8.0)])
        relaxed = merge._snap([span(2.0, 5.0)], env, self.preset.boundary_relax)
        assert relaxed[0].end <= 4.9 + 0.02, "the onset should survive the cut"

    def test_relaxing_only_ever_shrinks_a_removal(self) -> None:
        env = envelope(10.0, [(0.0, 2.1), (4.9, 8.0)])
        relaxed = merge._snap([span(2.0, 5.0)], env, self.preset.boundary_relax)
        assert relaxed[0].start >= 2.0 - 0.08
        assert relaxed[0].end <= 5.0 + 0.08

    def test_a_boundary_already_in_silence_does_not_move(self) -> None:
        env = envelope(10.0, [(0.0, 1.0), (6.0, 8.0)])
        relaxed = merge._snap([span(2.0, 5.0)], env, self.preset.boundary_relax)
        # Only the existing snap-to-quietest nudge applies, within its radius.
        assert relaxed[0].start == pytest.approx(2.0, abs=0.08)
        assert relaxed[0].end == pytest.approx(5.0, abs=0.08)

    def test_a_removal_buried_in_speech_is_not_inverted(self) -> None:
        """Relaxing both ends of a span that is loud throughout would collapse
        it; the span must come back intact instead."""
        env = envelope(10.0, [(0.0, 10.0)])
        relaxed = merge._snap([span(2.0, 2.4)], env, self.preset.boundary_relax)
        assert relaxed[0].start < relaxed[0].end

    def test_snapping_never_lands_inside_a_word(self) -> None:
        """The dip between two syllables is a local minimum but is still
        speech, so it is not a legal place to cut."""
        env = speech_with_dip()  # a quieter instant, still far above silence
        assert env.quietest_time_near(5.03, max_level=env.silence_threshold) == pytest.approx(5.03)
        # Without the guard it would happily move onto that dip.
        assert env.quietest_time_near(5.03) == pytest.approx(5.0, abs=0.02)


class TestWordlessSegments:
    """What survives between two adjacent cuts is often not speech at all."""

    preset = Preset()

    def test_a_short_segment_with_no_word_is_dropped(self) -> None:
        segments = [
            KeepSegment(start=0.0, end=5.0),
            KeepSegment(start=6.0, end=6.4),  # debris between two cuts
            KeepSegment(start=8.0, end=10.0),
        ]
        words = [Word(text="hello", start=1.0, end=1.4), Word(text="there", start=8.2, end=8.6)]
        kept = merge._drop_glitches(segments, self.preset, words)
        assert [(s.start, s.end) for s in kept] == [(0.0, 5.0), (8.0, 10.0)]

    def test_a_long_wordless_segment_survives(self) -> None:
        """Speech the recogniser dropped entirely has no word over it, and is
        exactly what must not be thrown away."""
        segments = [KeepSegment(start=0.0, end=5.0), KeepSegment(start=6.0, end=8.5)]
        words = [Word(text="hello", start=1.0, end=1.4)]
        kept = merge._drop_glitches(segments, self.preset, words)
        assert len(kept) == 2

    def test_a_segment_holding_a_whole_word_is_kept_however_short(self) -> None:
        segments = [KeepSegment(start=0.0, end=0.4)]
        words = [Word(text="hi", start=0.05, end=0.35)]
        assert merge._drop_glitches(segments, self.preset, words) == segments

    def test_a_clipped_word_does_not_count_as_content(self) -> None:
        """The tail of a word the cut was meant to remove is debris, not a
        reason to keep the fragment it landed in."""
        segments = [KeepSegment(start=1.0, end=1.5)]
        words = [Word(text="completely", start=0.4, end=1.06)]
        assert merge._drop_glitches(segments, self.preset, words) == []

    def test_without_words_the_duration_rule_still_applies(self) -> None:
        segments = [KeepSegment(start=0.0, end=0.1), KeepSegment(start=1.0, end=5.0)]
        kept = merge._drop_glitches(segments, self.preset, None)
        assert [(s.start, s.end) for s in kept] == [(1.0, 5.0)]
