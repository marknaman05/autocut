"""Reconciling detector output into an edit decision list."""

from __future__ import annotations

import pytest

from autocut.analyze import merge
from autocut.config import Preset
from autocut.models import Reason, RemovalSpan


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
