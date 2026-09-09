"""The time map is the spine of the pipeline -- if it drifts, captions drift.

The timeline under test keeps three windows out of a 20s source:

    source:  [0 ---- 5]  xxxx  [8 ---- 12]  xxxx  [15 ---- 20]
    output:  [0 ---- 5]        [5 ----  9]        [9 ---- 14]
"""

from __future__ import annotations

import pytest

from autocut.models import KeepSegment, TimeMap

SEGMENTS = [
    KeepSegment(start=0.0, end=5.0),
    KeepSegment(start=8.0, end=12.0),
    KeepSegment(start=15.0, end=20.0),
]


@pytest.fixture
def tm() -> TimeMap:
    return TimeMap(SEGMENTS)


def test_output_duration_is_sum_of_kept(tm: TimeMap) -> None:
    assert tm.output_duration == pytest.approx(14.0)


@pytest.mark.parametrize(
    ("source", "output"),
    [
        (0.0, 0.0),
        (2.5, 2.5),
        (5.0, 5.0),    # end of first segment
        (8.0, 5.0),    # start of second segment lands on the same cut point
        (10.0, 7.0),
        (12.0, 9.0),
        (15.0, 9.0),
        (17.5, 11.5),
        (20.0, 14.0),
    ],
)
def test_to_output_at_and_between_boundaries(tm: TimeMap, source: float, output: float) -> None:
    assert tm.to_output(source) == pytest.approx(output)


def test_times_inside_a_removed_span_collapse_onto_the_cut(tm: TimeMap) -> None:
    for t in (5.5, 6.0, 7.9):
        assert tm.to_output(t) == pytest.approx(5.0)
    for t in (12.1, 14.0, 14.999):
        assert tm.to_output(t) == pytest.approx(9.0)


def test_to_output_is_monotonic_across_the_whole_source(tm: TimeMap) -> None:
    previous = -1.0
    t = 0.0
    while t <= 20.0:
        current = tm.to_output(t)
        assert current >= previous
        previous = current
        t += 0.05


def test_out_of_range_times_clamp(tm: TimeMap) -> None:
    assert tm.to_output(-3.0) == pytest.approx(0.0)
    assert tm.to_output(99.0) == pytest.approx(14.0)
    assert tm.to_source(-3.0) == pytest.approx(0.0)
    assert tm.to_source(99.0) == pytest.approx(20.0)


@pytest.mark.parametrize("source", [0.0, 1.0, 4.99, 8.0, 9.5, 11.99, 15.0, 19.0, 20.0])
def test_round_trip_through_kept_time_is_lossless(tm: TimeMap, source: float) -> None:
    assert tm.to_source(tm.to_output(source)) == pytest.approx(source)


def test_a_cut_point_resolves_to_the_incoming_segment(tm: TimeMap) -> None:
    """Every interior boundary is genuinely ambiguous: output time 9.0 is both
    the end of segment 2 (source 12.0) and the start of segment 3 (source 15.0).
    We resolve it to the incoming segment, because that is the frame actually on
    screen at that instant -- which is what the reframe stage needs."""
    assert tm.to_source(9.0) == pytest.approx(15.0)
    assert tm.to_source(9.0 - 1e-6) == pytest.approx(12.0, abs=1e-5)


def test_is_kept(tm: TimeMap) -> None:
    assert tm.is_kept(0.0) and tm.is_kept(4.0) and tm.is_kept(5.0)
    assert not tm.is_kept(5.5) and not tm.is_kept(13.0)
    assert tm.is_kept(8.0) and tm.is_kept(19.9)


def test_map_span_inside_one_segment(tm: TimeMap) -> None:
    assert tm.map_span(9.0, 10.0) == pytest.approx((6.0, 7.0))


def test_map_span_returns_none_for_a_fully_removed_word(tm: TimeMap) -> None:
    assert tm.map_span(6.0, 7.0) is None


def test_map_span_clips_a_word_straddling_a_cut(tm: TimeMap) -> None:
    # Mostly inside the first segment: clipped to it, not stretched across.
    assert tm.map_span(4.5, 6.5) == pytest.approx((4.5, 5.0))
    # Mostly inside the second: clipped to the second segment's start.
    assert tm.map_span(7.5, 9.0) == pytest.approx((5.0, 6.0))


def test_identity_map_is_a_no_op() -> None:
    tm = TimeMap.identity(30.0)
    assert tm.output_duration == pytest.approx(30.0)
    for t in (0.0, 12.34, 30.0):
        assert tm.to_output(t) == pytest.approx(t)


def test_empty_map_is_safe() -> None:
    tm = TimeMap([])
    assert tm.output_duration == 0.0
    assert tm.to_output(5.0) == 0.0
    assert tm.to_source(5.0) == 0.0
    assert tm.map_span(1.0, 2.0) is None
    assert not tm.is_kept(0.0)


def test_overlapping_segments_are_rejected() -> None:
    with pytest.raises(ValueError):
        TimeMap([KeepSegment(start=0.0, end=5.0), KeepSegment(start=4.0, end=8.0)])
