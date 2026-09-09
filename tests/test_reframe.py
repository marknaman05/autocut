"""Framing maths.

Face detection itself needs a real face and is verified by running the
pipeline; what is tested here is everything that turns a track into ffmpeg
expressions -- where the bugs are silent and show up as a drifting or
distorted picture rather than an error.
"""

from __future__ import annotations

import pytest

from autocut.config import PunchInConfig, ReframeConfig
from autocut.models import KeepSegment, TimeMap, ZoomSpan
from autocut.render import reframe


def track(points: list[tuple[float, float, float]]) -> reframe.FaceTrack:
    return reframe.FaceTrack(
        times=[t for t, _, _ in points],
        x=[x for _, x, _ in points],
        y=[y for _, _, y in points],
        detection_rate=1.0,
    )


class TestSmooth:
    def test_movement_inside_the_dead_zone_is_ignored(self) -> None:
        config = ReframeConfig(dead_zone=0.1)
        jittery = track([(i * 0.25, 0.5 + (0.02 if i % 2 else -0.02), 0.4) for i in range(20)])
        smoothed = reframe.smooth(jittery, config)
        assert len(set(smoothed.x)) == 1, "small jitter must not move the framing at all"

    def test_large_movement_is_followed_slowly(self) -> None:
        config = ReframeConfig(dead_zone=0.02, smoothing=0.1)
        moving = track([(0.0, 0.2, 0.4)] + [(i * 0.25, 0.8, 0.4) for i in range(1, 40)])
        smoothed = reframe.smooth(moving, config)
        assert smoothed.x[0] == pytest.approx(0.2)
        # It gets there, but never in one step.
        assert smoothed.x[1] - smoothed.x[0] < 0.1
        assert smoothed.x[-1] > 0.7
        assert all(b >= a for a, b in zip(smoothed.x, smoothed.x[1:]))


class TestSimplify:
    def test_a_static_track_collapses_to_its_endpoints(self) -> None:
        simplified = reframe.simplify(track([(i * 0.25, 0.5, 0.4) for i in range(100)]))
        assert len(simplified) == 2

    def test_the_keyframe_limit_is_respected(self) -> None:
        # A track that moves on every sample would otherwise produce hundreds
        # of keyframes, and an ffmpeg expression too large to be practical.
        wandering = track([(i * 0.25, i / 400, 0.4 + i / 800) for i in range(400)])
        simplified = reframe.simplify(wandering, limit=20)
        assert len(simplified) <= 20
        assert simplified.times[0] == pytest.approx(0.0)
        assert simplified.times[-1] == pytest.approx(wandering.times[-1])


class TestPiecewise:
    def test_a_step_expression_holds_each_value(self) -> None:
        expression = reframe._piecewise([(0.0, 1.0), (5.0, 1.2)], "t", step=True)
        assert "if(lt(t,5.00000),1.00000,1.20000)" == expression

    def test_a_linear_expression_interpolates(self) -> None:
        expression = reframe._piecewise([(0.0, 0.0), (2.0, 1.0)], "t", step=False)
        assert "0.50000" in expression, "expected a slope of 0.5 per second"

    def test_a_single_keyframe_is_a_constant(self) -> None:
        assert reframe._piecewise([(0.0, 0.42)], "t", step=False) == "0.42000"

    def test_no_keyframes_is_safe(self) -> None:
        assert reframe._piecewise([], "t", step=False) == "0"


class TestPlan:
    config = ReframeConfig()

    def test_a_landscape_source_is_scaled_to_cover_the_output(self) -> None:
        plan = reframe.plan(1920, 1080, None, [], self.config, 30.0)
        # Scaled by height, so the 9:16 window fits inside the frame.
        assert plan.prescale_height == 1920
        assert plan.prescale_width >= self.config.width

    def test_a_narrow_source_is_scaled_to_cover_the_output(self) -> None:
        plan = reframe.plan(720, 1600, None, [], self.config, 30.0)
        assert plan.prescale_width >= self.config.width
        assert plan.prescale_height >= self.config.height

    def test_prescale_dimensions_are_even(self) -> None:
        # Odd dimensions break yuv420p encoding.
        for width, height in ((1280, 720), (1920, 1080), (1440, 1080), (1234, 567)):
            plan = reframe.plan(width, height, None, [], self.config, 30.0)
            assert plan.prescale_width % 2 == 0
            assert plan.prescale_height % 2 == 0

    def test_the_zoom_filter_is_skipped_when_nothing_zooms(self) -> None:
        zooms = [ZoomSpan(start=0, end=10, zoom=1.0)]
        plan = reframe.plan(1280, 720, None, zooms, self.config, 30.0)
        assert plan.static_zoom
        assert "zoompan" not in plan.filter_string(1080, 1920, 30.0)

    def test_the_zoom_filter_is_used_when_something_zooms(self) -> None:
        zooms = [ZoomSpan(start=0, end=5, zoom=1.0), ZoomSpan(start=5, end=10, zoom=1.12)]
        plan = reframe.plan(1280, 720, None, zooms, self.config, 30.0)
        assert not plan.static_zoom
        assert "zoompan" in plan.filter_string(1080, 1920, 30.0)

    def test_the_crop_is_always_the_full_output_size(self) -> None:
        """The crop establishes the aspect ratio, so it must never be squeezed
        -- that was a real bug: zoompan crops at the *input's* aspect ratio and
        silently distorted the picture."""
        plan = reframe.plan(1280, 720, None, [], self.config, 30.0)
        assert "crop=w=1080:h=1920" in plan.filter_string(1080, 1920, 30.0)

    def test_an_untracked_plan_is_centred(self) -> None:
        plan = reframe.plan(1280, 720, None, [], self.config, 30.0)
        assert "0.5" in plan.x

    def test_a_tracked_plan_follows_the_face(self) -> None:
        moving = track([(0.0, 0.3, 0.4), (2.0, 0.7, 0.4), (4.0, 0.7, 0.4)])
        plan = reframe.plan(1280, 720, moving, [], self.config, 30.0)
        assert "t" in plan.x and "if(" in plan.x

    def test_the_window_is_clamped_inside_the_frame(self) -> None:
        plan = reframe.plan(1280, 720, None, [], self.config, 30.0)
        for expression in (plan.x, plan.y):
            assert expression.startswith("max(0")
            assert "min(" in expression


def test_punchins_only_change_on_a_cut() -> None:
    from autocut.analyze import punchins

    segments = [
        KeepSegment(start=0, end=4),
        KeepSegment(start=6, end=10),
        KeepSegment(start=12, end=16),
    ]
    spans = punchins.schedule(segments, TimeMap(segments), PunchInConfig(min_hold=2.5))
    assert spans
    # Every hold boundary lands on a segment boundary in output time.
    boundaries = {0.0, 4.0, 8.0, 12.0}
    for span in spans:
        assert span.start in boundaries
    assert spans[-1].end == pytest.approx(12.0)


def test_short_segments_do_not_restart_the_zoom() -> None:
    from autocut.analyze import punchins

    # A run of very short shots must not make the framing strobe.
    segments = [KeepSegment(start=i * 1.0, end=i * 1.0 + 0.5) for i in range(10)]
    spans = punchins.schedule(
        segments, TimeMap(segments), PunchInConfig(min_hold=2.5, min_segment=1.2)
    )
    assert len(spans) == 1, "no shot was long enough to justify a punch-in"
