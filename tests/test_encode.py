"""Where the caption band lands in the frame."""

from __future__ import annotations

import pytest

from autocut.config import CaptionConfig
from autocut.render.encode import caption_overlay_y


class TestCaptionOverlayY:
    def test_bottom_is_margin_above_the_bottom_edge(self) -> None:
        assert caption_overlay_y(1920, 200, CaptionConfig(position="bottom", margin_v=420)) == 1300

    def test_top_is_margin_below_the_top_edge(self) -> None:
        assert caption_overlay_y(1920, 200, CaptionConfig(position="top", margin_v=420)) == 420

    def test_centre_ignores_the_margin(self) -> None:
        assert caption_overlay_y(1920, 200, CaptionConfig(position="centre", margin_v=420)) == 860

    def test_a_band_taller_than_the_margin_moves_inward(self) -> None:
        assert caption_overlay_y(1920, 1600, CaptionConfig(position="bottom", margin_v=420)) == 0
        assert caption_overlay_y(1920, 1600, CaptionConfig(position="top", margin_v=420)) == 320

    def test_a_band_taller_than_the_frame_starts_at_the_top(self) -> None:
        for position in ("bottom", "centre", "top"):
            assert caption_overlay_y(1920, 2500, CaptionConfig(position=position)) == 0

    def test_an_unknown_position_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="sideways"):
            caption_overlay_y(1920, 200, CaptionConfig(position="sideways"))
