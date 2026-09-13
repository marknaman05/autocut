"""Composing a preset with a caption style."""

from __future__ import annotations

from dataclasses import replace

import pytest

from autocut.config import CAPTION_STYLE_LABELS, CAPTION_STYLES, DEFAULT, CaptionConfig


class TestWithCaptionStyle:
    def test_takes_the_styles_look(self) -> None:
        preset = DEFAULT.with_caption_style("hormozi")
        assert preset.caption.font == CAPTION_STYLES["hormozi"].font
        assert preset.caption.uppercase

    def test_keeps_the_presets_timing_and_review_fields(self) -> None:
        tuned = replace(
            DEFAULT,
            caption=replace(DEFAULT.caption, offset=0.25, review_confidence=0.8,
                            min_visible_fraction=0.9, enabled=False),
        )
        styled = tuned.with_caption_style("neon")
        assert styled.caption.offset == 0.25
        assert styled.caption.review_confidence == 0.8
        assert styled.caption.min_visible_fraction == 0.9
        assert styled.caption.enabled is False
        assert styled.caption.glow == CAPTION_STYLES["neon"].glow

    def test_classic_is_the_default_look(self) -> None:
        assert DEFAULT.with_caption_style("classic").caption == CaptionConfig()

    def test_composes_with_an_edit_preset_in_either_order(self) -> None:
        a = DEFAULT.gentle().with_caption_style("pill")
        b = DEFAULT.with_caption_style("pill").gentle()
        assert a == b
        assert a.silence.min_gap == 0.6
        assert a.caption.active_box_colour == "#7C3AED"

    def test_an_unknown_style_is_refused(self) -> None:
        with pytest.raises(ValueError, match="carrier-pigeon"):
            DEFAULT.with_caption_style("carrier-pigeon")


def test_every_style_has_a_label() -> None:
    assert list(CAPTION_STYLE_LABELS) == list(CAPTION_STYLES)
