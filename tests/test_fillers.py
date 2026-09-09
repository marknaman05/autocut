"""Filler removal, and the guards that stop it eating real words."""

from __future__ import annotations

import pytest

from autocut.analyze import fillers
from autocut.config import FillerConfig
from autocut.models import Word


def build(*specs: tuple[str, float, float]) -> list[Word]:
    return [Word(text=text, start=start, end=end) for text, start, end in specs]


def removed_text(words: list[Word], config: FillerConfig) -> list[str]:
    """The words each proposed span covers."""
    spans = fillers.detect(words, config)
    return [
        " ".join(w.text for w in words if span.start <= w.start < span.end)
        for span in spans
    ]


class TestUnambiguousFillers:
    def test_um_is_removed(self) -> None:
        words = build(("about,", 4.7, 5.1), ("um,", 5.1, 5.5), ("three", 5.6, 5.9))
        assert removed_text(words, FillerConfig()) == ["um,"]

    def test_a_drawn_out_token_is_left_alone(self) -> None:
        # Held for over a second, this is deliberate, not a stumble.
        words = build(("well", 0.0, 0.4), ("uh", 0.5, 1.8), ("yes", 1.9, 2.2))
        assert removed_text(words, FillerConfig(max_duration=0.6)) == []

    def test_an_ambiguous_word_opening_the_transcript_counts_as_padding(self) -> None:
        # There is nothing before the first word, so "So, ..." is treated as
        # the throat-clearing it usually is.
        words = build(("so", 0.0, 0.3), ("here", 0.35, 0.6), ("we", 0.6, 0.75),
                      ("go", 0.75, 1.0))
        assert removed_text(words, FillerConfig()) == ["so"]

    def test_the_pause_before_a_filler_goes_with_it(self) -> None:
        """Removing only the word would leave the pauses either side of it back
        to back -- a longer gap than either, which the silence detector has
        already decided not to cut."""
        words = build(("about,", 4.5, 4.9), ("um,", 5.2, 5.5), ("three", 5.6, 5.9))
        spans = fillers.detect(words, FillerConfig())
        assert spans[0].start == pytest.approx(4.9), "the lead-in pause should go too"
        assert spans[0].end == pytest.approx(5.5)


class TestAmbiguousWords:
    def test_like_as_a_verb_survives(self) -> None:
        words = build(("I", 0.0, 0.2), ("like", 0.2, 0.5), ("it", 0.5, 0.7))
        assert removed_text(words, FillerConfig()) == []

    def test_like_as_padding_is_removed(self) -> None:
        # Bounded by pauses, this is verbal padding.
        words = build(("was", 0.0, 0.3), ("like", 0.9, 1.2), ("really", 1.8, 2.2))
        assert removed_text(words, FillerConfig()) == ["like"]

    def test_a_comma_marks_padding(self) -> None:
        # Whisper punctuates verbal padding even when the timings are tight.
        words = build(("was,", 0.0, 0.3), ("like,", 0.35, 0.6), ("really", 0.65, 1.0))
        assert removed_text(words, FillerConfig()) == ["like,"]


class TestPhrases:
    def test_you_know_is_removed_as_one_unit(self) -> None:
        words = build(("it's", 0.0, 0.3), ("you", 0.9, 1.05), ("know", 1.05, 1.3),
                      ("fine", 1.9, 2.2))
        assert removed_text(words, FillerConfig()) == ["you know"]

    def test_a_phrase_is_matched_before_its_parts(self) -> None:
        words = build(("it's", 0.0, 0.3), ("sort", 0.9, 1.1), ("of", 1.1, 1.25),
                      ("done", 1.9, 2.2))
        spans = fillers.detect(words, FillerConfig())
        assert len(spans) == 1


class TestConfiguration:
    def test_disabled_removes_nothing(self) -> None:
        words = build(("um,", 0.0, 0.3), ("hello", 0.9, 1.2))
        assert fillers.detect(words, FillerConfig(enabled=False)) == []

    def test_no_words_is_safe(self) -> None:
        assert fillers.detect([], FillerConfig()) == []

    def test_clean_speech_is_untouched(self) -> None:
        words = build(("today", 0.0, 0.4), ("we", 0.4, 0.6), ("begin", 0.6, 1.0))
        assert fillers.detect(words, FillerConfig()) == []
