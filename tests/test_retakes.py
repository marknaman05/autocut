"""Retake detection, and the guards that make a local model safe to trust.

These matter more than most: a false positive here deletes something the
speaker actually said, and the model proposing them is the least reliable part
of the system.  Every test below is a rule the model is not allowed to break.
"""

from __future__ import annotations

import pytest

from autocut.analyze import retakes
from autocut.config import RetakeConfig
from autocut.models import Reason, Word


def speech(text: str, *, start: float = 0.0, rate: float = 0.3, pause_before: set[int] = frozenset()) -> list[Word]:
    """Lay a sentence out on a timeline, one word every ``rate`` seconds."""
    words: list[Word] = []
    cursor = start
    for index, token in enumerate(text.split()):
        if index in pause_before:
            cursor += 0.6
        words.append(Word(text=token, start=cursor, end=cursor + rate * 0.8))
        cursor += rate
    return words


class TestNgramRepeats:
    config = RetakeConfig()

    def test_a_restarted_sentence_is_caught(self) -> None:
        words = speech(
            "the first thing you should the first thing you should do is write it down",
            pause_before={5},
        )
        spans = retakes._ngram_repeats(words, self.config)
        assert spans, "the abandoned attempt should be found"
        span = spans[0]
        assert span.reason is Reason.NGRAM_REPEAT
        # It removes the first attempt and leaves the good take intact.
        assert span.start == pytest.approx(words[0].start)
        assert span.end == pytest.approx(words[4].end)

    def test_a_repeat_with_no_pause_is_not_a_retake(self) -> None:
        # Said straight through, this is a rhetorical repetition, not a stumble.
        words = speech("the first thing you the first thing you matters")
        assert retakes._ngram_repeats(words, self.config) == []

    def test_a_distant_repeat_is_not_a_retake(self) -> None:
        words = speech("the first thing you should", pause_before=set())
        filler = speech("and then a lot of other things happened here", start=30.0)
        later = speech("the first thing you should do", start=60.0, pause_before={0})
        assert retakes._ngram_repeats(words + filler + later, self.config) == []

    def test_clean_speech_produces_nothing(self) -> None:
        words = speech("today we are going to talk about three completely different ideas")
        assert retakes._ngram_repeats(words, self.config) == []

    def test_very_short_input_is_safe(self) -> None:
        assert retakes._ngram_repeats(speech("hello there"), self.config) == []


class TestIsSuperseded:
    config = RetakeConfig()

    def test_a_span_repeated_later_is_superseded(self) -> None:
        words = speech("the first thing you should the first thing you should do", pause_before={5})
        assert retakes._is_superseded(words, 0, 4, self.config)

    def test_a_span_said_only_once_is_not(self) -> None:
        """The exact false positive a local model produced in testing: it
        wanted to cut the greeting as a 'repeated introduction'."""
        words = speech("hello and welcome back to the channel today we talk about workflows")
        assert not retakes._is_superseded(words, 0, 6, self.config)

    def test_a_repeat_beyond_the_window_does_not_count(self) -> None:
        words = speech("the first thing you should") + speech(
            "the first thing you should do", start=90.0
        )
        assert not retakes._is_superseded(words, 0, 4, self.config)

    def test_a_paraphrased_retake_is_superseded(self) -> None:
        """The whole point of the fuzzy guard: real retakes are rarely
        verbatim -- "the first thing you should do is" becomes "so the first
        thing to do is" -- and an exact-match guard would reject this."""
        words = speech(
            "the first thing you should do is so the first thing to do is write it down",
            pause_before={7},
        )
        assert retakes._is_superseded(words, 0, 6, self.config)

    def test_a_span_of_only_stopwords_is_rejected(self) -> None:
        # "is to the" has no content words to measure overlap on at all.
        words = speech("is to the so it was to a the", pause_before={5})
        assert not retakes._is_superseded(words, 0, 2, self.config)

    def test_low_overlap_is_not_enough(self) -> None:
        # Only "thing" is shared; that is not the same statement restarted.
        words = speech("the first thing you liked", pause_before={4}) + speech(
            "another thing entirely happened", start=10.0
        )
        assert not retakes._is_superseded(words, 0, 4, self.config)


class TestValidateChunk:
    config = RetakeConfig()

    @pytest.fixture
    def words(self) -> list[Word]:
        return speech(
            "the first thing you should the first thing you should do is write it down",
            pause_before={5},
        )

    def test_a_good_span_is_accepted(self, words) -> None:
        accepted = retakes._validate_chunk(
            [{"start": 0, "end": 4, "why": "false start"}], words, 0, len(words), self.config
        )
        assert accepted == [(0, 4, "false start")]

    def test_a_span_nothing_repeats_is_rejected(self, words) -> None:
        # Words 10-14 are the tail of the good take; nothing follows them.
        assert retakes._validate_chunk(
            [{"start": 10, "end": 14, "why": "invented"}], words, 0, len(words), self.config
        ) == []

    def test_an_out_of_range_span_is_fatal(self, words) -> None:
        with pytest.raises(retakes.RetakeValidationError):
            retakes._validate_chunk(
                [{"start": 0, "end": 999}], words, 0, len(words), self.config
            )

    def test_a_malformed_span_is_fatal(self, words) -> None:
        with pytest.raises(retakes.RetakeValidationError):
            retakes._validate_chunk([{"start": "x"}], words, 0, len(words), self.config)

    def test_a_span_with_no_pause_after_it_is_rejected(self) -> None:
        words = speech("the first thing you the first thing you do")
        assert retakes._validate_chunk(
            [{"start": 0, "end": 3, "why": "no beat"}], words, 0, len(words), self.config
        ) == []


class TestDetect:
    def test_an_unreachable_model_still_finds_obvious_retakes(self) -> None:
        config = RetakeConfig(ollama_host="http://127.0.0.1:1")  # nothing listens here
        words = speech(
            "the first thing you should the first thing you should do is write it down",
            pause_before={5},
        )
        spans = retakes.detect(words, config)
        assert spans, "the deterministic detector must work without a model"
        assert all(span.reason is Reason.NGRAM_REPEAT for span in spans)

    def test_disabled_detects_nothing(self) -> None:
        words = speech("the first thing you should the first thing you should do", pause_before={5})
        assert retakes.detect(words, RetakeConfig(enabled=False)) == []

    def test_choose_model_fails_cleanly_with_no_ollama(self) -> None:
        with pytest.raises((retakes.RetakeValidationError, OSError)):
            retakes.choose_model(RetakeConfig(ollama_host="http://127.0.0.1:1"))

    def test_an_explicit_model_is_used_without_asking_ollama(self) -> None:
        config = RetakeConfig(model="my-model:7b", ollama_host="http://127.0.0.1:1")
        assert retakes.choose_model(config) == "my-model:7b"
