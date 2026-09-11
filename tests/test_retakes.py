"""Retake detection, and the guards that make a local model safe to trust.

These matter more than most: a false positive here deletes something the
speaker actually said, and the model proposing them is the least reliable part
of the system.  Every test below is a rule the model is not allowed to break.
"""

from __future__ import annotations

from dataclasses import replace

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

    def test_a_long_restarted_sentence_is_caught(self) -> None:
        """A fluffed take runs as long as it runs.

        This is the real case from a render where a 15-word abandoned take
        survived into the output: a word-count cap used to reject any repeat
        whose attempts were more than ``ngram_size * 3`` words apart, which is
        an ordinary length for a sentence someone gave up on.
        """
        words = speech(
            "and every few days a report tells you that the hiring pattern has "
            "changed completely "
            "and every few days a new report tells me that hiring has "
            "completely changed",
            pause_before={15},
        )
        spans = retakes._ngram_repeats(words, self.config)
        assert spans, "the abandoned 15-word take should be found"
        assert spans[0].start == pytest.approx(words[0].start)
        assert spans[0].end == pytest.approx(words[14].end)

    def test_an_anaphoric_refrain_is_not_a_retake(self) -> None:
        """Deliberate repetition for rhythm must survive.

        This is what the word-distance cap was really protecting, and what
        ``_is_superseded`` now protects instead: each clause repeats the frame
        but not the content, so no window scores highly enough to be a retake.
        """
        words = speech(
            "one is news one is headlines one is replacing jobs entirely",
            pause_before={3, 6},
        )
        assert retakes._ngram_repeats(words, self.config) == []

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
        assert accepted == [(0, 4, "false start", retakes._CONFIDENCE_REPEATED)]

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


class TestReplacementClaim:
    """A retake the transcript does not repeat, but the model points at.

    This is the path that catches a reworded restart -- "The tool cuts video
    into parts." replaced by forty different words saying the same thing --
    which no lexical overlap measure can confirm.
    """

    config = RetakeConfig()

    @pytest.fixture
    def words(self) -> list[Word]:
        # Nothing in the second half repeats the first, so only a named
        # replacement can get the span accepted.
        return speech("the tool cuts video into parts", pause_before={6}) + speech(
            "what really happens under here is rather different entirely", start=8.0
        )

    def claim(self, **overrides) -> dict:
        return {
            "start": 0, "end": 5, "why": "reworded restart",
            "replaced_by_start": 6, "replaced_by_end": 14, **overrides,
        }

    def test_a_named_later_span_is_accepted_with_lower_confidence(self, words) -> None:
        assert not retakes._is_superseded(words, 0, 5, self.config)
        assert retakes._validate_chunk(
            [self.claim()], words, 0, len(words), self.config
        ) == [(0, 5, "reworded restart", retakes._CONFIDENCE_CLAIMED)]

    def test_a_replacement_inside_the_cut_is_rejected(self, words) -> None:
        # Pointing at words that are themselves being deleted leaves no good
        # take at all, which is the shape of an invented retake.
        assert retakes._validate_chunk(
            [self.claim(replaced_by_start=1, replaced_by_end=4)],
            words, 0, len(words), self.config,
        ) == []

    def test_a_replacement_out_of_range_is_rejected(self, words) -> None:
        assert retakes._validate_chunk(
            [self.claim(replaced_by_end=999)], words, 0, len(words), self.config
        ) == []

    def test_a_missing_replacement_is_rejected(self, words) -> None:
        assert retakes._validate_chunk(
            [{"start": 0, "end": 5, "why": "unsupported"}],
            words, 0, len(words), self.config,
        ) == []

    def test_a_replacement_far_later_is_rejected(self, words) -> None:
        # The speaker returning to the subject a minute on is not a retake.
        config = replace(self.config, supersede_window=1.0)
        assert retakes._validate_chunk(
            [self.claim()], words, 0, len(words), config
        ) == []


class TestNumbering:
    def test_a_long_pause_is_marked_for_the_model(self) -> None:
        words = speech("the tool cuts video", pause_before={2})
        assert "[pause" in retakes._numbered(words)

    def test_ordinary_word_gaps_are_not_marked(self) -> None:
        assert "[pause" not in retakes._numbered(speech("the tool cuts video"))


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


class TestOpenRouterBackend:
    """The hosted backend is a different wire format wrapping the same
    contract: same prompt, same schema, same validation on the way out."""

    config = RetakeConfig(backend="openrouter", timeout=5.0)

    @pytest.fixture
    def words(self) -> list[Word]:
        # Long enough that a five-word false start stays under the removal
        # ceiling -- the ratio guard is exercised in its own test below.
        return speech(
            "the first thing you should the first thing you should do is write it "
            "down before you forget it and then move on to the next item on the list",
            pause_before={5},
        )

    def _reply(self, monkeypatch, removals, *, status=200):
        """Stub urlopen with one OpenRouter-shaped chat completion."""
        import io
        import json as _json

        body = _json.dumps(
            {"choices": [{"message": {"content": _json.dumps({"removals": removals})}}]}
        ).encode()

        class _Response(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["headers"] = {k.lower(): v for k, v in request.header_items()}
            captured["payload"] = _json.loads(request.data)
            return _Response(body)

        monkeypatch.setattr(retakes.urllib.request, "urlopen", fake_urlopen)
        return captured

    def test_a_proposal_from_openrouter_is_validated_and_accepted(self, monkeypatch, words) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        self._reply(monkeypatch, [{"start": 0, "end": 4, "why": "false start"}])

        spans = retakes._llm_spans(words, self.config)
        assert [s.reason for s in spans] == [Reason.RETAKE]
        assert spans[0].start == pytest.approx(words[0].start)

    def test_an_invented_retake_is_still_rejected(self, monkeypatch, words) -> None:
        """The model saying so is not enough -- the transcript has to repeat it."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        self._reply(monkeypatch, [{"start": 10, "end": 14, "why": "invented"}])
        assert retakes._llm_spans(words, self.config) == []

    def test_the_request_carries_the_key_schema_and_model(self, monkeypatch, words) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        captured = self._reply(monkeypatch, [])
        retakes._llm_spans(words, self.config)

        assert captured["headers"]["authorization"] == "Bearer sk-test"
        assert captured["payload"]["model"] == self.config.openrouter_model
        assert captured["payload"]["response_format"]["json_schema"]["strict"] is True
        assert captured["payload"]["temperature"] == 0.0

    def test_a_missing_key_raises_rather_than_calling_out(self, monkeypatch, words) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(retakes.RetakeValidationError, match="OPENROUTER_API_KEY"):
            retakes._llm_spans(words, self.config)

    def test_detect_falls_back_when_the_key_is_missing(self, monkeypatch, words) -> None:
        """A missing key is a config gap, not a crash: the n-gram detector
        still carries the obvious retakes."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        spans = retakes.detect(words, self.config)
        assert spans and all(s.reason is Reason.NGRAM_REPEAT for s in spans)

    def test_an_http_error_is_turned_into_a_fallback(self, monkeypatch, words) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        def boom(request, timeout=None):
            raise retakes.urllib.error.HTTPError(request.full_url, 429, "slow down", {}, None)

        monkeypatch.setattr(retakes.urllib.request, "urlopen", boom)
        spans = retakes.detect(words, self.config)
        assert all(s.reason is Reason.NGRAM_REPEAT for s in spans)

    def test_non_json_content_is_rejected(self, monkeypatch, words) -> None:
        import io

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        class _Response(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(
            retakes.urllib.request, "urlopen",
            lambda request, timeout=None: _Response(
                b'{"choices": [{"message": {"content": "sorry, no JSON here"}}]}'
            ),
        )
        with pytest.raises(retakes.RetakeValidationError):
            retakes._llm_spans(words, self.config)

    def test_an_unknown_backend_is_an_error(self, words) -> None:
        with pytest.raises(retakes.RetakeValidationError, match="unknown retake backend"):
            retakes._llm_spans(words, RetakeConfig(backend="gemini"))
