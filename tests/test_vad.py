"""Repairing recogniser word timings against a voice detector.

The detector itself is not under test here -- these build a ``SpeechTrack``
from a known probability array, so the suite never needs the model or the
network.  What is under test is the rule applied on top of it, which is where
the judgement lives: how far an edge may move, which edges may move at all,
and what must never happen to a word.
"""

from __future__ import annotations

import numpy as np
import pytest

from autocut.asr.vad import FRAME_SECONDS, SpeechTrack, align_to_speech
from autocut.models import Word


def track(pattern: list[tuple[float, float]], duration: float = 5.0) -> SpeechTrack:
    """A track that is silent except for the given voiced spans."""
    probs = np.zeros(int(duration / FRAME_SECONDS), dtype=np.float32)
    for start, end in pattern:
        probs[int(start / FRAME_SECONDS) : int(end / FRAME_SECONDS)] = 1.0
    return SpeechTrack(probs=probs)


def word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


class TestSpeechTrack:
    def test_speech_onset_finds_the_next_voice(self) -> None:
        t = track([(1.0, 2.0)])
        assert t.speech_onset(0.0, limit=2.0) == pytest.approx(1.0, abs=FRAME_SECONDS)

    def test_speech_onset_gives_up_rather_than_guessing(self) -> None:
        """No voice within reach is the detector failing to find something,
        which is not the same as having found silence."""
        assert track([(4.0, 5.0)]).speech_onset(0.0, limit=0.5) is None

    def test_speech_run_end_stops_where_the_voice_stops(self) -> None:
        t = track([(1.0, 2.0), (4.0, 5.0)])
        assert t.speech_run_end(1.0) == pytest.approx(2.0, abs=FRAME_SECONDS)

    def test_speech_run_end_steps_over_a_plosive(self) -> None:
        """A stop consonant reads as non-speech; ending the word there would
        take the back half off an ordinary word."""
        t = track([(1.0, 1.4), (1.45, 2.0)])  # 50ms gap
        assert t.speech_run_end(1.0) == pytest.approx(2.0, abs=FRAME_SECONDS)

    def test_speech_run_end_does_not_reach_across_a_real_pause(self) -> None:
        t = track([(1.0, 1.4), (3.0, 4.0)])
        assert t.speech_run_end(1.0) == pytest.approx(1.4, abs=FRAME_SECONDS)

    def test_speech_run_end_is_none_when_it_does_not_start_on_a_voice(self) -> None:
        assert track([(3.0, 4.0)]).speech_run_end(1.0) is None

    def test_an_empty_track_answers_nothing(self) -> None:
        empty = SpeechTrack(probs=np.zeros(0, dtype=np.float32))
        assert empty.speech_onset(0.0, 1.0) is None
        assert empty.speech_run_end(0.0) is None
        assert not empty.is_speech(0.0)


class TestAlignToSpeech:
    def test_an_early_onset_is_pulled_onto_the_voice(self) -> None:
        """The defect this exists for: measured onsets ran 0.39-0.90s early."""
        words = [word("The", 0.0, 1.2)]
        aligned = align_to_speech(words, track([(0.9, 1.2)]))
        assert aligned[0].start == pytest.approx(0.9, abs=FRAME_SECONDS)
        assert aligned[0].end == pytest.approx(1.2, abs=0.05)

    def test_an_onset_beyond_reach_is_left_alone(self) -> None:
        words = [word("The", 0.0, 3.0)]
        aligned = align_to_speech(words, track([(2.5, 3.0)]), max_shift=0.5)
        assert aligned[0].start == pytest.approx(0.0)

    def test_an_overlong_word_is_trimmed_to_its_leading_burst(self) -> None:
        """A 7s "that" was the case this replaces: the timestamp ran on
        through a pause until the next word."""
        words = [word("that", 1.0, 5.0)]
        aligned = align_to_speech(words, track([(1.0, 1.5), (3.0, 4.0)]))
        assert aligned[0].end == pytest.approx(1.5, abs=FRAME_SECONDS)

    def test_an_overlong_word_does_not_stretch_to_later_speech(self) -> None:
        """Words the recogniser dropped often sit inside an over-long word's
        span; taking the last voice would swallow them into the word."""
        words = [word("that", 1.0, 5.0)]
        aligned = align_to_speech(words, track([(1.0, 1.5), (3.0, 4.5)]))
        assert aligned[0].end < 2.0

    def test_an_ordinary_word_keeps_its_end(self) -> None:
        """Applied to ordinary words the trim over-reaches -- the detector dips
        mid-word often enough that a 0.6s word lost 0.4s whose audio was louder
        than the part kept."""
        words = [word("every", 1.0, 1.6)]
        aligned = align_to_speech(words, track([(1.0, 1.2)]), trim_longer_than=0.8)
        assert aligned[0].end == pytest.approx(1.6)

    def test_words_never_reorder_or_overlap(self) -> None:
        words = [word("one", 0.0, 1.0), word("two", 1.0, 2.0), word("three", 2.0, 3.0)]
        aligned = align_to_speech(words, track([(0.5, 0.9), (1.4, 1.9), (2.2, 2.9)]))
        for earlier, later in zip(aligned, aligned[1:]):
            assert earlier.end <= later.start + 1e-6
            assert earlier.start < earlier.end

    def test_a_word_is_never_emptied(self) -> None:
        words = [word("x", 1.0, 1.05)]
        aligned = align_to_speech(words, track([(4.0, 5.0)]))
        assert aligned[0].end > aligned[0].start

    def test_an_empty_track_changes_nothing(self) -> None:
        words = [word("one", 0.0, 1.0), word("two", 2.0, 3.0)]
        empty = SpeechTrack(probs=np.zeros(0, dtype=np.float32))
        assert align_to_speech(words, empty) == words

    def test_edges_only_ever_move_inward(self) -> None:
        """Widening a word would claim audio the recogniser never heard."""
        words = [word("a", 1.0, 2.0), word("b", 3.0, 4.0)]
        aligned = align_to_speech(words, track([(0.0, 5.0)]))
        for before, after in zip(words, aligned):
            assert after.start >= before.start - 1e-6
            assert after.end <= before.end + 1e-6
