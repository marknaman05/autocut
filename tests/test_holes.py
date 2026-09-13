"""Second-pass transcription of stretches the first pass returned no words for."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from pytest import approx as pytest_approx

from autocut.asr import holes
from autocut.audio import Envelope
from autocut.models import Word

RATE = 16000


def write_wav(path: Path, loud: list[tuple[float, float]], duration: float) -> None:
    """Room tone everywhere, speech-level noise across each ``loud`` span."""
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 0.001, int(duration * RATE))
    for start, end in loud:
        samples[int(start * RATE):int(end * RATE)] = rng.normal(0, 0.2, int((end - start) * RATE))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())


class FakeTranscriber:
    """Answers with whatever it was told to, and remembers what it was given."""

    def __init__(self, reply: list[Word]) -> None:
        self.reply = reply
        self.clips: list[Path] = []

    def transcribe(self, audio: Path, *, language=None) -> list[Word]:
        self.clips.append(audio)
        return list(self.reply)


def words(*specs):
    return [Word(text=t, start=s, end=e, probability=p) for t, s, e, p in specs]


class TestFindHoles:
    def test_a_wordless_stretch_with_speech_is_a_hole(self, tmp_path) -> None:
        audio = tmp_path / "a.wav"
        write_wav(audio, [(0.5, 2.0), (2.5, 5.0), (5.5, 7.0)], 8.0)
        envelope = Envelope.from_wav(audio)
        spoken = words(("hi", 0.5, 2.0, 0.9), ("bye", 5.5, 7.0, 0.9))
        assert holes.find_holes(spoken, 8.0, envelope, min_gap=0.8, sustained=0.3) == [(2.0, 5.5)]

    def test_a_quiet_gap_is_not_a_hole(self, tmp_path) -> None:
        audio = tmp_path / "a.wav"
        write_wav(audio, [(0.5, 2.0), (5.5, 7.0)], 8.0)
        envelope = Envelope.from_wav(audio)
        spoken = words(("hi", 0.5, 2.0, 0.9), ("bye", 5.5, 7.0, 0.9))
        assert holes.find_holes(spoken, 8.0, envelope, min_gap=0.8, sustained=0.3) == []

    def test_a_short_gap_is_left_alone(self, tmp_path) -> None:
        audio = tmp_path / "a.wav"
        write_wav(audio, [(0.5, 7.0)], 8.0)
        envelope = Envelope.from_wav(audio)
        spoken = words(("hi", 0.5, 2.0, 0.9), ("bye", 2.5, 7.0, 0.9))
        assert holes.find_holes(spoken, 8.0, envelope, min_gap=0.8, sustained=0.3) == []

    def test_the_ends_of_the_file_count(self, tmp_path) -> None:
        audio = tmp_path / "a.wav"
        write_wav(audio, [(0.0, 2.0), (6.0, 8.0)], 8.0)
        envelope = Envelope.from_wav(audio)
        spoken = words(("mid", 3.0, 4.0, 0.9))
        assert holes.find_holes(spoken, 8.0, envelope, min_gap=0.8, sustained=0.3) == [(0.0, 3.0), (4.0, 8.0)]


class TestFillHoles:
    def _setup(self, tmp_path):
        audio = tmp_path / "a.wav"
        write_wav(audio, [(0.5, 2.0), (2.5, 5.0), (5.5, 7.0)], 8.0)
        envelope = Envelope.from_wav(audio)
        spoken = words(("hi", 0.5, 2.0, 0.9), ("bye", 5.5, 7.0, 0.9))
        return audio, envelope, spoken

    def test_the_hole_is_re_read_and_spliced_in_at_real_times(self, tmp_path) -> None:
        audio, envelope, spoken = self._setup(tmp_path)
        # Times relative to the clip, which starts PAD before the hole.
        heard = words(("the", 0.6, 0.9, 0.9), ("repeat", 0.9, 1.6, 0.95))
        transcriber = FakeTranscriber(heard)
        result = holes.fill_holes(
            spoken, audio, 8.0, envelope, transcriber, tmp_path,
            language="en", min_gap=0.8, sustained=0.3,
        )
        assert [w.text for w in result] == ["hi", "the", "repeat", "bye"]
        the = result[1]
        assert the.start == pytest_approx(2.0 - holes.PAD + 0.6)
        assert len(transcriber.clips) == 1
        with wave.open(str(transcriber.clips[0])) as clip:
            seconds = clip.getnframes() / clip.getframerate()
        assert seconds == pytest_approx(3.5 + 2 * holes.PAD, abs=0.01)

    def test_words_heard_outside_the_hole_are_dropped(self, tmp_path) -> None:
        audio, envelope, spoken = self._setup(tmp_path)
        # "hi" again at the very start of the clip is the padding being read.
        heard = words(("hi", 0.0, 0.1, 0.9), ("the", 0.6, 0.9, 0.9), ("repeat", 0.9, 1.6, 0.9))
        result = holes.fill_holes(
            spoken, audio, 8.0, envelope, FakeTranscriber(heard), tmp_path,
            language="en", min_gap=0.8, sustained=0.3,
        )
        assert [w.text for w in result] == ["hi", "the", "repeat", "bye"]

    def test_a_single_doubtful_word_is_a_guess_not_a_reading(self, tmp_path) -> None:
        audio, envelope, spoken = self._setup(tmp_path)
        result = holes.fill_holes(
            spoken, audio, 8.0, envelope, FakeTranscriber(words(("so", 0.5, 1.9, 0.32))), tmp_path,
            language="en", min_gap=0.8, sustained=0.3,
        )
        assert result == spoken

    def test_doubtful_words_are_a_guess_too(self, tmp_path) -> None:
        audio, envelope, spoken = self._setup(tmp_path)
        heard = words(("um", 0.5, 0.9, 0.2), ("ah", 1.0, 1.4, 0.3))
        result = holes.fill_holes(
            spoken, audio, 8.0, envelope, FakeTranscriber(heard), tmp_path,
            language="en", min_gap=0.8, sustained=0.3,
        )
        assert result == spoken

    def test_no_holes_means_no_second_pass(self, tmp_path) -> None:
        audio = tmp_path / "a.wav"
        write_wav(audio, [(0.5, 7.0)], 8.0)
        envelope = Envelope.from_wav(audio)
        spoken = words(("one", 0.5, 3.5, 0.9), ("two", 3.5, 7.0, 0.9))
        transcriber = FakeTranscriber(words(("x", 0, 1, 1)))
        assert holes.fill_holes(
            spoken, audio, 8.0, envelope, transcriber, tmp_path,
            language="en", min_gap=0.8, sustained=0.3,
        ) == spoken
        assert transcriber.clips == []



