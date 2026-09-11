"""Dead-air detection, including the pauses that hide inside a word.

The between-word cases are straightforward.  The intra-word ones are here
because they are the failure that actually shipped: Whisper hands back a word
whose end timestamp runs on through a pause -- a seven-second "that" -- and a
detector that only looks at the gaps *between* words cannot see it at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from autocut.analyze import silence
from autocut.audio import FRAME_SECONDS, Envelope
from autocut.config import SilenceConfig
from autocut.models import Reason, Word

SPEECH_DB = -20.0
SILENT_DB = -60.0
#: Clearly above the silence threshold but well below speech -- a breath or a
#: click, the kind of sound bridging exists to reach across.
_AUDIBLE = -38.0


#: Appended past the tested timeline so the envelope has a measurable noise
#: floor.  Without one, ``noise_floor`` and ``speech_level`` collapse onto the
#: same value, and every level derived from the gap between them -- the silence
#: threshold, the speech-presence level -- becomes meaningless.  Real
#: recordings run 18-36 dB of separation; a fixture with 0 dB tests nothing.
_ROOM_TONE_SECONDS = 2.0


def envelope(duration: float, quiet: list[tuple[float, float]]) -> Envelope:
    """An envelope that is speech-loud everywhere except ``quiet``.

    A stretch of room tone is appended beyond ``duration`` so the envelope has
    a realistic noise floor; detectors are only ever asked about ``[0,
    duration)``, so it does not otherwise take part in the test.
    """
    frames = int(duration / FRAME_SECONDS)
    db = np.full(frames + int(_ROOM_TONE_SECONDS / FRAME_SECONDS), SILENT_DB, dtype=np.float32)
    db[:frames] = SPEECH_DB
    for start, end in quiet:
        db[int(start / FRAME_SECONDS) : int(end / FRAME_SECONDS)] = SILENT_DB
    return Envelope(db)


class TestWithinWord:
    config = SilenceConfig()

    def test_a_pause_inside_an_overlong_word_is_found(self) -> None:
        # One "word" spanning four seconds, silent from 1.0 to 3.0.
        words = [Word(text="that", start=0.0, end=4.0)]
        spans = silence.detect(words, 4.0, self.config, envelope(4.0, [(1.0, 3.0)]))

        within = [s for s in spans if s.detail == "within-word"]
        assert len(within) == 1, "the dead air inside the word should be found"
        assert within[0].reason is Reason.SILENCE
        # Padded inward, so the speech either side is never clipped.
        assert within[0].start == pytest.approx(1.0 + self.config.pad, abs=0.02)
        assert within[0].end == pytest.approx(3.0 - self.config.pad, abs=0.02)

    def test_a_normally_timed_word_is_never_picked_apart(self) -> None:
        """The scan only runs on implausibly long words.

        A word of ordinary length is trusted even if its envelope dips, so a
        stop consonant or a quiet syllable can never split a real word in two.
        """
        words = [Word(text="stop", start=0.0, end=0.5)]
        spans = silence.detect(words, 0.5, self.config, envelope(0.5, [(0.1, 0.45)]))
        assert [s for s in spans if s.detail == "within-word"] == []

    def test_a_long_but_genuinely_voiced_word_survives(self) -> None:
        # A drawn-out "sooooo" is long, but there is no silence inside it.
        words = [Word(text="so", start=0.0, end=2.0)]
        spans = silence.detect(words, 2.0, self.config, envelope(2.0, []))
        assert [s for s in spans if s.detail == "within-word"] == []

    def test_a_dip_shorter_than_min_gap_is_left_alone(self) -> None:
        words = [Word(text="that", start=0.0, end=2.0)]
        quiet = [(1.0, 1.0 + self.config.min_gap * 0.5)]
        spans = silence.detect(words, 2.0, self.config, envelope(2.0, quiet))
        assert [s for s in spans if s.detail == "within-word"] == []

    def test_without_an_envelope_nothing_is_scanned(self) -> None:
        """No waveform, no intra-word claims -- the timestamps are all we have."""
        words = [Word(text="that", start=0.0, end=4.0)]
        spans = silence.detect(words, 4.0, self.config, None)
        assert [s for s in spans if s.detail == "within-word"] == []

    def test_spans_come_back_in_time_order(self) -> None:
        """Intra-word spans are found last but must not be returned last."""
        words = [
            Word(text="that", start=0.0, end=3.0),
            Word(text="then", start=5.0, end=5.4),
        ]
        env = envelope(6.0, [(1.0, 2.5), (3.0, 5.0)])
        spans = silence.detect(words, 6.0, self.config, env)
        assert [s.start for s in spans] == sorted(s.start for s in spans)


class TestMissedSpeech:
    """Speech the recogniser dropped must survive the cut.

    A gap in the transcript does not mean a gap in the audio.  Whisper drops
    words, and when it does, the "gap" between the words either side of them is
    part silence and part speech -- so the region cannot be cut wholesale.
    """

    config = SilenceConfig()

    def test_only_the_silent_part_of_a_gap_is_cut(self) -> None:
        # Two words 7s apart.  The first 2s of the gap are silent; the rest is
        # speech that never made it into the transcript.
        words = [
            Word(text="saying", start=0.0, end=0.4),
            Word(text="they", start=7.4, end=7.8),
        ]
        env = envelope(8.0, [(0.4, 2.4)])
        spans = silence.detect(words, 8.0, self.config, env)

        assert len(spans) == 1, "only the silent stretch should be cut"
        assert spans[0].start == pytest.approx(0.4 + self.config.pad, abs=0.02)
        assert spans[0].end == pytest.approx(2.4 - self.config.pad, abs=0.02)

    def test_a_mostly_quiet_gap_does_not_swallow_its_speech(self) -> None:
        """The regression this replaced an averaging veto to fix.

        Judging the gap as a whole -- cut it if under half of it is voiced --
        gets this wrong in the worst way: the long silence at the front drags
        the average under the threshold and the speech at the back is deleted
        along with it.
        """
        words = [
            Word(text="that", start=0.0, end=0.4),
            Word(text="they", start=7.0, end=7.4),
        ]
        # 3.5s silent, then 3.1s of speech: 47% voiced overall, but the speech
        # is entirely real.
        env = envelope(8.0, [(0.4, 3.9)])
        spans = silence.detect(words, 8.0, self.config, env)

        assert len(spans) == 1
        assert spans[0].end <= 3.9, "the speech after the silence must survive"

    def test_a_fully_voiced_gap_is_left_alone(self) -> None:
        words = [
            Word(text="one", start=0.0, end=0.4),
            Word(text="two", start=3.0, end=3.4),
        ]
        assert silence.detect(words, 4.0, self.config, envelope(4.0, [])) == []

    def test_without_an_envelope_the_whole_gap_is_cut(self) -> None:
        """No waveform, no better information: trust the transcript."""
        words = [
            Word(text="one", start=0.0, end=0.4),
            Word(text="two", start=3.0, end=3.4),
        ]
        gaps = [s for s in silence.detect(words, 4.0, self.config, None) if s.detail == "gap"]
        assert len(gaps) == 1
        assert gaps[0].start == pytest.approx(0.4 + self.config.pad, abs=0.02)
        assert gaps[0].end == pytest.approx(3.0 - self.config.pad, abs=0.02)


class TestNoisyRecordings:
    """A recording with little headroom between room tone and speech.

    Energy thresholding degrades badly here, which is why this detector is
    transcript-driven.  The regression these cover is a real one: an 18-second
    clip at 17.9 dB separation, with a four-second pause in it, came back with
    nothing to cut at all.
    """

    config = SilenceConfig()

    def noisy(self, duration: float, quiet: list[tuple[float, float]]) -> Envelope:
        """Speech only ~16 dB above the room, and a room that is never quiet."""
        frames = int(duration / FRAME_SECONDS)
        db = np.full(frames, -28.0, dtype=np.float32)
        for start, end in quiet:
            db[int(start / FRAME_SECONDS) : int(end / FRAME_SECONDS)] = -44.0
        return Envelope(db)

    def test_the_threshold_never_sinks_below_the_noise_floor(self) -> None:
        """Below it, no frame in the file is ever silent and the whole
        detector switches off without saying so."""
        env = self.noisy(10.0, [(2.0, 6.0)])
        assert env.speech_level - env.noise_floor < 18.0, "fixture must be low-SNR"
        assert env.silence_threshold > env.noise_floor

    def test_a_pause_in_a_noisy_room_is_still_found(self) -> None:
        words = [
            Word(text="colleges", start=1.0, end=1.9),
            Word(text="this", start=6.1, end=6.6),
        ]
        spans = silence.detect(words, 8.0, self.config, self.noisy(8.0, [(2.0, 6.0)]))
        assert [s for s in spans if s.detail == "gap"], "the four-second pause must be cut"

    def test_a_gap_the_waveform_says_nothing_about_is_still_cut(self) -> None:
        """No stretch of the gap falls below the threshold -- the room is
        simply loud.  The transcript says no words are here, and with the
        waveform proving nothing either way the transcript is the better
        witness."""
        words = [
            Word(text="one", start=0.5, end=1.0),
            Word(text="two", start=5.0, end=5.5),
        ]
        env = self.noisy(6.0, [])  # nothing anywhere is below threshold
        assert not env.silent_runs(1.0, 5.0, self.config.min_gap)
        gaps = [s for s in silence.detect(words, 6.0, self.config, env) if s.detail == "gap"]
        assert len(gaps) == 1

    def test_sustained_speech_level_audio_still_protects_a_gap(self) -> None:
        """The escape hatch above must not swallow words the recogniser
        dropped -- those hold at speech level for an appreciable stretch,
        where room tone only ever blips across it."""
        frames = int(6.0 / FRAME_SECONDS)
        db = np.full(frames, -44.0, dtype=np.float32)
        db[: int(1.0 / FRAME_SECONDS)] = -28.0          # "one"
        db[int(2.0 / FRAME_SECONDS) : int(3.5 / FRAME_SECONDS)] = -28.0  # dropped words
        db[int(5.0 / FRAME_SECONDS) :] = -28.0          # "two"
        env = Envelope(db)
        words = [
            Word(text="one", start=0.4, end=0.9),
            Word(text="two", start=5.1, end=5.6),
        ]
        spans = silence.detect(words, 6.0, self.config, env)
        # The untranscribed speech at 2.0-3.5 must survive.
        assert not any(s.start < 3.4 and s.end > 2.1 for s in spans)

    def test_a_word_is_never_hollowed_out_by_the_fallback(self) -> None:
        """Inside a word the transcript asserts speech, so the waveform has to
        prove silence before anything is cut.  Without that, a noisy recording
        loses the middles of its words."""
        words = [Word(text="jobs", start=0.5, end=2.0)]
        env = self.noisy(3.0, [])  # nothing below threshold anywhere
        spans = silence.detect(words, 3.0, self.config, env)
        assert [s for s in spans if s.detail == "within-word"] == []


class TestSilentRuns:
    def test_a_run_at_the_very_end_is_reported(self) -> None:
        """A quiet stretch running to the end of the window has no closing
        edge to trigger on, so it is easy to drop by accident."""
        env = envelope(2.0, [(1.0, 2.0)])
        runs = env.silent_runs(0.0, 2.0, 0.35)
        assert len(runs) == 1
        assert runs[0][0] == pytest.approx(1.0, abs=0.02)
        assert runs[0][1] == pytest.approx(2.0, abs=0.02)

    def test_runs_are_clipped_to_the_requested_window(self) -> None:
        env = envelope(4.0, [(0.5, 3.5)])
        runs = env.silent_runs(1.0, 2.0, 0.35)
        assert len(runs) == 1
        assert runs[0][0] >= 1.0 - 0.02
        assert runs[0][1] <= 2.0 + 0.02

    def test_a_quiet_click_does_not_split_a_pause(self) -> None:
        """Room tone flutters over the silence threshold constantly.

        The threshold is set low on purpose, so that quiet speech is never cut.
        The cost is that a breath in the middle of a long pause reads as sound;
        if that split the pause, the edit would leave a fragment of room tone
        stranded between two cuts.
        """
        db = np.full(400, SILENT_DB, dtype=np.float32)
        db[195:205] = _AUDIBLE  # a 0.1s breath, well below speech level
        env = Envelope(np.concatenate([np.full(100, SPEECH_DB, dtype=np.float32), db]))
        runs = env.silent_runs(1.0, 5.0, 0.35, bridge=0.5)
        assert len(runs) == 1, "the breath should not break the pause in two"

    def test_real_speech_still_splits_a_pause(self) -> None:
        """Bridging must not reach across something that is actually speech.

        Half a second of it: loudness alone cannot make this call, since a
        cough outpeaks most speech, so what marks a word is holding near
        speech level rather than spiking to it.
        """
        db = np.full(400, SILENT_DB, dtype=np.float32)
        db[150:200] = SPEECH_DB  # 0.5s -- a word, not a blip
        env = Envelope(np.concatenate([np.full(100, SPEECH_DB, dtype=np.float32), db]))
        runs = env.silent_runs(1.0, 5.0, 0.35, bridge=0.5)
        assert len(runs) == 2, "speech in the middle of a pause must survive"

    def test_a_loud_but_brief_noise_does_not_split_a_pause(self) -> None:
        """A cough is louder than most speech and must still be cut.

        Measured on a real recording: the cough peaked at -21 dB, above three
        of four stretches of speech the recogniser had dropped, and held at
        speech level for 0.19s where those stretches held 0.37-1.05s.
        """
        db = np.full(400, SILENT_DB, dtype=np.float32)
        db[195:214] = 0.0  # 0.19s, louder than speech itself
        env = Envelope(np.concatenate([np.full(100, SPEECH_DB, dtype=np.float32), db]))
        runs = env.silent_runs(1.0, 5.0, 0.35, bridge=1.0)
        assert len(runs) == 1, "a brief noise is not a reason to keep the pause"

    def test_a_long_quiet_island_is_not_bridged(self) -> None:
        """Bridging is for clicks and breaths, not for stretches of audio."""
        db = np.full(400, SILENT_DB, dtype=np.float32)
        db[150:250] = _AUDIBLE  # a full second, quiet but sustained
        env = Envelope(np.concatenate([np.full(100, SPEECH_DB, dtype=np.float32), db]))
        assert _AUDIBLE > env.silence_threshold, "fixture must be above threshold"
        runs = env.silent_runs(1.0, 5.0, 0.35, bridge=0.5)
        assert len(runs) == 2

    def test_bridging_is_off_by_default(self) -> None:
        db = np.full(400, SILENT_DB, dtype=np.float32)
        db[195:205] = _AUDIBLE
        env = Envelope(np.concatenate([np.full(100, SPEECH_DB, dtype=np.float32), db]))
        assert len(env.silent_runs(1.0, 5.0, 0.35)) == 2

    def test_an_empty_envelope_finds_nothing(self) -> None:
        assert Envelope(np.zeros(0, dtype=np.float32)).silent_runs(0.0, 1.0, 0.35) == []
