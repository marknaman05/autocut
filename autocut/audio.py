"""Audio energy analysis over the extracted 16 kHz mono WAV.

Used for three things: sanity-checking silence detection against the actual
waveform, snapping cut points to quiet samples so joins do not click, and
nudging ASR word starts onto real speech onsets.
"""

from __future__ import annotations

import wave
from bisect import bisect_left
from pathlib import Path

import numpy as np

#: Envelope resolution.  10 ms is finer than any threshold we care about and
#: keeps a one hour video's envelope well under a megabyte.
FRAME_SECONDS = 0.01


class Envelope:
    """A per-frame RMS envelope of the analysis audio, in dBFS."""

    def __init__(self, db: np.ndarray, frame_seconds: float = FRAME_SECONDS) -> None:
        self.db = db
        self.frame_seconds = frame_seconds
        self.duration = len(db) * frame_seconds
        #: Noise floor estimated from the quietest decile, so a hissy room does
        #: not read as speech and a very clean room is not over-cut.
        self.noise_floor = float(np.percentile(db, 10)) if len(db) else -90.0
        self.speech_level = float(np.percentile(db, 90)) if len(db) else -20.0

    @classmethod
    def from_wav(cls, path: Path, frame_seconds: float = FRAME_SECONDS) -> Envelope:
        with wave.open(str(path), "rb") as wav:
            if wav.getsampwidth() != 2:
                raise ValueError("expected 16-bit PCM audio from the ingest stage")
            rate = wav.getframerate()
            channels = wav.getnchannels()
            raw = wav.readframes(wav.getnframes())

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)

        frame_length = max(int(rate * frame_seconds), 1)
        usable = len(samples) - (len(samples) % frame_length)
        if usable == 0:
            return cls(np.zeros(0, dtype=np.float32), frame_seconds)
        frames = samples[:usable].reshape(-1, frame_length)
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        db = 20.0 * np.log10(np.maximum(rms, 1e-6))
        return cls(db.astype(np.float32), frame_seconds)

    def _index(self, t: float) -> int:
        return min(max(int(t / self.frame_seconds), 0), max(len(self.db) - 1, 0))

    def level_at(self, t: float) -> float:
        return float(self.db[self._index(t)]) if len(self.db) else -90.0

    @property
    def silence_threshold(self) -> float:
        """Anything below this is treated as non-speech.  Sits a little above
        the measured noise floor, but never so high that quiet speech is cut."""
        return min(self.noise_floor + 8.0, self.speech_level - 18.0)

    def is_silent(self, t: float) -> bool:
        return self.level_at(t) < self.silence_threshold

    def loudest_between(self, start: float, end: float) -> float:
        if not len(self.db):
            return -90.0
        lo, hi = self._index(start), self._index(end)
        if hi <= lo:
            return float(self.db[lo])
        return float(np.max(self.db[lo : hi + 1]))

    def quietest_time_near(self, t: float, radius: float = 0.06) -> float:
        """The quietest instant within ``radius`` of ``t``.

        Cutting here instead of at an arbitrary sample keeps joins from
        clicking, and moves the boundary by at most a couple of frames.
        """
        if not len(self.db):
            return t
        lo, hi = self._index(t - radius), self._index(t + radius)
        if hi <= lo:
            return t
        window = self.db[lo : hi + 1]
        return (lo + int(np.argmin(window))) * self.frame_seconds

    def onset_near(self, t: float, radius: float = 0.12) -> float:
        """The start of the speech burst nearest ``t``.

        Whisper's word timings come from attention alignment and drift by tens
        of milliseconds, which is invisible in a cut but very visible in a
        karaoke caption.  Walking back to where the level actually rises above
        the noise floor recovers most of that.
        """
        if not len(self.db):
            return t
        threshold = self.silence_threshold + 4.0
        centre = self._index(t)
        lo = self._index(t - radius)
        hi = self._index(t + radius)
        # Walk backwards from t while we are still inside the same burst.
        index = centre
        while index > lo and self.db[index - 1] >= threshold:
            index -= 1
        if index < centre:
            return index * self.frame_seconds
        # Otherwise the word starts a little late: walk forward to the burst.
        while index < hi and self.db[index] < threshold:
            index += 1
        return index * self.frame_seconds


def snap_boundaries(times: list[float], envelope: Envelope, radius: float = 0.06) -> list[float]:
    """Snap a sorted list of cut times to nearby quiet instants, preserving
    order (a snap is never allowed to reorder two adjacent boundaries)."""
    snapped: list[float] = []
    for t in times:
        candidate = envelope.quietest_time_near(t, radius)
        if snapped and candidate < snapped[-1]:
            candidate = t
        snapped.append(candidate)
    return snapped


def index_of_nearest(sorted_times: list[float], t: float) -> int:
    """Index of the value in ``sorted_times`` closest to ``t``."""
    if not sorted_times:
        raise ValueError("no times to search")
    i = bisect_left(sorted_times, t)
    if i == 0:
        return 0
    if i >= len(sorted_times):
        return len(sorted_times) - 1
    before, after = sorted_times[i - 1], sorted_times[i]
    return i - 1 if (t - before) <= (after - t) else i
