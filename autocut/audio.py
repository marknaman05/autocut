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
        """Anything below this is treated as non-speech.

        Anchored to how loud the speech is, not to how quiet the room is.
        Anchoring it to the floor was tried and fails from both directions,
        because the measured floor is not reliably the room: on a noisy
        recording the floor sits so close to speech that the threshold lands
        *beneath* it and no frame in the file is ever silent; on a recording
        with digitally silent passages the floor is the digital silence, and a
        threshold just above that reads ordinary room tone as speech.  Both
        failures are the same shape -- a pause plainly audible to anyone
        watching comes back uncut -- and neither announces itself.

        Speech level is the stable landmark.  A threshold a good margin below
        it separates the two populations on quiet and noisy recordings alike;
        the floor is kept only as a lower bound, and speech itself as an upper
        one, for the degenerate case where a file has no headroom at all.
        """
        # The upper bound matters for a file with no headroom at all, where
        # the floor bound would otherwise land *above* the signal and call
        # every frame silent.
        return min(max(self.speech_level - 25.0, self.noise_floor + 3.0), self.speech_level - 3.0)

    def is_silent(self, t: float) -> bool:
        return self.level_at(t) < self.silence_threshold

    @property
    def midpoint_level(self) -> float:
        """Halfway between room tone and speech, in dB."""
        return (self.noise_floor + self.speech_level) / 2.0

    @property
    def speech_confidence_level(self) -> float:
        """Above this, a sound is loud enough to be worth protecting as speech.

        Sits well clear of ``silence_threshold``, which is deliberately set low
        so that quiet speech is never mistaken for silence.  That caution has a
        cost: room tone flutters over the threshold constantly, and a detector
        that treats every flutter as speech chops a clean pause into fragments.
        """
        return self.silence_threshold + 10.0

    def silent_runs(
        self,
        start: float,
        end: float,
        min_duration: float,
        bridge: float = 0.0,
        sustained: float = 0.3,
    ) -> list[tuple[float, float]]:
        """Stretches of ``[start, end)`` at least ``min_duration`` long that sit
        below the silence threshold.

        Two runs separated by less than ``bridge`` seconds are treated as one,
        provided the sound between them is not *sustained* at speech level for
        ``sustained`` seconds.  Without bridging, a breath or a cough in the
        middle of a long pause splits it into two cuts with a fragment
        stranded between them -- a worse edit and an audible stutter.

        Loudness cannot make this call.  A cough measured here peaked at
        -21 dB, louder than three of four stretches of real speech the
        recogniser had dropped; judging by peak keeps the cough and would have
        to keep it to stay safe.  What separates them is shape: a cough is one
        impulse with a 50 ms attack and a decay, while a spoken word holds
        near speech level for an appreciable fraction of a second.
        """
        if not len(self.db):
            return []
        lo, hi = self._index(start), self._index(end)
        if hi <= lo:
            return []

        quiet = self.db[lo:hi] < self.silence_threshold
        min_frames = max(int(round(min_duration / self.frame_seconds)), 1)

        runs: list[tuple[float, float]] = []
        run_start: int | None = None
        for offset, is_quiet in enumerate(quiet):
            if is_quiet:
                if run_start is None:
                    run_start = offset
            elif run_start is not None:
                if offset - run_start >= min_frames:
                    runs.append((
                        (lo + run_start) * self.frame_seconds,
                        (lo + offset) * self.frame_seconds,
                    ))
                run_start = None
        if run_start is not None and len(quiet) - run_start >= min_frames:
            runs.append((
                (lo + run_start) * self.frame_seconds,
                (lo + len(quiet)) * self.frame_seconds,
            ))

        if bridge <= 0.0 or len(runs) < 2:
            return runs

        merged = [runs[0]]
        for run in runs[1:]:
            previous_end = merged[-1][1]
            island = run[0] - previous_end
            held = self.longest_run_above(
                self.speech_confidence_level, previous_end, run[0]
            )
            if island <= bridge and held < sustained:
                merged[-1] = (merged[-1][0], run[1])
            else:
                merged.append(run)
        return merged

    def longest_run_above(self, level: float, start: float, end: float) -> float:
        """Duration of the longest unbroken stretch of ``[start, end)`` louder
        than ``level``.

        Distinguishes a noisy room from speech the recogniser dropped, which a
        peak or an average cannot: room tone crosses any threshold you pick,
        but only in isolated frames, while a spoken word holds above it for a
        appreciable fraction of a second.  Measured across two recordings, the
        two populations do not overlap -- dropped words ran 0.40s and longer,
        room tone never exceeded 0.12s.
        """
        if not len(self.db):
            return 0.0
        lo, hi = self._index(start), self._index(end)
        if hi <= lo:
            return 0.0

        longest = current = 0
        for loud in self.db[lo:hi] > level:
            current = current + 1 if loud else 0
            longest = max(longest, current)
        return longest * self.frame_seconds

    def loudest_between(self, start: float, end: float) -> float:
        if not len(self.db):
            return -90.0
        lo, hi = self._index(start), self._index(end)
        if hi <= lo:
            return float(self.db[lo])
        return float(np.max(self.db[lo : hi + 1]))

    def quietest_time_near(
        self, t: float, radius: float = 0.06, max_level: float | None = None
    ) -> float:
        """The quietest instant within ``radius`` of ``t``.

        Cutting here instead of at an arbitrary sample keeps joins from
        clicking, and moves the boundary by at most a couple of frames.

        ``max_level`` refuses the move when even the quietest instant nearby is
        louder than that.  Without it, a boundary that lands in the middle of a
        word gets snapped to the dip between two syllables -- which is a local
        minimum, but is still speech, so the join clips the word instead of
        landing cleanly beside it.  A boundary with no quiet audio near it is
        better left exactly where the detector put it.
        """
        if not len(self.db):
            return t
        lo, hi = self._index(t - radius), self._index(t + radius)
        if hi <= lo:
            return t
        window = self.db[lo : hi + 1]
        quietest = int(np.argmin(window))
        if max_level is not None and window[quietest] >= max_level:
            return t
        return (lo + quietest) * self.frame_seconds

    def retreat_to_quiet(self, t: float, limit: float) -> float:
        """Walk *backwards* from ``t`` to where the audio last went quiet.

        Used to pull the end of a removal off a word it would otherwise clip.
        A word's onset routinely begins before the timestamp the recogniser
        gives it, so resuming exactly at that timestamp cuts the front off the
        word.  Returns ``t`` unchanged if it is already quiet, or if nothing
        quiet is found within ``limit``.
        """
        if not len(self.db) or self.level_at(t) < self.silence_threshold:
            return t
        floor = self._index(t - limit)
        index = self._index(t)
        while index > floor and self.db[index] >= self.silence_threshold:
            index -= 1
        if self.db[index] >= self.silence_threshold:
            return t
        return index * self.frame_seconds

    def advance_to_quiet(self, t: float, limit: float) -> float:
        """Walk *forwards* from ``t`` to where the audio next goes quiet.

        The mirror of :meth:`retreat_to_quiet`, for the start of a removal: a
        word rings on past the timestamp it ends at, and cutting the instant
        the timestamp says truncates the tail mid-decay.
        """
        if not len(self.db) or self.level_at(t) < self.silence_threshold:
            return t
        ceiling = self._index(t + limit)
        index = self._index(t)
        while index < ceiling and self.db[index] >= self.silence_threshold:
            index += 1
        if self.db[index] >= self.silence_threshold:
            return t
        return index * self.frame_seconds

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
