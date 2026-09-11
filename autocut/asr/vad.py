"""Voice activity detection, for repairing the recogniser's word timings.

Whisper places a word's *text* reliably and its *boundaries* poorly.  Measured
across three recordings, word onsets ran 0.39-0.90s early -- the audio in the
stretch Whisper claimed was 12-20 dB quieter than the audio in the stretch it
shared with a voice detector, so the word had simply not started yet.  Word
ends drift the other way, running on through a pause until the next word.

Both defects were previously corrected against the energy envelope, which can
only ask "is this loud?".  That question has no stable answer across
recordings: a noisy room and a room with digitally silent passages need
opposite thresholds, and a cough is louder than most speech.  A voice detector
asks "is this a voice?" instead, which is the question that was always meant.

Silero is a 2 MB TorchScript model and runs at roughly 190x realtime on CPU,
so this costs a few seconds on a long video against the minutes Whisper
already takes.  It is loaded directly rather than through ``torch.hub``, whose
wrapper pulls in torchaudio for file loading this module does not need.

Nothing here is required: every entry point degrades to the envelope-based
repair when torch is missing or the model cannot be fetched.  A worse edit is
an acceptable outcome; a failed render is not.
"""

from __future__ import annotations

import logging
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import Word

log = logging.getLogger(__name__)

#: The single TorchScript file, taken from the upstream repository.  Only this
#: is fetched -- not the package, which imports torchaudio at module scope.
MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/"
    "src/silero_vad/data/silero_vad.jit"
)

#: Silero consumes exactly this many samples per step at 16 kHz.
_WINDOW = 512
_RATE = 16000
FRAME_SECONDS = _WINDOW / _RATE


class VADUnavailable(RuntimeError):
    """The detector could not be loaded, so timings stay as the recogniser left them."""


@dataclass
class SpeechTrack:
    """Per-frame probability that a voice is speaking.

    Resolution is 32 ms -- coarser than the energy envelope's 10 ms, which is
    why this decides *what is speech* while the envelope still decides
    *exactly where to splice*.
    """

    probs: np.ndarray
    threshold: float = 0.5
    frame_seconds: float = FRAME_SECONDS

    def __len__(self) -> int:
        return len(self.probs)

    @property
    def duration(self) -> float:
        return len(self.probs) * self.frame_seconds

    def _index(self, t: float) -> int:
        return min(max(int(t / self.frame_seconds), 0), max(len(self.probs) - 1, 0))

    def is_speech(self, t: float) -> bool:
        return bool(len(self.probs)) and self.probs[self._index(t)] >= self.threshold

    def speech_onset(self, start: float, limit: float) -> float | None:
        """First instant at or after ``start`` that carries a voice.

        ``None`` if none is found within ``limit`` seconds, which is the
        signal to leave the timestamp alone rather than guess.
        """
        if not len(self.probs):
            return None
        first, last = self._index(start), self._index(start + limit)
        for index in range(first, min(last + 1, len(self.probs))):
            if self.probs[index] >= self.threshold:
                return index * self.frame_seconds
        return None

    def speech_run_end(self, start: float, *, bridge: float = 0.15) -> float | None:
        """Where the run of speech beginning at ``start`` stops.

        Walks forward, stepping over silences shorter than ``bridge`` so an
        ordinary stop consonant does not end the run.  ``None`` if ``start``
        carries no voice at all.

        This is deliberately the *leading* run rather than the last voice
        anywhere in the word.  A word whose end ran on through a pause often
        has unrelated speech later inside its span -- words the recogniser
        dropped -- and taking the last voice would stretch the word over them
        instead of ending it where the speaker stopped.
        """
        if not len(self.probs) or not self.is_speech(start):
            return None

        tolerance = max(int(round(bridge / self.frame_seconds)), 1)
        index = self._index(start)
        last_voiced = index
        while index < len(self.probs):
            if self.probs[index] >= self.threshold:
                last_voiced = index
            elif index - last_voiced >= tolerance:
                break
            index += 1
        return (last_voiced + 1) * self.frame_seconds


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise VADUnavailable("expected 16-bit PCM audio from the ingest stage")
        if wav.getframerate() != _RATE:
            raise VADUnavailable(f"expected {_RATE} Hz audio, got {wav.getframerate()}")
        channels = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


def _load_model():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - torch is installed here
        raise VADUnavailable("torch is not installed") from error

    cache = Path(torch.hub.get_dir()) / "autocut" / "silero_vad.jit"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        log.info("fetching the voice detector (2 MB, once)")
        try:
            torch.hub.download_url_to_file(MODEL_URL, str(cache), progress=False)
        except Exception as error:
            raise VADUnavailable(f"could not fetch the model: {error}") from error

    try:
        model = torch.jit.load(str(cache))
    except Exception as error:
        # A truncated download stays broken until it is cleared.
        cache.unlink(missing_ok=True)
        raise VADUnavailable(f"could not load the model: {error}") from error
    model.eval()
    return torch, model


def analyse(audio: Path, threshold: float = 0.5) -> SpeechTrack:
    """Run the detector over the analysis WAV.

    Raises :class:`VADUnavailable` rather than failing the render; callers are
    expected to fall back to the envelope.
    """
    torch, model = _load_model()
    samples = _read_wav(Path(audio))

    tensor = torch.from_numpy(samples)
    probs: list[float] = []
    model.reset_states()
    with torch.no_grad():
        for offset in range(0, len(tensor) - _WINDOW + 1, _WINDOW):
            probs.append(float(model(tensor[offset : offset + _WINDOW], _RATE)))

    log.info(
        "voice detector: %d frames, %.0f%% of the audio carries a voice",
        len(probs),
        100.0 * float(np.mean(np.asarray(probs) >= threshold)) if probs else 0.0,
    )
    return SpeechTrack(probs=np.asarray(probs, dtype=np.float32), threshold=threshold)


def align_to_speech(
    words: list[Word],
    track: SpeechTrack,
    *,
    max_shift: float = 1.0,
    trim_longer_than: float = 0.8,
) -> list[Word]:
    """Trim each word back to the part of it that carries a voice.

    Only the leading and trailing edges move, and only inward.  A word is never
    split: short internal dips are stepped over, so a plosive does not take the
    middle out of an ordinary word.

    ``max_shift`` bounds how far an edge may travel.  A word that finds no
    voice within that distance is left exactly as the recogniser placed it --
    the detector has failed to find something, which is not the same as having
    found silence.

    Ends are only pulled in on words longer than ``trim_longer_than``, because
    no single word is spoken for that long and the timestamp has run on through
    a pause.  Applied to ordinary words the same rule over-trims: the detector
    dips mid-word often enough that "every" lost 0.4s whose audio peaked at
    -15 dB, louder than the part that was kept.
    """
    if not len(track):
        return words

    aligned: list[Word] = []
    for index, word in enumerate(words):
        previous_end = aligned[-1].end if aligned else 0.0
        next_start = words[index + 1].start if index + 1 < len(words) else float("inf")

        start, end = word.start, word.end
        onset = track.speech_onset(start, min(max_shift, max(end - start, 0.0)))
        if onset is not None and onset > start:
            start = min(onset, end - 0.02)
        if end - start > trim_longer_than:
            run_end = track.speech_run_end(start)
            if run_end is not None and run_end < end:
                end = max(run_end, start + 0.02)

        # Never reorder, never overlap a neighbour, never empty a word.
        start = max(start, previous_end)
        end = max(min(end, next_start), start + 0.02)
        aligned.append(
            Word(text=word.text, start=start, end=end, probability=word.probability)
        )
    return aligned
