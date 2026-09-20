"""Hosted transcriber: Whisper large-v3 through OpenRouter's speech-to-text API.

Same model family as the local MLX default, so the timings have the same
character (cross-attention DTW, tens of milliseconds of drift) and everything
downstream -- ``refine_timings``, hole filling, VAD alignment -- behaves the
same.  What changes is where it runs: nothing on this machine, so the server
no longer needs Apple silicon and a 15-minute video transcribes in seconds
rather than minutes.  Costs about a cent per 15 minutes of audio.

The endpoint takes the audio base64-encoded inside a JSON body, so the 16 kHz
analysis WAV (~2 MB per minute) is re-encoded to 32 kbps mono MP3 (~240 KB
per minute) first; Whisper hears 16 kHz mono either way.  Long recordings are
sent in chunks that split on the quietest moment near the chunk boundary, so
no word is cut in half, and each chunk's times are shifted back into place.

Set ``AUTOCUT_ASR_BACKEND=openrouter`` (and ``OPENROUTER_API_KEY``) to use it.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import urllib.error
import urllib.request
import wave
from pathlib import Path

from .. import ffmpeg
from ..models import Word
from .base import clean_words

log = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/whisper-large-v3"
URL = "https://openrouter.ai/api/v1/audio/transcriptions"
#: Chunk length.  Whisper is trained on 30 s windows and the API handles far
#: longer, but a smaller upload keeps each request quick and a retry cheap.
CHUNK_SECONDS = 20 * 60
#: How far either side of a chunk boundary to look for the quietest moment.
SPLIT_SEARCH = 15.0
BITRATE = "32k"


class OpenRouterWhisperTranscriber:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

    def transcribe(self, audio: Path, *, language: str | None = None) -> list[Word]:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set; the openrouter ASR backend needs it")
        duration = _wav_duration(audio)
        words: list[Word] = []
        with tempfile.TemporaryDirectory(prefix="autocut-asr-") as tmp:
            for index, (start, end) in enumerate(_chunks(audio, duration)):
                clip = Path(tmp) / f"{index:02d}.mp3"
                _encode(audio, start, end, clip)
                log.info("transcribing %s [%.0fs-%.0fs] with %s", audio.name, start, end, self.model)
                result = self._request(clip, language)
                for word in _iter_words(result):
                    words.append(Word(text=word.text, start=word.start + start, end=word.end + start,
                                      probability=word.probability))
        return clean_words(words, duration=duration)

    def _request(self, clip: Path, language: str | None) -> dict:
        body = {
            "model": self.model,
            "input_audio": {"data": base64.b64encode(clip.read_bytes()).decode("ascii"), "format": "mp3"},
            "response_format": "verbose_json",
            "timestamp_granularities": ["word", "segment"],
        }
        if language:
            body["language"] = language
        request = urllib.request.Request(
            URL, data=json.dumps(body).encode(), method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/marknaman05/autocut",
                "X-Title": "autocut",
            },
        )
        last: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    result = json.load(response)
                usage = result.get("usage") or {}
                log.info("openrouter asr: %.0fs of audio, $%.4f", usage.get("seconds", 0), usage.get("cost", 0))
                return result
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:300]
                last = RuntimeError(f"openrouter asr {error.code}: {detail}")
                if error.code < 500 and error.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError) as error:
                last = RuntimeError(f"openrouter asr unreachable: {error}")
            log.warning("openrouter asr attempt %d failed: %s", attempt + 1, last)
        raise last or RuntimeError("openrouter asr failed")


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def _chunks(audio: Path, duration: float) -> list[tuple[float, float]]:
    """(start, end) spans covering the file, boundaries moved to quiet audio."""
    if duration <= CHUNK_SECONDS + SPLIT_SEARCH:
        return [(0.0, duration)]
    from ..audio import Envelope  # local import: keeps the module light to import

    envelope = Envelope.from_wav(audio)
    spans: list[tuple[float, float]] = []
    start = 0.0
    while duration - start > CHUNK_SECONDS + SPLIT_SEARCH:
        target = start + CHUNK_SECONDS
        cut = _quietest(envelope, target - SPLIT_SEARCH, target + SPLIT_SEARCH)
        spans.append((start, cut))
        start = cut
    spans.append((start, duration))
    return spans


def _quietest(envelope, lo: float, hi: float) -> float:
    """The quietest instant in [lo, hi], by the analysis envelope (dBFS)."""
    best, best_level = (lo + hi) / 2, float("inf")
    t = lo
    while t <= hi:
        level = envelope.level_at(t)
        if level < best_level:
            best, best_level = t, level
        t += envelope.frame_seconds
    return best


def _encode(audio: Path, start: float, end: float, out: Path) -> None:
    ffmpeg.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
         "-i", str(audio), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", BITRATE, str(out)],
        description="ffmpeg (asr upload)",
    )


def _iter_words(result: dict):
    """Words from a verbose_json response.  ``words`` is the flat list the
    provider returns when it supports word timestamps; fall back to the
    segments' own ``words`` if it is absent."""
    words = result.get("words")
    if not words:
        words = [w for segment in result.get("segments", []) for w in (segment.get("words") or [])]
    if not words:
        raise RuntimeError("openrouter asr returned no word timestamps; the routed provider does not support them")
    for word in words:
        text = (word.get("word") or word.get("text") or "").strip()
        start, end = word.get("start"), word.get("end")
        if not text or start is None or end is None:
            continue
        yield Word(text=text, start=float(start), end=float(end), probability=float(word.get("probability", 1.0)))
