"""The ASR seam.

Backends only have to turn a 16 kHz WAV into word-level timings; everything
else in the pipeline talks to this protocol.  That seam exists specifically so
the transcriber can be swapped without touching the detectors -- word-timing
accuracy is the main quality risk in the whole system, and swapping backends is
the escape hatch if the default drifts.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..audio import Envelope
from ..models import Word

log = logging.getLogger(__name__)

_PUNCTUATION = re.compile(r"[^\w']+", re.UNICODE)


@runtime_checkable
class Transcriber(Protocol):
    """Turns analysis audio into word-level timings."""

    def transcribe(self, audio: Path, *, language: str | None = None) -> list[Word]:
        ...


def normalize(text: str) -> str:
    """Lowercase, punctuation-free form used for lexicon and n-gram matching."""
    return _PUNCTUATION.sub("", text.strip().lower())


def clean_words(raw: list[Word], *, duration: float | None = None) -> list[Word]:
    """Drop empty tokens, clamp to the media duration, and enforce monotonic,
    non-overlapping timings.

    Backends occasionally emit a zero-length word or a timestamp slightly past
    the end of the audio; every detector downstream assumes sane ordering, so
    it is fixed once here rather than defended against everywhere.
    """
    cleaned: list[Word] = []
    previous_end = 0.0
    for word in raw:
        if not normalize(word.text):
            continue
        start = max(word.start, previous_end)
        end = max(word.end, start)
        if duration is not None:
            start = min(start, duration)
            end = min(end, duration)
        if end <= start:
            end = start + 0.02
        cleaned.append(
            Word(text=word.text.strip(), start=start, end=end, probability=word.probability)
        )
        previous_end = end
    return cleaned


def refine_timings(words: list[Word], envelope: Envelope, *, radius: float = 0.12) -> list[Word]:
    """Nudge word starts onto real speech onsets.

    Only the start moves, and only within ``radius``; the end is pulled along
    only far enough to keep the word non-empty.  This is a correction for
    alignment drift, not a re-alignment -- a word is never allowed to jump past
    its neighbours.
    """
    refined: list[Word] = []
    for index, word in enumerate(words):
        previous_end = refined[-1].end if refined else 0.0
        next_start = words[index + 1].start if index + 1 < len(words) else word.end + radius

        start = envelope.onset_near(word.start, radius)
        start = min(max(start, previous_end), word.end - 0.01, next_start - 0.02)
        start = max(start, 0.0)
        refined.append(
            Word(text=word.text, start=start, end=max(word.end, start + 0.02),
                 probability=word.probability)
        )
    return refined


def save_words(words: list[Word], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([w.model_dump() for w in words], indent=1))


def load_words(path: Path) -> list[Word]:
    return [Word(**item) for item in json.loads(path.read_text())]
