"""Dead-air detection.

Deliberately driven by *gaps between transcribed words* rather than by raw
energy thresholding.  A pause between words is exactly what we want to tighten,
and word boundaries give us that directly; energy thresholds instead fight room
tone, breaths and mouth noise, and cut into quiet speech.

The waveform is still consulted, but only as a veto: if a "gap" turns out to be
full of sound, the recogniser probably missed some speech and we leave it be.
"""

from __future__ import annotations

import logging

import numpy as np

from ..audio import Envelope
from ..config import SilenceConfig
from ..models import Reason, RemovalSpan, Word

log = logging.getLogger(__name__)

#: If more than this fraction of a gap is above the silence threshold, treat it
#: as missed speech rather than dead air and leave it alone.
_VOICED_VETO = 0.5


def _is_actually_silent(envelope: Envelope | None, start: float, end: float) -> bool:
    if envelope is None or not len(envelope.db):
        return True
    lo, hi = envelope._index(start), envelope._index(end)
    if hi <= lo:
        return True
    window = envelope.db[lo:hi]
    voiced = float(np.mean(window > envelope.silence_threshold))
    if voiced > _VOICED_VETO:
        log.debug("gap %.2f-%.2f is %.0f%% voiced; not cutting", start, end, voiced * 100)
        return False
    return True


def detect(
    words: list[Word],
    duration: float,
    config: SilenceConfig,
    envelope: Envelope | None = None,
) -> list[RemovalSpan]:
    """Find dead air in ``words``, including the head and tail of the video."""
    if not words:
        return []

    spans: list[RemovalSpan] = []

    def consider(start: float, end: float, detail: str) -> None:
        if end - start <= config.min_gap:
            return
        # Leave breathing room on each side so speech never sounds clipped.
        cut_start = start + config.pad
        cut_end = end - config.pad
        if cut_end - cut_start <= 0.02:
            return
        if not _is_actually_silent(envelope, start, end):
            return
        spans.append(
            RemovalSpan(
                start=cut_start, end=cut_end, reason=Reason.SILENCE, detail=detail
            )
        )

    # Leading dead air: trimmed to the pad, not to zero, so the first word has
    # a moment of air before it.
    consider(0.0, words[0].start, "leading")

    for previous, following in zip(words, words[1:]):
        consider(previous.end, following.start, "gap")

    consider(words[-1].end, duration, "trailing")

    log.info("silence detector proposed %d spans", len(spans))
    return spans
