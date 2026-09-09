"""Filler word and verbal padding removal.

Two failure modes matter here and they pull in opposite directions: leaving an
audible "um" in, and cutting the word "like" out of "I like it".  The detector
is therefore split by confidence.  Unambiguous fillers go on sight; words that
are also ordinary English are only cut when the surrounding pauses or the
recogniser's own punctuation mark them as padding.

Each removal swallows the pause *before* the filler as well.  Removing only the
word itself would leave the two pauses that surrounded it back to back -- a
longer gap than either, and one the silence detector has already declined to
touch.
"""

from __future__ import annotations

import logging

from ..asr.base import normalize
from ..config import FillerConfig
from ..models import Reason, RemovalSpan, Word

log = logging.getLogger(__name__)


def _gap_before(words: list[Word], index: int) -> float:
    if index == 0:
        return float("inf")
    return words[index].start - words[index - 1].end


def _gap_after(words: list[Word], index: int) -> float:
    if index >= len(words) - 1:
        return float("inf")
    return words[index + 1].start - words[index].end


def _comma_bounded(words: list[Word], start: int, end: int) -> bool:
    """Whisper reliably punctuates verbal padding as ", um," -- a strong signal
    that the speaker paused around it, independent of the timings."""
    before = words[start - 1].text.rstrip() if start > 0 else ","
    after = words[end].text.rstrip() if end < len(words) else ","
    return before.endswith((",", ".", "?", "!")) or after.endswith(",")


def _is_padding(words: list[Word], start: int, end: int, config: FillerConfig) -> bool:
    """Whether tokens ``[start:end]`` read as padding rather than content."""
    surrounded = (
        _gap_before(words, start) >= config.ambiguous_pause
        or _gap_after(words, end - 1) >= config.ambiguous_pause
    )
    return surrounded or _comma_bounded(words, start, end)


def _span(words: list[Word], start: int, end: int, detail: str) -> RemovalSpan:
    """Removal covering tokens ``[start:end]`` plus the pause leading into them."""
    lead = words[start - 1].end if start > 0 else words[start].start
    return RemovalSpan(
        start=min(lead, words[start].start),
        end=words[end - 1].end,
        reason=Reason.FILLER,
        detail=detail,
    )


def detect(words: list[Word], config: FillerConfig) -> list[RemovalSpan]:
    if not config.enabled or not words:
        return []

    normalized = [normalize(word.text) for word in words]
    lexicon = {w.lower() for w in config.lexicon}
    ambiguous = {w.lower() for w in config.ambiguous}
    phrases = [tuple(p) for p in config.phrases]

    spans: list[RemovalSpan] = []
    index = 0
    while index < len(words):
        # Multi-word phrases first, so "you know" is not partly matched.
        matched = False
        for phrase in phrases:
            end = index + len(phrase)
            if tuple(normalized[index:end]) != phrase:
                continue
            if words[end - 1].end - words[index].start > config.max_duration * len(phrase):
                continue
            if _is_padding(words, index, end, config):
                spans.append(_span(words, index, end, " ".join(phrase)))
                index = end
                matched = True
            break
        if matched:
            continue

        token = normalized[index]
        word = words[index]
        if token in lexicon and word.duration <= config.max_duration:
            spans.append(_span(words, index, index + 1, token))
        elif (
            token in ambiguous
            and word.duration <= config.max_duration
            and _is_padding(words, index, index + 1, config)
        ):
            spans.append(_span(words, index, index + 1, token))
        index += 1

    log.info("filler detector proposed %d spans", len(spans))
    return spans
