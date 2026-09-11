"""Dead-air detection.

Driven by the transcript rather than by raw energy thresholding alone: a pause
*between words* is what we actually want to tighten, and word boundaries give
us that directly, where a bare energy threshold has no idea whether the quiet
stretch it found is a pause worth cutting or a beat inside a sentence.

The waveform then decides where, precisely, the dead air is.  Within every
region the transcript nominates, only the stretches that are genuinely below
the noise floor are cut.  That distinction matters: a "gap" is frequently part
silence and part speech the recogniser dropped, and cutting the whole gap
because most of it was quiet deletes the words that were in it.

Two kinds of region are nominated.  The obvious one is the gap between two
words, plus the head and tail of the video.  The other is the inside of a word
whose timestamp is implausibly long -- see ``asr.base.trim_overlong``, which
repairs the common case, but a word can also fall silent in the middle.
"""

from __future__ import annotations

import logging

from ..audio import Envelope
from ..config import SilenceConfig
from ..models import Reason, RemovalSpan, Word

log = logging.getLogger(__name__)

#: Trailing characters that do not change whether a word ended a sentence --
#: Whisper puts the closing quote after the full stop, as anyone would.
_CLOSERS = "\"')]}\u00bb\u201d\u2019"


def ends_sentence(text: str) -> bool:
    """Whether this word is the last one of a sentence."""
    return text.rstrip().rstrip(_CLOSERS).endswith((".", "?", "!", "\u2026"))


def _floor_after(word: Word, config: SilenceConfig) -> float:
    """The earliest a cut may start, given the word it follows."""
    return word.end + config.sentence_pause if ends_sentence(word.text) else 0.0


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
    has_envelope = envelope is not None and len(envelope.db) > 0

    def cut(start: float, end: float, detail: str, floor: float = 0.0) -> None:
        if end - start <= config.min_gap:
            return
        # Leave breathing room on each side so speech never sounds clipped.
        # ``floor`` is the earliest a cut may begin, which is how a sentence
        # keeps its beat: after a full stop it sits a good half-second past
        # the word, so tightening the pause can never take all of it.
        cut_start = max(start + config.pad, floor)
        cut_end = end - config.pad
        if cut_end - cut_start <= 0.02:
            return
        spans.append(
            RemovalSpan(
                start=cut_start, end=cut_end, reason=Reason.SILENCE, detail=detail
            )
        )

    def consider(
        start: float,
        end: float,
        detail: str,
        *,
        transcribed: bool = False,
        floor: float = 0.0,
    ) -> None:
        """Cut the dead air in ``[start, end)`` -- not necessarily all of it.

        Without a waveform we can only take the transcript's word for it and
        cut the whole region.  With one, we prefer to cut just the stretches
        that fall below the silence threshold, so speech the recogniser missed
        survives even though no word was transcribed over it.

        When no stretch falls below the threshold at all, the waveform has not
        proved there is speech here -- only that the room is loud.  Deciding
        that means "leave it alone" hands the whole judgement to an energy
        threshold, which is precisely what this detector is built to avoid: on
        a noisy recording nothing is ever below threshold and every pause
        survives.  So the transcript gets the final word, unless the region
        holds a *sustained* run at speech level, which is what dropped words
        actually look like.

        That last step applies only where the transcript claims nothing is
        spoken.  ``transcribed`` marks a region the recogniser placed a word
        over -- the inside of an over-long word -- and there the waveform must
        prove silence before anything is cut, or a noisy recording would have
        the middles of its words removed.
        """
        if end - start <= config.min_gap:
            return
        if not has_envelope:
            cut(start, end, detail, floor)
            return

        runs = envelope.silent_runs(
            start, end, config.min_gap,
            bridge=config.bridge, sustained=config.sustained_speech,
        )
        if runs:
            for run_start, run_end in runs:
                cut(run_start, run_end, detail, floor)
            return
        if transcribed:
            return

        speech = envelope.longest_run_above(
            envelope.speech_confidence_level, start, end
        )
        if speech >= config.sustained_speech:
            log.debug(
                "leaving %.2f-%.2f: holds %.2fs of speech-level audio",
                start, end, speech,
            )
            return
        cut(start, end, detail, floor)

    # Leading dead air: trimmed to the pad, not to zero, so the first word has
    # a moment of air before it.
    consider(0.0, words[0].start, "leading")

    for previous, following in zip(words, words[1:]):
        consider(
            previous.end, following.start, "gap",
            floor=_floor_after(previous, config),
        )

    # The last sentence gets its beat too, rather than the video stopping the
    # instant the final word does.
    consider(
        words[-1].end, duration, "trailing", floor=_floor_after(words[-1], config)
    )

    # Dead air hiding *inside* a word.  Gap-based detection is blind to this:
    # if a word's end timestamp runs on through a pause there is no gap to
    # find, and the pause survives into the render.  Only implausibly long
    # words are scanned, so a normally-timed word is never picked apart.
    if has_envelope:
        for word in words:
            if word.duration > config.max_word_duration:
                consider(word.start, word.end, "within-word", transcribed=True)

    spans.sort(key=lambda span: span.start)
    log.info("silence detector proposed %d spans", len(spans))
    return spans
