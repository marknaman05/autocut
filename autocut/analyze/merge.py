"""Reconcile every detector's proposals into one edit decision list.

Detectors are deliberately independent and a little eager; this module is where
their output is made safe.  Two rules matter most:

* boundaries are snapped to quiet instants, so joins do not click;
* there is a hard ceiling on how much of the video may disappear, enforced by
  dropping whole detectors in priority order rather than by trimming spans.
  If something has gone wrong upstream, the failure mode is a video that is cut
  too little, never one that is cut to shreds.
"""

from __future__ import annotations

import logging

from ..audio import Envelope
from ..config import Preset
from ..models import KeepSegment, Reason, RemovalSpan, Word

log = logging.getLogger(__name__)

#: Order in which detectors are sacrificed when the removal budget is blown.
#: Least trustworthy first.
_SACRIFICE_ORDER = (Reason.RETAKE, Reason.NGRAM_REPEAT, Reason.FILLER)


def union(spans: list[RemovalSpan]) -> list[RemovalSpan]:
    """Collapse overlapping and touching spans into a minimal covering set.

    The surviving span keeps the reason of its longest contributor, which is
    what the UI shows when explaining a cut.
    """
    if not spans:
        return []

    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    merged: list[RemovalSpan] = [ordered[0].model_copy()]
    for span in ordered[1:]:
        current = merged[-1]
        if span.start <= current.end:
            if span.duration > current.duration:
                current.reason = span.reason
                current.detail = span.detail
            current.end = max(current.end, span.end)
            current.confidence = min(current.confidence, span.confidence)
        else:
            merged.append(span.model_copy())
    return merged


def invert(removals: list[RemovalSpan], duration: float) -> list[KeepSegment]:
    """Turn removal spans into the keep segments between them."""
    segments: list[KeepSegment] = []
    cursor = 0.0
    for span in removals:
        start = max(span.start, 0.0)
        if start > cursor:
            segments.append(KeepSegment(start=cursor, end=min(start, duration)))
        cursor = max(cursor, min(span.end, duration))
    if cursor < duration:
        segments.append(KeepSegment(start=cursor, end=duration))
    return [s for s in segments if s.duration > 0]


#: How much of a word a removal was created to delete may survive relaxation.
#: Backing a boundary off a word's onset is worth a few milliseconds of the
#: word before it; backing off far enough to leave the word audible is not
#: avoiding a clip any more, it is failing to make the cut.
_RELAX_TOLERANCE = 0.1


def _snap(
    removals: list[RemovalSpan],
    envelope: Envelope | None,
    relax: float = 0.0,
    words: list[Word] | None = None,
) -> list[RemovalSpan]:
    """Move each boundary onto quiet audio, so the join does not click.

    Two things happen here, in order.  First the boundary is nudged to the
    quietest instant within a couple of frames -- but only if there *is* one,
    since snapping to the dip between two syllables lands inside the word
    rather than beside it.

    Then, if the boundary is still sitting on sound, it is relaxed off it: the
    start of a removal moves later and the end moves earlier, until both are
    quiet.  A word's audible extent is wider than the timestamps say -- it
    begins a little before its start and rings on after its end -- so a cut
    made exactly at a timestamp truncates a decay or clips an onset, which is
    what an abrupt edit sounds like.  Relaxing only ever shrinks a removal, so
    the worst case is keeping a few milliseconds too many.

    A boundary is only moved if it does not invert the span or collide with the
    neighbouring one, so neither step can reorder or empty a span.
    """
    if envelope is None or not len(envelope.db):
        return removals

    snapped: list[RemovalSpan] = []
    for index, span in enumerate(removals):
        previous_end = snapped[-1].end if snapped else 0.0
        next_start = removals[index + 1].start if index + 1 < len(removals) else float("inf")

        quiet = envelope.silence_threshold
        start = envelope.quietest_time_near(span.start, max_level=quiet)
        end = envelope.quietest_time_near(span.end, max_level=quiet)
        if not (previous_end <= start < end <= next_start):
            start, end = span.start, span.end

        if relax > 0.0:
            relaxed_start = envelope.advance_to_quiet(start, relax)
            relaxed_end = envelope.retreat_to_quiet(end, relax)
            # A filler or retake removal covers speech on purpose, and the
            # audio at its edges is the very thing being deleted.  Relaxing
            # off that shrinks the cut instead of protecting a neighbour, so
            # it may not retreat past the words the removal exists to remove.
            covered = [
                word for word in (words or [])
                if word.start >= span.start and word.end <= span.end
            ]
            if covered:
                relaxed_start = min(relaxed_start, covered[0].start + _RELAX_TOLERANCE)
                relaxed_end = max(relaxed_end, covered[-1].end - _RELAX_TOLERANCE)
            if relaxed_start < relaxed_end:
                start, end = relaxed_start, relaxed_end

        snapped.append(span.model_copy(update={"start": start, "end": end}))
    return snapped


def _drop_glitches(
    segments: list[KeepSegment],
    preset: Preset,
    words: list[Word] | None = None,
) -> list[KeepSegment]:
    """Remove keep segments too short to read as anything but a glitch.

    Duration alone is not enough to judge this.  What survives between two
    adjacent cuts is often a fragment carrying no word at all -- a breath, or
    the clipped tail of the very word the cut was meant to remove, because the
    boundary landed a few milliseconds early.  Spliced between the two cuts,
    that fragment is audible as a chirp, and it is the usual reason an
    otherwise good edit sounds abrupt.

    So a segment holding a complete word is content and is kept at any length,
    while one holding none must be substantial to survive -- long enough that
    it is more likely speech the recogniser missed than debris.
    """
    if words is None:
        return [s for s in segments if s.duration >= preset.min_segment]

    kept: list[KeepSegment] = []
    for segment in segments:
        if segment.duration < preset.min_segment:
            continue
        has_word = any(
            word.start >= segment.start - 0.01 and word.end <= segment.end + 0.01
            for word in words
        )
        if not has_word and segment.duration < preset.min_wordless_segment:
            log.debug(
                "dropping %.2f-%.2f: %.2fs with no whole word in it",
                segment.start, segment.end, segment.duration,
            )
            continue
        kept.append(segment)
    return kept


def coalesce(segments: list[KeepSegment]) -> list[KeepSegment]:
    """Join segments that touch, so a run of them becomes one.

    Two adjacent parts kept either side of a cut nobody made are not two
    segments -- they are one uninterrupted stretch of video.  Leaving them
    split would put a cut point where there is no cut, which the punch-in
    scheduler reads as an edit and reacts to with a zoom change.
    """
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    joined = [ordered[0].model_copy()]
    for segment in ordered[1:]:
        current = joined[-1]
        if segment.start <= current.end + 1e-6:
            current.end = max(current.end, segment.end)
        else:
            joined.append(segment.model_copy())
    return joined


def _pause_splits(words: list[Word], start: float, end: float, min_pause: float) -> list[float]:
    """Times inside ``(start, end)`` where the speaker paused long enough to
    mark a new part.

    Splitting only at sentence-ending punctuation was tried first, and missed
    the case that matters most: a word gap with no punctuation before it at
    all, sitting in the middle of a transcribed sentence, is the signature of
    speech the recogniser silently dropped.  In one recording a 3.18s gap
    after "down" -- ten times longer than every other pause in the clip that
    was *not* at a sentence boundary -- turned out to hold a second,
    unrecognised attempt at the line, invisible to every detector because
    none of them can act on words that were never transcribed.

    So every gap this long becomes its own part, not only the ones that land
    on a period.  It has to be its own part rather than merely a boundary
    between its neighbours: a boundary alone would glue the silent stretch
    onto whichever sentence comes before it, which is exactly what let a
    5.29s pause hide inside "...understand?" in one recording and a 3.18s one
    hide inside "...rabbit slows down and takes rest" in another.  Both
    boundaries of the gap are returned, so the gap becomes a part with no
    words in it -- silence.detect having already left it alone is what tells
    you it is not silence -- which is the nudge to listen to it before
    trusting the sentence it was hiding in.

    The gap before the first word and after the last word of ``(start, end)``
    counts too.  ``(start, end)`` is bounded by removals, not by words, so the
    stretch between a removal ending and the next real word beginning is just
    as capable of hiding dropped speech as a gap between two transcribed
    words -- and did, once VAD-corrected timings moved a word's start later
    without moving the removal boundary that used to sit right next to it.
    """
    inside = [word for word in words if word.start >= start and word.end <= end]
    if not inside:
        return []

    splits: list[float] = []
    if inside[0].start - start >= min_pause:
        splits.append(inside[0].start)
    for word, following in zip(inside, inside[1:]):
        if following.start - word.end >= min_pause:
            splits.append(word.end)
            splits.append(following.start)
    if end - inside[-1].end >= min_pause:
        splits.append(inside[-1].end)
    return splits


def parts(
    removals: list[RemovalSpan],
    duration: float,
    words: list[Word] | None = None,
    min_pause: float = 0.0,
    part_pause: float = 0.0,
) -> list[tuple[float, float, RemovalSpan | None]]:
    """Split the whole timeline into contiguous pieces, in order.

    Every instant of the video belongs to exactly one piece, and a piece is
    either a stretch the detectors left alone or one they proposed removing --
    the third element is the proposal, or ``None`` for untouched video.

    Given ``words``, untouched stretches are split again wherever the speaker
    paused for at least ``part_pause``, so the pieces line up with what was
    said rather than with wherever a detector happened to cut -- and so a
    pause holding words the recogniser missed becomes its own part rather
    than vanishing into the sentence around it.  Removals are never split: a
    cut is one decision however many sentences it spans.

    ``min_pause`` sets the shortest silence worth asking about.  Pauses below
    it are left in the video and never appear as a part -- the natural beat
    between two sentences is rhythm, not dead air, and surfacing every one of
    them buries the cuts that actually matter in a list of decisions nobody
    wants to make.  Only silence is treated this way; a short filler or retake
    is precisely the kind of small cut worth offering.

    This is the timeline as a person reviews it: an ordered list of parts to
    tick or untick, rather than a list of deletions to argue with.
    """
    pieces: list[tuple[float, float, RemovalSpan | None]] = []

    def add_kept(start: float, end: float) -> None:
        cursor = start
        if words:
            for split in _pause_splits(words, start, end, part_pause):
                if split > cursor:
                    pieces.append((cursor, split, None))
                    cursor = split
        pieces.append((cursor, end, None))

    worth_asking = [
        span
        for span in removals
        if span.reason is not Reason.SILENCE or span.duration >= min_pause
    ]

    cursor = 0.0
    for span in sorted(worth_asking, key=lambda s: s.start):
        start = max(span.start, 0.0)
        end = min(span.end, duration)
        if start > cursor:
            add_kept(cursor, start)
        if end > start:
            pieces.append((start, end, span))
        cursor = max(cursor, end)
    if cursor < duration:
        add_kept(cursor, duration)
    return [(a, b, span) for a, b, span in pieces if b - a > 1e-6]


def segments_for(
    removals: list[RemovalSpan],
    duration: float,
    preset: Preset,
    words: list[Word] | None = None,
) -> list[KeepSegment]:
    """The keep segments left by an already-decided set of removals.

    Used when a person has chosen which proposed cuts to apply.  Their choice
    is final -- no budget ceiling, no snapping, no detector sacrificed -- so
    this deliberately skips everything :func:`build` does to make a detector's
    output safe.  The spans arriving here have been through that already, and
    then been approved one at a time.

    Rejecting a cut simply lets the segments either side of it join up, which
    ``invert`` handles by construction.
    """
    ordered = sorted(removals, key=lambda span: span.start)
    segments = _drop_glitches(invert(ordered, duration), preset, words)
    if not segments:
        log.error("every segment was cut; falling back to the uncut timeline")
        return [KeepSegment(start=0.0, end=duration)]
    return segments


def build(
    removals: list[RemovalSpan],
    duration: float,
    preset: Preset,
    envelope: Envelope | None = None,
    words: list[Word] | None = None,
) -> tuple[list[KeepSegment], list[RemovalSpan]]:
    """Build the edit decision list.

    Returns the keep segments and the removals actually applied -- which may be
    fewer than proposed, if the removal budget forced a detector to be dropped.
    """
    applied = list(removals)
    budget = duration * preset.max_total_removal_ratio

    for sacrifice in (None, *_SACRIFICE_ORDER):
        if sacrifice is not None:
            applied = [span for span in applied if span.reason != sacrifice]
            log.warning(
                "removal budget exceeded; dropping all %s spans and retrying", sacrifice
            )
        merged = _snap(union(applied), envelope, preset.boundary_relax, words)
        removed_total = sum(span.duration for span in merged)
        if removed_total <= budget or not applied:
            break

    segments = _drop_glitches(invert(merged, duration), preset, words)
    if not segments:
        log.error("every segment was cut; falling back to the uncut timeline")
        return [KeepSegment(start=0.0, end=duration)], []

    kept = sum(segment.duration for segment in segments)
    log.info(
        "edit decision list: %d segments, %.1fs of %.1fs kept (%.0f%% removed)",
        len(segments), kept, duration, (1 - kept / duration) * 100 if duration else 0,
    )
    return segments, merged
