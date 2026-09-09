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
from ..models import KeepSegment, Reason, RemovalSpan

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


def _snap(removals: list[RemovalSpan], envelope: Envelope | None) -> list[RemovalSpan]:
    """Nudge each boundary to the quietest nearby instant.

    A boundary is only moved if it does not invert the span or collide with the
    neighbouring one, so snapping can never reorder or empty a span.
    """
    if envelope is None or not len(envelope.db):
        return removals

    snapped: list[RemovalSpan] = []
    for index, span in enumerate(removals):
        previous_end = snapped[-1].end if snapped else 0.0
        next_start = removals[index + 1].start if index + 1 < len(removals) else float("inf")

        start = envelope.quietest_time_near(span.start)
        end = envelope.quietest_time_near(span.end)
        if not (previous_end <= start < end <= next_start):
            start, end = span.start, span.end
        snapped.append(span.model_copy(update={"start": start, "end": end}))
    return snapped


def _drop_glitches(segments: list[KeepSegment], min_segment: float) -> list[KeepSegment]:
    """Remove keep segments too short to read as anything but a glitch."""
    return [s for s in segments if s.duration >= min_segment]


def build(
    removals: list[RemovalSpan],
    duration: float,
    preset: Preset,
    envelope: Envelope | None = None,
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
        merged = _snap(union(applied), envelope)
        removed_total = sum(span.duration for span in merged)
        if removed_total <= budget or not applied:
            break

    segments = _drop_glitches(invert(merged, duration), preset.min_segment)
    if not segments:
        log.error("every segment was cut; falling back to the uncut timeline")
        return [KeepSegment(start=0.0, end=duration)], []

    kept = sum(segment.duration for segment in segments)
    log.info(
        "edit decision list: %d segments, %.1fs of %.1fs kept (%.0f%% removed)",
        len(segments), kept, duration, (1 - kept / duration) * 100 if duration else 0,
    )
    return segments, merged
