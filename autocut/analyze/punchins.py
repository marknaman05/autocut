"""Zoom punch-in scheduling.

A jump cut with no visual change reads as a glitch; the same cut with a small
change of framing reads as an edit.  So the punch-in schedule is derived from
the cut points themselves -- zoom only ever changes *on* a cut, never during a
shot, which is what makes it look deliberate rather than like a camera move.

Restraint is the whole game: a minimum hold time keeps the framing from
strobing across a run of short segments.
"""

from __future__ import annotations

import logging

from ..config import PunchInConfig
from ..models import KeepSegment, TimeMap, ZoomSpan

log = logging.getLogger(__name__)


def schedule(
    segments: list[KeepSegment], time_map: TimeMap, config: PunchInConfig
) -> list[ZoomSpan]:
    """Assign a zoom level to each output-time segment.

    Always returns a schedule covering the whole output, so the reframe stage
    can treat it as the single source of truth for scale.
    """
    if not segments:
        return []

    base = config.levels[0] if config.levels else 1.0
    if not config.enabled or len(config.levels) < 2:
        return [ZoomSpan(start=0.0, end=time_map.output_duration, zoom=base)]

    spans: list[ZoomSpan] = []
    level_index = 0
    held_for = 0.0

    for segment in segments:
        start = time_map.to_output(segment.start)
        end = start + segment.duration

        # Change framing only when the previous one has been on screen long
        # enough, and only if this shot is long enough to register.
        if (
            spans
            and held_for >= config.min_hold
            and segment.duration >= config.min_segment
        ):
            level_index = (level_index + 1) % len(config.levels)
            held_for = 0.0

        zoom = config.levels[level_index]
        if spans and spans[-1].zoom == zoom:
            # Same framing as the previous shot: extend it rather than emitting
            # a second span, so the renderer sees one continuous hold.
            spans[-1].end = end
        else:
            spans.append(ZoomSpan(start=start, end=end, zoom=zoom))
        held_for += segment.duration

    if spans:
        spans[-1].end = max(spans[-1].end, time_map.output_duration)
    log.info("scheduled %d punch-in holds", len(spans))
    return spans
