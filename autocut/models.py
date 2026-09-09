"""Core data structures shared by every pipeline stage.

The most important type here is :class:`TimeMap`.  Cutting the video changes the
timebase, so anything produced before the cut (word timings, face tracks) is in
*source* time while anything the renderer consumes afterwards (captions, the
punch-in schedule) must be in *output* time.  Every conversion between the two
goes through ``TimeMap`` -- it is the single place that knowledge lives.
"""

from __future__ import annotations

from bisect import bisect_right
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class Reason(StrEnum):
    """Why a span was proposed for removal.  Kept on the span for debugging and
    so the UI can explain a cut without re-running the detector."""

    SILENCE = "silence"
    FILLER = "filler"
    RETAKE = "retake"
    NGRAM_REPEAT = "ngram_repeat"


class Word(BaseModel):
    """One transcribed word with its source-time span."""

    text: str
    start: float
    end: float
    probability: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, end: float, info) -> float:
        start = info.data.get("start")
        if start is not None and end < start:
            raise ValueError(f"word end {end} precedes start {start}")
        return end


class RemovalSpan(BaseModel):
    """A source-time range a detector wants gone."""

    start: float
    end: float
    reason: Reason
    confidence: float = 1.0
    detail: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


class KeepSegment(BaseModel):
    """A source-time range that survives into the output.  The ordered list of
    these *is* the edit decision list."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class TimeMap:
    """Bidirectional map between source time and post-cut output time.

    Built once from the merged edit decision list and then handed to every
    downstream stage.  Times inside a removed region collapse onto the cut
    point, which keeps the map total and monotonic -- callers never have to
    handle a ``None`` for an out-of-range timestamp.
    """

    def __init__(self, segments: list[KeepSegment]) -> None:
        self.segments = list(segments)
        self._src_starts: list[float] = []
        self._src_ends: list[float] = []
        self._out_starts: list[float] = []

        elapsed = 0.0
        previous_end = float("-inf")
        for segment in self.segments:
            if segment.start < previous_end:
                raise ValueError("keep segments must be sorted and non-overlapping")
            previous_end = segment.end
            self._src_starts.append(segment.start)
            self._src_ends.append(segment.end)
            self._out_starts.append(elapsed)
            elapsed += segment.duration
        self.output_duration = elapsed

    def __len__(self) -> int:
        return len(self.segments)

    @classmethod
    def identity(cls, duration: float) -> TimeMap:
        """A map for an uncut timeline -- source and output times are equal."""
        return cls([KeepSegment(start=0.0, end=duration)])

    def _index_at_or_before(self, t: float) -> int:
        """Index of the last segment starting at or before ``t``; -1 if ``t``
        precedes every segment."""
        return bisect_right(self._src_starts, t) - 1

    def is_kept(self, t: float) -> bool:
        i = self._index_at_or_before(t)
        return i >= 0 and t <= self._src_ends[i]

    def to_output(self, t: float) -> float:
        """Source time -> output time.  Times inside a removed region map to the
        cut point, i.e. the output time of the preceding segment's end."""
        if not self.segments:
            return 0.0
        i = self._index_at_or_before(t)
        if i < 0:
            return 0.0
        if t <= self._src_ends[i]:
            return self._out_starts[i] + (t - self._src_starts[i])
        # In the gap after segment i (or past the end): collapse onto the cut.
        return self._out_starts[i] + self.segments[i].duration

    def to_source(self, t_out: float) -> float:
        """Output time -> source time."""
        if not self.segments:
            return 0.0
        t_out = min(max(t_out, 0.0), self.output_duration)
        i = max(bisect_right(self._out_starts, t_out) - 1, 0)
        offset = t_out - self._out_starts[i]
        return min(self._src_starts[i] + offset, self._src_ends[i])

    def kept_fraction(self, start: float, end: float) -> float:
        """How much of a source span survives the cut, as a fraction.

        Snapping cut points to quiet samples can leave a few milliseconds of a
        removed word alive.  Callers use this to tell "this word survived" from
        "a sliver of this word survived" -- a distinction that matters for
        captions, which would otherwise flash a word that was cut.
        """
        total = end - start
        if total <= 0:
            return 1.0 if self.is_kept(start) else 0.0
        kept = sum(
            max(0.0, min(end, segment.end) - max(start, segment.start))
            for segment in self.segments
        )
        return kept / total

    def map_span(self, start: float, end: float) -> tuple[float, float] | None:
        """Map a source-time span into output time, or ``None`` if it did not
        survive the cut.

        A span straddling a cut is clipped to whichever keep segment it overlaps
        most, so a word that was half-removed still gets a sane caption slot
        rather than a zero-length or wildly stretched one.
        """
        best_index = -1
        best_overlap = 0.0
        for i, segment in enumerate(self.segments):
            if segment.start >= end:
                break
            overlap = min(end, segment.end) - max(start, segment.start)
            if overlap > best_overlap:
                best_overlap, best_index = overlap, i
        if best_index < 0 or best_overlap <= 0.0:
            return None
        segment = self.segments[best_index]
        clipped_start = max(start, segment.start)
        clipped_end = min(end, segment.end)
        return self.to_output(clipped_start), self.to_output(clipped_end)


class ZoomSpan(BaseModel):
    """A stretch of *output* time held at one zoom level.

    Punch-ins are scheduled after the cut, so unlike everything else produced
    by the analysis stage these times are already in the output timebase.
    """

    start: float
    end: float
    zoom: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


class Timeline(BaseModel):
    """Everything known about one video as it moves through the pipeline.

    Serialised to ``timeline.json`` in the job directory after each stage so a
    run can be resumed or inspected without redoing the expensive work.
    """

    source: Path
    duration: float
    fps: float
    width: int
    height: int
    audio: Path | None = None
    words: list[Word] = Field(default_factory=list)
    removals: list[RemovalSpan] = Field(default_factory=list)
    keep_segments: list[KeepSegment] = Field(default_factory=list)

    def time_map(self) -> TimeMap:
        if not self.keep_segments:
            return TimeMap.identity(self.duration)
        return TimeMap(self.keep_segments)

    @property
    def kept_duration(self) -> float:
        return sum(segment.duration for segment in self.keep_segments)
