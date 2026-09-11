"""The review step: splitting a timeline into parts, and stitching the kept ones.

The detectors are good enough to propose and not good enough to decide, so the
pipeline stops between the two.  Review is framed around what the final video
is *made of* rather than what was taken out of it, so every part of the
timeline is offered -- including the ones a detector wanted gone, which is what
makes a wrongly-removed retake recoverable.

What matters here is that a person's answer is taken literally: every guard
that makes a *detector* safe would, applied to a decision someone made
deliberately, quietly overrule them.
"""

from __future__ import annotations

import pytest

from autocut.analyze import merge
from autocut.config import Preset
from autocut.models import KeepSegment, Reason, RemovalSpan, Timeline, Word
from server.jobs import Job


def span(start: float, end: float, reason: Reason = Reason.SILENCE, detail: str = "") -> RemovalSpan:
    return RemovalSpan(start=start, end=end, reason=reason, detail=detail)


class TestSegmentsFor:
    preset = Preset()

    def test_approved_cuts_are_applied(self) -> None:
        segments = merge.segments_for([span(2, 4)], duration=10, preset=self.preset)
        assert [(s.start, s.end) for s in segments] == [(0, 2), (4, 10)]

    def test_a_rejected_cut_lets_its_neighbours_join_up(self) -> None:
        """Rejecting the middle cut must leave one continuous segment across
        it, not a seam where the cut used to be."""
        segments = merge.segments_for([span(2, 3), span(7, 8)], duration=10, preset=self.preset)
        assert [(s.start, s.end) for s in segments] == [(0, 2), (3, 7), (8, 10)]

        without_middle = merge.segments_for([span(7, 8)], duration=10, preset=self.preset)
        assert [(s.start, s.end) for s in without_middle] == [(0, 7), (8, 10)]

    def test_approving_nothing_keeps_the_whole_video(self) -> None:
        segments = merge.segments_for([], duration=10, preset=self.preset)
        assert [(s.start, s.end) for s in segments] == [(0, 10)]

    def test_unsorted_input_is_handled(self) -> None:
        """The UI sends back whatever order the user's clicks arrived in."""
        segments = merge.segments_for([span(7, 8), span(2, 3)], duration=10, preset=self.preset)
        assert [(s.start, s.end) for s in segments] == [(0, 2), (3, 7), (8, 10)]

    def test_a_persons_choice_is_not_second_guessed_by_the_budget(self) -> None:
        """``build`` drops whole detectors when they ask for too much, because
        a runaway detector should cut nothing rather than everything.  A person
        approving cuts one at a time is not a runaway detector, and the ceiling
        must not silently discard what they asked for."""
        heavy = [span(1, 9, Reason.RETAKE)]  # 80%, far over max_total_removal_ratio
        segments = merge.segments_for(heavy, duration=10, preset=self.preset)
        assert [(s.start, s.end) for s in segments] == [(0, 1), (9, 10)]

    def test_cutting_everything_falls_back_to_the_uncut_timeline(self) -> None:
        segments = merge.segments_for([span(0, 10)], duration=10, preset=self.preset)
        assert [(s.start, s.end) for s in segments] == [(0, 10)]


class TestParts:
    """Splitting the timeline into the pieces a person ticks."""

    def test_every_instant_belongs_to_exactly_one_part(self) -> None:
        pieces = merge.parts([span(2, 3), span(7, 8)], duration=10)
        assert [(a, b) for a, b, _ in pieces] == [(0, 2), (2, 3), (3, 7), (7, 8), (8, 10)]

    def test_a_part_knows_whether_a_detector_wanted_it_gone(self) -> None:
        pieces = merge.parts([span(2, 3)], duration=10)
        assert [removal is None for _, _, removal in pieces] == [True, False, True]

    def test_a_removal_at_the_very_start_produces_no_empty_part(self) -> None:
        pieces = merge.parts([span(0, 2)], duration=10)
        assert [(a, b) for a, b, _ in pieces] == [(0, 2), (2, 10)]

    def test_a_removal_running_to_the_end_produces_no_empty_part(self) -> None:
        pieces = merge.parts([span(8, 10)], duration=10)
        assert [(a, b) for a, b, _ in pieces] == [(0, 8), (8, 10)]

    def test_no_removals_is_one_part(self) -> None:
        assert [(a, b) for a, b, _ in merge.parts([], duration=10)] == [(0, 10)]

    def test_a_removal_past_the_end_is_clipped(self) -> None:
        pieces = merge.parts([span(8, 99)], duration=10)
        assert [(a, b) for a, b, _ in pieces] == [(0, 8), (8, 10)]

    def test_unsorted_removals_still_produce_an_ordered_timeline(self) -> None:
        pieces = merge.parts([span(7, 8), span(2, 3)], duration=10)
        assert [a for a, _, _ in pieces] == sorted(a for a, _, _ in pieces)


class TestPauseSplitting:
    """Parts line up with pauses, not with wherever a detector cut, and not
    only with sentence-ending punctuation.

    Splitting only at punctuation was tried first and missed the case that
    matters most: a long pause with no punctuation before it, in the middle of
    a transcribed sentence, is what speech the recogniser silently dropped
    looks like from the transcript's side.  A cut this generates in review
    starts mid-sentence sometimes -- "is headlines, one is replacing jobs," --
    which is the same trade every real cut mid-sentence already makes.
    """

    preset = Preset()
    PAUSE = 0.35

    def sentence_words(self) -> list[Word]:
        """Three short sentences, each followed by a real pause -- long enough
        to split on, and distinct from the short gaps between words in the
        same sentence, which must not be split."""
        text = "the world is here. we are in college. this is a quote."
        words, cursor, duration = [], 0.0, 0.1
        for token in text.split():
            words.append(Word(text=token, start=cursor, end=cursor + duration))
            gap = 0.5 if token.endswith(".") else 0.03
            cursor += duration + gap
        return words

    def test_an_untouched_stretch_splits_at_real_pauses(self) -> None:
        """Each sentence lands as its own part -- with the pause between two
        of them as an explicit part of its own too, matching how any other
        long, wordless pause is handled.  ``merge.parts`` never sees the
        detectors here (called with no removals), so it cannot know these
        particular pauses are ordinary rather than a place words went
        missing; that judgement is exactly why the gap gets a part instead of
        vanishing silently into whichever sentence sits next to it."""
        words = self.sentence_words()
        pieces = merge.parts(
            [], duration=words[-1].end + 0.1, words=words, part_pause=self.PAUSE
        )
        texts = [
            " ".join(w.text for w in words if w.start >= a and w.end <= b)
            for a, b, _ in pieces
        ]
        assert texts == [
            "the world is here.",
            "",
            "we are in college.",
            "",
            "this is a quote.",
        ]

    def test_a_pause_with_no_punctuation_still_splits(self) -> None:
        """The case this rule exists for: a long pause sitting mid-sentence,
        where a second attempt at the line went untranscribed."""
        words = [
            Word(text="the", start=0.0, end=0.2),
            Word(text="turtle", start=0.25, end=0.5),
            Word(text="wins", start=0.55, end=0.8),  # no sentence-ending mark
            Word(text="and", start=4.0, end=4.2),  # 3.2s later
            Word(text="takes", start=4.25, end=4.5),
            Word(text="rest.", start=4.55, end=4.8),
        ]
        pieces = merge.parts([], duration=5.0, words=words, part_pause=self.PAUSE)
        texts = [
            " ".join(w.text for w in words if w.start >= a and w.end <= b)
            for a, b, _ in pieces
        ]
        # The pause itself is a part -- not merged into the sentence before
        # it, which is the mistake that let a real one hide undetected.
        assert texts == ["the turtle wins", "", "and takes rest."]
        assert pieces[1][1] - pieces[1][0] == pytest.approx(3.2, abs=0.05)

    def test_a_short_gap_within_a_sentence_does_not_split(self) -> None:
        """Below ``part_pause``, a gap is ordinary speech rhythm, not a
        pause worth asking about."""
        words = [
            Word(text="and", start=0.0, end=0.2),
            Word(text="then", start=0.25, end=0.45),
            Word(text="also", start=0.5, end=0.7),
        ]
        pieces = merge.parts([], duration=1.0, words=words, part_pause=self.PAUSE)
        assert len(pieces) == 1

    def test_without_words_nothing_is_split(self) -> None:
        """The CLI and the tests that predate this still get whole stretches."""
        pieces = merge.parts([], duration=10)
        assert len(pieces) == 1

    def test_a_removal_is_never_split_by_a_pause(self) -> None:
        """A cut is one decision, however many sentences -- or pauses -- it
        covers."""
        words = self.sentence_words()
        cut = span(0.0, words[8].end, Reason.NGRAM_REPEAT)  # two whole sentences
        pieces = merge.parts(
            [cut], duration=words[-1].end + 0.1, words=words, part_pause=self.PAUSE
        )
        removals = [(a, b) for a, b, r in pieces if r is not None]
        assert len(removals) == 1

    def test_parts_still_tile_the_timeline_after_splitting(self) -> None:
        words = self.sentence_words()
        duration = words[-1].end + 0.1
        pieces = merge.parts([], duration=duration, words=words, part_pause=self.PAUSE)
        assert pieces[0][0] == 0.0
        assert pieces[-1][1] == pytest.approx(duration)
        for (_, end, _), (start, _, _) in zip(pieces, pieces[1:]):
            assert end == pytest.approx(start)


class TestCoalesce:
    """Two parts kept either side of a cut nobody made are one stretch."""

    def test_touching_segments_become_one(self) -> None:
        joined = merge.coalesce([KeepSegment(start=0, end=5), KeepSegment(start=5, end=9)])
        assert [(s.start, s.end) for s in joined] == [(0, 9)]

    def test_separated_segments_stay_apart(self) -> None:
        joined = merge.coalesce([KeepSegment(start=0, end=5), KeepSegment(start=7, end=9)])
        assert [(s.start, s.end) for s in joined] == [(0, 5), (7, 9)]

    def test_a_run_of_touching_segments_collapses_to_one(self) -> None:
        joined = merge.coalesce([
            KeepSegment(start=0, end=2), KeepSegment(start=2, end=4),
            KeepSegment(start=4, end=6),
        ])
        assert [(s.start, s.end) for s in joined] == [(0, 6)]

    def test_unsorted_input_is_handled(self) -> None:
        joined = merge.coalesce([KeepSegment(start=5, end=9), KeepSegment(start=0, end=5)])
        assert [(s.start, s.end) for s in joined] == [(0, 9)]

    def test_nothing_coalesces_to_nothing(self) -> None:
        assert merge.coalesce([]) == []


@pytest.fixture
def job(tmp_path) -> Job:
    words = [
        Word(text="the", start=0.0, end=0.3),
        Word(text="first", start=0.3, end=0.6),
        Word(text="thing", start=0.6, end=0.9),
        # The good take runs contiguously from where the retake removal ends
        # to where the trailing-silence removal begins, with no gap either
        # word can leave unaccounted for -- this fixture is about the debris
        # and revision logic below, not about the untranscribed-gap detection
        # covered in its own tests above.
        Word(text="the", start=1.0, end=1.6),
        Word(text="first", start=1.6, end=2.2),
        Word(text="thing", start=2.2, end=2.8),
        Word(text="is", start=2.8, end=3.4),
    ]
    timeline = Timeline(
        source=tmp_path / "input.mp4", duration=5.0, fps=30.0, width=1920, height=1080,
        words=words,
        removals=[
            span(0.0, 1.0, Reason.NGRAM_REPEAT, "the first thing"),
            span(3.4, 4.4, Reason.SILENCE, "trailing"),
        ],
        # The edit decision list as the merge stage actually leaves it: the
        # 0.6s tail after the last removal is debris and never made it in, so
        # the removals and the segments together do not tile the timeline.
        keep_segments=[KeepSegment(start=1.0, end=3.4)],
    )
    return Job(
        id="test", filename="input.mp4", source=tmp_path / "input.mp4",
        work_dir=tmp_path, timeline=timeline,
    )


class TestPartsPayload:
    """What the review UI is given: the video as an ordered list of parts."""

    def test_a_part_carries_the_words_spoken_in_it(self, job) -> None:
        assert job.parts()[0]["text"] == "the first thing"

    def test_video_in_the_edit_starts_kept(self, job) -> None:
        """The stretch between the two proposals is video nobody objected to."""
        middle = job.parts()[1]
        assert middle["reason"] is None
        assert middle["keep"] is True

    def test_a_gap_the_transcript_cannot_account_for_is_flagged(self, job) -> None:
        """The bug report this exists for: a retake spoken a second time went
        untranscribed, sitting mid-sentence with no punctuation before it, and
        the only sign anything was there at all was a pause ten times longer
        than any other non-sentence-boundary gap in the clip.  It must come
        back as its own part -- not silently absorbed into the sentence
        around it -- so a person reviewing the edit can hear it and decide."""
        job.timeline.words = [
            Word(text="the", start=0.0, end=0.2),
            Word(text="turtle", start=0.25, end=0.5),
            Word(text="wins", start=0.55, end=0.8),  # no sentence-ending mark
            Word(text="and", start=4.0, end=4.2),  # 3.2s later -- the anomaly
            Word(text="takes", start=4.25, end=4.5),
            Word(text="rest.", start=4.55, end=4.8),
        ]
        job.timeline.duration = 5.0
        job.timeline.removals = []
        job.timeline.keep_segments = [KeepSegment(start=0.0, end=5.0)]

        parts = job.parts()
        gap = next(p for p in parts if p["text"] == "" and 0.8 <= p["start"] < 4.0)
        assert gap["reason"] == "untranscribed"
        assert gap["keep"] is True, "real audio defaults to kept, for a person to judge"
        assert gap["duration"] == pytest.approx(3.2, abs=0.05)

    def test_debris_between_cuts_starts_dropped_and_says_why(self, job) -> None:
        """A sliver the merge stage discarded is in neither the removals nor
        the segments.  Defaulting it to kept would put back exactly the
        fragments that make an edit sound abrupt."""
        offcut = job.parts()[3]
        assert offcut["reason"] == "offcut"
        assert offcut["keep"] is False

    def test_the_default_selection_is_the_pipelines_own_edit(self, job) -> None:
        kept = [p for p in job.parts() if p["keep"]]
        assert sum(p["duration"] for p in kept) == pytest.approx(
            sum(s.duration for s in job.timeline.keep_segments)
        )

    def test_a_part_a_detector_wanted_gone_starts_dropped(self, job) -> None:
        first = job.parts()[0]
        assert first["reason"] == "ngram_repeat"
        assert first["keep"] is False

    def test_a_dropped_part_is_still_offered(self, job) -> None:
        """This is what makes a wrongly-removed retake recoverable: it is in
        the list, unticked, rather than absent."""
        assert any(p["reason"] and not p["keep"] for p in job.parts())

    def test_the_parts_tile_the_whole_timeline(self, job) -> None:
        parts = job.parts()
        assert parts[0]["start"] == 0.0
        assert parts[-1]["end"] == pytest.approx(job.timeline.duration)
        for earlier, later in zip(parts, parts[1:]):
            assert earlier["end"] == pytest.approx(later["start"])

    def test_a_previous_answer_is_reflected_back(self, job) -> None:
        """So revising an edit starts from what was chosen last time."""
        job.keep = [0]
        assert [p["keep"] for p in job.parts()] == [True, False, False, False]  # noqa: E501

    def test_ids_are_the_order_of_the_parts(self, job) -> None:
        assert [p["id"] for p in job.parts()] == [0, 1, 2, 3]

    def test_a_partly_overlapped_word_is_not_reported_in_a_part(self, job) -> None:
        """Only words wholly inside a part are spoken in it; a word straddling
        the boundary belongs to neither.

        A filler, not a silence: a cut this short would be absorbed if it were
        a pause, which is the subject of its own test below.
        """
        job.timeline.removals = [span(0.0, 0.5, Reason.FILLER, "um")]
        assert job.parts()[0]["text"] == "the"

    def test_a_job_with_nothing_proposed_yet_has_no_parts(self, tmp_path) -> None:
        bare = Job(id="x", filename="a.mp4", source=tmp_path / "a.mp4", work_dir=tmp_path)
        assert bare.parts() == []
