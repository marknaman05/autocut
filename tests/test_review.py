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

import asyncio
import json

import pytest

from autocut.analyze import merge
from autocut.config import Preset
from tests.test_merge import envelope
from autocut.audio import Envelope
from autocut.models import KeepSegment, Reason, RemovalSpan, Timeline, Word
from server.jobs import Job, JobManager


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


def make_job(tmp_path) -> Job:
    """A job at the review stage, with one retake and one trailing silence."""
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


@pytest.fixture
def job(tmp_path) -> Job:
    return make_job(tmp_path)


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


class TestPartBoundariesLeaveSpeech:
    """A part boundary must not land in the middle of a word.

    A pause split is placed at a word timestamp, and a word is audible either
    side of the one it is given: it starts a little early and rings on after
    it ends.  Splicing exactly at the timestamp clips the tail of the sentence
    before the boundary or the onset of the one after it -- measured on a real
    recording, the end of a part sat at -29 dB with the silence threshold at
    -53 dB, which is the last word of the sentence being cut in half.

    Relaxing a boundary only ever *grows* the speech a part holds, so the worst
    outcome is a few milliseconds of room tone kept.
    """

    PAUSE = 0.35
    RELAX = 0.25
    #: Indices of the two parts that hold a sentence.  The others are the
    #: silence before the first word, the pause between the sentences, and the
    #: silence after the last word -- each a part in its own right.
    SPOKEN = (1, 3)

    def words(self) -> list[Word]:
        # Timestamps deliberately tighter than the audio: each sentence's
        # first word starts after its onset and its last word ends before its
        # tail has died away, which is how Whisper actually places them.
        return [
            Word(text="the", start=1.0, end=1.3),
            Word(text="turtle", start=1.35, end=1.7),
            Word(text="and", start=3.1, end=3.4),
            Word(text="rest.", start=3.45, end=3.8),
        ]

    def envelope(self) -> Envelope:
        return envelope(5.0, [(0.9, 1.9), (3.0, 3.95)])

    def boundaries(self, **kwargs) -> list[tuple[float, float]]:
        return [
            (a, b)
            for a, b, _ in merge.parts(
                [], duration=5.0, words=self.words(), part_pause=self.PAUSE, **kwargs
            )
        ]

    def test_unrelaxed_boundaries_land_inside_the_words(self) -> None:
        """What the bug looked like, so the fix below is not vacuous."""
        level = self.envelope()
        loud = [
            edge
            for index in self.SPOKEN
            for edge in self.boundaries()[index]
            if level.level_at(edge) >= level.silence_threshold
        ]
        assert loud, "the unfixed boundaries should sit on speech"

    def test_every_spoken_part_is_bounded_by_silence(self) -> None:
        level = self.envelope()
        parts = self.boundaries(envelope=level, relax=self.RELAX)
        for index in self.SPOKEN:
            for edge in parts[index]:
                assert level.level_at(edge) < level.silence_threshold, (
                    f"part {index} still joins on speech at {edge:.2f}"
                )

    def test_a_spoken_part_grows_to_cover_the_whole_word(self) -> None:
        """Specifically: past the tail of the last word and before the onset
        of the first, which is the audio the timestamps leave out."""
        parts = self.boundaries(envelope=self.envelope(), relax=self.RELAX)
        first_start, first_end = parts[1]
        assert first_start < 1.0 and first_end > 1.7
        second_start, second_end = parts[3]
        assert second_start < 3.1 and second_end > 3.8

    def test_the_pause_between_them_survives_as_its_own_part(self) -> None:
        """Both boundaries move inward toward each other and neither may
        cross: the pause has to stay a part, or a part someone reviewed stops
        existing between review and render."""
        parts = self.boundaries(envelope=self.envelope(), relax=self.RELAX)
        assert len(parts) == len(self.boundaries())
        start, end = parts[2]
        assert end > start

    def test_relaxing_never_shrinks_a_spoken_part(self) -> None:
        plain = self.boundaries()
        relaxed = self.boundaries(envelope=self.envelope(), relax=self.RELAX)
        for index in self.SPOKEN:
            assert relaxed[index][0] <= plain[index][0]
            assert relaxed[index][1] >= plain[index][1]

    def test_the_parts_still_tile_the_timeline(self) -> None:
        """Every instant belongs to exactly one part, before and after."""
        parts = self.boundaries(envelope=self.envelope(), relax=self.RELAX)
        assert parts[0][0] == 0.0
        assert parts[-1][1] == pytest.approx(5.0)
        for (_, end), (start, _) in zip(parts, parts[1:]):
            assert start == pytest.approx(end)

    def test_without_an_envelope_the_boundaries_are_unchanged(self) -> None:
        """The waveform is optional everywhere else in the pipeline; a job
        whose audio has gone missing still reviews and renders."""
        assert self.boundaries(envelope=None, relax=self.RELAX) == self.boundaries()


class TestCaptionCorrections:
    """Fixing a word the recogniser misheard, before the captions are drawn.

    Whisper places a word's text reliably except where it does not -- names,
    jargon and product names are exactly what a talking-head video is full of,
    and "AutoCut" comes back as "auto cut" or "audocut" often enough that a
    caption pass is unusable without a way to fix one word.

    The rule that shapes the whole feature: only ``text`` may move. Timings are
    what the karaoke highlight runs on and what every cut was decided against,
    so a correction that could shift one would quietly invalidate the edit that
    was just reviewed.
    """

    #: The fixture's transcript: an abandoned attempt, then the good take.
    ORIGINAL = ["the", "first", "thing", "the", "first", "thing", "is"]

    def correct(self, job, edits) -> None:
        asyncio.run(JobManager(job.work_dir).edit_captions(job, edits))

    def test_a_misheard_word_is_replaced(self, job) -> None:
        self.correct(job, [{"index": 1, "text": "worst"}])
        assert [w.text for w in job.timeline.words] == [
            "the", "worst", "thing", "the", "first", "thing", "is",
        ]

    def test_the_timing_is_left_exactly_alone(self, job) -> None:
        """The point of the whole restriction, asserted directly."""
        before = [(w.start, w.end) for w in job.timeline.words]
        self.correct(job, [{"index": 1, "text": "considerably-longer-word"}])
        assert [(w.start, w.end) for w in job.timeline.words] == before

    def test_the_correction_reaches_the_file_the_render_reads(self, job) -> None:
        """``render_edit`` loads the transcript off disk, not out of memory, so
        a correction that only updated the object would caption the video with
        the word it was meant to replace."""
        self.correct(job, [{"index": 0, "text": "The"}])
        stored = json.loads((job.work_dir / "timeline.json").read_text())
        assert stored["words"][0]["text"] == "The"

    def test_several_words_go_in_one_correction(self, job) -> None:
        self.correct(job, [{"index": 0, "text": "one"}, {"index": 6, "text": "two"}])
        assert [w.text for w in job.timeline.words] == [
            "one", "first", "thing", "the", "first", "thing", "two",
        ]

    def test_a_blank_word_is_refused(self, job) -> None:
        """A word keeps its slot on the timeline whatever happens to its text,
        and the caption has to draw something in it."""
        with pytest.raises(ValueError):
            self.correct(job, [{"index": 1, "text": "   "}])

    def test_a_word_that_does_not_exist_is_refused(self, job) -> None:
        with pytest.raises(ValueError):
            self.correct(job, [{"index": 99, "text": "nope"}])

    def test_a_malformed_correction_is_refused(self, job) -> None:
        with pytest.raises(ValueError):
            self.correct(job, [{"text": "no index"}])

    def test_nothing_is_written_when_a_correction_is_refused(self, job) -> None:
        """The whole batch fails together: a half-applied correction would
        leave the transcript in a state nobody asked for."""
        with pytest.raises(ValueError):
            self.correct(job, [{"index": 0, "text": "fine"}, {"index": 99, "text": "bad"}])
        assert [w.text for w in job.timeline.words] == self.ORIGINAL
        assert not (job.work_dir / "timeline.json").exists()

    def test_a_job_with_no_transcript_yet_is_refused(self, tmp_path) -> None:
        bare = Job(id="x", filename="a.mp4", source=tmp_path / "a.mp4", work_dir=tmp_path)
        with pytest.raises(ValueError):
            self.correct(bare, [{"index": 0, "text": "nope"}])


class TestCaptionPayload:
    """What the review screen needs to offer a word for correction."""

    def test_each_part_carries_its_words_and_their_places(self, job) -> None:
        spoken = next(p for p in job.parts() if p["words"])
        assert [w["text"] for w in spoken["caption"]] == ["the", "first", "thing"]

    def test_the_indices_address_the_transcript_not_the_part(self, job) -> None:
        """A part is not a stable address -- re-running the detectors renumbers
        them -- so a correction names the word's place in the transcript."""
        indices = [w["index"] for p in job.parts() for w in p["caption"]]
        assert indices == sorted(indices)
        for part in job.parts():
            for word in part["caption"]:
                assert job.timeline.words[word["index"]].text == word["text"]

    def test_a_part_with_no_speech_offers_nothing_to_correct(self, job) -> None:
        silent = next(p for p in job.parts() if not p["words"])
        assert silent["caption"] == []

    def test_a_correction_shows_up_in_the_parts_that_follow(self, job) -> None:
        asyncio.run(JobManager(job.work_dir).edit_captions(job, [{"index": 1, "text": "worst"}]))
        spoken = next(p for p in job.parts() if p["words"])
        assert "worst" in spoken["text"]
        assert [w["text"] for w in spoken["caption"]] == ["the", "worst", "thing"]

    def test_a_word_the_recogniser_doubted_is_marked(self, job) -> None:
        """Whisper reports a probability per word and it is worth believing:
        on one recording the median word scored 0.99 and the only two below
        0.5 were the only two words in the clip that were wrong."""
        job.timeline.words[1] = job.timeline.words[1].model_copy(
            update={"probability": 0.2}
        )
        spoken = next(p for p in job.parts() if p["words"])
        assert [w["uncertain"] for w in spoken["caption"]] == [False, True, False]

    def test_a_confident_word_is_not_marked(self, job) -> None:
        spoken = next(p for p in job.parts() if p["words"])
        assert not any(w["uncertain"] for w in spoken["caption"])


class TestCaptionStyleChoice:
    """The look chosen on the review screen travels with the render."""

    def test_the_style_is_recorded_and_shown(self, job) -> None:
        asyncio.run(JobManager(job.work_dir).approve(job, [1], "hormozi"))
        assert job.caption_style == "hormozi"
        assert job.snapshot()["caption_style"] == "hormozi"
        assert "bold caps captions" in job.message
        assert job.preset_config.caption.uppercase

    def test_the_default_is_classic(self, job) -> None:
        asyncio.run(JobManager(job.work_dir).approve(job, [1]))
        assert job.caption_style == "classic"

    def test_an_unknown_style_changes_nothing(self, job) -> None:
        job.keep = [0]
        with pytest.raises(ValueError, match="nope"):
            asyncio.run(JobManager(job.work_dir).approve(job, [1], "nope"))
        assert job.keep == [0]
        assert job.caption_style == "classic"
