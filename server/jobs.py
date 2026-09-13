"""Job registry and worker.

One machine, so there is no task broker: jobs live in a dict, their files in
a work directory, and a row per job in SQLite so a restart picks them back
up.  What does need care is that the pipeline is blocking and GPU-bound, so
it runs in a worker thread and exactly one job runs at a time -- two
concurrent renders on one machine finish no sooner and make progress
reporting meaningless -- and, now that several people share the machine, it
is also what keeps one person's ten-minute upload from taking the whole
thing over.

Progress arrives on the worker thread and has to reach subscribers on the event
loop, which is what ``call_soon_threadsafe`` below is for.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from autocut.analyze import merge
from autocut.audio import Envelope
from autocut.config import CAPTION_STYLE_LABELS, CAPTION_STYLES, DEFAULT, Preset
from autocut.models import Timeline
from autocut.ingest import probe
from autocut.pipeline import Progress, Result, part_envelope, propose, render_edit
from autocut import publish
from autocut.publish.instagram import InstagramError

from . import retention
from .auth import LOCAL
from .store import Row, Store

log = logging.getLogger(__name__)

PRESETS = {
    "default": DEFAULT,
    "gentle": DEFAULT.gentle(),
    "aggressive": DEFAULT.aggressive(),
}


@dataclass
class Publication:
    """One attempt to put a finished video on Instagram."""

    #: publishing -> published, or error.
    status: str = "publishing"
    message: str = "starting"
    caption: str = ""
    permalink: str | None = None
    media_id: str | None = None
    error: str | None = None

    def snapshot(self) -> dict:
        return {
            "status": self.status,
            "message": self.message,
            "caption": self.caption,
            "permalink": self.permalink,
            "media_id": self.media_id,
            "error": self.error,
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> Publication:
        fields = ("status", "message", "caption", "permalink", "media_id", "error")
        return cls(**{k: data[k] for k in fields if data.get(k) is not None})


class QuotaExceeded(ValueError):
    """An upload refused for want of room, with a message naming the limit."""


class BadUpload(ValueError):
    """An upload refused because it is not a video this can work on."""


@dataclass
class Job:
    id: str
    filename: str
    source: Path
    work_dir: Path
    #: Whose job this is; every route checks it.  ``local`` when running
    #: single-user.
    owner: str = LOCAL
    preset: str = "default"
    #: queued -> analyzing -> review -> rendering -> done, or error at any point.
    #: ``review`` is where the job waits for a person to say which of the
    #: proposed cuts to make; nothing expensive runs until they have.
    status: str = "queued"
    stage: str = "queued"
    percent: int = 0
    message: str = "waiting to start"
    error: str | None = None
    #: The finished video and what it is made of, set when a render
    #: finishes and kept in the row so a restart still has them.
    output: Path | None = None
    summary: dict | None = None
    #: Size of the upload, for the per-user budget.
    bytes: int = 0
    #: The proposals, and the parts a person chose to keep.  Kept apart so an
    #: edit can be revised: the proposals never change, the answers do.
    timeline: Timeline | None = None
    keep: list[int] | None = None
    #: The caption look chosen on the review screen; a key of CAPTION_STYLES.
    caption_style: str = "classic"
    #: Where the finished video has been sent, if anywhere.  Separate from the
    #: job's own status: a failed publish does not un-finish a render, and the
    #: file is still there to download.
    publish: Publication | None = None
    #: How many jobs are ahead of this one, while it waits; None otherwise.
    position: int | None = None
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)
    #: Cached by :meth:`envelope`.  The flag is separate from the value so that
    #: a genuine "no audio on disk" is cached too, rather than re-probed on
    #: every poll of the review screen.
    _envelope: Envelope | None = field(default=None, repr=False)
    _envelope_loaded: bool = field(default=False, repr=False)

    def snapshot(self) -> dict:
        """The job as the browser sees it."""
        data = {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "percent": self.percent,
            "message": self.message,
            "error": self.error,
            "created": self.created.isoformat(),
            "expires": retention.expires_at(self.created).isoformat(),
            "caption_style": self.caption_style,
        }
        if self.position is not None:
            data["position"] = self.position
        if self.timeline is not None:
            data["parts"] = len(self.parts())
            data["source_duration"] = round(self.timeline.duration, 1)
        if self.summary is not None:
            data["summary"] = self.summary
        if self.publish is not None:
            data["publish"] = self.publish.snapshot()
        return data

    @staticmethod
    def summarise(result: Result) -> dict:
        timeline = result.timeline
        return {
            "source_duration": round(timeline.duration, 1),
            "output_duration": round(timeline.kept_duration, 1),
            "removed_percent": round(result.removed_fraction * 100),
            "segments": len(timeline.keep_segments),
            "words": len(timeline.words),
            "tracked": result.tracked,
            "elapsed": round(result.elapsed),
            #: The file's own name, which changes with every render; the
            #: browser puts it in the result URL so a new cut is never
            #: served from cache.
            "output": result.output.name,
        }

    @property
    def active(self) -> bool:
        """Occupying, or waiting for, the worker."""
        return self.status in ("queued", "analyzing", "rendering")

    def row(self) -> Row:
        return Row(
            id=self.id, owner=self.owner, filename=self.filename, preset=self.preset,
            caption_style=self.caption_style, status=self.status, stage=self.stage,
            message=self.message, error=self.error, keep=self.keep, summary=self.summary,
            output=self.output.name if self.output else None,
            publish=self.publish.snapshot() if self.publish else None,
            created=self.created, bytes=self.bytes,
        )


    def envelope(self) -> Envelope | None:
        """The analysis envelope, loaded once and kept.

        Part boundaries are relaxed off speech against it, and ``parts()`` is
        called on every poll of the review screen; reading the WAV each time
        would make the list cost as much as the analysis did.
        """
        if not self._envelope_loaded:
            self._envelope = (
                part_envelope(self.timeline) if self.timeline is not None else None
            )
            self._envelope_loaded = True
        return self._envelope

    @property
    def preset_config(self) -> Preset:
        """The resolved preset this job was submitted with, in its caption style."""
        return PRESETS.get(self.preset, DEFAULT).with_caption_style(self.caption_style)

    def parts(self) -> list[dict]:
        """The timeline as an ordered list of parts, for review.

        Ticked parts are what the final video is stitched from.  The detectors'
        proposals decide only what starts ticked: a stretch they left alone is
        in, a stretch they wanted gone is out.  Both are shown and both can be
        flipped, so a wrongly-removed retake is one click from coming back.

        Each part carries the words spoken in it, because that -- not a
        timestamp -- is how you recognise the piece you are deciding about.
        """
        if self.timeline is None:
            return []

        words = self.timeline.words
        threshold = self.preset_config.caption.review_confidence
        chosen = set(self.keep) if self.keep is not None else None
        pieces = merge.parts(
            self.timeline.removals, self.timeline.duration, words,
            min_pause=self.preset_config.review_min_pause,
            part_pause=self.preset_config.part_pause,
            envelope=self.envelope(),
            relax=self.preset_config.boundary_relax,
        )

        def in_the_edit(start: float, end: float) -> bool:
            """Whether most of a part survived into the pipeline's own edit.

            By overlap rather than by midpoint: a part can now span a pause too
            short to be worth asking about, and its midpoint may land inside
            that pause even though the part is overwhelmingly speech.
            """
            span = end - start
            if span <= 0:
                return False
            covered = sum(
                max(0.0, min(end, segment.end) - max(start, segment.start))
                for segment in self.timeline.keep_segments
            )
            return covered / span > 0.5

        parts = []
        for index, (start, end, removal) in enumerate(pieces):
            # Indices travel with the words: a caption correction names the
            # word it replaces, and a part is not a stable address for one --
            # re-running the detectors renumbers the parts, while a word keeps
            # its place in the transcript.
            numbered = [
                (i, w) for i, w in enumerate(words)
                if w.start >= start and w.end <= end
            ]
            inside = [w for _, w in numbered]
            # What starts ticked is the pipeline's own edit decision list, not
            # simply "whatever no detector objected to".  The two differ: a
            # sliver left between two cuts is in neither the removals nor the
            # segments, because the merge stage drops it as debris, and
            # defaulting it to kept would put back exactly the fragments that
            # make an edit sound abrupt.
            kept = in_the_edit(start, end)
            if removal is not None:
                reason, detail = str(removal.reason), removal.detail
            elif not kept:
                # Excluded from the pipeline's own edit as debris -- too short
                # to judge, whether or not it happens to have no words either.
                reason, detail = "offcut", "too short to keep"
            elif not inside and (end - start) >= 0.05:
                # Kept, no words, and no removal -- which means the silence
                # detector looked at this exact stretch and did not call it
                # silence.  The only way both can be true is real audio the
                # recogniser failed to transcribe: a dropped word, a second
                # attempt at a line, background speech.  Flagged rather than
                # shown as plain kept video, because nothing else in the
                # pipeline has seen this part at all.
                reason, detail = "untranscribed", "no words recognised here"
            else:
                reason, detail = None, ""
            parts.append(
                {
                    "id": index,
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "duration": round(end - start, 2),
                    "text": " ".join(w.text for w in inside),
                    "words": len(inside),
                    #: Each word and where it sits in the transcript, so the
                    #: review screen can offer them one at a time for
                    #: correction before the captions are drawn.
                    "caption": [
                        {
                            "index": i,
                            "text": w.text,
                            #: The recogniser's own doubt about this word.
                            #: Nothing acts on it -- it marks where to look,
                            #: which is the difference between proof-reading a
                            #: transcript and scanning one.
                            "uncertain": w.probability < threshold,
                        }
                        for i, w in numbered
                    ],
                    #: Why this part is not in the video; absent when it is.
                    "reason": reason,
                    "detail": detail,
                    "keep": index in chosen if chosen is not None else kept,
                }
            )
        return parts


class JobManager:
    """Owns every job and the single worker that renders them."""

    def __init__(self, root: Path, db: Path | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = Store(db or self.root / "autocut.db")
        self.jobs: dict[str, Job] = {}
        #: The queue's contents, in order, so a job can be told how many
        #: are ahead of it; ``asyncio.Queue`` cannot be looked into.
        self._pending: list[str] = []
        #: (job id, phase) -- one worker runs both phases, so a render can
        #: never start while another job is still transcribing.
        self._queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        #: In-flight publishes.  Held so the event loop cannot collect a task
        #: that nothing else references before it finishes.
        self._publishing: set[asyncio.Task] = set()
        self._sweeper: asyncio.Task | None = None

    def load(self) -> None:
        """Pick up every job the previous process knew about.

        Whatever was mid-flight when the process died is put back where a
        person can act on it: a job that was still being analysed has
        nothing on disk worth keeping and asks for the upload again; one that
        was rendering has its proposals and its answers, so it goes back to
        review with a note to render again.
        """
        for row in self.store.all():
            work_dir = self.root / row.id
            source = next(work_dir.glob("input.*"), work_dir / "input.mp4")
            job = Job(
                id=row.id, filename=row.filename, source=source, work_dir=work_dir,
                owner=row.owner, preset=row.preset, caption_style=row.caption_style,
                status=row.status, stage=row.stage, message=row.message, error=row.error,
                keep=row.keep, summary=row.summary, bytes=row.bytes, created=row.created,
                output=work_dir / row.output if row.output else None,
                publish=Publication.from_snapshot(row.publish) if row.publish else None,
            )
            timeline_path = work_dir / "timeline.json"
            if timeline_path.exists():
                try:
                    job.timeline = Timeline(**json.loads(timeline_path.read_text()))
                except (OSError, ValueError) as error:
                    log.warning("job %s: could not read timeline: %s", job.id, error)

            if job.status == "analyzing" or (job.status == "queued" and job.timeline is None):
                job.status, job.error, job.message = "error", "interrupted by a restart", "upload it again"
            elif job.status in ("queued", "rendering"):
                job.status, job.stage, job.percent = "review", "review", 100
                job.message = "render interrupted by a restart; render again"
                job.error = None
            elif job.status == "done" and (job.output is None or not job.output.exists()):
                job.status, job.error, job.message = "error", "the finished video is gone", "render again"
                job.summary = None
            elif job.status == "review" and job.timeline is None:
                job.status, job.error, job.message = "error", "the analysis is gone", "upload it again"
            if job.publish is not None and job.publish.status == "publishing":
                job.publish.status, job.publish.error, job.publish.message = "error", "interrupted by a restart", "failed"
            self.jobs[job.id] = job
            self.store.save(job.row())
        if self.jobs:
            log.info("picked up %d jobs from the previous run", len(self.jobs))

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_worker())
        if self._sweeper is None or self._sweeper.done():
            self._sweeper = asyncio.create_task(self._sweep_forever())

    async def stop(self) -> None:
        for task in (self._worker, self._sweeper):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._worker = self._sweeper = None

    def owned_by(self, owner: str) -> list[Job]:
        return [job for job in self.jobs.values() if job.owner == owner]

    def check_room(self, owner: str) -> None:
        """Refuse an upload before a byte lands if the owner has no room."""
        mine = self.owned_by(owner)
        active = sum(1 for job in mine if job.active)
        if active >= retention.max_active_per_user():
            raise QuotaExceeded(
                f"you already have {active} videos in progress; wait for one to finish"
            )
        if len(mine) >= retention.max_jobs_per_user():
            raise QuotaExceeded(
                f"you have {len(mine)} videos, the most this keeps; delete one to upload another"
            )
        used = sum(job.bytes for job in mine)
        if used >= retention.max_bytes_per_user():
            raise QuotaExceeded(
                f"your videos take {retention.format_bytes(used)}, the most this keeps; "
                "delete one to upload another"
            )

    async def submit(self, filename: str, stream, preset: str = "default", owner: str = LOCAL) -> Job:
        """Save an upload and queue it for analysis.

        The file is checked as it arrives and again once it has landed:
        streamed to disk with a running count so a runaway upload is cut off
        at the cap rather than after, then probed so that something that is
        not a video, or is longer than a Reel can be, never reaches the
        worker.  Either failure removes what was written.
        """
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}")
        self.check_room(owner)

        job_id = uuid.uuid4().hex[:12]
        work_dir = self.root / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        # Keep the original suffix: ffmpeg uses it as a demuxer hint.
        suffix = Path(filename).suffix or ".mp4"
        source = work_dir / f"input{suffix}"
        cap = retention.max_upload_bytes()

        def receive() -> int:
            written = 0
            with source.open("wb") as destination:
                while chunk := stream.read(1024 * 1024):
                    written += len(chunk)
                    if written > cap:
                        raise BadUpload(
                            f"that file is over the {retention.format_bytes(cap)} upload limit"
                        )
                    destination.write(chunk)
            return written

        try:
            size = await asyncio.to_thread(receive)
            probed = await asyncio.to_thread(probe, source)
            if probed.duration > retention.max_duration():
                raise BadUpload(
                    f"that video is {probed.duration / 60:.1f} minutes; the most this takes "
                    f"is {retention.max_duration() / 60:.0f} (a Reel's limit)"
                )
        except BadUpload:
            retention.remove_job_dir(work_dir)
            raise
        except Exception as error:  # noqa: BLE001 - anything ffprobe rejects is "not a video"
            retention.remove_job_dir(work_dir)
            # The detail names paths on this machine; it belongs in the log.
            log.info("refused upload %r from %s: %s", filename, owner, error)
            raise BadUpload("that does not look like a video this can read") from error

        job = Job(
            id=job_id, filename=filename, source=source, work_dir=work_dir,
            owner=owner, preset=preset, bytes=size,
        )
        self.jobs[job_id] = job
        await self._enqueue(job, "propose")
        self.start()
        log.info("queued job %s (%s) for %s", job_id, filename, owner)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    async def delete(self, job: Job) -> None:
        """Remove a job and everything it stored."""
        if job.status in ("analyzing", "rendering"):
            raise ValueError(f"the job is {job.status}; wait for it to finish")
        if job.id in self._pending:
            # Queued but not started: pull it before the worker gets there.
            self._pending.remove(job.id)
            self._positions()
        self.jobs.pop(job.id, None)
        self.store.delete(job.id)
        await asyncio.to_thread(retention.remove_job_dir, job.work_dir)
        job.status, job.message = "deleted", "deleted"
        for queue in list(job.subscribers):
            queue.put_nowait(job.snapshot())
        log.info("deleted job %s (%s)", job.id, job.filename)

    async def _enqueue(self, job: Job, phase: str) -> None:
        self._pending.append(job.id)
        self._positions()
        await self._queue.put((job.id, phase))

    def _positions(self) -> None:
        """Tell every waiting job how many are ahead of it."""
        for index, job_id in enumerate(self._pending):
            job = self.jobs.get(job_id)
            if job is not None and job.position != index:
                job.position = index
                self._publish(job)

    async def _sweep_forever(self) -> None:
        """Expire old jobs, hourly.  Runs once at startup too."""
        while True:
            try:
                await self.sweep()
            except Exception:  # noqa: BLE001 - a failed sweep must not stop the sweeper
                log.exception("retention sweep failed")
            await asyncio.sleep(3600)

    async def sweep(self, now: datetime | None = None) -> int:
        """Delete every job past its retention window; returns how many."""
        expired = [
            job for job in list(self.jobs.values())
            if retention.expired(job.created, now) and not job.active
        ]
        for job in expired:
            await self.delete(job)
        if expired:
            log.info("expired %d jobs", len(expired))
        return len(expired)

    async def approve(self, job: Job, keep: list[int], style: str = "classic") -> Job:
        """Accept the parts a person chose to keep, and queue the render."""
        if job.timeline is None:
            raise ValueError("this job has no parts to review yet")
        # Checked before anything on the job changes, so a bad style leaves
        # the previous answers intact.
        if style not in CAPTION_STYLES:
            raise ValueError(f"unknown caption style {style!r}")
        count = len(job.parts())
        unknown = [index for index in keep if not 0 <= index < count]
        if unknown:
            raise ValueError(f"no such part: {unknown}")
        if not keep:
            raise ValueError("keep at least one part")

        job.keep = sorted(set(keep))
        job.caption_style = style
        # A render replaces the previous one: the old file goes, so the work
        # directory never holds two finished videos to confuse, and so does
        # anything published from it.  The new one gets its own name (see
        # ``_render``), so no browser can mistake a cached copy for it.
        if job.output is not None:
            try:
                job.output.unlink(missing_ok=True)
            except OSError as error:
                log.warning("job %s: could not remove %s: %s", job.id, job.output, error)
        job.output = None
        job.summary = None
        job.publish = None
        job.status = "queued"
        job.stage = "queued"
        job.percent = 0
        job.message = (
            f"stitching {len(job.keep)} of {count} parts, "
            f"{CAPTION_STYLE_LABELS[style].lower()} captions"
        )
        job.error = None
        self._publish(job)
        await self._enqueue(job, "render")
        self.start()
        return job

    async def edit_captions(self, job: Job, edits: list[dict]) -> Job:
        """Correct the text of individual words, before anything is rendered.

        Only ``text`` moves.  A word's timing is what the karaoke caption is
        built on and what every cut was decided against, so a correction that
        could change it would silently invalidate the edit a person just
        reviewed -- and the thing being fixed here is a recogniser that heard
        the word wrong, not one that placed it wrong.  Words cannot be added or
        removed for the same reason: there is no timing to give a new one.

        The correction is written back to ``timeline.json`` rather than kept in
        memory, because that file is what the render stage reads.  It is the
        one thing about the proposal a person is allowed to overwrite: the cuts
        are a decision to revise, but a misheard word was never right.
        """
        if job.timeline is None:
            raise ValueError("this job has no transcript to correct yet")

        words = list(job.timeline.words)
        for edit in edits:
            try:
                index = int(edit["index"])
                text = str(edit["text"]).strip()
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"malformed correction {edit!r}") from error
            if not 0 <= index < len(words):
                raise ValueError(f"no such word: {index}")
            if not text:
                raise ValueError("a caption word cannot be blank")
            words[index] = words[index].model_copy(update={"text": text})

        job.timeline.words = words
        (job.work_dir / "timeline.json").write_text(
            job.timeline.model_dump_json(indent=1)
        )
        log.info("job %s: corrected %d caption words", job.id, len(edits))
        self._publish(job)
        return job

    async def publish(self, job: Job, caption: str) -> Job:
        """Send a finished video to Instagram, in the background.

        Not queued behind the render worker: publishing is network-bound, not
        GPU-bound, so there is no reason to make it wait for someone else's
        transcription.  One publish per job at a time, though -- a second
        click while the first is uploading would post the video twice.
        """
        if job.status != "done" or job.output is None:
            raise ValueError(f"job is {job.status}, not finished")
        if job.publish is not None and job.publish.status == "publishing":
            raise ValueError("this video is already being published")
        if not publish.configured():
            raise ValueError("Instagram is not connected; see the README")

        job.publish = Publication(caption=caption)
        self._publish(job)
        task = asyncio.create_task(self._publish_to_instagram(job))
        self._publishing.add(task)
        task.add_done_callback(self._publishing.discard)
        return job

    async def _publish_to_instagram(self, job: Job) -> None:
        publication = job.publish
        assert publication is not None
        loop = asyncio.get_running_loop()

        def report(message: str) -> None:
            def apply() -> None:
                publication.message = message
                self._publish(job)

            loop.call_soon_threadsafe(apply)

        try:
            posted = await asyncio.to_thread(
                publish.backend().publish_reel,
                job.output,
                publication.caption,
                report=report,
            )
        except InstagramError as error:
            log.warning("job %s: publish failed: %s", job.id, error)
            publication.status, publication.error = "error", str(error)
            publication.message = "failed"
        except Exception as error:  # noqa: BLE001 - reported to the browser, not swallowed
            log.exception("job %s: publish failed", job.id)
            publication.status, publication.error = "error", str(error)
            publication.message = "failed"
        else:
            publication.status = "published"
            publication.message = "published"
            publication.media_id = posted.media_id
            publication.permalink = posted.permalink
            log.info("job %s published to Instagram: %s", job.id, posted.permalink)
        self._publish(job)

    def subscribe(self, job: Job) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        job.subscribers.append(queue)
        # Send the current state immediately, so a late subscriber is not left
        # staring at an empty progress bar until the next event.
        queue.put_nowait(job.snapshot())
        return queue

    def unsubscribe(self, job: Job, queue: asyncio.Queue) -> None:
        if queue in job.subscribers:
            job.subscribers.remove(queue)

    def _publish(self, job: Job) -> None:
        if job.id in self.jobs:
            try:
                self.store.save(job.row())
            except Exception:  # noqa: BLE001 - a row that fails to write must not stop the render
                log.exception("job %s: could not save", job.id)
        snapshot = job.snapshot()
        for queue in list(job.subscribers):
            queue.put_nowait(snapshot)

    async def _run_worker(self) -> None:
        while True:
            job_id, phase = await self._queue.get()
            if job_id in self._pending:
                self._pending.remove(job_id)
            job = self.jobs.get(job_id)
            if job is None:
                self._queue.task_done()
                continue
            job.position = None
            self._positions()
            try:
                if phase == "propose":
                    await self._propose(job)
                else:
                    await self._render(job)
            except Exception as error:  # noqa: BLE001 - one bad job must not stop the worker
                log.exception("job %s failed", job.id)
                job.status, job.error = "error", str(error)
                job.message = "failed"
                self._publish(job)
            finally:
                self._queue.task_done()

    def _reporter(self, job: Job):
        """A progress callback that publishes from the worker thread."""
        loop = asyncio.get_running_loop()

        def on_progress(progress: Progress) -> None:
            def apply() -> None:
                job.stage = progress.stage
                job.percent = progress.percent
                job.message = progress.message
                self._publish(job)

            loop.call_soon_threadsafe(apply)

        return on_progress

    async def _propose(self, job: Job) -> None:
        job.status = "analyzing"
        self._publish(job)

        timeline = await asyncio.to_thread(
            propose, job.source, job.work_dir, PRESETS[job.preset], self._reporter(job)
        )
        job.timeline = timeline
        job.status = "review"
        job.stage = "review"
        job.percent = 100
        count = len(job.parts())
        job.message = f"{count} parts to review"
        self._publish(job)
        log.info("job %s split into %d parts", job.id, count)

    async def _render(self, job: Job) -> None:
        job.status = "rendering"
        self._publish(job)

        # Named by the moment it was made, so every render of a job is a
        # different URL -- a browser that cached ``final.mp4`` would otherwise
        # keep showing the previous cut after "Back to the edit".
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        result = await asyncio.to_thread(
            render_edit,
            job.work_dir,
            job.keep or [],
            job.preset_config,
            self._reporter(job),
            output_name=f"final-{stamp}.mp4",
        )
        job.output = result.output
        job.summary = Job.summarise(result)
        job.status = "done"
        job.stage = "done"
        job.percent = 100
        job.message = "finished"
        self._publish(job)
        # The upload stays -- a second render cuts from it -- but the
        # intermediates are re-creatable and add up across users.
        freed = await asyncio.to_thread(retention.drop_intermediates, job.work_dir)
        log.info("job %s finished in %.0fs; freed %s", job.id, result.elapsed, retention.format_bytes(freed))
