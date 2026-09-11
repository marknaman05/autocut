"""Job registry and worker.

Single user, single machine, so there is no database and no task broker: jobs
live in a dict and their files live on disk.  What does need care is that the
pipeline is blocking and GPU-bound, so it runs in a worker thread and exactly
one job runs at a time -- two concurrent renders on one machine finish no
sooner and make progress reporting meaningless.

Progress arrives on the worker thread and has to reach subscribers on the event
loop, which is what ``call_soon_threadsafe`` below is for.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from autocut.analyze import merge
from autocut.config import DEFAULT, Preset
from autocut.models import Timeline
from autocut.pipeline import Progress, Result, propose, render_edit

log = logging.getLogger(__name__)

PRESETS = {
    "default": DEFAULT,
    "gentle": DEFAULT.gentle(),
    "aggressive": DEFAULT.aggressive(),
}


@dataclass
class Job:
    id: str
    filename: str
    source: Path
    work_dir: Path
    preset: str = "default"
    #: queued -> analyzing -> review -> rendering -> done, or error at any point.
    #: ``review`` is where the job waits for a person to say which of the
    #: proposed cuts to make; nothing expensive runs until they have.
    status: str = "queued"
    stage: str = "queued"
    percent: int = 0
    message: str = "waiting to start"
    error: str | None = None
    result: Result | None = None
    #: The proposals, and the parts a person chose to keep.  Kept apart so an
    #: edit can be revised: the proposals never change, the answers do.
    timeline: Timeline | None = None
    keep: list[int] | None = None
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)

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
        }
        if self.timeline is not None:
            data["parts"] = len(self.parts())
            data["source_duration"] = round(self.timeline.duration, 1)
        if self.result is not None:
            timeline = self.result.timeline
            data["summary"] = {
                "source_duration": round(timeline.duration, 1),
                "output_duration": round(timeline.kept_duration, 1),
                "removed_percent": round(self.result.removed_fraction * 100),
                "segments": len(timeline.keep_segments),
                "words": len(timeline.words),
                "tracked": self.result.tracked,
                "elapsed": round(self.result.elapsed),
            }
        return data


    @property
    def preset_config(self) -> Preset:
        """The resolved preset this job was submitted with."""
        return PRESETS.get(self.preset, DEFAULT)

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
        chosen = set(self.keep) if self.keep is not None else None
        pieces = merge.parts(
            self.timeline.removals, self.timeline.duration, words,
            min_pause=self.preset_config.review_min_pause,
            part_pause=self.preset_config.part_pause,
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
            inside = [w for w in words if w.start >= start and w.end <= end]
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
                    #: Why this part is not in the video; absent when it is.
                    "reason": reason,
                    "detail": detail,
                    "keep": index in chosen if chosen is not None else kept,
                }
            )
        return parts


class JobManager:
    """Owns every job and the single worker that renders them."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        #: (job id, phase) -- one worker runs both phases, so a render can
        #: never start while another job is still transcribing.
        self._queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_worker())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def submit(self, filename: str, stream, preset: str = "default") -> Job:
        """Save an upload and queue it for rendering."""
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}")

        job_id = uuid.uuid4().hex[:12]
        work_dir = self.root / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        # Keep the original suffix: ffmpeg uses it as a demuxer hint.
        suffix = Path(filename).suffix or ".mp4"
        source = work_dir / f"input{suffix}"
        with source.open("wb") as destination:
            await asyncio.to_thread(shutil.copyfileobj, stream, destination)

        job = Job(
            id=job_id, filename=filename, source=source, work_dir=work_dir, preset=preset
        )
        self.jobs[job_id] = job
        await self._queue.put((job_id, "propose"))
        self.start()
        log.info("queued job %s (%s)", job_id, filename)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    async def approve(self, job: Job, keep: list[int]) -> Job:
        """Accept the parts a person chose to keep, and queue the render."""
        if job.timeline is None:
            raise ValueError("this job has no parts to review yet")
        count = len(job.parts())
        unknown = [index for index in keep if not 0 <= index < count]
        if unknown:
            raise ValueError(f"no such part: {unknown}")
        if not keep:
            raise ValueError("keep at least one part")

        job.keep = sorted(set(keep))
        job.status = "queued"
        job.stage = "queued"
        job.percent = 0
        job.message = f"stitching {len(job.keep)} of {count} parts"
        job.error = None
        self._publish(job)
        await self._queue.put((job.id, "render"))
        self.start()
        return job

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
        snapshot = job.snapshot()
        for queue in list(job.subscribers):
            queue.put_nowait(snapshot)

    async def _run_worker(self) -> None:
        while True:
            job_id, phase = await self._queue.get()
            job = self.jobs.get(job_id)
            if job is None:
                continue
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

        result = await asyncio.to_thread(
            render_edit,
            job.work_dir,
            job.keep or [],
            PRESETS[job.preset],
            self._reporter(job),
        )
        job.result = result
        job.status = "done"
        job.stage = "done"
        job.percent = 100
        job.message = "finished"
        self._publish(job)
        log.info("job %s finished in %.0fs", job.id, result.elapsed)
