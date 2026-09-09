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

from autocut.config import DEFAULT, Preset
from autocut.pipeline import Progress, Result, run

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
    status: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    percent: int = 0
    message: str = "waiting to start"
    error: str | None = None
    result: Result | None = None
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


class JobManager:
    """Owns every job and the single worker that renders them."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
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
        await self._queue.put(job_id)
        self.start()
        log.info("queued job %s (%s)", job_id, filename)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

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
            job_id = await self._queue.get()
            job = self.jobs.get(job_id)
            if job is None:
                continue
            try:
                await self._render(job)
            except Exception as error:  # noqa: BLE001 - one bad job must not stop the worker
                log.exception("job %s failed", job.id)
                job.status, job.error = "error", str(error)
                job.message = "failed"
                self._publish(job)
            finally:
                self._queue.task_done()

    async def _render(self, job: Job) -> None:
        loop = asyncio.get_running_loop()
        job.status = "running"
        self._publish(job)

        def on_progress(progress: Progress) -> None:
            # Called on the worker thread; hop back to the loop to publish.
            def apply() -> None:
                job.stage = progress.stage
                job.percent = progress.percent
                job.message = progress.message
                self._publish(job)

            loop.call_soon_threadsafe(apply)

        result = await asyncio.to_thread(
            run, job.source, job.work_dir, PRESETS[job.preset], on_progress
        )
        job.result = result
        job.status = "done"
        job.stage = "done"
        job.percent = 100
        job.message = "finished"
        self._publish(job)
        log.info("job %s finished in %.0fs", job.id, result.elapsed)
