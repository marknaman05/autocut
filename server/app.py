"""The local web app: drop a video in, watch it render, download the result.

Bound to localhost and intended for one person on one machine, so there is no
authentication and no upload size limit beyond what the pipeline itself
enforces.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from .jobs import PRESETS, JobManager

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
WORK_ROOT = Path("work")

manager = JobManager(WORK_ROOT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    manager.start()
    yield
    await manager.stop()


app = FastAPI(title="autocut", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC / "index.html").read_text()


@app.post("/jobs")
async def create_job(file: UploadFile, preset: str = Form("default")) -> dict:
    if preset not in PRESETS:
        raise HTTPException(400, f"unknown preset {preset!r}")
    if not file.filename:
        raise HTTPException(400, "no file was uploaded")

    job = await manager.submit(file.filename, file.file, preset)
    return job.snapshot()


@app.get("/jobs")
async def list_jobs() -> list[dict]:
    return [
        job.snapshot()
        for job in sorted(manager.jobs.values(), key=lambda j: j.created, reverse=True)
    ]


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job.snapshot()


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    """Server-sent events carrying this job's progress."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")

    async def stream():
        queue = manager.subscribe(job)
        try:
            while True:
                if await request.is_disconnected():
                    break
                snapshot = await queue.get()
                yield f"data: {json.dumps(snapshot)}\n\n"
                if snapshot["status"] in ("done", "error"):
                    break
        finally:
            manager.unsubscribe(job, queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        # Without this, a proxy or the browser may buffer the whole stream and
        # the progress bar only moves once, at the end.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/jobs/{job_id}/result")
async def job_result(job_id: str) -> FileResponse:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job.status != "done" or job.result is None:
        raise HTTPException(409, f"job is {job.status}, not finished")

    return FileResponse(
        job.result.output,
        media_type="video/mp4",
        filename=f"{Path(job.filename).stem}-vertical.mp4",
    )
