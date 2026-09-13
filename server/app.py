"""The local web app: drop a video in, choose the cuts, download the result.

Uploading a video only gets as far as splitting it into parts.  Nothing is
rendered until someone has been through them and ticked the ones the final
video is made of -- the detectors are good enough to propose and not good
enough to decide, and the expensive stages are wasted on an edit that is going
to be rejected.

Bound to localhost and intended for one person on one machine, so there is no
authentication and no upload size limit beyond what the pipeline itself
enforces.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from autocut import publish
from autocut.config import CAPTION_STYLE_LABELS, CAPTION_STYLES
from autocut.publish.instagram import InstagramError

from .jobs import PRESETS, JobManager
from .samples import caption_sample

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


@app.get("/jobs/{job_id}/parts")
async def job_parts(job_id: str) -> dict:
    """The timeline split into parts, for choosing what the final video keeps."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job.timeline is None:
        raise HTTPException(409, f"job is {job.status}; nothing to review yet")

    return {
        "duration": round(job.timeline.duration, 2),
        "parts": job.parts(),
    }


@app.get("/caption-styles")
async def caption_styles() -> list[dict]:
    """The caption looks on offer, in display order."""
    return [{"name": name, "label": CAPTION_STYLE_LABELS[name]} for name in CAPTION_STYLES]


@app.get("/caption-styles/{name}.png")
async def caption_style_sample(name: str) -> Response:
    """A sample of one style, drawn by the renderer that captions the video."""
    if name not in CAPTION_STYLES:
        raise HTTPException(404, "no such caption style")
    image = await asyncio.to_thread(caption_sample, name)
    return Response(image, media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@app.post("/jobs/{job_id}/render")
async def render_job(
    job_id: str,
    keep: list[int] = Body(..., embed=True),
    style: str = Body("classic", embed=True),
) -> dict:
    """Stitch the chosen parts, in order, and render them in a caption style."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job.status in ("analyzing", "rendering"):
        raise HTTPException(409, f"job is already {job.status}")

    try:
        await manager.approve(job, keep, style)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return job.snapshot()


@app.post("/jobs/{job_id}/captions")
async def edit_captions(job_id: str, edits: list[dict] = Body(..., embed=True)) -> dict:
    """Correct misheard words before the captions are drawn.

    Returns the parts again rather than nothing, so the review screen redraws
    from the server's copy instead of trusting what it just typed.
    """
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job.status in ("analyzing", "rendering"):
        # The render stage reads the transcript off disk as it starts; editing
        # it underneath a running render would caption a video with words that
        # were not the ones cut.
        raise HTTPException(409, f"job is {job.status}; corrections must wait")

    try:
        await manager.edit_captions(job, edits)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return {"duration": round(job.timeline.duration, 2), "parts": job.parts()}


@app.get("/jobs/{job_id}/preview")
async def job_preview(job_id: str) -> FileResponse:
    """The review copy of the source, for hearing a cut before making it."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")

    preview = job.work_dir / "preview.mp4"
    if not preview.exists():
        # The proxy is best-effort; fall back to the original, which the
        # browser may or may not be able to play.
        if not job.source.exists():
            raise HTTPException(404, "no preview available")
        return FileResponse(job.source)
    return FileResponse(preview, media_type="video/mp4")


@app.get("/instagram")
async def instagram_status() -> dict:
    """Whether Instagram is connected, and as whom.

    Verified against Instagram rather than just "is the token set", so a
    revoked or expired token shows up here, before a render is spent on it.
    """
    try:
        backend = publish.backend()
        if not backend.configured():
            return {
                "connected": False,
                "error": "no INSTAGRAM_ACCESS_TOKEN or UPLOAD_POST_API_KEY in .env",
            }
        who = await asyncio.to_thread(backend.account)
    except InstagramError as error:
        return {"connected": False, "error": str(error)}
    name = next(n for n, m in publish.BACKENDS.items() if m is backend)
    return {"connected": True, "username": who.username, "user_id": who.user_id, "via": name}


@app.post("/jobs/{job_id}/publish")
async def publish_job(job_id: str, caption: str = Body("", embed=True)) -> dict:
    """Post the finished video to Instagram as a Reel."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    try:
        await manager.publish(job, caption.strip())
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return job.publish.snapshot()


@app.get("/jobs/{job_id}/publish")
async def publish_status(job_id: str) -> dict:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job.publish is None:
        raise HTTPException(404, "this video has not been published")
    return job.publish.snapshot()


@app.get("/jobs/{job_id}/result")
async def job_result(job_id: str) -> FileResponse:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job.status != "done" or job.result is None:
        raise HTTPException(409, f"job is {job.status}, not finished")

    # The download carries the render's timestamp, so two cuts of the same
    # source saved to the same folder do not overwrite each other.
    stamp = job.result.output.stem.removeprefix("final-")
    return FileResponse(
        job.result.output,
        media_type="video/mp4",
        filename=f"{Path(job.filename).stem}-{stamp}.mp4" if stamp else f"{Path(job.filename).stem}.mp4",
        headers={"Cache-Control": "no-store"},
    )
