"""The web app: drop a video in, choose the cuts, download the result.

Uploading a video only gets as far as splitting it into parts.  Nothing is
rendered until someone has been through them and ticked the ones the final
video is made of -- the detectors are good enough to propose and not good
enough to decide, and the expensive stages are wasted on an edit that is going
to be rejected.

Several people can use one instance.  Who they are comes from a proxy that
has already signed them in (see ``auth``); without one configured the app is
the single-user tool it started as.  Every job belongs to whoever uploaded
it, and a job that is not yours does not exist as far as the API is
concerned -- 404, not 403, so an id reveals nothing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Body, Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from autocut import publish
from autocut.config import CAPTION_STYLE_LABELS, CAPTION_STYLES
from autocut.publish.instagram import InstagramError

from .auth import User, current_user, header_name
from .jobs import PRESETS, BadUpload, Job, JobManager, QuotaExceeded
from .samples import caption_sample

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
#: Absolute so a service unit's working directory cannot move the data.
WORK_ROOT = Path(os.environ.get("AUTOCUT_WORK_ROOT") or "work").resolve()
DB_PATH = Path(os.environ["AUTOCUT_DB"]).resolve() if os.environ.get("AUTOCUT_DB") else None

manager = JobManager(WORK_ROOT, DB_PATH)

Viewer = Annotated[User, Depends(current_user)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if header_name():
        log.info("multi-user: trusting %s for identity", header_name())
    else:
        log.info("single-user: no AUTOCUT_USER_HEADER set")
    manager.load()
    manager.start()
    yield
    await manager.stop()


app = FastAPI(title="autocut", lifespan=lifespan)


def require_job(job_id: str, user: User) -> Job:
    """The job, if it is this user's; otherwise it does not exist."""
    job = manager.get(job_id)
    if job is None or job.owner != user.email:
        raise HTTPException(404, "no such job")
    return job


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/me")
async def me(user: Viewer) -> dict:
    """Who the page is talking to, so it can say so."""
    return {"email": user.email, "can_publish": user.can_publish, "local": user.is_local}


@app.post("/jobs")
async def create_job(user: Viewer, file: UploadFile, preset: str = Form("default")) -> dict:
    if preset not in PRESETS:
        raise HTTPException(400, f"unknown preset {preset!r}")
    if not file.filename:
        raise HTTPException(400, "no file was uploaded")

    try:
        job = await manager.submit(file.filename, file.file, preset, owner=user.email)
    except QuotaExceeded as error:
        raise HTTPException(429, str(error)) from error
    except BadUpload as error:
        raise HTTPException(413 if "upload limit" in str(error) else 400, str(error)) from error
    return job.snapshot()


@app.get("/jobs")
async def list_jobs(user: Viewer) -> list[dict]:
    return [
        job.snapshot()
        for job in sorted(manager.owned_by(user.email), key=lambda j: j.created, reverse=True)
    ]


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, user: Viewer) -> dict:
    return require_job(job_id, user).snapshot()


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str, user: Viewer) -> dict:
    """Remove a job and every file it kept."""
    job = require_job(job_id, user)
    try:
        await manager.delete(job)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return {"deleted": job_id}


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str, user: Viewer, request: Request) -> StreamingResponse:
    """Server-sent events carrying this job's progress."""
    job = require_job(job_id, user)

    async def stream():
        queue = manager.subscribe(job)
        try:
            while True:
                if await request.is_disconnected():
                    break
                snapshot = await queue.get()
                yield f"data: {json.dumps(snapshot)}\n\n"
                if snapshot["status"] in ("done", "error", "deleted"):
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
async def job_parts(job_id: str, user: Viewer) -> dict:
    """The timeline split into parts, for choosing what the final video keeps."""
    job = require_job(job_id, user)
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
    user: Viewer,
    keep: list[int] = Body(..., embed=True),
    style: str = Body("classic", embed=True),
) -> dict:
    """Stitch the chosen parts, in order, and render them in a caption style."""
    job = require_job(job_id, user)
    if job.status in ("analyzing", "rendering"):
        raise HTTPException(409, f"job is already {job.status}")

    try:
        await manager.approve(job, keep, style)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return job.snapshot()


@app.post("/jobs/{job_id}/captions")
async def edit_captions(job_id: str, user: Viewer, edits: list[dict] = Body(..., embed=True)) -> dict:
    """Correct misheard words before the captions are drawn.

    Returns the parts again rather than nothing, so the review screen redraws
    from the server's copy instead of trusting what it just typed.
    """
    job = require_job(job_id, user)
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
async def job_preview(job_id: str, user: Viewer) -> FileResponse:
    """The review copy of the source, for hearing a cut before making it."""
    job = require_job(job_id, user)

    preview = job.work_dir / "preview.mp4"
    if not preview.exists():
        # The proxy is best-effort; fall back to the original, which the
        # browser may or may not be able to play.
        if not job.source.exists():
            raise HTTPException(404, "no preview available")
        return FileResponse(job.source)
    return FileResponse(preview, media_type="video/mp4")


@app.get("/instagram")
async def instagram_status(user: Viewer) -> dict:
    """Whether Instagram is connected, and as whom.

    Verified against Instagram rather than just "is the token set", so a
    revoked or expired token shows up here, before a render is spent on it.
    The connection is the operator's own account; everyone else is told so
    and downloads instead.
    """
    if not user.can_publish:
        return {"connected": False, "error": "publishing is only available to the operator", "operator_only": True}
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
async def publish_job(job_id: str, user: Viewer, caption: str = Body("", embed=True)) -> dict:
    """Post the finished video to Instagram as a Reel."""
    job = require_job(job_id, user)
    if not user.can_publish:
        raise HTTPException(403, "publishing is only available to the operator")
    try:
        await manager.publish(job, caption.strip())
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return job.publish.snapshot()


@app.get("/jobs/{job_id}/publish")
async def publish_status(job_id: str, user: Viewer) -> dict:
    job = require_job(job_id, user)
    if job.publish is None:
        raise HTTPException(404, "this video has not been published")
    return job.publish.snapshot()


@app.get("/jobs/{job_id}/result")
async def job_result(job_id: str, user: Viewer) -> FileResponse:
    job = require_job(job_id, user)
    if job.status != "done" or job.output is None:
        raise HTTPException(409, f"job is {job.status}, not finished")

    # The download carries the render's timestamp, so two cuts of the same
    # source saved to the same folder do not overwrite each other.
    stamp = job.output.stem.removeprefix("final-")
    return FileResponse(
        job.output,
        media_type="video/mp4",
        filename=f"{Path(job.filename).stem}-{stamp}.mp4" if stamp else f"{Path(job.filename).stem}.mp4",
        headers={"Cache-Control": "no-store"},
    )
