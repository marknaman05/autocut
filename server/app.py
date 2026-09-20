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

from . import billing, retention, uploads, watermark
from .auth import User, current_user, header_name, plan_source
from .jobs import PRESETS, BadUpload, Job, JobManager, QuotaExceeded
from .samples import caption_sample

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
#: Absolute so a service unit's working directory cannot move the data.
WORK_ROOT = Path(os.environ.get("AUTOCUT_WORK_ROOT") or "work").resolve()
DB_PATH = Path(os.environ["AUTOCUT_DB"]).resolve() if os.environ.get("AUTOCUT_DB") else None

manager = JobManager(WORK_ROOT, DB_PATH)
parts = uploads.Uploads(WORK_ROOT)


def has_paid(email: str) -> bool:
    sub = manager.store.subscription(email)
    return bool(sub) and sub["status"] in billing.PRO_STATUSES


plan_source(has_paid)

Viewer = Annotated[User, Depends(current_user)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if header_name():
        log.info("multi-user: trusting %s for identity", header_name())
    else:
        log.info("single-user: no AUTOCUT_USER_HEADER set")
    if billing.enabled():
        log.info("billing: Dodo Payments (%s)", billing.environment())
    else:
        log.info("billing: off; Pro is the AUTOCUT_PRO list")
    parts.sweep()
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


@app.get("/pricing", response_class=HTMLResponse)
async def pricing() -> str:
    """The plans, and where the download button sends a free user."""
    return (STATIC / "pricing.html").read_text()


@app.get("/me")
async def me(user: Viewer) -> dict:
    """Who the page is talking to, so it can say so."""
    return {
        "email": user.email,
        "can_publish": user.can_publish,
        "plan": user.plan,
        # A trial's remaining allowance; nothing for anyone else.
        "credits": credits(user),
        "local": user.is_local,
        # Anything larger goes up in parts of this size (see ``uploads``).
        "part_size": uploads.PART_SIZE,
        "billing": billing_status(user),
    }


def billing_status(user: User) -> dict:
    """What the pricing page needs: whether Upgrade goes anywhere, and
    whether this person has a subscription to manage."""
    sub = manager.store.subscription(user.email) if not user.is_local else None
    return {
        "enabled": billing.enabled(),
        "environment": billing.environment() if billing.enabled() else None,
        "subscription": {"status": sub["status"], "updated": sub["updated"]} if sub else None,
    }


@app.post("/billing/checkout")
async def billing_checkout(user: Viewer) -> dict:
    """A hosted checkout for Pro; the page sends the browser to the URL."""
    if not billing.enabled():
        raise HTTPException(503, "payments are not set up on this instance")
    if user.is_local:
        raise HTTPException(400, "the local user is already Pro")
    if has_paid(user.email):
        raise HTTPException(409, "you already have an active subscription")
    try:
        url = await asyncio.to_thread(billing.checkout_url, user.email)
    except Exception as exc:  # the SDK's errors are all subclasses of Exception
        log.exception("checkout session failed")
        raise HTTPException(502, f"could not start checkout: {exc}")
    return {"url": url}


@app.post("/billing/portal")
async def billing_portal(user: Viewer) -> dict:
    """Dodo's customer portal, for cancelling or changing the card."""
    sub = manager.store.subscription(user.email)
    if not billing.enabled() or not sub or not sub["customer_id"]:
        raise HTTPException(404, "no subscription to manage")
    try:
        url = await asyncio.to_thread(billing.portal_url, sub["customer_id"])
    except Exception as exc:
        log.exception("portal session failed")
        raise HTTPException(502, f"could not open the billing portal: {exc}")
    return {"url": url}


@app.post("/billing/webhook")
async def billing_webhook(request: Request) -> dict:
    """Dodo's word on a subscription.  No user here: the signature is the
    authentication, and the event says whose subscription it is."""
    body = await request.body()
    try:
        event = await asyncio.to_thread(billing.verify, body, dict(request.headers))
    except Exception as exc:
        log.warning("webhook refused: %s", exc)
        raise HTTPException(401, "invalid webhook signature")
    event_id = request.headers.get("webhook-id", "")
    if event_id and not manager.store.note_webhook(event_id):
        return {"received": True, "duplicate": True}
    sub = billing.subscription_from(event)
    if sub is not None:
        manager.store.save_subscription(sub)
        log.info("billing: %s -> %s is %s", event.get("type"), sub["owner"], sub["status"])
    else:
        log.info("billing: %s (ignored)", event.get("type"))
    return {"received": True}


def credits(user: User) -> dict | None:
    if not user.uploads_are_metered:
        return None
    total = retention.trial_credits()
    used = manager.uploads_by(user.email)
    return {"used": used, "total": total, "left": max(0, total - used)}


def require_credit(user: User) -> None:
    """A trial with nothing left is asked to upgrade, before any bytes move.

    Only uploads are metered: rendering what is already here again is
    free, so a trial that has spent its videos still gets to finish them.
    """
    allowance = credits(user)
    if allowance is not None and allowance["left"] == 0:
        raise HTTPException(
            402, f"your {allowance['total']} trial videos are used; upgrade to upload another",
        )


@app.post("/jobs")
async def create_job(user: Viewer, file: UploadFile, preset: str = Form("default")) -> dict:
    if preset not in PRESETS:
        raise HTTPException(400, f"unknown preset {preset!r}")
    if not file.filename:
        raise HTTPException(400, "no file was uploaded")
    require_credit(user)

    try:
        job = await manager.submit(file.filename, file.file, preset, owner=user.email)
    except ValueError as error:
        raise submit_errors(error) from error
    return job.snapshot()


def submit_errors(error: ValueError) -> HTTPException:
    """The job manager's refusals, as HTTP."""
    if isinstance(error, QuotaExceeded):
        return HTTPException(429, str(error))
    return HTTPException(413 if "upload limit" in str(error) else 400, str(error))


@app.post("/uploads")
async def start_upload(
    user: Viewer,
    filename: str = Body(..., embed=True),
    size: int = Body(..., embed=True),
    preset: str = Body("default", embed=True),
) -> dict:
    """Begin a large upload, to be sent in parts.

    Refuses everything it can up front -- an unknown preset, a full quota, a
    file over the cap -- so nothing is uploaded that would be turned away
    once it arrived.
    """
    if preset not in PRESETS:
        raise HTTPException(400, f"unknown preset {preset!r}")
    if not filename:
        raise HTTPException(400, "no file was uploaded")
    require_credit(user)
    try:
        manager.check_room(user.email)
    except QuotaExceeded as error:
        raise HTTPException(429, str(error)) from error
    try:
        return await asyncio.to_thread(parts.start, user.email, filename, size, preset)
    except ValueError as error:
        raise HTTPException(413 if "upload limit" in str(error) else 400, str(error)) from error


def require_upload(upload_id: str, user: User) -> uploads.Pending:
    try:
        return parts.get(upload_id, user.email)
    except uploads.NoSuchUpload:
        raise HTTPException(404, "no such upload") from None


@app.put("/uploads/{upload_id}/{n}")
async def put_part(upload_id: str, n: int, user: Viewer, request: Request) -> dict:
    """One part of the file, raw in the body; ``n`` counts from 1."""
    pending = require_upload(upload_id, user)
    try:
        expected = pending.part_size(n)
    except uploads.BadPart as error:
        raise HTTPException(400, str(error)) from error
    chunks, received = [], 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > expected:
            raise HTTPException(400, f"part {n} is larger than the {retention.format_bytes(expected)} it should be")
        chunks.append(chunk)
    try:
        await asyncio.to_thread(parts.write_part, pending, n, b"".join(chunks))
    except uploads.BadPart as error:
        raise HTTPException(400, str(error)) from error
    return {"part": n}


@app.post("/uploads/{upload_id}/finish")
async def finish_upload(upload_id: str, user: Viewer) -> dict:
    """Every part has landed: queue the assembled file as a job.

    From the job manager's point of view this is an ordinary upload; the
    stream just happens to come from a file rather than the request.  The
    file is removed whether or not the job was accepted.
    """
    pending = require_upload(upload_id, user)
    try:
        stream = await asyncio.to_thread(parts.open, pending)
        with stream:
            job = await manager.submit(pending.filename, stream, pending.preset, owner=user.email)
    except ValueError as error:
        raise submit_errors(error) from error
    finally:
        parts.discard(pending)
    return job.snapshot()


@app.delete("/uploads/{upload_id}")
async def abandon_upload(upload_id: str, user: Viewer) -> dict:
    """The browser gave up part-way; drop what has landed."""
    parts.discard(require_upload(upload_id, user))
    return {"abandoned": upload_id}


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


def finished_output(job_id: str, user: User) -> Job:
    job = require_job(job_id, user)
    if job.status != "done" or job.output is None:
        raise HTTPException(409, f"job is {job.status}, not finished")
    return job


@app.get("/jobs/{job_id}/watch")
async def job_watch(job_id: str, user: Viewer) -> FileResponse:
    """The finished video, to play in the page.

    Pro plays the render itself.  Free plays a watermarked, low-resolution
    copy: the player's URL shows in any network tab, so whatever this serves
    is, in effect, downloadable -- and that must not be the real file.
    """
    job = finished_output(job_id, user)
    path = job.output
    if not user.can_download:
        path = await asyncio.to_thread(watermark.ensure_preview, job.output)
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "no-store"})


@app.get("/jobs/{job_id}/result")
async def job_result(job_id: str, user: Viewer) -> FileResponse:
    """The finished video, as a download.  Pro only; the page sends a free
    user to the pricing page instead of here, and this is the backstop."""
    job = finished_output(job_id, user)
    if not user.can_download:
        raise HTTPException(402, "downloading needs the Pro plan; see /pricing")

    # The download carries the render's timestamp, so two cuts of the same
    # source saved to the same folder do not overwrite each other.
    stamp = job.output.stem.removeprefix("final-")
    return FileResponse(
        job.output,
        media_type="video/mp4",
        filename=f"{Path(job.filename).stem}-{stamp}.mp4" if stamp else f"{Path(job.filename).stem}.mp4",
        headers={"Cache-Control": "no-store"},
    )
