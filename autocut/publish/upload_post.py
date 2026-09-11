"""Publish a finished video through Upload-Post (upload-post.com).

Upload-Post is a hosted relay: you connect your social accounts to it once, in
its dashboard, and it holds the platform tokens and does the platform-specific
publishing dance.  Against talking to Instagram directly (``instagram.py``)
that trades two things -- the video passes through a third party, and there is
a subscription -- for two others: no Meta developer app to set up, and the
same call reaches TikTok, YouTube and the rest by adding a platform name.

One multipart POST does the whole job.  It answers synchronously with the
post's URL when the platform is quick, and falls over to a ``request_id`` to
poll when it is not; both are handled here so the caller sees one result.

Same surface as ``instagram.py`` -- ``configured``, ``account``,
``publish_reel`` -- so the server and CLI can pick either without caring.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path

from .instagram import Account, InstagramError, Published, Reporter

log = logging.getLogger(__name__)

API_KEY_ENV = "UPLOAD_POST_API_KEY"
#: The Upload-Post *profile* to post as -- a name you chose on its "manage
#: users" page, not your Instagram handle.  Optional when the key has exactly
#: one profile with Instagram connected.
USER_ENV = "UPLOAD_POST_USER"
API_URL = os.environ.get("UPLOAD_POST_API_URL", "https://api.upload-post.com/api")

POLL_INTERVAL = 5.0
POLL_TIMEOUT = 600.0
CAPTION_LIMIT = 2200


def configured() -> bool:
    return bool(os.environ.get(API_KEY_ENV))


def _key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise InstagramError(f"{API_KEY_ENV} is not set; add it to .env to publish via Upload-Post")
    return key


def _call(
    path: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 60.0,
) -> dict:
    headers = {"Authorization": f"Apikey {_key()}"}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(
        f"{API_URL}/{path.lstrip('/')}", data=data, method=method, headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        raw = error.read().decode(errors="replace")
        try:
            parsed = json.loads(raw)
            message = parsed.get("error") or parsed.get("message") or raw[:200]
        except (json.JSONDecodeError, AttributeError):
            message = raw[:200]
        raise InstagramError(f"Upload-Post returned {error.code}: {message}") from error
    except urllib.error.URLError as error:
        raise InstagramError(f"could not reach Upload-Post: {error.reason}") from error
    return body if isinstance(body, dict) else {"result": body}


def _multipart(fields: list[tuple[str, str]], file_field: str, path: Path) -> tuple[bytes, str]:
    """Encode a multipart/form-data body by hand; urllib has no helper for it."""
    boundary = f"----autocut{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields:
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
        f"filename=\"{path.name}\"\r\nContent-Type: video/mp4\r\n\r\n".encode()
    )
    parts.append(path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _profiles() -> list[dict]:
    reply = _call("uploadposts/users")
    return [p for p in reply.get("profiles", []) if isinstance(p, dict)]


def _instagram_handle(profile: dict) -> str | None:
    """The Instagram handle a profile has connected, or None if it has not."""
    linked = (profile.get("social_accounts") or {}).get("instagram")
    if not linked:
        return None
    if isinstance(linked, str):
        return linked
    return linked.get("handle") or linked.get("username") or linked.get("display_name") or "instagram"


def account() -> Account:
    """The profile that will be posted as, and its Instagram handle.

    ``user_id`` here is the Upload-Post profile name, because that is what the
    upload call wants -- the same slot the Meta client fills with the account
    id.  Chosen from the environment if set, otherwise the only profile with
    Instagram connected; more than one and you have to say which.
    """
    profiles = _profiles()
    wanted = os.environ.get(USER_ENV)
    if wanted:
        match = next((p for p in profiles if p.get("username") == wanted), None)
        if match is None:
            names = ", ".join(p.get("username", "?") for p in profiles) or "none"
            raise InstagramError(
                f"Upload-Post has no profile called {wanted!r} (it has: {names}); check {USER_ENV}"
            )
        handle = _instagram_handle(match)
        if handle is None:
            raise InstagramError(
                f"Upload-Post profile {wanted!r} has no Instagram account connected; "
                "connect one at app.upload-post.com"
            )
        return Account(user_id=wanted, username=handle)

    with_instagram = [(p, _instagram_handle(p)) for p in profiles]
    with_instagram = [(p, h) for p, h in with_instagram if h]
    if not with_instagram:
        raise InstagramError(
            "no Upload-Post profile has an Instagram account connected yet; "
            "add one at app.upload-post.com under Manage users"
        )
    if len(with_instagram) > 1:
        names = ", ".join(p["username"] for p, _ in with_instagram)
        raise InstagramError(
            f"several Upload-Post profiles have Instagram connected ({names}); "
            f"set {USER_ENV} to the one to post as"
        )
    profile, handle = with_instagram[0]
    return Account(user_id=str(profile["username"]), username=handle)


def publish_reel(
    video: Path,
    caption: str = "",
    *,
    share_to_feed: bool = True,
    report: Reporter | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Published:
    video = Path(video)
    if not video.exists():
        raise InstagramError(f"no such file: {video}")
    if len(caption) > CAPTION_LIMIT:
        raise InstagramError(f"caption is over Instagram's {CAPTION_LIMIT} character limit")
    say = report or (lambda _: None)

    say("checking the Upload-Post profile")
    who = account()

    size = video.stat().st_size
    say(f"uploading {size / 1e6:.1f} MB to post as @{who.username}")
    fields = [
        ("user", who.user_id),
        ("platform[]", "instagram"),
        ("title", caption),
        ("media_type", "REELS"),
        ("share_to_feed", "true" if share_to_feed else "false"),
        ("async_upload", "false"),
    ]
    body, content_type = _multipart(fields, "video", video)
    reply = _call("upload", method="POST", data=body, content_type=content_type, timeout=600.0)

    if reply.get("request_id") and "results" not in reply:
        say("waiting for Instagram to process the video")
        reply = _wait(str(reply["request_id"]), sleep)

    return _result(reply)


def _result(reply: dict) -> Published:
    """Read one Instagram outcome out of either the sync or the polled shape."""
    results = reply.get("results")
    if isinstance(results, dict):
        outcome = results.get("instagram") or {}
    elif isinstance(results, list):
        outcome = next((r for r in results if r.get("platform") == "instagram"), {})
    else:
        outcome = {}

    if not outcome.get("success", reply.get("success", False)):
        detail = outcome.get("error") or outcome.get("message") or reply.get("error") or reply.get("message") or repr(reply)[:200]
        raise InstagramError(f"Instagram rejected the video: {detail}")

    media_id = str(outcome.get("post_id") or outcome.get("publish_id") or outcome.get("container_id") or reply.get("request_id") or "")
    permalink = outcome.get("url")
    log.info("published via Upload-Post: %s", permalink or media_id or "ok")
    return Published(media_id=media_id, permalink=permalink)


def _wait(request_id: str, sleep: Callable[[float], None]) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT
    while True:
        status = _call(f"uploadposts/status?request_id={request_id}")
        if status.get("status") == "completed":
            # The polled shape carries no post id of its own; the request id
            # is the only handle left for finding it in Upload-Post's history.
            status.setdefault("request_id", request_id)
            return status
        if time.monotonic() >= deadline:
            raise InstagramError(
                f"Upload-Post is still processing after {POLL_TIMEOUT:.0f}s; "
                "check app.upload-post.com, the post may still appear"
            )
        sleep(POLL_INTERVAL)
