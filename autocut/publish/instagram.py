"""Publish a finished video to Instagram as a Reel.

The Instagram API publishes in two steps -- create a *container* describing the
post, then publish the container once Instagram has finished processing it.
Containers normally fetch the video from a public URL, which a tool bound to
localhost cannot offer; the *resumable upload* variant instead accepts the
bytes directly, so nothing here needs a tunnel or a bucket.  That is the only
reason this module exists as more than two requests.

Authentication is a long-lived access token pasted into ``.env``.  This is a
single-user tool, and Meta's own dashboard hands one out for an Instagram
professional account (see the README); an OAuth dance would add an HTTPS
redirect and a Meta app review for exactly one user.  Tokens last sixty days;
``autocut instagram refresh`` extends one for another sixty.

Wire calls go through ``urllib`` like the rest of the project.  Every request
is funnelled through ``_call`` so that the tests can swap one function out.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

TOKEN_ENV = "INSTAGRAM_ACCESS_TOKEN"
USER_ID_ENV = "INSTAGRAM_USER_ID"
#: Meta versions the API roughly quarterly and keeps a version alive for
#: about two years; a stale default only stops working when they retire it.
API_VERSION = os.environ.get("INSTAGRAM_API_VERSION", "v25.0")

#: Instagram Login and Facebook Login are different products with different
#: hosts; the token decides which one it works against.  ``graph.instagram.com``
#: is the one the README walks through.  Resumable uploads go to Meta's upload
#: host either way.
GRAPH_HOST = os.environ.get("INSTAGRAM_GRAPH_HOST", "https://graph.instagram.com")
UPLOAD_HOST = "https://rupload.facebook.com"

#: Meta says to poll once a minute for at most five minutes.  A Reel this size
#: is usually ready inside thirty seconds, so poll a little faster than that
#: and give up after the same five minutes.
POLL_INTERVAL = 5.0
POLL_TIMEOUT = 300.0

#: Instagram caps Reel captions at 2,200 characters.
CAPTION_LIMIT = 2200


class InstagramError(RuntimeError):
    """Anything that stops a publish, with a message fit to show a person."""


@dataclass(frozen=True)
class Account:
    user_id: str
    username: str


@dataclass(frozen=True)
class Published:
    media_id: str
    permalink: str | None


Reporter = Callable[[str], None]


def configured() -> bool:
    """Whether a token is present at all -- not whether it works."""
    return bool(os.environ.get(TOKEN_ENV))


def _token() -> str:
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise InstagramError(
            f"{TOKEN_ENV} is not set; add it to .env to publish to Instagram"
        )
    return token


def _call(
    url: str,
    *,
    method: str = "GET",
    params: dict[str, str] | None = None,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict:
    """One request to Meta, with its error format turned into an exception.

    Meta reports failures as a JSON body ``{"error": {"message": ...}}`` on a
    4xx/5xx status; the message is written for developers but is far better
    than the status code alone ("The video file you selected is in a format
    that we don't support" beats a 400).
    """
    if params:
        query = urllib.parse.urlencode(params)
        if method == "GET":
            url = f"{url}?{query}"
            body = None
        else:
            body = query.encode()
            headers = {"Content-Type": "application/x-www-form-urlencoded", **(headers or {})}
    else:
        body = data

    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        raw = error.read().decode(errors="replace")
        try:
            message = json.loads(raw)["error"]["message"]
        except (json.JSONDecodeError, KeyError, TypeError):
            message = raw[:200]
        raise InstagramError(f"Instagram returned {error.code}: {message}") from error
    except urllib.error.URLError as error:
        raise InstagramError(f"could not reach Instagram: {error.reason}") from error


def _graph(path: str, **kwargs) -> dict:
    return _call(f"{GRAPH_HOST}/{API_VERSION}/{path.lstrip('/')}", **kwargs)


def account() -> Account:
    """Who the token belongs to.  Also the cheapest way to check it works."""
    token = _token()
    me = _graph("me", params={"fields": "id,user_id,username", "access_token": token})
    # With Instagram Login ``id`` is an app-scoped id and ``user_id`` is the
    # professional account the publishing endpoints want.  With Facebook Login
    # there is no ``user_id``, and the account id must come from the
    # environment because a Facebook user can manage several.
    user_id = os.environ.get(USER_ID_ENV) or me.get("user_id") or me.get("id")
    if not user_id:
        raise InstagramError(
            f"could not work out the Instagram account id; set {USER_ID_ENV} in .env"
        )
    return Account(user_id=str(user_id), username=str(me.get("username", "")))


def refresh_token() -> int:
    """Extend a long-lived Instagram Login token; returns seconds until expiry.

    Meta refuses to refresh a token less than a day old, so this is safe to
    call routinely: the error is reported, not raised, and the old token keeps
    working.  A refreshed token is only ever held in the environment for this
    process -- it is not written back to ``.env``, because overwriting a file
    someone edits by hand from a background task is the kind of surprise a
    local tool should not spring.  The README says how to store it.
    """
    token = _token()
    reply = _graph(
        "refresh_access_token",
        params={"grant_type": "ig_refresh_token", "access_token": token},
    )
    fresh = reply.get("access_token")
    if fresh:
        os.environ[TOKEN_ENV] = str(fresh)
    return int(reply.get("expires_in", 0))


def publish_reel(
    video: Path,
    caption: str = "",
    *,
    share_to_feed: bool = True,
    report: Reporter | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Published:
    """Upload ``video`` and publish it as a Reel; returns the new post.

    The steps, each of which can fail on its own and says so:

    1. create a REELS container with ``upload_type=resumable``;
    2. POST the file bytes to the upload host against that container;
    3. wait for Instagram to transcode it (``status_code`` becomes FINISHED);
    4. publish the container, which is what makes the post appear.

    ``report`` is told what is happening in words, for a progress line.
    """
    video = Path(video)
    if not video.exists():
        raise InstagramError(f"no such file: {video}")
    if len(caption) > CAPTION_LIMIT:
        raise InstagramError(f"caption is over Instagram's {CAPTION_LIMIT} character limit")
    say = report or (lambda _: None)
    token = _token()

    say("checking the Instagram account")
    who = account()

    say(f"creating a Reel on @{who.username}" if who.username else "creating a Reel")
    params = {
        "media_type": "REELS",
        "upload_type": "resumable",
        "share_to_feed": "true" if share_to_feed else "false",
        "access_token": token,
    }
    if caption:
        params["caption"] = caption
    container = _graph(f"{who.user_id}/media", method="POST", params=params)
    container_id = str(container.get("id", ""))
    if not container_id:
        raise InstagramError(f"Instagram did not return a container id: {container!r}")

    size = video.stat().st_size
    say(f"uploading {size / 1e6:.1f} MB")
    _call(
        f"{UPLOAD_HOST}/ig-api-upload/{API_VERSION}/{container_id}",
        method="POST",
        data=video.read_bytes(),
        headers={
            "Authorization": f"OAuth {token}",
            "offset": "0",
            "file_size": str(size),
            "Content-Type": "application/octet-stream",
        },
        # A Reel can be a few hundred megabytes; give the upload room.
        timeout=600.0,
    )

    say("waiting for Instagram to process the video")
    _wait_until_finished(container_id, token, sleep)

    say("publishing")
    published = _graph(
        f"{who.user_id}/media_publish",
        method="POST",
        params={"creation_id": container_id, "access_token": token},
    )
    media_id = str(published.get("id", ""))
    if not media_id:
        raise InstagramError(f"Instagram did not return a media id: {published!r}")

    # The permalink is what a person actually wants back.  It is a separate
    # read, and a failure here is not a failed publish -- the post is up.
    permalink = None
    try:
        details = _graph(media_id, params={"fields": "permalink", "access_token": token})
        permalink = details.get("permalink")
    except InstagramError as error:
        log.warning("published %s but could not read its permalink: %s", media_id, error)

    log.info("published Reel %s (%s)", media_id, permalink or "no permalink")
    return Published(media_id=media_id, permalink=permalink)


def _wait_until_finished(container_id: str, token: str, sleep: Callable[[float], None]) -> None:
    deadline = time.monotonic() + POLL_TIMEOUT
    while True:
        status = _graph(
            container_id, params={"fields": "status_code,status", "access_token": token}
        )
        code = status.get("status_code")
        if code == "FINISHED":
            return
        if code in ("ERROR", "EXPIRED"):
            detail = status.get("status") or code
            raise InstagramError(f"Instagram rejected the video: {detail}")
        if time.monotonic() >= deadline:
            raise InstagramError(
                f"Instagram is still processing the video after {POLL_TIMEOUT:.0f}s; "
                "check the account before trying again, it may still appear"
            )
        sleep(POLL_INTERVAL)
