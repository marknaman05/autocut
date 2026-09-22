"""Sign in with Google, done by the app.

The plain OAuth 2.0 authorization-code flow, no library: send the browser to
Google with a random ``state``; Google sends it back with a code; swap the
code for tokens; ask Google's tokeninfo endpoint who the ID token belongs to
(that call is the signature check, so no JWT parsing here); set the session
cookie from ``auth``.  Only verified addresses are accepted.

Needs, in the environment:

    AUTOCUT_GOOGLE_CLIENT_ID      a *Web application* OAuth client
    AUTOCUT_GOOGLE_CLIENT_SECRET
    AUTOCUT_SESSION_SECRET        32+ random characters
    AUTOCUT_PUBLIC_URL            where Google should send people back
                                  (redirect URI: <public url>/auth/callback)
"""

from __future__ import annotations

import base64
import logging
import os
import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from . import auth

log = logging.getLogger(__name__)
router = APIRouter()

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
STATE_COOKIE = "autocut_oauth_state"


def public_url(request: Request) -> str:
    return (os.environ.get("AUTOCUT_PUBLIC_URL") or str(request.base_url)).rstrip("/")


def secure(request: Request) -> bool:
    return public_url(request).startswith("https://")


LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>autocut · sign in</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&display=swap">
<style>
  :root{--paper:#fff;--tint:#fafafa;--border:#e5e5e5;--muted:#737373;--ink:#171717;--sans:"Geist",ui-sans-serif,system-ui,sans-serif;--serif:"Instrument Serif",ui-serif,Georgia,serif}
  *{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--tint);color:var(--ink);font:15px/1.6 var(--sans);letter-spacing:-.01em;-webkit-font-smoothing:antialiased}
  .card{width:min(420px,calc(100vw - 32px));background:var(--paper);border:1px solid var(--border);border-radius:16px;padding:36px 32px;box-shadow:0 20px 25px -5px rgba(212,212,212,.5)}
  .wordmark{font-family:var(--serif);font-size:28px}h1{font-family:var(--serif);font-weight:400;font-size:34px;line-height:1.1;margin:22px 0 8px}
  p{color:var(--muted);margin:0 0 26px}
  .btn{display:flex;align-items:center;justify-content:center;gap:10px;width:100%;height:46px;border:1px solid var(--ink);border-radius:8px;background:var(--ink);color:#fff;font:500 15px var(--sans);text-decoration:none;transition:background .15s}
  .btn:hover{background:#000}.fine{margin:18px 0 0;font-size:12px;color:var(--muted)}.err{margin:0 0 16px;padding:10px 12px;border-radius:8px;background:#fff1f2;color:#be123c;font-size:13px}
</style></head><body><main class="card">
<div class="wordmark">autocut</div>
<h1>Drop in a raw take.<br><em>Get back the cut.</em></h1>
<p>Sign in to upload videos and keep your edits.</p>
{{error}}
<a class="btn" href="/auth/google?next={{next}}">
<svg width="18" height="18" viewBox="0 0 48 48"><path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9.1 3.6l6.8-6.8C35.8 2.4 30.3 0 24 0 14.6 0 6.5 5.4 2.6 13.2l7.9 6.1C12.4 13.5 17.7 9.5 24 9.5z"/><path fill="#4285F4" d="M46.5 24.5c0-1.6-.1-3.1-.4-4.5H24v9h12.7c-.6 3-2.3 5.5-4.8 7.2l7.5 5.8c4.4-4.1 7.1-10.1 7.1-17.5z"/><path fill="#FBBC05" d="M10.5 28.7A14.5 14.5 0 0 1 9.5 24c0-1.6.3-3.2.8-4.7l-7.9-6.1A24 24 0 0 0 0 24c0 3.9.9 7.5 2.6 10.8l7.9-6.1z"/><path fill="#34A853" d="M24 48c6.5 0 11.9-2.1 15.9-5.8l-7.5-5.8c-2.1 1.4-4.9 2.3-8.4 2.3-6.3 0-11.6-4-13.5-9.7l-7.9 6.1C6.5 42.6 14.6 48 24 48z"/></svg>
Continue with Google</a>
<p class="fine">We use your address only to keep your videos yours. Nothing is posted anywhere without you pressing the button.</p>
</main></body></html>"""


@router.get("/login", response_class=HTMLResponse)
async def login(request: Request, next: str = "/", error: str = "") -> str:
    if auth.mode() != "google":
        raise HTTPException(404, "sign-in is not configured on this instance")
    messages = {
        "denied": "Google sign-in was cancelled.",
        "state": "That sign-in attempt expired — try again.",
        "unverified": "That Google account's email is not verified.",
        "failed": "Google sign-in failed — try again.",
    }
    banner = f'<div class="err">{messages.get(error, "")}</div>' if error in messages else ""
    return LOGIN_PAGE.replace("{{error}}", banner).replace("{{next}}", _safe_next(next))


@router.get("/auth/google")
async def start(request: Request, next: str = "/") -> Response:
    if auth.mode() != "google":
        raise HTTPException(404, "sign-in is not configured on this instance")
    state = secrets.token_urlsafe(24)
    params = {
        "client_id": os.environ[auth.GOOGLE_CLIENT_ID_ENV],
        "redirect_uri": public_url(request) + "/auth/callback",
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    }
    response = RedirectResponse(AUTH_URL + "?" + urlencode(params), status_code=302)
    # "<state>.<base64 next>": only cookie-legal characters, so the value is
    # never quoted on the way out and mangled on the way back.
    packed = base64.urlsafe_b64encode(_safe_next(next).encode()).decode().rstrip("=")
    response.set_cookie(STATE_COOKIE, f"{state}.{packed}", max_age=600, httponly=True,
                        secure=secure(request), samesite="lax")
    return response


@router.get("/auth/callback")
async def callback(request: Request, code: str = "", state: str = "", error: str = "") -> Response:
    if error:
        return RedirectResponse("/login?error=denied", status_code=302)
    saved = request.cookies.get(STATE_COOKIE, "")
    saved_state, _, packed = saved.partition(".")
    try:
        next_url = base64.urlsafe_b64decode((packed + "=" * (-len(packed) % 4)).encode()).decode() if packed else "/"
    except (ValueError, UnicodeDecodeError):
        next_url = "/"
    if not code or not state or not saved_state or not secrets.compare_digest(state, saved_state):
        return RedirectResponse("/login?error=state", status_code=302)
    try:
        email = await google_email(code, public_url(request) + "/auth/callback")
    except UnverifiedEmail:
        return RedirectResponse("/login?error=unverified", status_code=302)
    except Exception as exc:  # network, bad code, misconfigured client
        log.warning("google sign-in failed: %s", exc)
        return RedirectResponse("/login?error=failed", status_code=302)

    response = RedirectResponse(_safe_next(next_url), status_code=302)
    response.set_cookie(auth.SESSION_COOKIE, auth.make_session(email), max_age=auth.SESSION_DAYS * 86400,
                        httponly=True, secure=secure(request), samesite="lax")
    response.delete_cookie(STATE_COOKIE)
    log.info("signed in: %s", email)
    return response


@router.post("/logout")
async def logout(request: Request) -> Response:
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(auth.SESSION_COOKIE)
    return response


class UnverifiedEmail(Exception):
    pass


async def google_email(code: str, redirect_uri: str) -> str:
    """Exchange the code, then have Google tell us who the ID token is for."""
    async with httpx.AsyncClient(timeout=20) as client:
        tokens = await client.post(TOKEN_URL, data={
            "code": code,
            "client_id": os.environ[auth.GOOGLE_CLIENT_ID_ENV],
            "client_secret": os.environ[auth.GOOGLE_CLIENT_SECRET_ENV],
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        })
        tokens.raise_for_status()
        id_token = tokens.json()["id_token"]
        info = await client.get(TOKENINFO_URL, params={"id_token": id_token})
        info.raise_for_status()
        claims = info.json()
    if claims.get("aud") != os.environ[auth.GOOGLE_CLIENT_ID_ENV]:
        raise RuntimeError("id token was issued for a different client")
    if claims.get("email_verified") not in ("true", True):
        raise UnverifiedEmail(claims.get("email", ""))
    return claims["email"].strip().lower()


def _safe_next(url: str) -> str:
    """Only same-site paths; anything else goes home."""
    return url if url.startswith("/") and not url.startswith("//") else "/"
