"""Who is asking.

There is no login here.  Identity arrives as a request header set by a proxy
that has already done the login -- Cloudflare Access puts the signed-in
address in ``Cf-Access-Authenticated-User-Email``, Tailscale's ``serve`` in
``Tailscale-User-Login`` -- and the app trusts that header for one reason
only: the proxy is the only thing that can reach it.  So the two rules that
make this safe are stated here once and repeated in the README:

1. ``AUTOCUT_USER_HEADER`` unset means single-user mode.  Every request is
   the user ``local``, who may do everything.  This is what running it on
   your own machine at ``localhost:8000`` has always been.
2. ``AUTOCUT_USER_HEADER`` set means the app must be reachable **only**
   through the proxy that sets it (bind to 127.0.0.1 and let the tunnel in).
   A request without the header is refused outright.  Never bind to
   0.0.0.0 with the header set: anyone could then send the header themselves.

``AUTOCUT_OWNERS`` lists the addresses that may use the operator's own
Instagram connection; everyone else downloads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from fastapi import HTTPException, Request

HEADER_ENV = "AUTOCUT_USER_HEADER"
OWNERS_ENV = "AUTOCUT_OWNERS"
#: The identity of whoever runs the app on their own machine.
LOCAL = "local"


@dataclass(frozen=True)
class User:
    email: str
    #: May publish through the operator's Instagram connection.
    can_publish: bool

    @property
    def is_local(self) -> bool:
        return self.email == LOCAL


def header_name() -> str | None:
    """The trusted header, or ``None`` in single-user mode."""
    return os.environ.get(HEADER_ENV) or None


def owners() -> set[str]:
    raw = os.environ.get(OWNERS_ENV, "")
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def current_user(request: Request) -> User:
    """FastAPI dependency: the user behind a request.

    Refuses rather than guesses: with a header configured and absent, the
    request did not come through the proxy, and there is no safe answer to
    "who is this".
    """
    header = header_name()
    if header is None:
        return User(email=LOCAL, can_publish=True)
    value = request.headers.get(header, "").strip().lower()
    if not value:
        raise HTTPException(401, "not signed in")
    return User(email=value, can_publish=value in owners())
