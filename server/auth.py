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
Instagram connection; everyone else downloads -- if they are on the paid
plan.  ``AUTOCUT_PRO`` lists those; owners and the local user are on it
without being listed.  A paid subscription (see ``billing``) makes someone Pro too: the app
registers a lookup with ``plan_source`` and it is consulted before the trial
list.  ``AUTOCUT_TRIAL`` lists people on a trial: the whole
product, downloads included, for a few videos (``AUTOCUT_TRIAL_CREDITS``),
after which a new upload is what asks them to upgrade -- re-rendering what
they have stays free.  Everyone else may upload, review, render and watch
the result in the page, but the download is behind the pricing page.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from fastapi import HTTPException, Request

HEADER_ENV = "AUTOCUT_USER_HEADER"
OWNERS_ENV = "AUTOCUT_OWNERS"
PRO_ENV = "AUTOCUT_PRO"
TRIAL_ENV = "AUTOCUT_TRIAL"
#: The identity of whoever runs the app on their own machine.
LOCAL = "local"


@dataclass(frozen=True)
class User:
    email: str
    #: May publish through the operator's Instagram connection.
    can_publish: bool
    #: ``"free"``, ``"trial"`` or ``"pro"``.
    plan: str = "pro"

    @property
    def is_local(self) -> bool:
        return self.email == LOCAL

    @property
    def can_download(self) -> bool:
        return self.plan in ("pro", "trial")

    @property
    def uploads_are_metered(self) -> bool:
        return self.plan == "trial"


def header_name() -> str | None:
    """The trusted header, or ``None`` in single-user mode."""
    return os.environ.get(HEADER_ENV) or None


def _addresses(env: str) -> set[str]:
    raw = os.environ.get(env, "")
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def owners() -> set[str]:
    return _addresses(OWNERS_ENV)


def pro() -> set[str]:
    return _addresses(PRO_ENV) | owners()


def trial() -> set[str]:
    return _addresses(TRIAL_ENV)


#: Answers "has this address paid?"; the app points it at the subscriptions
#: table once the store exists.  Nobody has paid until then.
_paid: Callable[[str], bool] = lambda email: False


def plan_source(paid: Callable[[str], bool]) -> None:
    global _paid
    _paid = paid


def plan_for(email: str) -> str:
    if email in pro() or _paid(email):
        return "pro"
    if email in trial():
        return "trial"
    return "free"


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
    return User(
        email=value,
        can_publish=value in owners(),
        plan=plan_for(value),
    )
