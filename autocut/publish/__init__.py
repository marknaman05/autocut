"""Sending a finished video somewhere -- Instagram, one of two ways.

``instagram`` talks to Meta directly and needs a Meta developer app;
``upload_post`` goes through upload-post.com and needs a subscription there.
Which one is in use is decided by ``AUTOCUT_PUBLISH_BACKEND``, or failing that
by whichever key is in the environment, so pasting one key into ``.env`` is
enough to connect.
"""

from __future__ import annotations

import os
from types import ModuleType

from . import instagram, upload_post

BACKENDS: dict[str, ModuleType] = {
    "instagram": instagram,
    "upload-post": upload_post,
}


def backend() -> ModuleType:
    """The publish backend the environment selects.

    An explicit ``AUTOCUT_PUBLISH_BACKEND`` wins.  Otherwise whichever service
    has a key set; Upload-Post first, since anyone who has paid for it plainly
    means to use it.  With nothing set, the direct client is returned so that
    its "not configured" message is the one a person sees.
    """
    name = os.environ.get("AUTOCUT_PUBLISH_BACKEND")
    if name:
        try:
            return BACKENDS[name]
        except KeyError:
            choices = ", ".join(BACKENDS)
            raise instagram.InstagramError(
                f"unknown AUTOCUT_PUBLISH_BACKEND {name!r}; choose from {choices}"
            ) from None
    if upload_post.configured():
        return upload_post
    return instagram


def configured() -> bool:
    return backend().configured()
