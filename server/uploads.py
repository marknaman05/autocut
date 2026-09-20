"""Getting a big video onto this machine, one part at a time.

The app is reached through a Cloudflare tunnel, and Cloudflare will not
proxy a request body over 100 MB -- a raw phone take is several times that.
So the browser does not send a large video in one request.  It asks here
for an upload id, PUTs the file in 50 MB parts, each written at its offset
into one file under the work root, and then says finish: the assembled file
is handed to the job manager exactly as a plain upload would be, and the
pipeline never learns it arrived in pieces.

A part is one request, so a dropped connection costs one part, not the
whole upload; the browser retries just that part, and writing at an offset
makes a retry harmless.  A closed tab leaves a partial file behind; those
are removed when the app starts, since the ids that could finish them died
with the process.

What the browser is trusted with: nothing that matters.  The upload id it
hands back is looked up here, where the owner, name and size were recorded
when it started; a wrong id, or someone else's, is 404.  The size declared
at the start is the size that is enforced, part by part and at the end.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from . import retention

log = logging.getLogger(__name__)

#: Each part is one request through the proxy: comfortably under Cloudflare's
#: 100 MB, and small enough that a flaky connection loses little on a retry.
PART_SIZE = 50 * 1024 * 1024
DIR = "uploads"


class NoSuchUpload(LookupError):
    pass


class BadPart(ValueError):
    pass


@dataclass
class Pending:
    id: str
    owner: str
    filename: str
    size: int
    preset: str
    path: Path

    @property
    def parts(self) -> int:
        return (self.size + PART_SIZE - 1) // PART_SIZE

    def part_size(self, n: int) -> int:
        """How many bytes part ``n`` (1-based) must carry."""
        if n < 1 or n > self.parts:
            raise BadPart(f"part {n} of {self.parts} does not exist")
        return min(PART_SIZE, self.size - (n - 1) * PART_SIZE)


class Uploads:
    """The uploads in flight and the files they are landing in."""

    def __init__(self, root: Path) -> None:
        self.dir = root / DIR
        self.pending: dict[str, Pending] = {}

    def sweep(self) -> None:
        """Remove what a previous process left half-uploaded."""
        if self.dir.exists():
            left = sum(1 for _ in self.dir.iterdir())
            shutil.rmtree(self.dir, ignore_errors=True)
            if left:
                log.info("removed %d unfinished upload(s)", left)
        self.dir.mkdir(parents=True, exist_ok=True)

    def start(self, owner: str, filename: str, size: int, preset: str) -> dict:
        """Register an upload.  The size is checked here, before a byte moves,
        against the same cap the job manager enforces on arrival: an oversized
        file should be refused in a millisecond, not after ten minutes."""
        cap = retention.max_upload_bytes()
        if size > cap:
            raise ValueError(f"that file is over the {retention.format_bytes(cap)} upload limit")
        if size <= 0:
            raise ValueError("that file is empty")

        upload_id = uuid.uuid4().hex
        self.dir.mkdir(parents=True, exist_ok=True)
        pending = Pending(
            id=upload_id, owner=owner, filename=filename, size=size, preset=preset,
            path=self.dir / f"{upload_id}.part",
        )
        # Pre-size the file so a part can land at its offset in any order.
        with pending.path.open("wb") as handle:
            handle.truncate(size)
        self.pending[upload_id] = pending
        log.info("started upload %s (%s, %s) for %s in %d parts",
                 upload_id, filename, retention.format_bytes(size), owner, pending.parts)
        return {"id": upload_id, "part_size": PART_SIZE, "parts": pending.parts}

    def get(self, upload_id: str, owner: str) -> Pending:
        pending = self.pending.get(upload_id)
        if pending is None or pending.owner != owner:
            raise NoSuchUpload(upload_id)
        return pending

    def write_part(self, pending: Pending, n: int, data: bytes) -> None:
        """Land part ``n`` (1-based).  Overwriting is fine -- that is a retry."""
        expected = pending.part_size(n)
        if len(data) != expected:
            raise BadPart(
                f"part {n} is {retention.format_bytes(len(data))}, "
                f"not the {retention.format_bytes(expected)} it should be"
            )
        with pending.path.open("r+b") as handle:
            handle.seek((n - 1) * PART_SIZE)
            handle.write(data)

    def open(self, pending: Pending) -> BinaryIO:
        """The assembled file as a stream with ``read(n)``, for the job manager."""
        if pending.path.stat().st_size != pending.size:
            raise ValueError("the upload is not the size it started as")
        return pending.path.open("rb")

    def discard(self, pending: Pending) -> None:
        """Forget the upload and remove its file."""
        self.pending.pop(pending.id, None)
        pending.path.unlink(missing_ok=True)
