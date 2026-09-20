"""Keeping the disk bounded.

A job costs about as much disk as the video that was uploaded: the upload
itself dominates, the render's intermediates add a few percent, the result
a little more.  The upload has to stay while the job can still be
re-rendered -- a second render cuts from it -- but the intermediates are
re-creatable and go the moment a render finishes, and whole jobs go after
a retention window.  Uploads are also capped, and each user has a budget of
jobs and bytes, so the total is bounded by users x budget rather than by
how enthusiastic one person is.
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def retention_days() -> int:
    return _int_env("AUTOCUT_RETENTION_DAYS", 7)


def max_upload_bytes() -> int:
    return _int_env("AUTOCUT_MAX_UPLOAD_BYTES", 2 * 1024**3)


def max_duration() -> float:
    """Seconds; the default is Instagram's limit for a Reel."""
    return float(_int_env("AUTOCUT_MAX_DURATION", 15 * 60))


def max_jobs_per_user() -> int:
    return _int_env("AUTOCUT_MAX_JOBS_PER_USER", 10)


def max_bytes_per_user() -> int:
    return _int_env("AUTOCUT_MAX_BYTES_PER_USER", 10 * 1024**3)


def trial_credits() -> int:
    """Videos a trial may upload before being asked to upgrade."""
    return _int_env("AUTOCUT_TRIAL_CREDITS", 3)


def max_active_per_user() -> int:
    return _int_env("AUTOCUT_MAX_ACTIVE_PER_USER", 2)


#: Re-creatable from the upload; deleted once a render has finished.
INTERMEDIATES = ("audio.wav", "captions.txt")
INTERMEDIATE_GLOBS = ("cut*.mkv", "*.filter")
INTERMEDIATE_DIRS = ("captions", "holes")


def drop_intermediates(work_dir: Path) -> int:
    """Remove what a finished render no longer needs; returns bytes freed."""
    freed = 0
    paths: list[Path] = [work_dir / name for name in INTERMEDIATES]
    for pattern in INTERMEDIATE_GLOBS:
        paths.extend(work_dir.glob(pattern))
    for path in paths:
        if path.is_file():
            freed += path.stat().st_size
            path.unlink()
    for name in INTERMEDIATE_DIRS:
        directory = work_dir / name
        if directory.is_dir():
            freed += sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
            shutil.rmtree(directory)
    return freed


def directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def expires_at(created: datetime) -> datetime:
    return created + timedelta(days=retention_days())


def expired(created: datetime, now: datetime | None = None) -> bool:
    return (now or datetime.now(timezone.utc)) >= expires_at(created)


def remove_job_dir(work_dir: Path) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


def format_bytes(count: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1024 or unit == "GB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} GB"
