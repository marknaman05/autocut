"""Thin wrappers around the ffmpeg/ffprobe binaries.

Every subprocess in the project goes through here so failures surface with the
actual stderr rather than a bare non-zero exit code.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    """An ffmpeg/ffprobe invocation failed; carries the tail of its stderr."""


def require_binaries() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise FFmpegError(
            f"{' and '.join(missing)} not found on PATH. Install with: brew install ffmpeg"
        )


def run(args: list[str], *, description: str = "ffmpeg", include_stderr: bool = False) -> str:
    """Run a command, returning stdout.  Raises :class:`FFmpegError` on failure.

    ffmpeg reports almost everything on stderr -- including filter output such
    as loudnorm's measurements -- so ``include_stderr`` appends it for callers
    that need to read those.
    """
    log.debug("%s: %s", description, " ".join(args))
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-15:])
        raise FFmpegError(f"{description} failed (exit {result.returncode}):\n{tail}")
    return result.stdout + result.stderr if include_stderr else result.stdout


def ffprobe_json(path: Path, *args: str) -> dict[str, Any]:
    output = run(
        ["ffprobe", "-v", "error", "-of", "json", *args, str(path)],
        description=f"ffprobe {path.name}",
    )
    return json.loads(output or "{}")


def version() -> tuple[int, int]:
    """The installed ffmpeg's (major, minor) version."""
    global _version
    if _version is None:
        text = run(["ffmpeg", "-hide_banner", "-version"], description="ffmpeg -version")
        match = re.search(r"ffmpeg version n?(\d+)\.(\d+)", text)
        _version = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    return _version


_version: tuple[int, int] | None = None


def filter_script_args(script: Path) -> list[str]:
    """Arguments for reading a filter graph from a file.

    ffmpeg 7.1 introduced the generic ``-/option file`` form and 8.0 removed
    ``-filter_complex_script``, so which one works depends on the install.
    Reading from a file at all is what keeps a heavily chopped edit from
    blowing the command line length limit.
    """
    if version() >= (7, 1):
        return ["-/filter_complex", str(script)]
    return ["-filter_complex_script", str(script)]


def ffmpeg(args: list[str], *, description: str = "ffmpeg") -> None:
    """Run ffmpeg with the boilerplate flags we always want."""
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], description=description)
