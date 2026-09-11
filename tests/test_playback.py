"""Runs the review page's JavaScript playback tests, if node is available.

The preview player is a small state machine over the browser's media events,
and it broke once already in a way that only shows up when a seek's events
arrive after the turn that started it.  Testing it means running it, so this
shells out to node rather than re-describing the logic in Python.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SUITE = Path(__file__).parent / "js" / "playback.test.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_review_playback() -> None:
    result = subprocess.run(
        ["node", str(SUITE)], capture_output=True, text=True, cwd=SUITE.parents[2]
    )
    assert result.returncode == 0, result.stdout + result.stderr
