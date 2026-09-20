"""``AUTOCUT_RETAKE_BACKEND=none``: the n-gram detector alone, no model call.

For working on the page without paying for, or waiting on, a hosted model
per upload.  The switch must be total -- no request leaves -- and the
deterministic result must be the one that comes back.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from autocut import config as config_module
from autocut.analyze import retakes
from autocut.config import RetakeConfig
from autocut.models import Word


def words_with_a_retake() -> list[Word]:
    """The same six words twice, with the pause before the restart that the
    n-gram detector wants to see."""
    first = "so today we are going to".split()
    second = "so today we are going to talk about the thing".split()
    words = [Word(text=w, start=i * 0.4, end=i * 0.4 + 0.3) for i, w in enumerate(first)]
    base = len(first) * 0.4 + 0.8  # a clear breath before the retake
    words += [Word(text=w, start=base + i * 0.4, end=base + i * 0.4 + 0.3) for i, w in enumerate(second)]
    return words


def test_none_never_asks_a_model(monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("a model was called")

    monkeypatch.setattr(retakes, "_llm_spans", explode)
    spans = retakes.detect(words_with_a_retake(), RetakeConfig(backend="none"))
    # The deterministic detector still found the repeated run.
    assert spans and all(s.reason.name == "NGRAM_REPEAT" for s in spans)


def test_the_environment_selects_it(monkeypatch) -> None:
    monkeypatch.setenv("AUTOCUT_RETAKE_BACKEND", "none")
    assert config_module._retake_from_env(RetakeConfig()).backend == "none"


def test_other_backends_still_ask(monkeypatch) -> None:
    asked = []
    monkeypatch.setattr(retakes, "_llm_spans", lambda words, config: asked.append(config.backend) or [])
    retakes.detect(words_with_a_retake(), RetakeConfig(backend="openrouter"))
    assert asked == ["openrouter"]
