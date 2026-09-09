"""Default transcriber: Whisper on Apple MLX (Metal-accelerated).

Word timestamps here come from Whisper's cross-attention DTW.  They are good
enough to cut on but drift by tens of milliseconds, which karaoke captions
expose.  ``base.refine_timings`` corrects most of it against the waveform, and
``CaptionConfig.offset`` is the last-resort global nudge.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..models import Word
from .base import clean_words

log = logging.getLogger(__name__)

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


class MlxWhisperTranscriber:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def transcribe(self, audio: Path, *, language: str | None = None) -> list[Word]:
        try:
            import mlx_whisper
        except ImportError as error:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "mlx-whisper is not installed. Run: uv sync --extra asr"
            ) from error

        log.info("transcribing %s with %s", audio.name, self.model)
        result = mlx_whisper.transcribe(
            str(audio),
            path_or_hf_repo=self.model,
            word_timestamps=True,
            language=language,
            condition_on_previous_text=False,
        )
        return clean_words(list(_iter_words(result)))


def _iter_words(result: dict):
    """Flatten Whisper's segment/word structure.

    Tolerant of missing fields: a segment without word timings is skipped
    rather than crashing a job that is otherwise fine.
    """
    for segment in result.get("segments", []):
        words = segment.get("words")
        if not words:
            log.debug("segment %r had no word timings; skipping", segment.get("text", "")[:40])
            continue
        for word in words:
            text = (word.get("word") or word.get("text") or "").strip()
            start, end = word.get("start"), word.get("end")
            if not text or start is None or end is None:
                continue
            yield Word(
                text=text,
                start=float(start),
                end=float(end),
                probability=float(word.get("probability", 1.0)),
            )
