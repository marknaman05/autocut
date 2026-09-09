"""ASR backend registry."""

from __future__ import annotations

from .base import Transcriber, clean_words, load_words, normalize, refine_timings, save_words

__all__ = [
    "Transcriber",
    "clean_words",
    "get_transcriber",
    "load_words",
    "normalize",
    "refine_timings",
    "save_words",
]


def get_transcriber(backend: str, model: str) -> Transcriber:
    if backend == "mlx-whisper":
        from .mlx_whisper import MlxWhisperTranscriber

        return MlxWhisperTranscriber(model)
    raise ValueError(f"unknown ASR backend {backend!r}")
