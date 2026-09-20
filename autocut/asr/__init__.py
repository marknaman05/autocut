"""ASR backend registry."""

from __future__ import annotations

from .base import (
    Transcriber,
    clean_words,
    load_words,
    normalize,
    refine_timings,
    save_words,
    trim_overlong,
)
from .vad import SpeechTrack, VADUnavailable, align_to_speech, analyse

__all__ = [
    "SpeechTrack",
    "Transcriber",
    "VADUnavailable",
    "align_to_speech",
    "analyse",
    "clean_words",
    "get_transcriber",
    "load_words",
    "normalize",
    "refine_timings",
    "trim_overlong",
    "save_words",
]


def get_transcriber(backend: str, model: str) -> Transcriber:
    if backend == "mlx-whisper":
        from .mlx_whisper import MlxWhisperTranscriber

        return MlxWhisperTranscriber(model)
    if backend == "openrouter":
        from .openrouter_whisper import DEFAULT_MODEL, OpenRouterWhisperTranscriber

        # The preset's model names the MLX repo; it means nothing to the API.
        return OpenRouterWhisperTranscriber(model if model.startswith("openai/") else DEFAULT_MODEL)
    raise ValueError(f"unknown ASR backend {backend!r}")
