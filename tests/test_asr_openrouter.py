"""The OpenRouter ASR backend, without the network.

The request is intercepted so what is tested is ours: the audio is uploaded
as base64 MP3, the reply's word timestamps become ``Word``s, chunk offsets
are added back, and a reply without word timestamps fails loudly instead of
producing a video with no cuts.
"""

from __future__ import annotations

import base64
import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from autocut.asr import get_transcriber
from autocut.asr import openrouter_whisper as ow


def tone_wav(path: Path, seconds: float, rate: int = 16000) -> None:
    t = np.arange(int(seconds * rate)) / rate
    samples = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def intercept(monkeypatch):
    """Capture the request body and answer with a canned verbose_json."""
    seen: dict = {}

    def fake_urlopen(request, timeout=0):
        seen["body"] = json.loads(request.data)
        seen["headers"] = dict(request.header_items())
        reply = seen.get("reply") or {
            "text": "hello there world",
            "duration": 2.0,
            "words": [
                {"word": " hello", "start": 0.2, "end": 0.5},
                {"word": " there", "start": 0.5, "end": 0.8},
                {"word": " world", "start": 0.9, "end": 1.4},
            ],
            "segments": [{"start": 0.2, "end": 1.4, "text": " hello there world"}],
            "usage": {"seconds": 2.0, "cost": 0.00002},
        }
        return FakeResponse(json.dumps(reply).encode())

    monkeypatch.setattr(ow.urllib.request, "urlopen", fake_urlopen)
    return seen


def test_words_come_back_as_word_timings(tmp_path, intercept):
    audio = tmp_path / "a.wav"
    tone_wav(audio, 2.0)
    words = ow.OpenRouterWhisperTranscriber(api_key="k").transcribe(audio, language="en")
    assert [w.text for w in words] == ["hello", "there", "world"]
    assert words[0].start == pytest.approx(0.2) and words[2].end == pytest.approx(1.4)


def test_upload_is_base64_mp3_with_word_timestamps_requested(tmp_path, intercept):
    audio = tmp_path / "a.wav"
    tone_wav(audio, 1.0)
    ow.OpenRouterWhisperTranscriber(api_key="secret").transcribe(audio, language="hi")
    body = intercept["body"]
    assert body["model"] == "openai/whisper-large-v3"
    assert body["input_audio"]["format"] == "mp3"
    assert base64.b64decode(body["input_audio"]["data"])[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
    assert "word" in body["timestamp_granularities"]
    assert body["language"] == "hi"
    assert intercept["headers"]["Authorization"] == "Bearer secret"


def test_no_word_timestamps_is_an_error(tmp_path, intercept):
    intercept["reply"] = {"text": "hello", "segments": [{"start": 0, "end": 1, "text": "hello"}]}
    audio = tmp_path / "a.wav"
    tone_wav(audio, 1.0)
    with pytest.raises(RuntimeError, match="word timestamps"):
        ow.OpenRouterWhisperTranscriber(api_key="k").transcribe(audio)


def test_missing_key_is_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    audio = tmp_path / "a.wav"
    tone_wav(audio, 1.0)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        ow.OpenRouterWhisperTranscriber().transcribe(audio)


def test_long_audio_is_chunked_at_quiet_points_and_offsets_restored(tmp_path, intercept, monkeypatch):
    # 50 "seconds" with a 5-second chunk: the split must land in the silence
    # planted at 24-26 s, and the second chunk's words must be shifted back.
    monkeypatch.setattr(ow, "CHUNK_SECONDS", 25.0)
    monkeypatch.setattr(ow, "SPLIT_SEARCH", 3.0)
    rate = 16000
    t = np.arange(50 * rate) / rate
    samples = 0.3 * np.sin(2 * np.pi * 220 * t)
    samples[24 * rate:26 * rate] = 0.0
    audio = tmp_path / "long.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(rate)
        handle.writeframes((samples * 32767).astype("<i2").tobytes())

    spans = ow._chunks(audio, 50.0)
    assert len(spans) == 2 and 24.0 <= spans[0][1] <= 26.0 and spans[1] == (spans[0][1], 50.0)

    words = ow.OpenRouterWhisperTranscriber(api_key="k").transcribe(audio)
    # Both chunks return the same canned words; the second set is offset.
    assert len(words) == 6
    assert words[3].start == pytest.approx(spans[0][1] + 0.2, abs=1e-6)


def test_registry_maps_backend_and_ignores_mlx_model_name():
    transcriber = get_transcriber("openrouter", "mlx-community/whisper-large-v3-turbo")
    assert isinstance(transcriber, ow.OpenRouterWhisperTranscriber)
    assert transcriber.model == "openai/whisper-large-v3"
    assert get_transcriber("openrouter", "openai/whisper-large-v3").model == "openai/whisper-large-v3"
