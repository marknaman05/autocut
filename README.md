# autocut

Drop in a raw portrait talking-head video, get back a finished 1080x1920 MP4:
dead air, filler words and botched retakes removed, subtle zoom punch-ins over
the cuts, face-tracked framing, and karaoke captions that highlight one word at
a time.

Everything runs locally by default — transcription on MLX, retake detection on
a local LLM via Ollama, face tracking through OpenCV. The retake detector can
optionally call a hosted model through OpenRouter instead (see below); nothing
else leaves the machine.

## Requirements

- macOS on Apple Silicon (Metal-accelerated transcription)
- `ffmpeg` and `ffprobe` on `PATH` — any build; captions are drawn in Python, so
  libass is not needed
- [`uv`](https://docs.astral.sh/uv/)
- Optional, for smarter retake detection — pick one:
  - [Ollama](https://ollama.com) with a model pulled (`ollama pull qwen3:8b`), or
  - an [OpenRouter](https://openrouter.ai) API key, set as `OPENROUTER_API_KEY`
    and selected with `AUTOCUT_RETAKE_BACKEND=openrouter`.
  Without either, a deterministic detector still catches repeated-phrase retakes.

## Setup

```sh
uv sync --all-extras --group dev
```

The Whisper model (~1.5 GB) and the face detection model (~230 KB) are fetched
on first use and cached.

## Usage

Web app — upload, choose which cuts to make, render and download:

```sh
uv run uvicorn server.app:app --port 8000
```

Uploading only gets as far as splitting the video into parts. The list is the
final video: each ticked row is a piece that gets stitched, in order, with the
words spoken in it and a button to hear it. Parts the detectors wanted gone are
listed too, unticked and labelled with why — so a wrongly-removed retake is one
click from coming back. "Play the edit" runs the whole thing end to end before
you spend a render on it. The detectors are good enough to propose and not good
enough to decide.

CLI — renders straight through, applying every proposed cut:

```sh
uv run autocut render input.mp4 -o finished.mp4
uv run autocut render input.mp4 --preset gentle      # pauses only
uv run autocut render input.mp4 --preset aggressive  # tight social cut
```

Individual stages, for debugging:

```sh
uv run autocut probe input.mp4
uv run autocut transcribe input.mp4 --work-dir work/scratch
```

### Retake detection backend

Default is a local model via Ollama. To use a hosted model instead, copy
`.env.example` to `.env` and set:

```sh
AUTOCUT_RETAKE_BACKEND=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
# optional — the default is google/gemini-3.8-flash
AUTOCUT_RETAKE_MODEL=anthropic/claude-opus-5
```

`.env` is loaded at startup by both the CLI and the web app, and is gitignored.
Real environment variables override it, so `export OPENROUTER_API_KEY=...` works
too.

`google/gemini-3.8-flash` is the default: fast, a fraction of a cent per video,
dependable JSON. `deepseek/deepseek-v4.1-flash` is cheaper still;
`anthropic/claude-opus-5` has the best judgement on ambiguous retakes. Any
OpenRouter model with structured-output support works.

## How it works

```
input.mp4
  ├─ ingest      probe metadata (incl. rotation), extract 16 kHz mono WAV
  ├─ transcribe  MLX Whisper, word timings repaired against a voice detector
  ├─ analyze     silence + filler + retake detectors -> proposed cuts
  ·  · · · the web app splits this into parts and waits for you · · ·
  ├─ cut         a single ffmpeg select/aselect pass
  ├─ track       face detection on the cut video -> smoothed crop keyframes
  ├─ caption     Pillow renders one image per word state
  └─ compose     reframe + captions + loudness + H.264, in one pass
```

Everything hangs off `TimeMap` (`autocut/models.py`), which converts between
source and post-cut output time. Captions drift if it is wrong, so it is the
most heavily tested thing in the project.

Thresholds live in one place, `autocut/config.py`.

### Safety rails

The detectors are eager on purpose, and the merge stage is where that is made
safe:

- boundaries snap to quiet samples, so joins do not click;
- no more than 60% of a video can be removed — over budget, whole detectors are
  dropped in order of least trustworthiness;
- LLM retake proposals are only accepted if the transcript actually repeats the
  span later — a model, local or hosted, will otherwise invent retakes. The
  backend only changes *what proposes* a retake; every proposal runs the same
  gauntlet, and a missing key or an unreachable model degrades to the
  deterministic n-gram detector rather than failing the render.

Two of those rails are about ASR timings rather than detectors, and matter more
than they sound. Whisper places a word's text reliably and its boundaries
poorly: measured across three recordings, onsets ran 0.39–0.90s early and
over-long words ran on through pauses (a seven-second "that"). A word whose
timestamp covers a pause hides that pause from a gap-based detector entirely.

Both are repaired against [Silero VAD](https://github.com/snakers4/silero-vad) —
a 2 MB model at ~190× realtime — which asks "is this a voice?" rather than "is
this loud?". The latter has no stable answer: a noisy room and a room with
digitally silent passages need opposite thresholds, and a cough is louder than
most speech. The detector is optional; without it the timings are repaired
against the waveform instead, and the render never fails for want of it.

And a gap in the *transcript* is not a gap in the *audio* — Whisper drops words
— so only the genuinely silent stretches inside a gap are cut, never the gap as
a whole.

None of these apply to cuts a person approved in the web app: a deliberate
choice is taken literally.

## Development

```sh
uv run pytest              # fast: seeds a committed transcript
uv run pytest -m slow      # also runs the real recogniser
```

`samples/talking_head.mp4` is generated by `samples/make_sample.py` with a long
pause, two filler words and a restarted sentence planted in it;
`tests/test_pipeline_smoke.py` asserts each one is removed.

Every run keeps its intermediates in `work/<job id>/`, which is usually the
fastest way to see which stage caused a problem.
