# autocut

Drop in a raw portrait talking-head video, get back a finished 1080x1920 MP4:
dead air, filler words and botched retakes removed, subtle zoom punch-ins over
the cuts, face-tracked framing, and karaoke captions that highlight one word at
a time.

Everything runs locally by default — transcription on MLX, retake detection on
a local LLM via Ollama, face tracking through OpenCV. The retake detector can
optionally call a hosted model through OpenRouter, and a finished video can be
posted to Instagram (see below); nothing else leaves the machine.

## Requirements

- macOS on Apple Silicon (Metal-accelerated transcription)
- `ffmpeg` and `ffprobe` on `PATH` — any build; captions are drawn in Python, so
  libass is not needed. The caption fonts (Anton, Bebas Neue, Montserrat; all
  SIL OFL) ship in the package, so nothing needs installing for them
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

Next to Render is the caption style: classic, bold caps, pill, minimal, boxed
or neon. Each card is drawn by the same code that captions the video, so what
you pick is what you get. The styles are plain data in `CAPTION_STYLES`
(`autocut/config.py`) -- copy one and change a colour to add your own.

CLI — renders straight through, applying every proposed cut:

```sh
uv run autocut render input.mp4 -o finished.mp4
uv run autocut render input.mp4 --preset gentle      # pauses only
uv run autocut render input.mp4 --preset aggressive  # tight social cut
uv run autocut render input.mp4 --caption-style hormozi
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
`anthropic/claude-sonnet-5` has the best judgement on ambiguous retakes. Any
OpenRouter model with structured-output support works.

Raw model ability is rarely what limits this stage. On a recording where the
default missed a retake outright, so did a frontier model — and both found it
exactly once the prompt carried the *pause* before each word, which is what
marks a restart and which a bare transcript does not contain. Reach for a
bigger model only after checking the transcript actually shows the retake.

## Publishing to Instagram

Once a video is finished, the card offers a caption box and a **Publish as a
Reel** button; the card shows the link when the post is up. The same thing from
the shell:

```sh
uv run autocut publish finished.mp4 --caption "..."
uv run autocut instagram status     # which account, and through what
```

There are two ways to connect, and pasting either key into `.env` is enough --
autocut uses whichever it finds (`AUTOCUT_PUBLISH_BACKEND=upload-post|instagram`
forces one when both are set).

### Through Upload-Post (simplest)

[Upload-Post](https://upload-post.com) is a hosted relay: connect your Instagram
account to it once in its dashboard, and it holds the platform credentials.
No Meta developer app, and the same key can post to TikTok and YouTube later.
The trade-off is that the finished video passes through their servers, and it
is a paid service beyond the free tier.

1. Sign up at [app.upload-post.com](https://app.upload-post.com), open
   **Manage users**, create a profile and connect Instagram to it.
2. Copy the API key from the dashboard into `.env`:

   ```sh
   UPLOAD_POST_API_KEY=eyJ...
   # only if more than one profile has Instagram connected:
   # UPLOAD_POST_USER=default
   ```

3. `uv run autocut instagram status` -> `connected as @you via Upload-Post`.

### Directly through Meta

Nothing leaves your machine except the upload to Instagram itself. Needs an
Instagram **professional** account (Business or Creator -- Settings -> Account
type in the app; free and reversible) and a Meta app to act on it. About ten
minutes, once:

1. [developers.facebook.com/apps](https://developers.facebook.com/apps) ->
   **Create app** -> the *Instagram* use case ("Instagram API with Instagram
   Login").
2. In the dashboard, **Instagram -> API setup with Instagram login**. Under
   *Generate access tokens*, **Add account**, log in as the account to post to,
   and **Generate token**, ticking `instagram_business_basic` and
   `instagram_business_content_publish`.
3. Into `.env`:

   ```sh
   INSTAGRAM_ACCESS_TOKEN=IGAA...
   ```

4. `uv run autocut instagram status` -> `connected as @you via Meta`.

Accounts added this way are *testers* of your own app, so it never needs Meta's
review. Instagram normally fetches a video from a public URL, which a tool on
localhost cannot offer; autocut uses the API's resumable upload instead and
sends the bytes directly, so there is no tunnel or bucket in the path. The
token lasts 60 days; `uv run autocut instagram refresh` extends it and prints
the new value for `.env`. A token from **Facebook Login** works too, with
`INSTAGRAM_GRAPH_HOST=https://graph.facebook.com` and `INSTAGRAM_USER_ID` set.

Either way, a Reel is limited to 15 minutes and 1 GB, and an expired or
revoked key shows up on the finished card as "Not connected" with the
service's own reason.

## How it works

```
input.mp4
  ├─ ingest      probe metadata (incl. rotation), extract 16 kHz mono WAV
  ├─ transcribe  MLX Whisper, word timings repaired against a voice detector
  ├─ analyze     silence + filler + retake detectors -> proposed cuts
  ·  · · · the web app splits this into parts and waits for you · · ·
  ├─ cut         a single ffmpeg select/aselect pass
  ├─ track       face detection on the cut video -> smoothed crop keyframes
  ├─ caption     Pillow renders one image per word state, in the chosen style
  ├─ compose     reframe + captions + loudness + H.264, in one pass
  └─ publish     (optional) upload to Instagram as a Reel
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
- LLM retake proposals are only accepted if the good take is confirmed — either
  the transcript repeats the span's content later, or the model points at the
  specific later span that replaces it and those words check out. A model, local
  or hosted, will otherwise invent retakes. The second path exists because a
  restart is frequently a *replacement* rather than a rewording: "The AutoCut
  cuts the video into shorter parts." superseded by forty words sharing three of
  them is plainly the same point and passes no overlap threshold that would
  still reject a fabrication. Spans accepted that way carry lower confidence, so
  they are the first thing sacrificed if the removal budget is blown. The
  backend only changes *what proposes* a retake; every proposal runs the same
  gauntlet, and a missing key or an unreachable model degrades to the
  deterministic n-gram detector rather than failing the render.

- A sentence keeps its beat. The silence detector will not cut into the first
  half-second after a full stop (`sentence_pause`), and whatever makes a join —
  a retake cut, a part someone unticked — the final segments are checked again
  and room tone borrowed from either side of the cut until the sentence has
  its beat, provided that audio is genuinely free of speech. Measured on one
  recording, the gaps after "Cloude." and "AutoCut." arrived in the finished
  video at 0.12s and 0.00s; they now arrive at 0.82s and 0.50s. A pause the
  source does not have cannot be borrowed, and a mid-sentence join stays tight.

- Part boundaries are relaxed off speech before anything is spliced. A pause
  split is placed at a word timestamp, and a word is audible either side of the
  one it is given — it begins before its start and rings on past its end — so
  cutting exactly there clips the tail of the sentence before the join or the
  onset of the one after it. On one recording, part ends sat at −29 dB against a
  −53 dB silence threshold, which is the last word of a sentence cut in half.
  Each boundary walks outward to the nearest quiet instant, so relaxing only
  ever adds audio to a part; the pause between two parts keeps a sliver of
  itself, so no part a person reviewed can vanish before it is rendered.

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

The commonest dropped words are a whole repeated sentence. Whisper decodes
thirty seconds at a time and will not say the same thing twice within one
window, so a restarted line comes back once, with the last word's timestamp
stretched over the repeat. Once the timings are repaired that leaves a hole:
seconds of speech with no words. Any such hole is cut out and transcribed on
its own, where the recogniser — with nothing before it to repeat — reads the
second copy fine; the words go back in at their real times, the retake
detector sees the repeat, and whichever copy you keep has captions. A hole
whose second reading is a lone doubtful word is left as it was and shown as
"check this" rather than captioned with a guess.

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
