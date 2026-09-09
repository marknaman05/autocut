"""Generate the synthetic test clip used by the integration tests.

Recording a real take would make the tests unreproducible, so we synthesise one
with macOS ``say`` and plant exactly the defects the pipeline is supposed to
find: a long pause, two filler words, and a restarted sentence.  The tests
assert each planted defect ends up inside a removed span.

Run with:  uv run python samples/make_sample.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
VOICE = "Samantha"
RATE = 180


@dataclass
class Line:
    text: str
    #: Silence appended after this line, in seconds.
    pause: float = 0.25
    #: What this line plants, if anything: "silence", "filler" or "retake".
    defect: str | None = None


SCRIPT = [
    Line("Hello, and welcome back to the channel.", pause=1.6, defect="silence"),
    Line("Today I want to talk about, um, three ways to speed up your workflow.",
         defect="filler"),
    Line("The first thing you should.", pause=0.7, defect="retake"),
    Line("The first thing you should do is write everything down."),
    Line("The second one is to batch similar tasks together.", pause=1.4, defect="silence"),
    Line("And the third, uh, is to leave yourself a note before you stop.",
         defect="filler"),
    Line("That is everything for today. Thanks for watching."),
]


def run(args: list[str]) -> None:
    subprocess.run(args, check=True, capture_output=True)


def main() -> int:
    if sys.platform != "darwin":
        print("this generator needs macOS `say`", file=sys.stderr)
        return 1

    build = HERE / "build"
    build.mkdir(exist_ok=True)
    parts: list[Path] = []
    marks: list[dict] = []
    cursor = 0.0

    for index, line in enumerate(SCRIPT):
        speech = build / f"{index:02d}.aiff"
        run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(speech), line.text])
        duration = float(
            subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(speech)],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
        )
        parts.append(speech)
        if line.defect:
            # Where the planted defect lives, in source time.
            marks.append({
                "kind": line.defect,
                "text": line.text,
                "start": round(cursor, 3),
                "end": round(cursor + duration + line.pause, 3),
            })
        cursor += duration + line.pause

        silence = build / f"{index:02d}-gap.aiff"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"anullsrc=r=22050:cl=mono:d={line.pause}",
             str(silence)])
        parts.append(silence)

    concat_file = build / "parts.txt"
    concat_file.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    audio = build / "speech.wav"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "concat", "-safe", "0", "-i", str(concat_file),
         "-ac", "1", "-ar", "44100", str(audio)])

    output = HERE / "talking_head.mp4"
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        # A flat landscape backdrop: no face, which also exercises the
        # reframe stage's centre-crop fallback.
        "-f", "lavfi", "-i", f"color=c=0x20242c:s=1280x720:r=30:d={cursor:.2f}",
        "-i", str(audio),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "34", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-shortest",
        str(output),
    ])

    (HERE / "talking_head.marks.json").write_text(
        json.dumps({"duration": round(cursor, 3), "defects": marks}, indent=2)
    )
    print(f"wrote {output} ({output.stat().st_size / 1024:.0f} KB, {cursor:.1f}s)")
    print(f"planted {len(marks)} defects -> talking_head.marks.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
