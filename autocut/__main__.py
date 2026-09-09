"""Command line entry point.

The web app is the intended interface, but the CLI exposes each stage
separately so a bad result can be traced to the stage that caused it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import DEFAULT


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _cmd_probe(args: argparse.Namespace) -> int:
    from .ingest import probe

    timeline = probe(Path(args.input))
    print(json.dumps(timeline.model_dump(mode="json", exclude={"words"}), indent=2))
    return 0


def _cmd_transcribe(args: argparse.Namespace) -> int:
    from .asr import get_transcriber, refine_timings, save_words
    from .audio import Envelope
    from .ingest import ingest

    work_dir = Path(args.work_dir)
    timeline = ingest(Path(args.input), work_dir)

    transcriber = get_transcriber(DEFAULT.asr_backend, DEFAULT.asr_model)
    words = transcriber.transcribe(timeline.audio, language=DEFAULT.language)
    if not args.raw:
        words = refine_timings(words, Envelope.from_wav(timeline.audio))

    destination = work_dir / "transcript.json"
    save_words(words, destination)
    print(f"{len(words)} words -> {destination}")
    for word in words[: args.preview]:
        print(f"  {word.start:7.2f} {word.end:7.2f}  {word.text}")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    from .pipeline import run

    preset = DEFAULT
    if args.preset == "gentle":
        preset = preset.gentle()
    elif args.preset == "aggressive":
        preset = preset.aggressive()

    source = Path(args.input)
    work_dir = Path(args.work_dir or f"work/{source.stem}")
    result = run(source, work_dir, preset, output_name=Path(args.output).name)

    if args.output:
        destination = Path(args.output)
        if destination.resolve() != result.output.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(result.output.read_bytes())
            result.output = destination

    print(
        f"\n{result.output}\n"
        f"  {result.timeline.duration:.1f}s -> {result.timeline.kept_duration:.1f}s "
        f"({result.removed_fraction:.0%} removed) in {result.elapsed:.0f}s\n"
        f"  {len(result.timeline.keep_segments)} segments, "
        f"{'face tracked' if result.tracked else 'centre crop'}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autocut", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser("probe", help="show what we know about a video")
    probe_parser.add_argument("input")
    probe_parser.set_defaults(func=_cmd_probe)

    transcribe_parser = subparsers.add_parser("transcribe", help="dump word-level timings")
    transcribe_parser.add_argument("input")
    transcribe_parser.add_argument("--work-dir", default="work/cli")
    transcribe_parser.add_argument(
        "--raw", action="store_true", help="skip waveform-based timing refinement"
    )
    transcribe_parser.add_argument("--preview", type=int, default=20)
    transcribe_parser.set_defaults(func=_cmd_transcribe)

    render_parser = subparsers.add_parser("render", help="raw video in, finished video out")
    render_parser.add_argument("input")
    render_parser.add_argument("-o", "--output", default="finished.mp4")
    render_parser.add_argument("--work-dir", default=None)
    render_parser.add_argument(
        "--preset", choices=("default", "gentle", "aggressive"), default="default"
    )
    render_parser.set_defaults(func=_cmd_render)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return args.func(args)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
