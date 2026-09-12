"""Second-pass transcription of stretches the first pass returned no words for.

Whisper decodes a 30-second window at a time and, within one, is reluctant
to emit the same phrase twice in a row.  A restarted sentence -- said once,
then said again -- comes back as a single copy with the last word's timestamp
stretched over the repeat.  The timing repair then shrinks that word back to
its real length, which leaves a hole: seconds of speech-level audio with no
words in them.

The words are still there to be had.  Cut the hole out and transcribe it on
its own and the recogniser, with nothing before it to repeat, reads the second
copy perfectly well.  This module finds such holes and does exactly that,
then splices the words back in at their real times -- after which the retake
detector can see the repeat and propose dropping the first attempt, and
whichever copy is kept has captions.

Only holes that carry sustained speech are re-read, judged by the same test
the silence detector uses to protect speech it cannot see words for; a quiet
gap is a pause, not a hole, and re-reading it would only invite the
recogniser to invent something.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

from ..audio import Envelope
from ..models import Word
from .base import Transcriber, clean_words

log = logging.getLogger(__name__)

#: Audio taken either side of a hole so the recogniser hears the onset of the
#: first word and the tail of the last, which sit just outside the gap the
#: word timings describe.
PAD = 0.15
#: What the second pass must produce for a hole to count as read.  On one
#: recording a five-second hole of room noise came back as a single "so" at
#: probability 0.32; the swallowed retake it exists for came back as twelve
#: words averaging 0.9.
MIN_WORDS = 2
MIN_MEAN_PROBABILITY = 0.5


def find_holes(
    words: list[Word],
    duration: float,
    envelope: Envelope,
    *,
    min_gap: float,
    sustained: float,
) -> list[tuple[float, float]]:
    """Stretches with no words that nonetheless hold sustained speech."""
    edges = [0.0]
    for word in words:
        edges.extend((word.start, word.end))
    edges.append(duration)

    holes = []
    level = envelope.speech_confidence_level
    # ``edges`` alternates gap, word, gap, word, ... gap: the even-indexed
    # pairs are the gaps -- before the first word, between words, after the
    # last.
    for start, end in zip(edges[0::2], edges[1::2]):
        if end - start < min_gap:
            continue
        if envelope.longest_run_above(level, start, end) >= sustained:
            holes.append((start, end))
    return holes


def slice_wav(source: Path, start: float, end: float, destination: Path) -> float:
    """Write ``[start, end)`` of a WAV to ``destination``; returns the real start.

    Returned so the caller can offset timings: the slice may begin earlier
    than asked when ``start`` is close to the beginning of the file.
    """
    with wave.open(str(source), "rb") as reader:
        rate = reader.getframerate()
        total = reader.getnframes()
        first = max(int(start * rate), 0)
        last = min(int(end * rate), total)
        reader.setpos(first)
        frames = reader.readframes(max(last - first, 0))
        params = reader.getparams()
    with wave.open(str(destination), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)
    return first / rate


def fill_holes(
    words: list[Word],
    audio: Path,
    duration: float,
    envelope: Envelope,
    transcriber: Transcriber,
    work_dir: Path,
    *,
    language: str | None,
    min_gap: float,
    sustained: float,
) -> list[Word]:
    """Re-read every hole in ``words`` and splice what is heard back in.

    Words the second pass places outside the hole are discarded: they can
    only be the recogniser reaching for the neighbours it was given as
    padding, which the first pass already has.
    """
    holes = find_holes(words, duration, envelope, min_gap=min_gap, sustained=sustained)
    if not holes:
        return words

    scratch = work_dir / "holes"
    scratch.mkdir(exist_ok=True)
    found: list[Word] = []
    for index, (start, end) in enumerate(holes):
        clip = scratch / f"{index:02d}.wav"
        offset = slice_wav(audio, max(start - PAD, 0.0), min(end + PAD, duration), clip)
        heard = transcriber.transcribe(clip, language=language)
        inside = [
            Word(
                text=word.text,
                start=word.start + offset,
                end=word.end + offset,
                probability=word.probability,
            )
            for word in heard
            # Keep a word if most of it lies in the hole; the padding is there
            # to be heard, not to be transcribed.
            if (min(word.end + offset, end) - max(word.start + offset, start))
            > 0.5 * max(word.end - word.start, 0.02)
        ]
        # A reading, not a guess: one word out of several seconds, or words
        # the recogniser itself doubts, is what it produces when asked to
        # transcribe a breath or a chair creak, and would only put a caption
        # on noise.
        confident = (
            len(inside) >= MIN_WORDS
            and sum(w.probability for w in inside) / len(inside) >= MIN_MEAN_PROBABILITY
        )
        log.info(
            "hole %.2f-%.2f: second pass heard %d words%s: %s",
            start, end, len(inside), "" if confident else " (discarded as a guess)",
            " ".join(w.text for w in inside) or "(nothing)",
        )
        if confident:
            found.extend(inside)

    if not found:
        return words
    return clean_words(sorted(words + found, key=lambda w: w.start), duration=duration)
