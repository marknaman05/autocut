"""Retake and false-start removal -- the one stage that uses a language model.

A local model is the least reliable component in the pipeline, so nothing it
says is trusted directly.  Every proposal is checked against the transcript and
the timings, and any failure at all discards the model's output and falls back
to a deterministic detector.  The pipeline must never fail, or cut wildly,
because a local model had a bad day; the acceptable failure mode is cutting
less than it could have.

The deterministic fallback exploits the structure of a real retake: when you
fluff a line you restart it, so the same run of words appears twice within a
few seconds.  That repeated run is only the anchor, though -- what confirms a
retake is ``_is_superseded``, which asks whether the good take says the same
thing again, and accepts a paraphrase rather than demanding a verbatim repeat.
Both paths go through it: it is the guard that makes a weak local model safe to
use, and it is also what separates a genuine restart from a deliberate refrain,
a job the fallback used to do by capping how far apart the two attempts could
be -- which cut real retakes, because a fluffed sentence runs as long as it
runs.

There is a limit to what measuring repeated words can confirm, though.  A
restart is often not a rewording of the abandoned line but a *replacement* for
it -- "The AutoCut cuts the video into shorter parts." followed by forty words
that make the same point and share three of them.  No overlap threshold
separates that from an invented retake.  So the model is also asked to point
at the good take rather than merely assert one exists, and a proposal survives
if either the transcript repeats it or the span it names checks out; see
``_replacement_holds``.  Such spans carry lower confidence, and like every
other proposal they are offered to a person rather than applied.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from collections import Counter

from ..asr.base import normalize
from ..config import RetakeConfig
from ..models import Reason, RemovalSpan, Word

log = logging.getLogger(__name__)

#: Words per request.  Small enough that a local model keeps track, with an
#: overlap so a retake straddling a boundary is still seen whole.
_CHUNK_WORDS = 220
_CHUNK_OVERLAP = 30

_SYSTEM_PROMPT = """\
You edit raw talking-head video transcripts. The speaker sometimes fluffs a \
line, abandons it, and says it again. Your job is to find the abandoned \
attempts so they can be cut, leaving the clean take.

You are given the transcript as numbered words. A marker like [pause 1.2s] \
after a word is the silence before the next word. A pause noticeably longer \
than the others is the single strongest sign that what follows is a restart.

Return JSON only:
{"removals": [{"start": <first word index to cut>, "end": <last word index to \
cut>, "replaced_by_start": <first word index of the good take>, \
"replaced_by_end": <last word index of the good take>, "why": "<a few words>"}]}

Rules:
- Only mark a span when a LATER part of the transcript makes the same point \
properly, and name that later span in replaced_by_start/replaced_by_end. The \
good take must remain.
- The good take is usually NOT a verbatim repeat. It is normally reworded, and \
often several times longer than the attempt it replaces. Judge by meaning, not \
by matching words: "The tool cuts the video into shorter parts." is replaced by \
"So this application edits our video, it cuts them into smaller segments and \
removes the gaps...".
- Mark false starts, abandoned sentences, and stumbles.
- Do NOT mark filler words, pauses, or anything merely wordy. Something else \
handles those.
- Do NOT rewrite or reorder anything. Only choose spans to delete.
- If nothing was restarted, return {"removals": []}.\
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "removals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                    "replaced_by_start": {"type": "integer"},
                    "replaced_by_end": {"type": "integer"},
                    "why": {"type": "string"},
                },
                # Every property is required and none may be added: OpenRouter
                # passes ``strict: true`` to providers that enforce it, and a
                # schema they consider incomplete is rejected outright rather
                # than loosened.
                "required": [
                    "start", "end", "replaced_by_start", "replaced_by_end", "why",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["removals"],
    "additionalProperties": False,
}


def _numbered(words: list[Word], pause_marker: float = 0.3) -> str:
    """The chunk as the model sees it: numbered words, annotated with pauses.

    Without the pauses the model is judging a wall of text, and the thing that
    most reliably marks a restart is inaudible to it.  In one recording the
    abandoned line ended on the longest pause in the clip -- 1.14s against a
    0.26s runner-up -- and no model found it until that number was in the
    prompt; with it, every model tried found the span exactly.
    """
    lines = []
    for index, word in enumerate(words):
        line = f"{index}: {word.text}"
        if index + 1 < len(words):
            gap = words[index + 1].start - word.end
            if gap >= pause_marker:
                line += f"  [pause {gap:.1f}s]"
        lines.append(line)
    return "\n".join(lines)


class RetakeValidationError(ValueError):
    """The model's plan failed a sanity check and was discarded."""


#: Stripped out before measuring overlap in ``_is_superseded``, so a match
#: cannot be manufactured out of "the", "is", "a" and "to" alone -- content
#: words are what make two spans the same statement, not function words.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "was", "are", "were", "be", "been", "am",
    "to", "of", "in", "on", "at", "for", "with", "as", "and", "but", "so",
    "i", "you", "we", "it", "that", "this", "do", "does", "did", "um", "uh",
})


def _content_words(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token and token not in _STOPWORDS]


def _installed_models(config: RetakeConfig) -> list[str]:
    request = urllib.request.Request(f"{config.ollama_host.rstrip('/')}/api/tags")
    with urllib.request.urlopen(request, timeout=5) as response:
        body = json.loads(response.read())
    return [model.get("name", "") for model in body.get("models", [])]


def choose_model(config: RetakeConfig) -> str:
    """The model to use: the configured one, or the best one installed.

    Naming a model that has not been pulled would fail on every run and quietly
    demote us to the n-gram detector, so when nothing is configured we ask
    Ollama what it actually has.
    """
    if config.model:
        return config.model

    installed = _installed_models(config)
    if not installed:
        raise RetakeValidationError("Ollama has no models installed")

    # Exact match first, then a looser one so "qwen2.5:7b-instruct" counts.
    for preferred in config.preferred_models:
        if preferred in installed:
            return preferred
        family = preferred.split(":")[0]
        for name in installed:
            if name.split(":")[0] == family:
                return name
    return installed[0]


def _ask_ollama(prompt: str, model: str, config: RetakeConfig) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": _SCHEMA,
            "options": {"temperature": 0.0},
        }
    ).encode()

    request = urllib.request.Request(
        f"{config.ollama_host.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=config.timeout) as response:
        body = json.loads(response.read())
    content = body.get("message", {}).get("content", "")
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise RetakeValidationError(f"model did not return JSON: {content[:200]!r}") from error


def _ask_openrouter(prompt: str, config: RetakeConfig) -> dict:
    """Ask a hosted model, via OpenRouter's OpenAI-compatible chat endpoint.

    The wire format differs from Ollama's -- ``response_format`` rather than
    ``format``, the reply nested one level deeper -- but nothing else does: the
    same system prompt, the same numbered-word input, the same JSON schema
    back.  Every proposal it returns still goes through ``_validate_chunk``.
    """
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RetakeValidationError(
            f"{config.api_key_env} is not set; cannot reach OpenRouter"
        )

    payload = json.dumps(
        {
            "model": config.openrouter_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "retakes", "strict": True, "schema": _SCHEMA},
            },
        }
    ).encode()

    request = urllib.request.Request(
        config.openrouter_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # OpenRouter attributes calls to an app by these; harmless, and it
            # keeps autocut's usage legible in the dashboard.
            "X-Title": "autocut",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:200]
        raise RetakeValidationError(f"OpenRouter returned {error.code}: {detail!r}") from error

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RetakeValidationError(f"unexpected OpenRouter response: {str(body)[:200]!r}") from error
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError) as error:
        raise RetakeValidationError(f"model did not return JSON: {str(content)[:200]!r}") from error


def _is_superseded(words: list[Word], start: int, end: int, config: RetakeConfig) -> bool:
    """Whether the content of ``[start:end]`` is said again afterwards.

    This is the check that makes a weak local model safe to use.  Asked to find
    retakes in a clean transcript, a small model will happily invent one -- in
    testing, qwen2.5:7b proposed cutting "and welcome back to the channel" as a
    "repeated introduction" when nothing repeated it.

    A real retake is rarely repeated verbatim, though -- "The first thing you
    should do is--" becomes "So the first thing is...".  Requiring an exact run
    of words rejects exactly the paraphrased retakes a model is best at
    catching, so instead this slides a same-ish-sized window across what
    follows and measures how much of the span's *content* -- function words
    like "the"/"is"/"a" stripped out, so they cannot inflate the score --
    reappears in it.  Content the speaker only said once can never produce a
    high overlap, whatever the model's proposal claims.
    """
    span_tokens = _content_words([normalize(word.text) for word in words[start : end + 1]])
    if len(span_tokens) < config.min_overlap_words:
        # Too short or too generic to judge reliably: reject rather than guess.
        return False
    span_counts = Counter(span_tokens)

    deadline = words[end].end + config.ngram_window
    following = [
        normalize(word.text) for word in words[end + 1 :] if word.start <= deadline
    ]

    span_length = end - start + 1
    # The good take is routinely longer than the attempt it replaces -- it is
    # the one that got finished -- so the window searched has to be able to
    # grow well past the span's own length, or a reworded restart is measured
    # against only its first few words and scores nothing.
    max_window = min(span_length * 2 + 6, len(following))
    for window_size in range(max(1, span_length - 2), max_window + 1):
        for i in range(len(following) - window_size + 1):
            window_tokens = _content_words(following[i : i + window_size])
            if not window_tokens:
                continue
            overlap = sum((span_counts & Counter(window_tokens)).values())
            if overlap / len(span_tokens) >= config.overlap_threshold:
                return True
    return False


def _replacement_holds(
    words: list[Word], start: int, end: int, item: dict, config: RetakeConfig
) -> bool:
    """Whether the model's named good take is a real, later span of transcript.

    ``_is_superseded`` asks the only question a transcript can answer on its
    own -- do the same words come back? -- and a genuinely reworded retake can
    answer no.  In one recording "The AutoCut cuts the video into shorter
    parts." was replaced by forty words that share three content words with
    it: plainly the same point, and lexically far below any threshold that
    would still reject an invented retake.

    So the model is asked to point at the good take rather than merely assert
    one exists, and what is checked here is the pointing.  A model inventing a
    retake in a clean transcript has nothing to point at and has to fabricate
    indices, which fail these checks; a model that has actually found one names
    the span that replaces it.  The checks are structural on purpose -- the
    claim is *that these later words exist and say it again*, and only the
    first half of that is verifiable here.  The second half is why such spans
    are proposed to a person rather than applied, and carry lower confidence
    than a lexically confirmed one.
    """
    try:
        first = int(item["replaced_by_start"])
        last = int(item["replaced_by_end"])
    except (KeyError, TypeError, ValueError):
        return False

    if not (0 <= first <= last < len(words)):
        return False
    # It has to come after the attempt it replaces, and be a separate span:
    # a "good take" overlapping the cut would be deleted along with it.
    if first <= end:
        return False
    if last - first + 1 < config.min_overlap_words:
        return False
    # And it has to follow soon enough to be a restart rather than the speaker
    # returning to the subject minutes later, which is not a retake.
    if words[first].start - words[end].end > config.supersede_window:
        return False
    return True


#: Confidence for a span the transcript itself repeats, and for one only the
#: model says is replaced.  Both are proposals a person ticks or unticks; the
#: difference is which gets sacrificed first when the removal budget is blown,
#: and how loudly the review screen should hedge.
_CONFIDENCE_REPEATED = 0.7
_CONFIDENCE_CLAIMED = 0.45


def _validate_chunk(
    proposals: list[dict], words: list[Word], offset: int, count: int, config: RetakeConfig
) -> list[tuple[int, int, str, float]]:
    """Check one chunk's proposals, returning absolute word index ranges."""
    accepted: list[tuple[int, int, str, float]] = []
    for item in proposals:
        try:
            start = int(item["start"])
            end = int(item["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise RetakeValidationError(f"malformed span {item!r}") from error

        if not (0 <= start <= end < count):
            raise RetakeValidationError(f"span {start}-{end} outside the chunk of {count} words")

        absolute_start, absolute_end = offset + start, offset + end
        # A genuine restart has a beat before the good take begins.  Without
        # one, this is far more likely to be the model deleting content it
        # merely disliked.
        if absolute_end + 1 < len(words):
            pause = words[absolute_end + 1].start - words[absolute_end].end
            if pause < config.required_pause:
                log.warning(
                    "rejecting retake %r: only %.2fs before the restart",
                    item.get("why", ""), pause,
                )
                continue

        # Two ways for the good take to be confirmed.  The transcript repeating
        # the span's content is the strong one and needs nothing from the
        # model; the model pointing at a specific later span is the weak one,
        # and it exists because a reworded retake cannot pass the strong test.
        if _is_superseded(words, absolute_start, absolute_end, config):
            confidence = _CONFIDENCE_REPEATED
        elif _replacement_holds(words, absolute_start, absolute_end, item, config):
            confidence = _CONFIDENCE_CLAIMED
            log.info(
                "retake %r: no lexical repeat, but the model points at words %s-%s",
                item.get("why", ""), item.get("replaced_by_start"),
                item.get("replaced_by_end"),
            )
        else:
            log.warning(
                "rejecting retake %r: nothing later repeats or replaces it",
                item.get("why", ""),
            )
            continue

        accepted.append(
            (absolute_start, absolute_end, str(item.get("why", "")), confidence)
        )
    return accepted


def _ask(prompt: str, config: RetakeConfig, model: str | None) -> dict:
    """One request to whichever backend is configured."""
    if config.backend == "openrouter":
        return _ask_openrouter(prompt, config)
    if config.backend == "ollama":
        return _ask_ollama(prompt, model, config)
    raise RetakeValidationError(f"unknown retake backend {config.backend!r}")


def _llm_spans(words: list[Word], config: RetakeConfig) -> list[RemovalSpan]:
    # Ollama needs a model resolved up front (and may raise if none is
    # installed); OpenRouter carries its model in the request.
    model = choose_model(config) if config.backend == "ollama" else None
    spans: list[RemovalSpan] = []
    step = _CHUNK_WORDS - _CHUNK_OVERLAP

    for offset in range(0, len(words), step):
        chunk = words[offset : offset + _CHUNK_WORDS]
        if len(chunk) < 8:
            break
        result = _ask(_numbered(chunk), config, model)
        proposals = result.get("removals")
        if not isinstance(proposals, list):
            raise RetakeValidationError(f"'removals' was {type(proposals).__name__}, not a list")

        for start, end, why, confidence in _validate_chunk(
            proposals, words, offset, len(chunk), config
        ):
            spans.append(
                RemovalSpan(
                    start=words[start].start,
                    end=words[end].end,
                    reason=Reason.RETAKE,
                    confidence=confidence,
                    detail=why[:80],
                )
            )

    total_speech = sum(word.duration for word in words) or 1.0
    removed = sum(span.duration for span in spans)
    if removed / total_speech > config.max_removal_ratio:
        raise RetakeValidationError(
            f"plan removes {removed / total_speech:.0%} of speech, over the "
            f"{config.max_removal_ratio:.0%} ceiling"
        )
    return spans


def _ngram_repeats(words: list[Word], config: RetakeConfig) -> list[RemovalSpan]:
    """Deterministic fallback: an identical run of words repeated soon after
    means the first attempt was abandoned."""
    n = config.ngram_size
    if len(words) < n * 2:
        return []

    normalized = [normalize(word.text) for word in words]
    seen: dict[tuple[str, ...], list[int]] = {}
    for index in range(len(normalized) - n + 1):
        seen.setdefault(tuple(normalized[index : index + n]), []).append(index)

    spans: list[RemovalSpan] = []
    for gram, positions in seen.items():
        for earlier, later in zip(positions, positions[1:]):
            if later <= earlier:
                continue
            if words[later].start - words[earlier].start > config.ngram_window:
                continue
            pause = words[later].start - words[later - 1].end
            if pause < config.required_pause:
                continue
            # Only a restart, not a refrain.  A word-count cap was tried here
            # and cut real retakes: a fluffed sentence runs as long as it runs,
            # and 15 words is ordinary.  What actually separates the two cases
            # is whether the good take says the same thing again, which is
            # exactly what ``_is_superseded`` measures -- so ask it, rather
            # than guessing from how far apart the attempts are.
            if not _is_superseded(words, earlier, later - 1, config):
                continue
            spans.append(
                RemovalSpan(
                    start=words[earlier].start,
                    end=words[later - 1].end,
                    reason=Reason.NGRAM_REPEAT,
                    confidence=0.6,
                    detail=" ".join(gram),
                )
            )
    log.info("n-gram fallback proposed %d spans", len(spans))
    return spans


def detect(words: list[Word], config: RetakeConfig) -> list[RemovalSpan]:
    if not config.enabled or len(words) < 8:
        return []

    # The deterministic detector always runs.  It is precise but narrow -- it
    # only anchors on retakes that restart with the same run of words -- so the
    # model's job is to add the ones it misses, not to replace it.  If the model
    # is missing,
    # unreachable or talking nonsense, we still catch the obvious cases.
    spans = _ngram_repeats(words, config)

    try:
        spans += _llm_spans(words, config)
    except (RetakeValidationError, urllib.error.URLError, TimeoutError, OSError) as error:
        log.warning(
            "retake backend %r unusable (%s); using the n-gram detector alone",
            config.backend, error,
        )

    log.info("retake detector proposed %d spans", len(spans))
    return spans
