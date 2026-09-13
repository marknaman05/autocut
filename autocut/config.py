"""Every tunable threshold in the pipeline, in one place.

Detectors and renderers take a ``Preset`` rather than reading globals, so a job
can be re-run with different aggressiveness without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

try:
    # A .env file in the working directory is the intended place for
    # OPENROUTER_API_KEY and the AUTOCUT_* switches -- this is a local
    # single-user tool, so a dotfile beats exporting vars by hand.  Values
    # already in the real environment win over the file.
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:  # pragma: no cover - dotenv is a declared dep
    pass


@dataclass(frozen=True)
class SilenceConfig:
    #: Gaps between words shorter than this are left alone.
    min_gap: float = 0.35
    #: Breathing room left on each side of a removed gap.
    pad: float = 0.12
    #: Room left after a word that ends a sentence, in place of ``pad``.
    #: A full stop is not a breath: the pause after it is doing work, and
    #: trimming it to the same 0.12s as a mid-sentence hesitation is what makes
    #: an edit sound hurried.  Measured on one recording, the gaps after
    #: "Cloude." and "AutoCut." arrived in the finished video at 0.12s and
    #: 0.00s -- each sentence landing directly on the next.  This is a floor,
    #: not a target: a pause already shorter than this is left exactly as it
    #: was spoken, and a long one is still tightened, just not past here.
    sentence_pause: float = 0.5
    #: A noise shorter than this, in the middle of a pause, does not break the
    #: pause in two -- provided it is not sustained speech.  Wide enough to
    #: reach across a cough, which runs a good half-second including its decay.
    bridge: float = 1.0
    #: How long audio must hold at speech level to count as a spoken word
    #: rather than a noise.  Measured across three recordings, a cough and
    #: room tone held 0.00-0.19s while speech the recogniser dropped held
    #: 0.37-1.05s; this sits in the gap between those two populations.
    sustained_speech: float = 0.3
    #: A word longer than this is treated as suspect: no single word is spoken
    #: for a second, so the timestamp has almost certainly run on through a
    #: pause.  Such words are scanned internally for dead air.
    max_word_duration: float = 0.8


@dataclass(frozen=True)
class FillerConfig:
    enabled: bool = True
    #: Unambiguous fillers -- removed whenever they appear as a standalone token.
    lexicon: tuple[str, ...] = (
        "um", "uh", "uhm", "erm", "er", "ah", "mmm", "hmm",
    )
    #: Fillers that are also real words; only removed when a neighbouring pause
    #: marks them as verbal padding rather than content ("I like it" survives).
    ambiguous: tuple[str, ...] = (
        "like", "so", "right", "basically", "actually", "literally",
    )
    #: Multi-word filler phrases, matched on consecutive tokens.
    phrases: tuple[tuple[str, ...], ...] = (
        ("you", "know"),
        ("i", "mean"),
        ("sort", "of"),
        ("kind", "of"),
    )
    #: A token longer than this is probably emphasised content, not a filler.
    max_duration: float = 0.6
    #: Pause on either side required before an ambiguous word counts as filler.
    ambiguous_pause: float = 0.25


@dataclass(frozen=True)
class VADConfig:
    """Voice activity detection, used to repair the recogniser's word timings.

    Whisper places a word's text reliably and its boundaries poorly.  Correcting
    them against the energy envelope can only ask "is this loud?", which has no
    stable answer across recordings; a voice detector asks "is this a voice?".
    """

    enabled: bool = True
    #: Frame probability at or above which a frame is counted as speech.
    threshold: float = 0.5
    #: How far a word edge may travel to reach the nearest voice.  A word with
    #: no voice within this distance is left as the recogniser placed it.
    max_shift: float = 1.0


@dataclass(frozen=True)
class RetakeConfig:
    enabled: bool = True
    #: Which LLM proposes retakes.  ``"ollama"`` talks to a local model and
    #: needs nothing set up beyond Ollama itself; ``"openrouter"`` calls a
    #: hosted model over the OpenRouter API and wants ``OPENROUTER_API_KEY`` in
    #: the environment.  Either way the n-gram detector still runs underneath
    #: and every proposal goes through the same validation, so a missing key or
    #: a bad response degrades to the deterministic path rather than failing.
    backend: str = "ollama"

    # -- ollama backend -----------------------------------------------------
    #: Leave as None to use the best model Ollama actually has installed;
    #: naming one that is not pulled would silently fall back to the n-gram
    #: detector on every run.
    model: str | None = None
    #: Tried in order when ``model`` is None.  Ranked by how reliably they
    #: honour a JSON schema, which matters more here than raw ability.
    preferred_models: tuple[str, ...] = (
        "qwen3:8b", "qwen2.5:7b", "llama3.1:8b", "gemma3:4b", "llama3.2:3b",
    )
    ollama_host: str = "http://localhost:11434"

    # -- openrouter backend ----------------------------------------------
    #: Default is a fast, cheap model with dependable JSON adherence.  Raw
    #: ability turned out not to be the bottleneck: on a recording where this
    #: model missed a retake entirely, so did a frontier one, and both found
    #: it exactly once the prompt carried the pause lengths between words.
    #: Swap for ``deepseek/deepseek-v4.1-flash`` to go cheaper still, or
    #: ``anthropic/claude-sonnet-5`` for the most judgement on ambiguous
    #: transcripts; any OpenRouter model id with structured-output support
    #: works.
    openrouter_model: str = "google/gemini-3.8-flash"
    openrouter_url: str = "https://openrouter.ai/api/v1/chat/completions"
    #: Environment variable the API key is read from, so the key itself never
    #: lands in a preset or a checked-in config.
    api_key_env: str = "OPENROUTER_API_KEY"

    timeout: float = 120.0
    #: Reject the whole LLM plan if it wants to remove more than this fraction
    #: of speech -- a runaway local model should cut nothing, not everything.
    max_removal_ratio: float = 0.35
    #: A genuine restart has a beat before it; spans without one are rejected.
    required_pause: float = 0.2
    #: Deterministic fallback: an identical run of N+ words inside the window
    #: means the earlier attempt was abandoned.
    ngram_size: int = 4
    ngram_window: float = 15.0
    #: How much of a proposed span's content (function words stripped) must
    #: reappear afterwards for it to count as superseded.  1.0 would require a
    #: verbatim repeat; real retakes are usually paraphrased, so this is looser.
    #: Tuned against a small local model, which invents retakes freely and
    #: needs the strict setting to stay safe.
    paraphrase_threshold: float = 0.6
    #: The same, for a hosted model.  Looser, because it is a different
    #: trade: a weak local model's false positives are the thing to defend
    #: against, while a capable hosted one mostly loses real retakes to the
    #: strict threshold.  Nothing this detector proposes is applied without a
    #: person ticking it in the review screen, so the cost of the looser
    #: setting is a row to untick and the cost of the stricter one is a retake
    #: that is never offered at all.
    hosted_paraphrase_threshold: float = 0.45
    #: A span with fewer content words than this cannot be judged reliably and
    #: is rejected outright, rather than risking a false match on stopwords.
    min_overlap_words: int = 2
    #: How long after an abandoned line the good take may begin, when the model
    #: names it rather than the transcript repeating it.  Longer than
    #: ``ngram_window``: a restart with different words is often a longer,
    #: more careful second attempt, and the speaker may take a breath first.
    supersede_window: float = 45.0

    @property
    def overlap_threshold(self) -> float:
        """``paraphrase_threshold``, resolved for the backend in use."""
        return (
            self.hosted_paraphrase_threshold
            if self.backend == "openrouter"
            else self.paraphrase_threshold
        )


@dataclass(frozen=True)
class PunchInConfig:
    enabled: bool = True
    #: Zoom levels alternated across cuts.
    levels: tuple[float, ...] = (1.0, 1.12)
    #: Never change zoom more often than this, or the video strobes.
    min_hold: float = 2.5
    #: Segments shorter than this keep the previous zoom.
    min_segment: float = 1.2


@dataclass(frozen=True)
class CaptionConfig:
    """Captions are drawn with Pillow and composited by ffmpeg.

    Not libass: Homebrew's ffmpeg ships without it, and rendering the text
    ourselves keeps the pipeline working on whatever ffmpeg is installed.
    """

    enabled: bool = True
    #: Font file stem or family name; resolved by ``render.captions.find_font``,
    #: which looks in the bundled ``render/fonts`` directory before the system.
    font: str = "Arial Black"
    font_size: int = 82
    text_colour: str = "#FFFFFF"
    highlight_colour: str = "#FFD44A"
    outline_colour: str = "#000000"
    outline: int = 8
    shadow_offset: int = 4
    shadow_colour: str = "#00000099"
    #: Draw every word in capitals.  Applied at draw time only, so the
    #: transcript, the review screen and any caption correction keep the
    #: case that was spoken; only the picture changes.
    uppercase: bool = False
    #: Where the caption band sits: ``"bottom"``, ``"centre"`` or ``"top"``.
    #: ``margin_v`` is the distance from that edge, and ignored for centre.
    position: str = "bottom"
    #: Distance from the chosen edge of the frame, clear of platform UI.
    margin_v: int = 420
    #: Glyph colour of the active word when it sits on a box, where the
    #: highlight colour is usually the box and the text wants to stay white.
    #: ``None`` means ``highlight_colour``, which is what every style without
    #: a box wants.
    active_text_colour: str | None = None
    #: A rounded box drawn behind the active word -- the "pill" that a lot of
    #: short-form captioning does instead of recolouring the word.  ``None``
    #: draws nothing.
    active_box_colour: str | None = None
    #: A rounded box behind the whole line, for subtitle-block styles.  Takes
    #: ``#RRGGBBAA`` so the block can be translucent.  ``None`` draws nothing.
    line_box_colour: str | None = None
    #: How far a box extends past the glyphs on each side, and its corner
    #: radius.  One pair for both kinds of box so a pill inside a line block
    #: never disagrees with it.
    box_padding: int = 14
    box_radius: int = 18
    #: Blur radius of a soft copy of the outline drawn under the text, in the
    #: outline colour -- a neon halo.  0 disables it.  Each frame costs one
    #: blur of the band, a few milliseconds.
    glow: int = 0
    max_words_per_line: int = 4
    max_chars_per_line: int = 22
    #: A pause longer than this always breaks the caption line.
    line_break_gap: float = 0.4
    #: Global nudge applied to every caption, for ASR timing drift.
    offset: float = 0.0
    #: Below this recogniser confidence, a word is marked for checking in the
    #: review screen.  Whisper reports a probability per word and it is worth
    #: believing: on one recording the median word scored 0.99 and only two
    #: fell below this line -- "Cloud." for "Claude." and "charge" for "star" --
    #: which were exactly the two words in the clip that were wrong.  Set it
    #: higher to be shown more doubtful words, lower to be shown almost none;
    #: it changes nothing about what is rendered, only what is pointed at.
    review_confidence: float = 0.5
    #: A word must survive at least this much of the cut to be captioned.
    #: Snapping cut points can leave a few milliseconds of a removed word
    #: behind; without this, a cut "um" flashes up for a frame or two.
    min_visible_fraction: float = 0.5
    #: Scale of the active word at the peak of its pop, and how long the pop
    #: lasts.  Set ``pop_duration`` to 0 to disable the animation.
    pop_scale: float = 1.14
    pop_duration: float = 0.09


#: The caption looks a person can choose between, each a delta from the
#: default.  Order is display order.  A style is a *look* only: the timing
#: fields (``offset``, ``min_visible_fraction``) and ``review_confidence``
#: are carried over from the preset by ``Preset.with_caption_style`` rather
#: than reset here, so switching styles never moves a caption in time.
CAPTION_STYLES: dict[str, CaptionConfig] = {
    # The original: heavy sans, white words, the spoken one in yellow.
    "classic": CaptionConfig(),
    # Shouted caps, three words at a time, no pop -- the words are already
    # as big as they get.
    "hormozi": replace(
        CaptionConfig(), font="Anton-Regular", font_size=96, uppercase=True,
        outline=10, shadow_offset=6, max_words_per_line=3, pop_duration=0.0,
    ),
    # White words throughout; the spoken one sits on a purple pill instead
    # of changing colour.  A lighter pop, because the pill already moves.
    "pill": replace(
        CaptionConfig(), font="Montserrat-ExtraBold", active_box_colour="#7C3AED",
        active_text_colour="#FFFFFF", highlight_colour="#FFFFFF", outline=4, pop_scale=1.06,
    ),
    # Smaller, thinner, unanimated; the spoken word is not marked at all.
    # For talking heads whose words are not the point.
    "minimal": replace(
        CaptionConfig(), font="Montserrat-SemiBold", font_size=64, outline=2,
        shadow_offset=3, highlight_colour="#FFFFFF", pop_duration=0.0,
    ),
    # A subtitle block: the whole line on translucent black, no outline
    # needed because the box provides the contrast.
    "boxed": replace(
        CaptionConfig(), font="Montserrat-ExtraBold", font_size=72, line_box_colour="#000000B3",
        outline=0, shadow_offset=0, max_words_per_line=5, max_chars_per_line=28,
    ),
    # Tall condensed caps in the middle of the frame with a cyan halo; the
    # spoken word fills with a paler cyan -- the same cyan as the outline
    # would swallow its edges.
    "neon": replace(
        CaptionConfig(), font="BebasNeue-Regular", font_size=110, uppercase=True,
        outline_colour="#22D3EE", outline=4, glow=18, highlight_colour="#67E8F9",
        position="centre", shadow_offset=0,
    ),
}

#: What each style is called in the web app and the CLI's help.
CAPTION_STYLE_LABELS: dict[str, str] = {
    "classic": "Classic",
    "hormozi": "Bold caps",
    "pill": "Pill",
    "minimal": "Minimal",
    "boxed": "Boxed",
    "neon": "Neon",
}


@dataclass(frozen=True)
class ReframeConfig:
    enabled: bool = True
    width: int = 1080
    height: int = 1920
    #: Frames per second sampled for face detection (not the output fps).
    sample_fps: float = 4.0
    #: Exponential smoothing factor for the face track; lower is calmer.
    smoothing: float = 0.12
    #: Face movement smaller than this fraction of frame width is ignored.
    dead_zone: float = 0.06
    #: Below this detection rate, fall back to a static centre crop.
    min_detection_rate: float = 0.6
    #: Face is placed this far down the 9:16 frame (rule of thirds-ish).
    face_y_bias: float = 0.42


@dataclass(frozen=True)
class EncodeConfig:
    crf: int = 20
    preset: str = "veryfast"
    audio_bitrate: str = "192k"
    #: Integrated loudness target for social platforms.
    loudness_lufs: float = -14.0
    #: Quality of the post-cut intermediate; high, to avoid generation loss.
    intermediate_crf: int = 16


@dataclass(frozen=True)
class Preset:
    """The complete configuration for one render."""

    silence: SilenceConfig = field(default_factory=SilenceConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    filler: FillerConfig = field(default_factory=FillerConfig)
    retake: RetakeConfig = field(default_factory=RetakeConfig)
    punchin: PunchInConfig = field(default_factory=PunchInConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    reframe: ReframeConfig = field(default_factory=ReframeConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)

    #: ASR backend id, resolved by ``autocut.asr.get_transcriber``.
    asr_backend: str = "mlx-whisper"
    asr_model: str = "mlx-community/whisper-large-v3-turbo"
    language: str | None = "en"

    #: Keep segments shorter than this read as glitches and are dropped.
    min_segment: float = 0.25
    #: How far a cut boundary may be nudged to land on quiet audio rather than
    #: on a word's onset or its ringing tail, which the recogniser's timestamps
    #: do not cover.  Only ever shrinks a removal, so the cost of being wrong
    #: is a few milliseconds kept, not speech destroyed.
    boundary_relax: float = 0.25
    #: The shortest silence worth offering as its own decision in review.
    #: Pauses below this are left in the video: the beat between two sentences
    #: is rhythm rather than dead air, and listing every one of them buries the
    #: cuts that matter under decisions nobody wants to make.
    review_min_pause: float = 1.0
    #: How long a pause has to be before it starts a new part.  Applies to any
    #: gap between two words, not only ones at a sentence end -- a pause with
    #: no punctuation before it, longer than every ordinary pause in the clip,
    #: is what dropped speech looks like from the transcript's side, and it
    #: needs a part boundary just as much as a real sentence break does.
    part_pause: float = 0.35
    #: A keep segment containing no complete word has to be at least this long
    #: to be worth keeping.  Below it, the segment is debris left between two
    #: cuts -- a breath, or the clipped tail of a word that was removed -- and
    #: splicing it between them is what makes an edit sound abrupt.  Generous,
    #: because a stretch this long with no word in it is usually speech the
    #: recogniser dropped, which must survive.
    min_wordless_segment: float = 1.0
    #: Refuse to remove more than this fraction of the video, whatever the
    #: detectors say -- a last line of defence over all of them combined.
    max_total_removal_ratio: float = 0.6

    def gentle(self) -> Preset:
        """Cut only obvious dead air; no transcript-driven removals."""
        return replace(
            self,
            silence=replace(self.silence, min_gap=0.6),
            filler=replace(self.filler, enabled=False),
            retake=replace(self.retake, enabled=False),
        )

    def aggressive(self) -> Preset:
        """Tighten hard -- for fast-paced social cuts."""
        return replace(
            self,
            silence=replace(self.silence, min_gap=0.22, pad=0.06, sentence_pause=0.3),
            retake=replace(self.retake, max_removal_ratio=0.45),
        )

    def with_caption_style(self, name: str) -> Preset:
        """The same edit, captioned in a named style.

        Styles are whole ``CaptionConfig``s, so the fields that are not about
        looks -- whether captions are on, the timing nudge, the visibility
        threshold, what the review screen flags -- are copied over from this
        preset rather than taken from the style.  Otherwise choosing a look
        would silently undo a timing correction.
        """
        try:
            style = CAPTION_STYLES[name]
        except KeyError:
            choices = ", ".join(CAPTION_STYLES)
            raise ValueError(f"unknown caption style {name!r}; choose from {choices}") from None
        current = self.caption
        return replace(
            self,
            caption=replace(
                style,
                enabled=current.enabled,
                offset=current.offset,
                min_visible_fraction=current.min_visible_fraction,
                review_confidence=current.review_confidence,
            ),
        )


def _retake_from_env(retake: RetakeConfig) -> RetakeConfig:
    """Let the environment pick the retake backend without touching a preset.

    ``AUTOCUT_RETAKE_BACKEND=openrouter`` (with ``OPENROUTER_API_KEY`` set) is
    all it takes to move retake detection off the local model; a preset can
    still override this by setting ``retake=`` explicitly.
    """
    updates: dict[str, object] = {}
    if backend := os.environ.get("AUTOCUT_RETAKE_BACKEND"):
        updates["backend"] = backend
    if model := os.environ.get("AUTOCUT_RETAKE_MODEL"):
        updates["openrouter_model"] = model
    return replace(retake, **updates) if updates else retake


DEFAULT = replace(Preset(), retake=_retake_from_env(Preset().retake))
