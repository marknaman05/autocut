"""Every tunable threshold in the pipeline, in one place.

Detectors and renderers take a ``Preset`` rather than reading globals, so a job
can be re-run with different aggressiveness without touching code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class SilenceConfig:
    #: Gaps between words shorter than this are left alone.
    min_gap: float = 0.35
    #: Breathing room left on each side of a removed gap.
    pad: float = 0.12


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
class RetakeConfig:
    enabled: bool = True
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
    paraphrase_threshold: float = 0.6
    #: A span with fewer content words than this cannot be judged reliably and
    #: is rejected outright, rather than risking a false match on stopwords.
    min_overlap_words: int = 2


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
    #: Font family name; resolved to a file by ``render.captions.find_font``.
    font: str = "Arial Black"
    font_size: int = 82
    text_colour: str = "#FFFFFF"
    highlight_colour: str = "#FFD44A"
    outline_colour: str = "#000000"
    outline: int = 8
    shadow_offset: int = 4
    shadow_colour: str = "#00000099"
    #: Distance from the bottom of the frame, clear of platform UI.
    margin_v: int = 420
    max_words_per_line: int = 4
    max_chars_per_line: int = 22
    #: A pause longer than this always breaks the caption line.
    line_break_gap: float = 0.4
    #: Global nudge applied to every caption, for ASR timing drift.
    offset: float = 0.0
    #: A word must survive at least this much of the cut to be captioned.
    #: Snapping cut points can leave a few milliseconds of a removed word
    #: behind; without this, a cut "um" flashes up for a frame or two.
    min_visible_fraction: float = 0.5
    #: Scale of the active word at the peak of its pop, and how long the pop
    #: lasts.  Set ``pop_duration`` to 0 to disable the animation.
    pop_scale: float = 1.14
    pop_duration: float = 0.09


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
            silence=replace(self.silence, min_gap=0.22, pad=0.06),
            retake=replace(self.retake, max_removal_ratio=0.45),
        )


DEFAULT = Preset()
