"""Stage 5 -- 9:16 reframing with face tracking and punch-ins.

Face detection runs on the *already cut* video, so the track is in output time
and needs no mapping.  That also means the tracker never sees footage that was
removed, and cannot be pulled off course by it.

The framing is deliberately lazy.  A crop that follows a head exactly looks
like a nervous camera operator; the dead zone means small movements produce no
motion at all, and the smoothing turns large ones into a slow drift.  Zoom, by
contrast, is a step function that only changes on a cut -- that is what makes a
punch-in read as an edit rather than a camera move.

The resulting motion is expressed as ffmpeg ``zoompan`` expressions in terms of
the output frame number, so the whole reframe happens inside the single final
render pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import ReframeConfig
from ..models import ZoomSpan

log = logging.getLogger(__name__)


@dataclass
class FaceTrack:
    """Face centre over time, in normalised source coordinates (0..1)."""

    times: list[float]
    x: list[float]
    y: list[float]
    detection_rate: float

    def __len__(self) -> int:
        return len(self.times)


#: YuNet, from the OpenCV model zoo: ~230 KB, fetched once and cached.
#:
#: This runs through the OpenCV we already use to read frames.  mediapipe would
#: be the obvious alternative, but mediapipe 1.x aborts the process outright on
#: this machine (a Metal service failure inside the graph, not a Python
#: exception), which no amount of error handling can survive.
_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_MODEL_CACHE = Path.home() / ".cache" / "autocut" / "face_detection_yunet_2023mar.onnx"

_DETECTION_THRESHOLD = 0.6


def _face_model() -> Path:
    """Path to the face detection model, downloading it once if needed."""
    if _MODEL_CACHE.exists() and _MODEL_CACHE.stat().st_size > 0:
        return _MODEL_CACHE

    import urllib.request

    _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    log.info("fetching the face detection model (one time, ~230 KB)")
    temporary = _MODEL_CACHE.with_suffix(".partial")
    urllib.request.urlretrieve(_MODEL_URL, temporary)
    temporary.replace(_MODEL_CACHE)
    return _MODEL_CACHE


def _detect_faces_in(frames: list, width: int, height: int) -> list[tuple[float, float] | None]:
    """Face centre per frame, in normalised coordinates, or None if not found."""
    import cv2

    detector = cv2.FaceDetectorYN.create(
        str(_face_model()), "", (width, height), _DETECTION_THRESHOLD, 0.3, 5000
    )
    detector.setInputSize((width, height))

    results: list[tuple[float, float] | None] = []
    for frame in frames:
        _, faces = detector.detect(frame)
        if faces is None or len(faces) == 0:
            results.append(None)
            continue
        # Largest face wins: the speaker, not someone in the background.
        x, y, face_width, face_height = max(faces, key=lambda f: f[2] * f[3])[:4]
        results.append(
            ((x + face_width / 2) / width, (y + face_height / 2) / height)
        )
    return results


def detect_faces(video: Path, config: ReframeConfig) -> FaceTrack | None:
    """Sample the video and track the speaker's face.

    Returns ``None`` if detection is unavailable or too unreliable to use, in
    which case the caller falls back to a static centre crop.
    """
    try:
        import cv2
    except ImportError:
        log.warning("opencv not installed; using a centre crop (uv sync --extra vision)")
        return None

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        log.warning("could not open %s for face detection", video)
        return None

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(int(round(source_fps / config.sample_fps)), 1)
    frames, times = [], []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                frames.append(frame)
                times.append(index / source_fps)
            index += 1
    finally:
        capture.release()

    if not frames:
        return None
    height, width = frames[0].shape[:2]

    try:
        detections = _detect_faces_in(frames, width, height)
    except Exception as error:  # noqa: BLE001 - never fail a render over framing
        log.warning("face detection unavailable (%s); using a centre crop", error)
        return None

    found = [d for d in detections if d is not None]
    rate = len(found) / len(detections) if detections else 0.0
    if rate < config.min_detection_rate:
        log.info(
            "face found in only %.0f%% of sampled frames; using a centre crop", rate * 100
        )
        return None

    # Hold the last known position through frames where detection dropped out,
    # so a momentary miss does not yank the framing back to centre.
    xs, ys = [], []
    last = found[0]
    for detection in detections:
        if detection is not None:
            last = detection
        xs.append(last[0])
        ys.append(last[1])

    log.info("tracked a face across %d samples (%.0f%% detected)", len(times), rate * 100)
    return FaceTrack(times=times, x=xs, y=ys, detection_rate=rate)


def smooth(track: FaceTrack, config: ReframeConfig) -> FaceTrack:
    """Apply the dead zone and exponential smoothing to a raw track."""
    smoothed_x, smoothed_y = [], []
    current_x, current_y = track.x[0], track.y[0]

    for x, y in zip(track.x, track.y):
        # Inside the dead zone the framing simply does not move.
        if abs(x - current_x) > config.dead_zone:
            current_x += (x - current_x) * config.smoothing
        if abs(y - current_y) > config.dead_zone:
            current_y += (y - current_y) * config.smoothing
        smoothed_x.append(current_x)
        smoothed_y.append(current_y)

    return FaceTrack(
        times=track.times, x=smoothed_x, y=smoothed_y, detection_rate=track.detection_rate
    )


def simplify(track: FaceTrack, tolerance: float = 0.004, limit: int = 48) -> FaceTrack:
    """Reduce the track to the fewest keyframes that still describe it.

    The result becomes a nested ffmpeg expression, so its size matters: the
    tolerance is raised until the track fits within ``limit`` keyframes.
    """
    while True:
        times, xs, ys = [track.times[0]], [track.x[0]], [track.y[0]]
        for t, x, y in zip(track.times[1:], track.x[1:], track.y[1:]):
            if abs(x - xs[-1]) >= tolerance or abs(y - ys[-1]) >= tolerance:
                times.append(t)
                xs.append(x)
                ys.append(y)
        if track.times[-1] > times[-1]:
            times.append(track.times[-1])
            xs.append(track.x[-1])
            ys.append(track.y[-1])
        if len(times) <= limit or tolerance > 0.2:
            return FaceTrack(times=times, x=xs, y=ys, detection_rate=track.detection_rate)
        tolerance *= 1.8


def _piecewise(keyframes: list[tuple[float, float]], variable: str, step: bool) -> str:
    """Build an ffmpeg expression interpolating (or stepping) through keyframes.

    Emitted as a nested ``if`` chain, which is why the keyframe count is capped.
    """
    if not keyframes:
        return "0"
    if len(keyframes) == 1:
        return f"{keyframes[0][1]:.5f}"

    expression = f"{keyframes[-1][1]:.5f}"
    for (t0, v0), (t1, v1) in reversed(list(zip(keyframes, keyframes[1:]))):
        if step or t1 <= t0:
            value = f"{v0:.5f}"
        else:
            slope = (v1 - v0) / (t1 - t0)
            value = f"({v0:.5f}+({slope:.5f})*({variable}-{t0:.5f}))"
        expression = f"if(lt({variable},{t1:.5f}),{value},{expression})"
    return expression


@dataclass
class ReframePlan:
    """Everything the renderer needs to turn the source into a 9:16 frame.

    Three filters, in this order, because each can only do one of the jobs:

    ``scale``
        Enlarges the source so the un-zoomed 9:16 window is exactly the output
        size.  Everything downstream can then work in output pixels.
    ``crop``
        Follows the face.  Its size is fixed -- a filter cannot change frame
        size mid-stream -- so this is what establishes the 9:16 aspect.
    ``zoompan``
        Does the punch-in, and *only* the punch-in.  It crops ``in/zoom`` at the
        input's aspect ratio, so it cannot change 16:9 into 9:16; pointing it at
        an already-9:16 stream is what keeps the picture undistorted.
    """

    prescale_width: int
    prescale_height: int
    zoom: str
    x: str
    y: str
    #: True when every punch-in level is 1.0, so the zoom filter can be skipped.
    static_zoom: bool = False

    def filter_string(self, out_width: int, out_height: int, fps: float) -> str:
        # crop evaluates x/y per frame on every ffmpeg release (verified on
        # 7.1 and 9.0); it has never had an `eval` option, and passing one
        # is a hard error on builds before 9.
        chain = [
            f"scale={self.prescale_width}:{self.prescale_height}:flags=bicubic",
            f"crop=w={out_width}:h={out_height}:x='{self.x}':y='{self.y}'",
        ]
        if not self.static_zoom:
            # Zoom about the centre of the already-framed picture.
            chain.append(
                f"zoompan=z='{self.zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d=1:s={out_width}x{out_height}:fps={fps:.6f}"
            )
        return ",".join(chain)


def plan(
    source_width: int,
    source_height: int,
    track: FaceTrack | None,
    zooms: list[ZoomSpan],
    config: ReframeConfig,
    fps: float,
) -> ReframePlan:
    """Turn a face track and a zoom schedule into ffmpeg expressions.

    The source is first scaled so that the un-zoomed 9:16 window is exactly the
    output size.  That keeps ``zoompan``'s zoom factor at or above 1 whatever
    the source resolution -- below 1 it cannot fill the frame.
    """
    scale = max(config.width / source_width, config.height / source_height)
    prescale_width = int(round(source_width * scale / 2)) * 2
    prescale_height = int(round(source_height * scale / 2)) * 2

    # crop evaluates per frame and exposes `t` directly.  zoompan does not, but
    # its output frame counter `on` divided by the (fixed) fps is the same
    # thing, because it emits exactly one frame per input frame.
    zoom_levels = [(span.start, span.zoom) for span in zooms]
    static_zoom = not zoom_levels or all(z == 1.0 for _, z in zoom_levels)
    zoom_expression = _piecewise(zoom_levels, f"(on/{fps:.6f})", step=True)

    if track is None or len(track) == 0:
        centre_x, centre_y = "0.5", f"{config.face_y_bias:.4f}"
    else:
        simplified = simplify(smooth(track, config))
        centre_x = _piecewise(list(zip(simplified.times, simplified.x)), "t", step=False)
        centre_y = _piecewise(list(zip(simplified.times, simplified.y)), "t", step=False)

    # Window position in pre-scaled pixels, clamped so the crop never leaves the
    # frame.  The window is a fixed output-sized rectangle; only its position
    # moves, which is what keeps the aspect ratio honest.
    x_expression = (
        f"max(0\\,min({prescale_width - config.width}\\,"
        f"({centre_x})*{prescale_width}-{config.width}/2))"
    )
    # Faces sit above centre in a portrait frame, so the window is placed to put
    # the tracked point at `face_y_bias` down the output rather than halfway.
    y_expression = (
        f"max(0\\,min({prescale_height - config.height}\\,"
        f"({centre_y})*{prescale_height}-{config.height}*{config.face_y_bias:.4f}))"
    )

    return ReframePlan(
        prescale_width=prescale_width,
        prescale_height=prescale_height,
        zoom=zoom_expression,
        x=x_expression,
        y=y_expression,
        static_zoom=static_zoom,
    )
