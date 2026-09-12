"""Sample images of each caption style, for the review screen's picker.

Drawn by the same renderer that captions the video, so what the picker shows
is what the render produces -- a CSS imitation would drift from the real
thing the first time a style changed.  One canned line, one word active,
rendered once per process and kept.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO

from PIL import Image

from autocut.config import CAPTION_STYLES
from autocut.render.captions import CaptionWord, Line, Renderer

#: The width every sample is drawn at (the video's), and the width it is
#: served at.  Drawing at full size keeps outline and glow in the proportions
#: the video will have; halving it afterwards keeps the picker light.
RENDER_WIDTH = 1080
SAMPLE_WIDTH = 540
#: The web app's panel-hover colour, so the sample reads as part of the page.
BACKGROUND = "#1c2129"

_LINE = Line([CaptionWord(text, 0.0, 0.0) for text in ("this", "is", "how", "it", "looks")])
_ACTIVE = 2


@lru_cache(maxsize=None)
def caption_sample(name: str) -> bytes:
    """A PNG of ``name``'s caption look; ``KeyError`` for an unknown style.

    The style's line-length limits are ignored on purpose: every sample is
    the same five words on one line, so the cards in the picker line up.
    """
    config = CAPTION_STYLES[name]
    band = Renderer(config, RENDER_WIDTH).render(_LINE, _ACTIVE, popped=False)
    sample = Image.new("RGB", band.size, BACKGROUND)
    sample.paste(band, mask=band)
    scale = SAMPLE_WIDTH / band.width
    sample = sample.resize((SAMPLE_WIDTH, max(1, round(band.height * scale))), Image.LANCZOS)
    buffer = BytesIO()
    sample.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
