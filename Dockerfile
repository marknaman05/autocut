# autocut server, hosted.  Runs the web app and a Cloudflare Tunnel side by
# side, so the app is reachable only through the tunnel (and therefore
# through Cloudflare Access), exactly like the laptop setup it replaces.
# Transcription goes through OpenRouter, so no GPU, no MLX, no torch.
FROM python:3.12-slim-trixie

ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
# TARGETARCH is set by BuildKit (amd64/arm64); the legacy builder leaves it
# empty, so fall back to the build machine's own architecture.
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 curl ca-certificates \
    && ARCH="${TARGETARCH:-$(dpkg --print-architecture)}" \
    && curl -fsSL -o /usr/local/bin/cloudflared \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${ARCH}" \
    && chmod +x /usr/local/bin/cloudflared \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra server --extra vision
COPY autocut ./autocut
COPY server ./server
COPY samples ./samples
COPY README.md ./
RUN uv sync --frozen --no-dev --extra server --extra vision

# Data lives on the mounted volume; the paths are absolute on purpose.
# Identity is decided by the environment file: AUTOCUT_GOOGLE_CLIENT_ID/_SECRET
# + AUTOCUT_SESSION_SECRET for the app's own Google sign-in, or
# AUTOCUT_USER_HEADER when a proxy (Cloudflare Access) does the login.
ENV AUTOCUT_WORK_ROOT=/data/work AUTOCUT_DB=/data/autocut.db \
    AUTOCUT_ASR_BACKEND=openrouter AUTOCUT_RETAKE_BACKEND=openrouter
COPY deploy/start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
