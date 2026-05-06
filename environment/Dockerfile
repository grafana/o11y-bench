FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim@sha256:3e70f580d0e63d78408c35d332d780024b6e1d46d9744c888e22fa944393448e

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://raw.githubusercontent.com/grafana/gcx/main/scripts/install.sh | INSTALL_DIR=/usr/local/bin sh

RUN gcx skills install --all \
    && mkdir -p ~/.config/opencode/skills \
    && ln -s ~/.agents/skills/* ~/.config/opencode/skills/

WORKDIR /app
