FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim@sha256:76c7449e8bd509a860b61eb060599fd9b1750bcbafe28008f769d35db6109a3d

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://raw.githubusercontent.com/grafana/gcx/main/scripts/install.sh | INSTALL_DIR=/usr/local/bin sh

RUN gcx skills install --all \
    && mkdir -p ~/.config/opencode/skills \
    && ln -s ~/.agents/skills/* ~/.config/opencode/skills/

WORKDIR /app
