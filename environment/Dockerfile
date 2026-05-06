FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim@sha256:76c7449e8bd509a860b61eb060599fd9b1750bcbafe28008f769d35db6109a3d

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install gcx — use a local binary if present, otherwise fetch from GitHub.
# --mount=type=bind exposes the build context at /ctx without adding a layer.
RUN --mount=type=bind,source=.,target=/ctx \
    if [ -x /ctx/gcx ]; then \
      echo "Installing gcx from local binary"; \
      cp /ctx/gcx /usr/local/bin/gcx; \
    else \
      curl -fsSL https://raw.githubusercontent.com/grafana/gcx/main/scripts/install.sh | INSTALL_DIR=/usr/local/bin sh; \
    fi

RUN gcx skills install --all \
    && mkdir -p ~/.config/opencode/skills \
    && ln -s ~/.agents/skills/* ~/.config/opencode/skills/

WORKDIR /app
