FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim@sha256:76c7449e8bd509a860b61eb060599fd9b1750bcbafe28008f769d35db6109a3d

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install gcx — either from a local binary or from the remote install script.
#
# The --mount=type=bind makes files from the Docker build context (the environment/
# directory) available at /ctx inside this RUN step, without copying them into the
# image. If environment/gcx exists and is executable, we use it directly.
# Otherwise we fall back to the standard remote install.
#
# To use a local build: copy your gcx binary to environment/gcx before building,
# or set LOCAL_GCX=/path/to/binary when running harbor_preflight.sh.
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
