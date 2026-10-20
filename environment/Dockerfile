FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim@sha256:76c7449e8bd509a860b61eb060599fd9b1750bcbafe28008f769d35db6109a3d

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install gcx — use a local binary if present, otherwise the pinned release.
# --mount=type=bind exposes the build context at /ctx without adding a layer.
ARG TARGETARCH
ARG GCX_VERSION=1.6.0
# Hashes come from the release's checksums.txt.
ARG GCX_SHA256_amd64=335e051cda6edfb4ad5757540208961dbdfab6d89d4af11a4087de6ffa0ec111
ARG GCX_SHA256_arm64=116fd1a9f25d5f5e29802b705d9964578a01614f94499b85461b6abca1cb4a04
RUN --mount=type=bind,source=.,target=/ctx \
    if [ -x /ctx/gcx ]; then \
      echo "Installing gcx from local binary"; \
      cp /ctx/gcx /usr/local/bin/gcx; \
    else \
      GCX_SHA256=$([ "$TARGETARCH" = "arm64" ] && echo "$GCX_SHA256_arm64" || echo "$GCX_SHA256_amd64") && \
      curl -fsSL -o /tmp/gcx.tar.gz https://github.com/grafana/gcx/releases/download/v${GCX_VERSION}/gcx_${GCX_VERSION}_linux_${TARGETARCH}.tar.gz && \
      echo "$GCX_SHA256  /tmp/gcx.tar.gz" | sha256sum -c - && \
      tar xzf /tmp/gcx.tar.gz -C /usr/local/bin gcx && \
      rm /tmp/gcx.tar.gz; \
    fi

RUN gcx agent skills install --all \
    && mkdir -p ~/.config/opencode/skills \
    && ln -s ~/.agents/skills/* ~/.config/opencode/skills/

WORKDIR /app
