# o11y-bench: Self-contained observability stack for Harbor
# Prometheus + Loki + Tempo + Grafana + mcp-grafana (streamable-http)

FROM grafana/grafana:13.2.4@sha256:dbbc38ea21103cefb50396340cd88afec779c7e070b0a7bf5a75b4ce46651d3f

ARG TARGETARCH

USER root

# Install runtime dependencies (python3 for data gen).
# Download steps below rely on curl/tar/unzip already being present in the Grafana base image.
RUN apk add --no-cache \
    ca-certificates \
    python3

# Install mcp-grafana — use a local binary if present, otherwise fetch the
# pinned release. Release names use x86_64 instead of amd64.
# --mount=type=bind exposes the build context at /ctx without adding a layer.
ARG MCP_GRAFANA_VERSION=2.0.2
# SHA-256 of the release tarballs (see the release's checksums file); update with the version.
ARG MCP_GRAFANA_SHA256_x86_64=624e997b78bcadf236f82f8086827890ba77b6dfdc6fd6446f7c48dfd9e586f0
ARG MCP_GRAFANA_SHA256_arm64=96559ba56803c253ee4bfc2be971a5132bdd60a09b4a20375b6456564216ec48
RUN --mount=type=bind,source=.,target=/ctx \
    if [ -x /ctx/mcp-grafana ]; then \
      echo "Installing mcp-grafana from local binary"; \
      cp /ctx/mcp-grafana /usr/local/bin/mcp-grafana; \
    else \
      MCP_ARCH=$([ "$TARGETARCH" = "arm64" ] && echo "arm64" || echo "x86_64") && \
      MCP_SHA256=$([ "$MCP_ARCH" = "arm64" ] && echo "$MCP_GRAFANA_SHA256_arm64" || echo "$MCP_GRAFANA_SHA256_x86_64") && \
      curl -fsSL -o /tmp/mcp-grafana.tar.gz https://github.com/grafana/mcp-grafana/releases/download/v${MCP_GRAFANA_VERSION}/mcp-grafana_Linux_${MCP_ARCH}.tar.gz && \
      echo "$MCP_SHA256  /tmp/mcp-grafana.tar.gz" | sha256sum -c - && \
      tar xzf /tmp/mcp-grafana.tar.gz -C /usr/local/bin mcp-grafana && \
      rm /tmp/mcp-grafana.tar.gz; \
    fi && \
    chmod +x /usr/local/bin/mcp-grafana

# Install Prometheus
# SHA-256 of the release tarballs (see the release's sha256sums.txt); update with the version.
ARG PROMETHEUS_VERSION=3.15.0
ARG PROMETHEUS_SHA256_amd64=2a542df32eac02ee17b9d844fb2aa1de00dafa5476579ba8a3ba862e9d572ea0
ARG PROMETHEUS_SHA256_arm64=f1f90ec08e849d494ca66c611470afc50192f0355f1a61c33f2cbde02d067823
RUN PROM_SHA256=$([ "$TARGETARCH" = "arm64" ] && echo "$PROMETHEUS_SHA256_arm64" || echo "$PROMETHEUS_SHA256_amd64") && \
    curl -fsSL -o /tmp/prometheus.tar.gz https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-${TARGETARCH}.tar.gz && \
    echo "$PROM_SHA256  /tmp/prometheus.tar.gz" | sha256sum -c - && \
    tar xzf /tmp/prometheus.tar.gz --strip-components=1 -C /usr/local/bin \
        prometheus-${PROMETHEUS_VERSION}.linux-${TARGETARCH}/prometheus \
        prometheus-${PROMETHEUS_VERSION}.linux-${TARGETARCH}/promtool && \
    rm /tmp/prometheus.tar.gz

# Install Loki
# SHA-256 of the release zips (see the release's SHA256SUMS); update with the version.
ARG LOKI_VERSION=3.7.8
ARG LOKI_SHA256_amd64=62aea42c9cba52cd1642b3666ab37019a0ce4c24ab50b07e85dccc8d812f7d61
ARG LOKI_SHA256_arm64=a874c10ffefe4d66c914b8aac8897159451d3026469598341aa03c76084697de
RUN LOKI_SHA256=$([ "$TARGETARCH" = "arm64" ] && echo "$LOKI_SHA256_arm64" || echo "$LOKI_SHA256_amd64") && \
    curl -fsSL https://github.com/grafana/loki/releases/download/v${LOKI_VERSION}/loki-linux-${TARGETARCH}.zip \
    -o /tmp/loki.zip && \
    echo "$LOKI_SHA256  /tmp/loki.zip" | sha256sum -c - && \
    unzip -qp /tmp/loki.zip loki-linux-${TARGETARCH} > /usr/local/bin/loki && \
    chmod +x /usr/local/bin/loki && \
    rm /tmp/loki.zip

# Install Tempo
# SHA-256 of the release tarballs (see the release's SHA256SUMS); update with the version.
ARG TEMPO_VERSION=3.1.0
ARG TEMPO_SHA256_amd64=9604ab46348270a42954801bba91cd65335e16568ef886bdf71b988fdcb7ac4e
ARG TEMPO_SHA256_arm64=e4092dcf2b93e141be77e300b539f6c5cceefa9803417d7d635c6797d9bc19ad
RUN TEMPO_SHA256=$([ "$TARGETARCH" = "arm64" ] && echo "$TEMPO_SHA256_arm64" || echo "$TEMPO_SHA256_amd64") && \
    curl -fsSL -o /tmp/tempo.tar.gz https://github.com/grafana/tempo/releases/download/v${TEMPO_VERSION}/tempo_${TEMPO_VERSION}_linux_${TARGETARCH}.tar.gz && \
    echo "$TEMPO_SHA256  /tmp/tempo.tar.gz" | sha256sum -c - && \
    tar xzf /tmp/tempo.tar.gz -C /usr/local/bin tempo && \
    rm /tmp/tempo.tar.gz

# Create directories for all services
RUN mkdir -p \
    /prometheus \
    /loki/chunks /loki/rules /loki/compactor \
    /var/tempo/wal /var/tempo/blocks /var/tempo/generator/wal /var/tempo/generator/traces \
    /etc/prometheus /etc/loki /etc/tempo \
    /scripts

# Copy configuration files
COPY prometheus.yml /etc/prometheus/prometheus.yml
COPY loki-config.yaml /etc/loki/config.yaml
COPY tempo.yaml /etc/tempo/tempo.yaml
COPY datasources.yaml /etc/grafana/provisioning/datasources/datasources.yaml
COPY entrypoint.sh /entrypoint.sh
COPY grafana-env.sh /etc/grafana/o11y-env.sh
COPY bake-grafana.sh /tmp/bake-grafana.sh
COPY python/src/o11y_stack /scripts/o11y_stack
# Seed data at a fixed clock, shifted to the scenario clock at boot (see o11y_stack/seed.py)
RUN PYTHONPATH=/scripts python3 -m o11y_stack.seed dump
# The entrypoint probes which flags mcp-grafana accepts; do it once here instead of on every boot
# (an empty result would silently drop --allowed-hosts and every trial would get a 403, so fail the build)
RUN /usr/local/bin/mcp-grafana --help > /etc/mcp-grafana-help.txt 2>&1 || true; test -s /etc/mcp-grafana-help.txt
RUN chmod +x /entrypoint.sh /tmp/bake-grafana.sh && /tmp/bake-grafana.sh && rm /tmp/bake-grafana.sh

# Expose ports: Prometheus:9090, Loki:3100, Grafana:3000, Tempo:3200, MCP-Grafana:8080
EXPOSE 9090 3100 3000 3200 8080

# Five Go services share the host with other trials' stacks: capping each at 2 threads' worth of
# parallelism cuts boot CPU by about 15% and an 8-way parallel boot from 4.4 s to 3.9 s.
# Override with O11Y_STACK_GOMAXPROCS (compose and regrade pass it through).
ENV GOMAXPROCS=2

ENTRYPOINT ["/entrypoint.sh"]
