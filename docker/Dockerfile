# o11y-bench: Self-contained observability stack for Harbor
# Prometheus + Loki + Tempo + Grafana + mcp-grafana (streamable-http)

FROM grafana/grafana:12.4.2@sha256:83749231c3835e390a3144e5e940203e42b9589761f20ef3169c716e734ad505

ARG TARGETARCH

USER root

# Install runtime dependencies (python3 for data gen).
# Download steps below rely on curl/tar/unzip already being present in the Grafana base image.
RUN apk add --no-cache \
    ca-certificates \
    python3

# Install mcp-grafana (uses x86_64 instead of amd64 in release names)
ARG MCP_GRAFANA_VERSION=0.11.4
RUN MCP_ARCH=$([ "$TARGETARCH" = "arm64" ] && echo "arm64" || echo "x86_64") && \
    curl -fsSL https://github.com/grafana/mcp-grafana/releases/download/v${MCP_GRAFANA_VERSION}/mcp-grafana_Linux_${MCP_ARCH}.tar.gz \
    | tar xz -C /usr/local/bin mcp-grafana && \
    chmod +x /usr/local/bin/mcp-grafana

# Install Prometheus
ARG PROMETHEUS_VERSION=3.10.0
RUN curl -fsSL https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-${TARGETARCH}.tar.gz \
    | tar xz --strip-components=1 -C /usr/local/bin \
        prometheus-${PROMETHEUS_VERSION}.linux-${TARGETARCH}/prometheus \
        prometheus-${PROMETHEUS_VERSION}.linux-${TARGETARCH}/promtool

# Install Loki
ARG LOKI_VERSION=3.6.7
RUN curl -fsSL https://github.com/grafana/loki/releases/download/v${LOKI_VERSION}/loki-linux-${TARGETARCH}.zip \
    -o /tmp/loki.zip && \
    unzip -qp /tmp/loki.zip loki-linux-${TARGETARCH} > /usr/local/bin/loki && \
    chmod +x /usr/local/bin/loki && \
    rm /tmp/loki.zip

# Install Tempo
ARG TEMPO_VERSION=2.10.3
RUN curl -fsSL https://github.com/grafana/tempo/releases/download/v${TEMPO_VERSION}/tempo_${TEMPO_VERSION}_linux_${TARGETARCH}.tar.gz \
    | tar xz -C /usr/local/bin tempo

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
COPY python/src/o11y_stack /scripts/o11y_stack
RUN chmod +x /entrypoint.sh

# Expose ports: Prometheus:9090, Loki:3100, Grafana:3000, Tempo:3200, MCP-Grafana:8080
EXPOSE 9090 3100 3000 3200 8080

ENTRYPOINT ["/entrypoint.sh"]
