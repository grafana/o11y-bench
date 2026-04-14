#!/bin/bash
set -e

sanitize_max_attempts() {
    local name=$1
    local default_value=$2
    local value=${!name:-}

    if [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s\n' "$value"
        return 0
    fi

    if [ -n "$value" ]; then
        echo "WARN: invalid $name=$value, using default $default_value" >&2
    fi
    printf '%s\n' "$default_value"
}

LOKI_MAX_ATTEMPTS=$(sanitize_max_attempts "LOKI_MAX_ATTEMPTS" 120)
TEMPO_MAX_ATTEMPTS=$(sanitize_max_attempts "TEMPO_MAX_ATTEMPTS" 120)
TEMPO_OTLP_MAX_ATTEMPTS=$(sanitize_max_attempts "TEMPO_OTLP_MAX_ATTEMPTS" 120)
PROMETHEUS_MAX_ATTEMPTS=$(sanitize_max_attempts "PROMETHEUS_MAX_ATTEMPTS" 120)
GRAFANA_MAX_ATTEMPTS=$(sanitize_max_attempts "GRAFANA_MAX_ATTEMPTS" 180)
MCP_MAX_ATTEMPTS=$(sanitize_max_attempts "MCP_MAX_ATTEMPTS" 60)

wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-30}
    local log_file=$4
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo "$name ready"
            return 0
        fi
        sleep 0.5
        attempt=$((attempt + 1))
    done
    echo "ERROR: $name failed to start after $max_attempts attempts"
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        echo "=== $name logs (last 50 lines) ==="
        tail -50 "$log_file"
        echo "=== end $name logs ==="
    fi
    return 1
}

wait_for_tempo_otlp() {
    local max_attempts=${1:-30}
    local attempt=1
    # Minimal JSON trace accepted by OTLP HTTP; proves 4318 is serving, not only /ready.
    local probe='{"resourceSpans":[{"resource":{"attributes":[]},"scopeSpans":[{"scope":{"name":"test"},"spans":[{"traceId":"00000000000000000000000000000001","spanId":"0000000000000001","name":"test","startTimeUnixNano":"1000000000","endTimeUnixNano":"1000000001"}]}]}]}'

    while [ $attempt -le $max_attempts ]; do
        if curl -sf -X POST "http://localhost:4318/v1/traces" \
            -H "Content-Type: application/json" \
            -d "$probe" > /dev/null 2>&1; then
            echo "Tempo OTLP ready"
            return 0
        fi
        sleep 0.5
        attempt=$((attempt + 1))
    done
    echo "ERROR: Tempo OTLP not ready after $max_attempts attempts"
    return 1
}

wait_for_http_listener() {
    local url=$1
    local name=$2
    local max_attempts=${3:-30}
    local log_file=$4
    local attempt=1

    while [ "$attempt" -le "$max_attempts" ]; do
        if curl -s -o /dev/null "$url" 2>/dev/null; then
            echo "$name ready"
            return 0
        fi
        sleep 0.5
        attempt=$((attempt + 1))
    done
    echo "ERROR: $name failed to start after $max_attempts attempts"
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        echo "=== $name logs (last 50 lines) ==="
        tail -50 "$log_file"
        echo "=== end $name logs ==="
    fi
    return 1
}

provision_task_resources() {
    local setup_file="/task/setup.json"

    if [ -d "$setup_file" ]; then
        echo "ERROR: $setup_file is a directory (host bind mount source was missing when compose started)"
        return 1
    fi
    if [ ! -f "$setup_file" ]; then
        echo "No task setup file found"
        return 0
    fi

    PYTHONPATH=/scripts python3 -m o11y_stack.provision_task_resources "$setup_file"
}

echo "=== o11y-bench Environment ==="

mkdir -p /logs/artifacts/sidecar
ENTRYPOINT_LOG=/logs/artifacts/sidecar/entrypoint.log
: > "$ENTRYPOINT_LOG"
exec > >(tee "$ENTRYPOINT_LOG") 2>&1

# Start Loki, Tempo, and Grafana in parallel
echo "Starting services..."
/usr/local/bin/loki --config.file=/etc/loki/config.yaml > /logs/artifacts/sidecar/loki.log 2>&1 &

/usr/local/bin/tempo --config.file=/etc/tempo/tempo.yaml > /logs/artifacts/sidecar/tempo.log 2>&1 &

GF_PATHS_PROVISIONING="/etc/grafana/provisioning" \
GF_PATHS_PLUGINS="/var/lib/grafana/plugins" \
GF_AUTH_ANONYMOUS_ENABLED="true" \
GF_AUTH_ANONYMOUS_ORG_ROLE="Admin" \
GF_AUTH_BASIC_ENABLED="false" \
GF_INSTALL_PLUGINS="" \
GF_PLUGINS_PREINSTALL_DISABLED="true" \
GF_ANALYTICS_CHECK_FOR_UPDATES="false" \
GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES="false" \
/usr/share/grafana/bin/grafana-server \
    --homepath=/usr/share/grafana \
    --config=/usr/share/grafana/conf/defaults.ini \
    > /logs/artifacts/sidecar/grafana.log 2>&1 &

# Wait for Loki and Tempo (needed for data generation)
wait_for_service "http://localhost:3100/ready" "Loki" "$LOKI_MAX_ATTEMPTS" "/logs/artifacts/sidecar/loki.log"
wait_for_service "http://localhost:3200/ready" "Tempo" "$TEMPO_MAX_ATTEMPTS" "/logs/artifacts/sidecar/tempo.log"
wait_for_tempo_otlp "$TEMPO_OTLP_MAX_ATTEMPTS"

# Generate telemetry data (writes /tmp/env_timestamp when done)
echo "Generating telemetry data..."
PYTHONPATH=/scripts python3 -m o11y_stack.generate_data
if [ ! -f /tmp/env_timestamp ]; then
    echo "ERROR: data generation failed (no env_timestamp)"
    exit 1
fi

# Start Prometheus after data generation (so it loads TSDB blocks)
echo "Starting Prometheus..."
/usr/local/bin/prometheus \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --storage.tsdb.retention.time=99y \
    --web.enable-lifecycle \
    > /logs/artifacts/sidecar/prometheus.log 2>&1 &

# Wait for all services
wait_for_service "http://localhost:9090/-/ready" "Prometheus" "$PROMETHEUS_MAX_ATTEMPTS" "/logs/artifacts/sidecar/prometheus.log"
wait_for_service "http://localhost:3000/api/health" "Grafana" "$GRAFANA_MAX_ATTEMPTS" "/logs/artifacts/sidecar/grafana.log"
provision_task_resources

# Start mcp-grafana in streamable-http mode for Harbor agent access
echo "Starting mcp-grafana (streamable-http on :8080)..."
GRAFANA_URL=http://localhost:3000 /usr/local/bin/mcp-grafana \
    -t streamable-http \
    --address :8080 \
    --disable-sift \
    --disable-oncall \
    --disable-incident \
    --disable-asserts \
    --disable-pyroscope \
    > /logs/artifacts/sidecar/mcp-grafana.log 2>&1 &

# Wait for mcp-grafana to be listening (404 is fine — it serves on /mcp path)
wait_for_http_listener \
    "http://localhost:8080/" \
    "mcp-grafana" \
    "$MCP_MAX_ATTEMPTS" \
    "/logs/artifacts/sidecar/mcp-grafana.log"

echo "=== Environment Ready ==="
echo "  Grafana:     http://localhost:3000"
echo "  Prometheus:  http://localhost:9090"
echo "  Loki:        http://localhost:3100"
echo "  Tempo:       http://localhost:3200"
echo "  MCP-Grafana: http://localhost:8080"

# Keep container alive
wait
