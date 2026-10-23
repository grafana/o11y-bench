"""Pre-generated seed data, replayed at boot with every timestamp shifted to the scenario clock.

``dump`` (image build) runs the generator once at a fixed clock and keeps what it would have sent
to Prometheus, Loki and Tempo. ``metrics`` and ``ingest`` (boot) shift those by the whole-minute
difference between the scenario clock and the seed clock, so the data keeps the same offsets from
the clock that generate_data produces.
"""

import calendar
import os
import pickle
import random
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from . import generate_data as g

SEED_DIR = "/seed"
SEED_CLOCK = "2026-10-20T00:00:00Z"

OTLP_PROBE = (
    b'{"resourceSpans":[{"resource":{"attributes":[]},"scopeSpans":[{"scope":{"name":"test"},'
    b'"spans":[{"traceId":"00000000000000000000000000000001","spanId":"0000000000000001",'
    b'"name":"test","startTimeUnixNano":"1000000000","endTimeUnixNano":"1000000001"}]}]}]}'
)
_TEMPO_NS = re.compile(rb'(TimeUnixNano": ")(\d+)(")')
_LOKI_NS = re.compile(rb'(\[")(\d{19})(", ")')
_LINE_TS = re.compile(
    rb'(\\"timestamp\\": \\")(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(\.\d{3}Z)'
)


def build_seed(clock: str = SEED_CLOCK) -> dict[str, dict]:
    """Run the generator at `clock` and keep what it would have sent, one entry per consumer."""
    os.environ["O11Y_SCENARIO_TIME_ISO"] = clock
    random.seed(42)
    g.DUMP = dumped = {"traces": [], "logs": []}
    g.generate_all_data()

    groups: list[tuple[bytes, bytes]] = []  # (epoch, lines without the trailing timestamp)
    head: list[bytes] = []
    with open(g.METRICS_FILE, "rb") as f:
        for line in f:
            if line.startswith(b"#"):
                if line.startswith(b"# TYPE"):
                    head.append(line)
                continue
            body, _, ts = line.rstrip(b"\n").rpartition(b" ")
            if groups and groups[-1][0] == ts:
                groups[-1] = (ts, groups[-1][1] + b"\n" + body)
            else:
                groups.append((ts, body))
    os.remove(g.METRICS_FILE)
    meta = dumped["meta"]
    return {
        "metrics": {"meta": meta, "head": b"".join(head), "metrics": groups},
        "ingest": {"meta": meta, "traces": dumped["traces"], "logs": dumped["logs"]},
    }


def dump() -> None:
    os.makedirs(SEED_DIR, exist_ok=True)
    # One file per consumer: each boot-time process unpickles only what it replays.
    for kind, seed in build_seed().items():
        with open(f"{SEED_DIR}/{kind}.pkl", "wb") as f:
            pickle.dump(seed, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(kind: str) -> tuple[dict, int, datetime]:
    with open(f"{SEED_DIR}/{kind}.pkl", "rb") as f:
        seed = pickle.load(f)
    clock = g.data_end_time_utc()
    seed_clock = datetime.fromisoformat(seed["meta"]["end_time"].replace("Z", "+00:00"))
    return seed, int((clock - seed_clock).total_seconds()), clock


def shift_metrics(seed: dict, delta_s: int) -> bytes:
    parts = [seed["head"]]
    for ts, body in seed["metrics"]:
        suffix = b" %d.0\n" % (int(float(ts)) + delta_s)
        parts.append(body.replace(b"\n", suffix) + suffix)
    parts.append(b"# EOF\n")
    return b"".join(parts)


def shift_trace_body(body: bytes, delta_ns: int) -> bytes:
    return _TEMPO_NS.sub(lambda m: m[1] + b"%d" % (int(m[2]) + delta_ns) + m[3], body)


def shift_log_body(body: bytes, delta_s: int) -> bytes:
    cache: dict[bytes, bytes] = {}

    def line_ts(m: re.Match) -> bytes:
        key = m.group(0)[: -len(m[8])]
        if key not in cache:
            fields = tuple(int(m[i]) for i in range(2, 8))
            shifted = time.gmtime(calendar.timegm(fields) + delta_s)
            cache[key] = m[1] + time.strftime("%Y-%m-%dT%H:%M:%S", shifted).encode()
        return cache[key] + m[8]

    body = _LOKI_NS.sub(lambda m: m[1] + b"%d" % (int(m[2]) + delta_s * 10**9) + m[3], body)
    return _LINE_TS.sub(line_ts, body)


def import_metrics() -> None:
    seed, delta_s, clock = load("metrics")
    with open(g.METRICS_FILE, "wb") as f:
        f.write(shift_metrics(seed, delta_s))
    subprocess.run(
        [
            "promtool",
            "tsdb",
            "create-blocks-from",
            "--max-block-duration=54h",
            "openmetrics",
            g.METRICS_FILE,
            g.PROMETHEUS_DATA_DIR,
        ],
        check=True,
        capture_output=True,
    )
    os.remove(g.METRICS_FILE)
    with open("/tmp/env_timestamp", "w") as f:
        f.write(clock.strftime("%Y-%m-%dT%H:%M:%SZ"))
    g.log("  Metrics import complete")


def _post(url: str, body: bytes, timeout: float = 60) -> None:
    """POST, retrying while the service is still starting (refused, 503)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=30)
            return
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


def wait_ok(url: str, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


def push_traces(seed: dict, delta_s: int) -> None:
    # Shift while the service is still starting, so the bodies are ready when it is.
    bodies = [shift_trace_body(b, delta_s * 10**9) for b in seed["traces"]]
    wait_ok(f"{g.TEMPO_QUERY_URL}/ready")
    # The 1.1 entrypoint readied Tempo's OTLP port by POSTing this span (a 1970 trace that
    # shows up in tag discovery); keep sending it so the Tempo data stays the same.
    _post(f"{g.TEMPO_URL}/v1/traces", OTLP_PROBE)
    for body in bodies:
        _post(f"{g.TEMPO_URL}/v1/traces", body)


def push_logs(seed: dict, delta_s: int) -> None:
    bodies = [shift_log_body(b, delta_s) for b in seed["logs"]]
    wait_ok(f"{g.LOKI_URL}/ready")
    for body in bodies:
        _post(f"{g.LOKI_URL}/loki/api/v1/push", body)


def ingest() -> None:
    seed, delta_s, clock = load("ingest")
    with ThreadPoolExecutor(2) as pool:
        for pushed in [
            pool.submit(push_traces, seed, delta_s),
            pool.submit(push_logs, seed, delta_s),
        ]:
            pushed.result()
    meta = seed["meta"]
    g.log("  Waiting for Loki to serve every line...")
    g.wait_for_loki_complete(clock, meta["logs"])
    g.log("  Waiting for Tempo search to become ready...")
    g.wait_for_tempo_searchable(clock, expected_traces=meta["traces"])


if __name__ == "__main__":
    {"dump": dump, "metrics": import_metrics, "ingest": ingest}[sys.argv[1]]()
