from datetime import datetime

import pytest
from o11y_stack import generate_data, seed

SEED_CLOCK = "2026-10-20T00:00:00Z"
# A Monday night, a Saturday noon in another month, and a leap day: each shifts the seed by a
# different whole number of minutes across day, month and weekday boundaries.
CLOCKS = ["2026-10-19T23:01:00Z", "2026-11-28T12:30:00Z", "2028-02-29T00:17:00Z"]


@pytest.fixture(scope="module")
def build(tmp_path_factory: pytest.TempPathFactory):
    metrics_path = str(tmp_path_factory.mktemp("seed") / "metrics.txt")
    cache: dict[str, dict] = {}

    def run(clock: str) -> dict:
        if clock not in cache:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(generate_data, "METRICS_FILE", metrics_path)
                cache[clock] = seed.build_seed(clock)
        return cache[clock]

    return run


def shifted(clock: str, seed_clock: str, build) -> tuple[list[bytes], list[bytes], bytes]:
    delta_s = int(
        (
            datetime.fromisoformat(clock.replace("Z", "+00:00"))
            - datetime.fromisoformat(seed_clock.replace("Z", "+00:00"))
        ).total_seconds()
    )
    base = build(seed_clock)
    return (
        [seed.shift_trace_body(b, delta_s * 10**9) for b in base["ingest"]["traces"]],
        [seed.shift_log_body(b, delta_s) for b in base["ingest"]["logs"]],
        seed.shift_metrics(base["metrics"], delta_s),
    )


@pytest.mark.parametrize("clock", CLOCKS)
def test_the_seed_shifted_to_a_clock_is_what_the_generator_sends_at_that_clock(
    clock: str, build
) -> None:
    # Boot replays the seed instead of generating, so every body must come out byte for byte.
    traces, logs, metrics = shifted(clock, SEED_CLOCK, build)
    fresh = build(clock)

    assert traces == fresh["ingest"]["traces"]
    assert logs == fresh["ingest"]["logs"]
    assert metrics == seed.shift_metrics(fresh["metrics"], 0)
