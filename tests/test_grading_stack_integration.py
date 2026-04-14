"""Optional live-stack checks (O11Y_GRADING_SMOKE=1 + o11y-stack on localhost)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    os.environ.get("O11Y_GRADING_SMOKE", "").lower() not in ("1", "true", "yes"),
    reason="Set O11Y_GRADING_SMOKE=1 with stack reachable at default localhost ports",
)
def test_grading_stack_smoke_script_exits_zero() -> None:
    script = ROOT / "scripts" / "grading_stack_smoke.py"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "grading")}
    r = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        pytest.fail(
            f"grading_stack_smoke.py failed ({r.returncode})\n"
            f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
        )
