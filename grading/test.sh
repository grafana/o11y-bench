#!/bin/bash
# o11y-bench verifier — runs deterministic checks + LLM-as-judge grading
set -e

uv run /tests/verifier.py \
    --problem /tests/problem.yaml \
    --logs /logs/agent/ \
    --output /logs/verifier/
