#!/usr/bin/env python3
"""Generate Harbor task directories from o11y-bench task specs."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from grading import models

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPECS_ROOT = ROOT / "tasks-spec"
MATERIALIZED_TASKS_ROOT = ROOT / ".cache" / "tasks"

DEFAULT_AGENT_TIMEOUT_SEC = 600


def iter_spec_files(specs_path: Path) -> list[Path]:
    if specs_path.is_file() and specs_path.suffix == ".yaml":
        return [specs_path]
    if not specs_path.is_dir():
        return []
    return sorted(spec_file for spec_file in specs_path.rglob("*.yaml"))


def task_spec_ids(specs_path: Path) -> list[str]:
    ids: list[str] = []
    for spec_file in iter_spec_files(specs_path):
        with open(spec_file) as f:
            data = yaml.safe_load(f)
        ids.append(models.Problem.model_validate(data).id)
    return sorted(ids)


def generate_task_toml(spec: dict[str, Any], category: str) -> str:
    tags = spec.get("tags", [])
    tags_str = ", ".join(f'"{t}"' for t in tags)

    return f"""version = "1.0"

[metadata]
category = "{category}"
tags = [{tags_str}]
difficulty = "medium"

[verifier]
timeout_sec = 300.0

[verifier.env]
ANTHROPIC_API_KEY = "${{ANTHROPIC_API_KEY}}"
GRADING_MODEL = "claude-haiku-4-5-20251001"

[agent]
timeout_sec = {DEFAULT_AGENT_TIMEOUT_SEC}.0

[environment]
build_timeout_sec = 600.0
docker_image = "o11y-bench-main:latest"
cpus = 1
memory_mb = 2048
storage_mb = 10240
allow_internet = true

[[environment.mcp_servers]]
name = "mcp-grafana"
transport = "streamable-http"
url = "http://o11y-stack:8080/mcp"
"""


def generate_task(spec_path: Path, output_dir: Path) -> str:
    with open(spec_path) as f:
        spec = models.Problem.model_validate(yaml.safe_load(f)).model_dump()

    task_id = str(spec["id"])
    task_dir = output_dir / task_id

    if task_dir.exists():
        shutil.rmtree(task_dir)

    task_dir.mkdir(parents=True)
    (task_dir / "environment").mkdir()
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "grading").mkdir()

    (task_dir / "instruction.md").write_text(spec["statement"].strip() + "\n")
    (task_dir / "task.toml").write_text(generate_task_toml(spec, spec["category"]))

    for fname in ("Dockerfile", "docker-compose.yaml"):
        shutil.copy2(ROOT / "environment" / fname, task_dir / "environment" / fname)

    setup_payload = {key: value for key, value in spec.items() if key.startswith("setup_")}
    (task_dir / "environment" / "setup.json").write_text(json.dumps(setup_payload, indent=2) + "\n")

    shutil.copy2(spec_path, task_dir / "tests" / "problem.yaml")

    for path in sorted((ROOT / "grading").glob("*.py")):
        shutil.copy2(path, task_dir / "tests" / "grading" / path.name)

    shutil.copy2(
        ROOT / "grading" / "grader_prompt.txt", task_dir / "tests" / "grading" / "grader_prompt.txt"
    )
    shutil.copy2(ROOT / "grading" / "verifier_launcher.py", task_dir / "tests" / "verifier.py")
    shutil.copy2(ROOT / "grading" / "test.sh", task_dir / "tests" / "test.sh")

    return task_id


def sync_specs_to_output(specs_path: Path, output_dir: Path) -> int:
    spec_files = iter_spec_files(specs_path)
    if not spec_files:
        print(f"No YAML files found in {specs_path}")
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    synced_task_ids = {generate_task(spec_file, output_dir) for spec_file in spec_files}

    for existing_task_dir in output_dir.iterdir():
        if existing_task_dir.is_dir() and existing_task_dir.name not in synced_task_ids:
            shutil.rmtree(existing_task_dir)

    return len(spec_files)


def is_materialized_tasks_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            if (child / "task.toml").exists() and (child / "tests" / "problem.yaml").exists():
                return True
    return False


def materialized_output_dir_for_specs(specs_path: Path) -> Path:
    resolved = specs_path.resolve()
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:12]
    name = resolved.name.replace("_", "-") or "specs"
    return MATERIALIZED_TASKS_ROOT / f"{name}-{digest}"


def materialize_specs_path(specs_path: Path) -> Path:
    output_dir = materialized_output_dir_for_specs(specs_path)
    sync_specs_to_output(specs_path, output_dir)
    return output_dir


def normalize_source_path(path: Path | None) -> Path:
    if path is None:
        return DEFAULT_SPECS_ROOT
    return path if path.is_absolute() else (ROOT / path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Harbor tasks from o11y-bench task specs")
    parser.add_argument(
        "--path",
        type=Path,
        help="Spec source directory. Defaults to tasks-spec/.",
    )
    parser.add_argument("--output-dir", default="tasks", help="Output Harbor tasks directory")
    parser.add_argument(
        "--list-ids",
        action="store_true",
        help="Print task ids from the selected spec source and exit without syncing.",
    )
    args = parser.parse_args()

    source_path = normalize_source_path(args.path)
    output_dir = Path(args.output_dir)

    if args.list_ids:
        if not source_path.exists():
            print(f"Specs directory not found: {source_path}", file=sys.stderr)
            raise SystemExit(1)
        for tid in task_spec_ids(source_path):
            print(tid)
        return

    count = sync_specs_to_output(source_path, output_dir)
    print(f"Synced task specs to {output_dir}/ ({count} tasks)")


if __name__ == "__main__":
    main()
