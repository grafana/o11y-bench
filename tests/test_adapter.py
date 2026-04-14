from scripts import sync_tasks


def test_generate_task_creates_harbor_layout(tmp_path):
    problem_path = tmp_path / "problem.yaml"
    problem_path.write_text(
        "\n".join(
            [
                "id: sample-task",
                "category: prometheus_query",
                "tags: [promql]",
                "statement: |",
                "  Find the error rate for checkout.",
            ]
        )
    )

    output_dir = tmp_path / "tasks"
    task_id = sync_tasks.generate_task(problem_path, output_dir)

    task_dir = output_dir / task_id
    assert task_id == "sample-task"
    assert (task_dir / "instruction.md").read_text() == "Find the error rate for checkout.\n"
    assert (task_dir / "task.toml").exists()
    assert (task_dir / "environment" / "Dockerfile").exists()
    assert (task_dir / "tests" / "problem.yaml").exists()
    assert (task_dir / "tests" / "verifier.py").exists()


def test_generate_task_toml_keeps_category_and_fixed_timeout():
    task_toml = sync_tasks.generate_task_toml(
        {
            "category": "grafana_api",
            "tags": ["api"],
        },
        "grafana_api",
    )

    assert 'category = "grafana_api"' in task_toml
    assert "timeout_sec = 600.0" in task_toml
    assert 'url = "http://o11y-stack:8080/mcp"' in task_toml
