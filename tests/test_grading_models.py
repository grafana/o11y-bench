import importlib.util
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRADING_DIR = ROOT / "grading"


def load_grading_module(module_name: str):
    path = GRADING_DIR / f"{module_name}.py"
    sys.path.insert(0, str(GRADING_DIR))
    try:
        unique_name = f"test_{module_name}_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(unique_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_transcript_to_text_respects_budget_and_keeps_key_context() -> None:
    models = load_grading_module("models")
    transcript = models.Transcript(
        messages=[
            models.Message(role="user", content="Investigate the checkout service failure."),
            models.Message(
                role="assistant",
                content="Working through logs and traces." + " detail" * 500,
                thinking_content="reasoning" * 300,
                tool_calls=[
                    models.ToolCall(
                        id="tool-1",
                        name="query_logs",
                        arguments={"query": "error" * 200},
                    )
                ],
            ),
            models.Message(
                role="tool",
                tool_results=[
                    models.ToolResult(
                        tool_call_id="tool-1",
                        content="log-line" * 1000,
                    )
                ],
            ),
        ]
    )

    rendered = transcript.to_text(max_chars=500)

    assert len(rendered) <= 500
    assert "Investigate the checkout service failure." in rendered
    assert "query_logs" in rendered
