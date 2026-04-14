"""Behavior checks for MCP → LLM tool schema shaping in agent_runner."""

from agents.agent_runner import relax_mcp_tool_input_schema_for_llm


def test_relax_adds_additional_properties_to_bare_object_fields() -> None:
    schema = {
        "type": "object",
        "properties": {
            "dashboard": {"type": "object", "description": "Full dashboard JSON"},
            "uid": {"type": "string"},
        },
    }
    out = relax_mcp_tool_input_schema_for_llm(schema)
    assert out["properties"]["dashboard"].get("additionalProperties") is True
    assert "additionalProperties" not in out["properties"]["uid"]


def test_relax_does_not_mutate_original() -> None:
    schema = {
        "type": "object",
        "properties": {"dashboard": {"type": "object"}},
    }
    relax_mcp_tool_input_schema_for_llm(schema)
    assert "additionalProperties" not in schema["properties"]["dashboard"]
