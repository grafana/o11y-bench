from o11y_bench import config


def test_provider_variants_include_selected_low_reasoning_models() -> None:
    expected = {
        "anthropic": {
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        },
        "openai": {
            "gpt-5.4-2026-03-05",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
        },
        "google": {
            "gemini-3.1-flash-lite-preview",
            "gemini-3.1-pro-preview",
        },
    }

    for provider, expected_models in expected.items():
        low_effort_models = {
            model
            for model, reasoning_effort in config.provider_variants(provider)
            if reasoning_effort == "low"
        }
        assert expected_models.issubset(low_effort_models)
