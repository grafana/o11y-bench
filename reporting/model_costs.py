from typing import Final

# TODO: remove this after litellm supports cost estimation for gpt-5.4-mini and gpt-5.4-nano

USD_PER_1M_TOKENS: Final[dict[str, tuple[float, float, float]]] = {
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-mini-2026-03-17": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4-nano-2026-03-17": (0.20, 0.02, 1.25),
}


def normalize_model_name(model_name: str) -> str:
    if "/" in model_name:
        return model_name.rsplit("/", 1)[-1]
    return model_name


def estimate_cost_usd(
    model_name: str, n_input_tokens: int, n_cache_tokens: int, n_output_tokens: int
) -> float | None:
    rates = USD_PER_1M_TOKENS.get(normalize_model_name(model_name))
    if rates is None:
        return None

    input_rate, cached_input_rate, output_rate = rates
    uncached_input_tokens = max(0, n_input_tokens - n_cache_tokens)
    return (
        (uncached_input_tokens * input_rate)
        + (n_cache_tokens * cached_input_rate)
        + (n_output_tokens * output_rate)
    ) / 1_000_000
