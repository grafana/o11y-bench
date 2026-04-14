def normalize_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    total = sum(raw_weights.values())
    if total <= 0:
        return {key: 0.0 for key in raw_weights}

    normalized = {key: value / total for key, value in raw_weights.items()}
    weight_sum = sum(normalized.values())
    if weight_sum != 1.0 and weight_sum > 0:
        last_key = list(normalized)[-1]
        normalized[last_key] += 1.0 - weight_sum
    return normalized


def calculate_score(subscores: dict[str, float], weights: dict[str, float]) -> float:
    weighted_sum = sum(subscores[key] * weights[key] for key in subscores if key in weights)
    return max(0.0, min(1.0, weighted_sum))
