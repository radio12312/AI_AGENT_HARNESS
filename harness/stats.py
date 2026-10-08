"""Small statistics helpers."""
from __future__ import annotations

import math


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def reference_cost(prompt_tokens: int, completion_tokens: int, pricing: dict) -> float:
    return (prompt_tokens * float(pricing.get("input_per_1m_usd", 0))
            + completion_tokens * float(pricing.get("output_per_1m_usd", 0))) / 1_000_000
