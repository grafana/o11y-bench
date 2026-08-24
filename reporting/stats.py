"""Statistical helpers for the comparison report.

Two tests, both implemented against the standard library only:

- :func:`wilson_interval` -- a binomial proportion confidence interval, for showing how
  precisely a pass rate is pinned down by a fixed number of tasks.
- :func:`mcnemar_exact_p` -- an exact two-sided paired test, for asking whether two runs
  differ by more than chance on the same task set.

Neither ``scipy`` nor ``statsmodels`` is a dependency of this project, so the usual
``proportion_confint`` / ``contingency_tables.mcnemar`` are unavailable. Both functions
below are small enough to implement directly, and doing so keeps the dependency set
unchanged.
"""

import math
from collections.abc import Mapping

# Two-sided 95% normal quantile: the z with P(-z < Z < z) = 0.95.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Return the Wilson score interval for ``successes`` out of ``trials``.

    Preferred over the textbook normal (Wald) interval because Wald collapses to zero
    width at 0% and 100% -- precisely the proportions a small benchmark produces on
    tasks every model passes or every model fails.

    Returns ``(0.0, 1.0)`` for zero trials: no observation, no information.
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"invalid counts: {successes} of {trials}")
    if trials == 0:
        return (0.0, 1.0)

    n = float(trials)
    p = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_exact_p(only_first: int, only_second: int) -> float:
    """Return the two-sided p-value of an exact McNemar test.

    ``only_first`` and ``only_second`` are the discordant counts: tasks the first run
    passed and the second failed, and the reverse. Tasks both runs passed, or both
    failed, say nothing about which run is better and are excluded by construction --
    which is why this is a paired test over tasks rather than a comparison of two rates.

    Under the null hypothesis every disagreement is a coin flip, so the number falling
    one way is Binomial(n, 0.5) with ``n = only_first + only_second``. The p-value is the
    smaller tail doubled, clamped at 1.0.

    Returns 1.0 when nothing is discordant: the two runs passed exactly the same tasks.

    Exact by construction -- no chi-squared approximation, which is the right call at the
    disagreement counts a small benchmark produces.
    """
    if only_first < 0 or only_second < 0:
        raise ValueError(f"invalid discordant counts: {only_first}, {only_second}")

    n = only_first + only_second
    if n == 0:
        return 1.0

    # n is bounded by the number of tasks, so the exact 2**n term stays small.
    tail = sum(math.comb(n, i) for i in range(min(only_first, only_second) + 1))
    return min(1.0, 2.0 * tail / (1 << n))


def discordant_counts(first: Mapping[str, bool], second: Mapping[str, bool]) -> tuple[int, int]:
    """Count tasks that exactly one of two runs passed.

    Returns ``(only_first, only_second)`` over the tasks present in both mappings; tasks
    missing from either side are skipped, since an unpaired task cannot be compared.
    """
    shared = first.keys() & second.keys()
    only_first = sum(1 for name in shared if first[name] and not second[name])
    only_second = sum(1 for name in shared if second[name] and not first[name])
    return (only_first, only_second)
