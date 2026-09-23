"""Tests for reporting.stats.

Expected values were produced once with statsmodels 0.14.6 and hardcoded here:

    proportion_confint(k, n, alpha=0.05, method="wilson")
    contingency_tables.mcnemar([[0, b], [c, 0]], exact=True, correction=False).pvalue

statsmodels is not a dependency of this project, so it is not imported by these tests --
it was used only to generate the constants below.
"""

import math

import pytest

from reporting.stats import Z_95, discordant_counts, mcnemar_exact_p, wilson_interval


def close(actual, expected):
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


# --- Wilson interval -------------------------------------------------------------


@pytest.mark.parametrize(
    ("successes", "trials", "low", "high"),
    [
        # The two cases the Wald interval gets wrong: it reports zero width at both.
        (0, 10, 0.0, 0.27753279986288926),
        (10, 10, 0.7224672001371106, 1.0),
        # Realistic shapes for a benchmark with few tasks.
        (32, 63, 0.38762886274954733, 0.6273319118601534),
        (50, 63, 0.6783018567138666, 0.8752468038450618),
        (46, 63, 0.6097009078044822, 0.82416155587684),
        (1, 63, 0.002807494213869284, 0.08458525459438399),
    ],
)
def test_wilson_matches_statsmodels(successes, trials, low, high):
    got_low, got_high = wilson_interval(successes, trials)
    assert close(got_low, low)
    assert close(got_high, high)


def test_wilson_is_symmetric_under_complement():
    """Swapping successes for failures mirrors the interval about 0.5."""
    low, high = wilson_interval(31, 63)
    mirror_low, mirror_high = wilson_interval(63 - 31, 63)
    assert close(low, 1.0 - mirror_high)
    assert close(high, 1.0 - mirror_low)


def test_wilson_with_no_trials_spans_everything():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_stays_inside_the_unit_interval():
    for trials in range(1, 40):
        for successes in range(trials + 1):
            low, high = wilson_interval(successes, trials)
            assert 0.0 <= low <= high <= 1.0


def test_wilson_narrows_as_trials_grow():
    """Same proportion, more tasks, tighter interval."""
    small_low, small_high = wilson_interval(5, 10)
    large_low, large_high = wilson_interval(50, 100)
    assert (large_high - large_low) < (small_high - small_low)


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
    with pytest.raises(ValueError):
        wilson_interval(1, -10)


def test_z_95_is_the_two_sided_95_percent_quantile():
    assert close(Z_95, 1.959963984540054)


# --- Exact McNemar ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("only_first", "only_second", "p_value"),
    [
        (4, 0, 0.125),
        (9, 1, 0.021484375),
        (13, 3, 0.021270751953125),
        (18, 6, 0.022655844688415527),
        (7, 2, 0.1796875),
    ],
)
def test_mcnemar_matches_statsmodels(only_first, only_second, p_value):
    assert close(mcnemar_exact_p(only_first, only_second), p_value)


def test_mcnemar_with_no_disagreement_is_one():
    """Two runs that passed exactly the same tasks are indistinguishable, not different."""
    assert mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_doubling_is_clamped_at_one():
    """A one-all split doubles to 1.5 before clamping; a p-value above 1.0 is nonsense."""
    assert mcnemar_exact_p(1, 1) == 1.0


def test_mcnemar_is_symmetric():
    """The test asks whether the runs differ, not which one is ahead."""
    assert close(mcnemar_exact_p(9, 1), mcnemar_exact_p(1, 9))


def test_mcnemar_needs_a_net_gap_of_six_on_a_clean_sweep():
    """The floor on separability, and the reason a naive gate would empty the board.

    On a perfect sweep -- every task the loser passed, the winner passed too -- a net gap
    of five still cannot reach p < 0.05, and six is the first that can. Leaders separated
    by a handful of tasks therefore cannot be told apart on a benchmark this size.
    """
    assert mcnemar_exact_p(4, 0) == 0.125
    assert mcnemar_exact_p(5, 0) == 0.0625
    assert mcnemar_exact_p(6, 0) == 0.03125
    assert mcnemar_exact_p(5, 0) > 0.05
    assert mcnemar_exact_p(6, 0) < 0.05


def test_mcnemar_rejects_negative_counts():
    with pytest.raises(ValueError):
        mcnemar_exact_p(-1, 3)
    with pytest.raises(ValueError):
        mcnemar_exact_p(3, -1)


# --- Discordant counts -----------------------------------------------------------


def test_discordant_counts_ignores_agreement():
    first = {"a": True, "b": True, "c": False, "d": False}
    second = {"a": True, "b": False, "c": True, "d": False}
    assert discordant_counts(first, second) == (1, 1)


def test_discordant_counts_skips_unpaired_tasks():
    """A task only one run attempted cannot be paired, so it carries no vote."""
    first = {"a": True, "only_in_first": True}
    second = {"a": False, "only_in_second": True}
    assert discordant_counts(first, second) == (1, 0)


def test_discordant_counts_reverses_with_argument_order():
    first = {"a": True, "b": False, "c": True}
    second = {"a": False, "b": True, "c": True}
    assert discordant_counts(first, second) == (1, 1)
    assert discordant_counts(second, first) == (1, 1)


def test_discordant_counts_of_identical_runs_is_empty():
    run = {"a": True, "b": False, "c": True}
    assert discordant_counts(run, dict(run)) == (0, 0)
