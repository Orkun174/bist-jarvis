import pytest
import numpy as np
import pandas as pd

from jarvis.gate import compare_recipes, drift_check, health_check

DAYS = pd.bdate_range("2026-08-03", periods=60)


def scores(top_excess, rank_ic, days=DAYS):
    return pd.DataFrame({"top_excess": top_excess, "rank_ic": rank_ic, "n": 100}, index=days[: len(top_excess)])


def noisy(mean, seed, n=60, scale=0.01):
    return np.random.default_rng(seed).normal(mean, scale, n)


def test_clearly_better_challenger_is_promoted():
    champion = scores(noisy(0.000, 1), noisy(0.00, 2))
    challenger = scores(noisy(0.010, 3), noisy(0.02, 4))
    result = compare_recipes(champion, challenger)
    assert result["promote"] is True
    assert result["days"] == 60
    assert result["top_excess_diff"] > 0 and result["rank_ic_diff"] > 0
    assert result["p_top_excess"] >= 0.8 and result["p_rank_ic"] >= 0.8


def test_equal_recipes_are_not_promoted():
    same = scores(noisy(0.002, 1), noisy(0.01, 2))
    result = compare_recipes(same, same.copy())
    assert result["promote"] is False


def test_better_on_one_metric_only_is_not_promoted():
    champion = scores(noisy(0.000, 1), noisy(0.02, 2))
    challenger = scores(noisy(0.010, 3), noisy(0.00, 4))
    assert compare_recipes(champion, challenger)["promote"] is False


def test_worse_challenger_is_not_promoted():
    champion = scores(noisy(0.010, 1), noisy(0.02, 2))
    challenger = scores(noisy(0.000, 3), noisy(0.00, 4))
    assert compare_recipes(champion, challenger)["promote"] is False


def test_too_few_days_not_promoted():
    champion = scores(noisy(0.0, 1, 20), noisy(0.0, 2, 20))
    challenger = scores(noisy(0.05, 3, 20), noisy(0.05, 4, 20))
    result = compare_recipes(champion, challenger)
    assert result["promote"] is False and result["reason"] == "too_few_days"


def test_only_common_days_are_compared():
    champion = scores(noisy(0.0, 1), noisy(0.0, 2))
    challenger = scores(noisy(0.01, 3, 40), noisy(0.02, 4, 40))
    assert compare_recipes(champion, challenger)["days"] == 40


def test_bootstrap_is_deterministic():
    champion = scores(noisy(0.000, 1), noisy(0.00, 2))
    challenger = scores(noisy(0.002, 3), noisy(0.004, 4))
    assert compare_recipes(champion, challenger) == compare_recipes(champion, challenger)


def test_health_positive_ic_passes():
    result = health_check(scores(noisy(0.0, 1), noisy(0.03, 2)))
    assert result["ok"] is True and result["mean_rank_ic"] > 0 and result["days"] == 60


def test_health_negative_ic_fails():
    days = pd.bdate_range("2026-05-01", periods=120)
    result = health_check(scores(noisy(0.0, 1, 120), noisy(-0.03, 2, 120), days))
    assert result["ok"] is False


def test_health_short_history_passes_with_reason():
    result = health_check(scores(noisy(0.0, 1, 30), noisy(-0.05, 2, 30)))
    assert result["ok"] is True and result["reason"] == "insufficient_history"


def series(values):
    return pd.Series(values, index=[f"T{i:02d}" for i in range(len(values))])


def test_drift_similar_ranking_passes():
    old = series(np.linspace(0.3, 0.7, 30))
    new = old + np.random.default_rng(0).normal(0, 0.01, 30)
    result = drift_check(new, old)
    assert result["ok"] is True and result["rank_corr"] > 0.9


def test_drift_uncorrelated_fails():
    old = series(np.linspace(0.3, 0.7, 30))
    new = series(np.linspace(0.7, 0.3, 30))
    result = drift_check(new, old)
    assert result["ok"] is False and result["rank_corr"] < 0


def test_drift_without_previous_model_passes():
    result = drift_check(series(np.linspace(0.3, 0.7, 30)), None)
    assert result == {"ok": True, "reason": "no_previous_model"}


def test_drift_too_few_names_passes():
    result = drift_check(series([0.4, 0.6]), series([0.6, 0.4]))
    assert result["ok"] is True and result["reason"] == "too_few_names"


def test_confirmation_needs_both_metrics_ahead_over_the_long_window():
    import numpy as np
    import pandas as pd
    from jarvis.gate import confirm_recipes

    days = pd.bdate_range("2026-01-01", periods=100)
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"top_excess": rng.normal(0, 0.03, 100), "rank_ic": rng.normal(0.05, 0.1, 100)}, index=days)
    better = base + [0.002, 0.02]
    assert confirm_recipes(base, better)["ok"] is True
    lucky_ranker = base + [-0.002, 0.02]           # ranks better, picks worse
    verdict = confirm_recipes(base, lucky_ranker)
    assert verdict["ok"] is False and verdict["top_excess_diff"] == pytest.approx(-0.002)
    assert confirm_recipes(base.head(10), better.head(10)) == {"ok": False, "days": 10, "reason": "too_few_days"}
