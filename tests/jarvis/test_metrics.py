import numpy as np
import pandas as pd
import pytest

from jarvis.metrics import daily_scores, summarize, upper_threshold

DAY = pd.Timestamp("2026-11-13")


def predictions(probs, excess, day=DAY):
    return pd.DataFrame(
        {
            "date": day,
            "ticker": [f"T{i}" for i in range(len(probs))],
            "prob": probs,
            "fwd_excess": excess,
        }
    )


def test_upper_threshold_follows_vix_percentile():
    assert upper_threshold(0.0) == pytest.approx(0.52)
    assert upper_threshold(25.0) == pytest.approx(0.52)
    assert upper_threshold(50.0) == pytest.approx(0.585)
    assert upper_threshold(75.0) == pytest.approx(0.65)
    assert upper_threshold(100.0) == pytest.approx(0.65)
    assert upper_threshold(float("nan")) == pytest.approx(0.52)


def test_top_excess_subtracts_cost_once():
    frame = predictions([0.9, 0.8, 0.1, 0.2], [0.04, 0.02, -0.05, -0.01])
    scores = daily_scores(frame, top_n=2, cost_bps=20)
    assert scores.loc[DAY, "top_excess"] == pytest.approx(0.03 - 0.002)
    assert scores.loc[DAY, "n"] == 4


def test_top_n_larger_than_available_uses_all():
    frame = predictions([0.9, 0.8, 0.7], [0.03, 0.00, -0.03])
    scores = daily_scores(frame, top_n=5, cost_bps=0)
    assert scores.loc[DAY, "top_excess"] == pytest.approx(0.0)


def test_rank_ic_perfect_and_inverse():
    excess = [0.01, 0.02, 0.03, 0.04]
    assert daily_scores(predictions([0.1, 0.2, 0.3, 0.4], excess)).loc[DAY, "rank_ic"] == pytest.approx(1.0)
    assert daily_scores(predictions([0.4, 0.3, 0.2, 0.1], excess)).loc[DAY, "rank_ic"] == pytest.approx(-1.0)


def test_rank_ic_nan_below_three():
    scores = daily_scores(predictions([0.6, 0.4], [0.01, -0.01]))
    assert np.isnan(scores.loc[DAY, "rank_ic"])


def test_unlabelled_rows_are_ignored():
    frame = predictions([0.9, 0.8, 0.7, 0.6], [np.nan, 0.02, 0.01, 0.00])
    scores = daily_scores(frame, top_n=1, cost_bps=0)
    assert scores.loc[DAY, "top_excess"] == pytest.approx(0.02)
    assert scores.loc[DAY, "n"] == 3


def test_multiple_days_and_summary():
    first = predictions([0.9, 0.5, 0.1], [0.05, 0.0, -0.05], DAY)
    second = predictions([0.9, 0.5, 0.1], [-0.05, 0.0, 0.05], DAY + pd.Timedelta(days=3))
    scores = daily_scores(pd.concat([first, second]), top_n=1, cost_bps=0)
    assert list(scores["top_excess"]) == pytest.approx([0.05, -0.05])
    summary = summarize(scores)
    assert summary == {
        "days": 2,
        "mean_top_excess": pytest.approx(0.0),
        "mean_rank_ic": pytest.approx(0.0),
        "hit_rate": pytest.approx(0.5),
    }


def test_summary_of_empty_scores():
    summary = summarize(daily_scores(predictions([], [])))
    assert summary == {"days": 0, "mean_top_excess": None, "mean_rank_ic": None, "hit_rate": None}
