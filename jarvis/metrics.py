"""Evaluation metrics shared by the walk-forward exam, the ledger and the scorecard."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from jarvis.config import COST_BPS, TOP_N

SCORE_COLUMNS = ["top_excess", "rank_ic", "n"]


def upper_threshold(vix_percentile: float) -> float:
    """Minimum probability for a recommendation; rises with the VIX percentile."""
    if vix_percentile is None or not math.isfinite(vix_percentile):
        return 0.52
    fraction = min(max((float(vix_percentile) - 25.0) / 50.0, 0.0), 1.0)
    return 0.52 + 0.13 * fraction


def _rank_ic(group: pd.DataFrame) -> float:
    if len(group) < 3:
        return float("nan")
    probabilities = group["prob"].rank()
    outcomes = group["fwd_excess"].rank()
    if probabilities.nunique() < 2 or outcomes.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(probabilities, outcomes)[0, 1])


def daily_scores(
    predictions: pd.DataFrame, top_n: int = TOP_N, cost_bps: float = COST_BPS
) -> pd.DataFrame:
    """Per day: mean excess return of the top picks (net of cost) and rank IC."""
    labelled = predictions.dropna(subset=["fwd_excess", "prob"])
    records = {}
    for day, group in labelled.groupby("date", sort=True):
        ordered = group.sort_values(["prob", "ticker"], ascending=[False, True])
        top = ordered.head(top_n)
        records[day] = {
            "top_excess": float(top["fwd_excess"].mean()) - cost_bps / 10_000.0,
            "rank_ic": _rank_ic(group),
            "n": len(group),
        }
    if not records:
        return pd.DataFrame(columns=SCORE_COLUMNS, index=pd.DatetimeIndex([], name="date"))
    scores = pd.DataFrame.from_dict(records, orient="index")[SCORE_COLUMNS]
    scores.index.name = "date"
    return scores


def _mean(series: pd.Series) -> float | None:
    values = series.dropna()
    return None if values.empty else float(values.mean())


def summarize(scores: pd.DataFrame) -> dict:
    if scores.empty:
        return {"days": 0, "mean_top_excess": None, "mean_rank_ic": None, "hit_rate": None}
    return {
        "days": int(len(scores)),
        "mean_top_excess": _mean(scores["top_excess"]),
        "mean_rank_ic": _mean(scores["rank_ic"]),
        "hit_rate": float((scores["top_excess"] > 0).mean()),
    }
