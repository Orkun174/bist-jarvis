"""Evaluation metrics shared by the walk-forward exam, the ledger and the scorecard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from jarvis.config import COST_BPS, TOP_N

SCORE_COLUMNS = ["top_excess", "rank_ic", "n", "universe_excess", "top_vs_universe", "bottom_excess"]


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
    """Per day: mean excess return of the top picks (net of cost) and rank IC.

    `universe_excess` is the plain average over every ranked stock that day,
    `top_vs_universe` the top picks minus that average, and `bottom_excess` the
    average of the lowest-ranked `top_n` (all three before cost).
    They separate the model's skill from the universe's luck: the index is
    capital-weighted, so the average eligible stock can trail or lead it for
    months whatever the model does.

    When the predictions carry `fwd_excess_open`, `top_excess_open` is the same
    top-pick return measured from the next session's open.
    """
    labelled = predictions.dropna(subset=["fwd_excess", "prob"])
    from_open = "fwd_excess_open" in predictions.columns
    records = {}
    for day, group in labelled.groupby("date", sort=True):
        ordered = group.sort_values(["prob", "ticker"], ascending=[False, True])
        top = ordered.head(top_n)
        records[day] = {
            "top_excess": float(top["fwd_excess"].mean()) - cost_bps / 10_000.0,
            "rank_ic": _rank_ic(group),
            "n": len(group),
            "universe_excess": float(group["fwd_excess"].mean()),
            "top_vs_universe": float(top["fwd_excess"].mean() - group["fwd_excess"].mean()),
            "bottom_excess": float(ordered.tail(top_n)["fwd_excess"].mean()),
        }
        if from_open:
            records[day]["top_excess_open"] = float(top["fwd_excess_open"].mean()) - cost_bps / 10_000.0
    columns = SCORE_COLUMNS + (["top_excess_open"] if from_open else [])
    if not records:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))
    scores = pd.DataFrame.from_dict(records, orient="index")[columns]
    scores.index.name = "date"
    return scores


def _mean(series: pd.Series) -> float | None:
    values = series.dropna()
    return None if values.empty else float(values.mean())


def summarize(scores: pd.DataFrame) -> dict:
    if scores.empty:
        return {"days": 0, "mean_top_excess": None, "mean_rank_ic": None, "hit_rate": None}
    summary = {
        "days": int(len(scores)),
        "mean_top_excess": _mean(scores["top_excess"]),
        "mean_rank_ic": _mean(scores["rank_ic"]),
        "hit_rate": float((scores["top_excess"] > 0).mean()),
    }
    if "universe_excess" in scores.columns:
        summary["mean_universe_excess"] = _mean(scores["universe_excess"])
        summary["mean_top_vs_universe"] = _mean(scores["top_vs_universe"])
        summary["mean_bottom_excess"] = _mean(scores["bottom_excess"])
    if "top_excess_open" in scores.columns:
        from_open = scores["top_excess_open"].dropna()
        summary["mean_top_excess_open"] = _mean(from_open)
        summary["hit_rate_open"] = float((from_open > 0).mean()) if len(from_open) else None
    return summary
