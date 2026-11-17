"""The weekly recipe exam and the daily safety checks.

Recipes (hyperparameters, features, training window) are compared on equal
footing with a paired bootstrap over walk-forward days. Daily refits of the
same recipe are not "examined" against yesterday's weights -- both saw almost
the same data, so no fair exam exists -- they pass through safety checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jarvis.config import (
    BOOTSTRAP_CONFIDENCE, BOOTSTRAP_SAMPLES, DRIFT_MIN_RANK_CORR, RANDOM_SEED,
)

MIN_COMPARE_DAYS = 30
MIN_HEALTH_DAYS = 60
MIN_DRIFT_NAMES = 10


def _probability_positive(differences: np.ndarray, samples: int, rng) -> float:
    """Share of bootstrap resamples whose mean difference is above zero."""
    if len(differences) == 0:
        return 0.0
    picks = rng.integers(0, len(differences), size=(samples, len(differences)))
    return float((differences[picks].mean(axis=1) > 0).mean())


def _pick_metric(champion_scores: pd.DataFrame, challenger_scores: pd.DataFrame) -> str:
    """The return measure the exam uses: the quintile spread when both sides have it.

    Five picks a day are too noisy for 60 sessions (2026-11-17: a +0.6% edge is
    within the noise even over 245 sessions); the best fifth minus the worst
    fifth uses far more stocks. Scores without it fall back to the top picks.
    """
    for frame in (champion_scores, challenger_scores):
        if "quintile_spread" not in frame.columns or frame["quintile_spread"].dropna().empty:
            return "top_excess"
    return "quintile_spread"


def compare_recipes(
    champion_scores: pd.DataFrame,
    challenger_scores: pd.DataFrame,
    samples: int = BOOTSTRAP_SAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = RANDOM_SEED,
) -> dict:
    """Promote the challenger only when it ranks better on both measures, convincingly.

    The two measures are rank IC and the pick metric (see `_pick_metric`). The
    top picks' own return is reported but not required here; the long-window
    confirmation (`confirm_recipes`) makes sure it did not get worse.
    """
    common = champion_scores.index.intersection(challenger_scores.index)
    metric = _pick_metric(champion_scores, challenger_scores)
    result: dict = {"promote": False, "days": int(len(common)), "pick_metric": metric}
    if len(common) < MIN_COMPARE_DAYS:
        result["reason"] = "too_few_days"
        return result
    rng = np.random.default_rng(seed)

    def difference(column: str) -> pd.Series:
        return (challenger_scores.loc[common, column] - champion_scores.loc[common, column]).dropna()

    rank_ic = difference("rank_ic")
    picks = difference(metric)
    result.update(
        {
            "top_excess_diff": float(difference("top_excess").mean()) if len(common) else 0.0,
            "rank_ic_diff": float(rank_ic.mean()) if len(rank_ic) else 0.0,
            "pick_diff": float(picks.mean()) if len(picks) else 0.0,
            "p_rank_ic": _probability_positive(rank_ic.to_numpy(), samples, rng),
            "p_pick": _probability_positive(picks.to_numpy(), samples, rng),
        }
    )
    result["promote"] = bool(
        result["pick_diff"] > 0
        and result["rank_ic_diff"] > 0
        and result["p_pick"] >= confidence
        and result["p_rank_ic"] >= confidence
    )
    return result


def confirm_recipes(champion_scores: pd.DataFrame, challenger_scores: pd.DataFrame) -> dict:
    """Second opinion for a challenger that passed the exam, over a much longer window.

    Sixty sessions of top-pick returns can be won by luck: on 2026-11-17 two
    challengers that passed (or nearly passed) the exam turned out to pick
    *worse* top stocks than the champion over 245 sessions. A challenger is
    therefore promoted only if, over the long window too, it ranks better
    (rank IC, quintile spread) and its top picks are not behind. No confidence
    level is asked here; the exam already did.
    """
    common = champion_scores.index.intersection(challenger_scores.index)
    result: dict = {"ok": False, "days": int(len(common))}
    if len(common) < MIN_COMPARE_DAYS:
        result["reason"] = "too_few_days"
        return result
    excess = (challenger_scores.loc[common, "top_excess"] - champion_scores.loc[common, "top_excess"]).dropna()
    rank_ic = (challenger_scores.loc[common, "rank_ic"] - champion_scores.loc[common, "rank_ic"]).dropna()
    result["top_excess_diff"] = float(excess.mean()) if len(excess) else 0.0
    result["rank_ic_diff"] = float(rank_ic.mean()) if len(rank_ic) else 0.0
    ok = result["top_excess_diff"] >= 0 and result["rank_ic_diff"] > 0
    if _pick_metric(champion_scores, challenger_scores) == "quintile_spread":
        spread = (challenger_scores.loc[common, "quintile_spread"]
                  - champion_scores.loc[common, "quintile_spread"]).dropna()
        result["quintile_spread_diff"] = float(spread.mean()) if len(spread) else 0.0
        ok = ok and result["quintile_spread_diff"] > 0
    result["ok"] = bool(ok)
    return result


def health_check(scores: pd.DataFrame) -> dict:
    """Is the recipe still ranking stocks better than chance on recent days?"""
    rank_ic = scores["rank_ic"].dropna()
    days = int(len(rank_ic))
    mean = float(rank_ic.mean()) if days else None
    if days < MIN_HEALTH_DAYS:
        return {"ok": True, "reason": "insufficient_history", "mean_rank_ic": mean, "days": days}
    return {"ok": bool(mean > 0), "mean_rank_ic": mean, "days": days}


def drift_check(new_probs: pd.Series, old_probs: pd.Series | None) -> dict:
    """New weights should rank today's stocks broadly like yesterday's weights do."""
    if old_probs is None:
        return {"ok": True, "reason": "no_previous_model"}
    common = new_probs.index.intersection(old_probs.index)
    if len(common) < MIN_DRIFT_NAMES:
        return {"ok": True, "reason": "too_few_names"}
    new_rank, old_rank = new_probs.loc[common].rank(), old_probs.loc[common].rank()
    if new_rank.nunique() < 2 or old_rank.nunique() < 2:
        return {"ok": True, "reason": "constant_predictions"}
    correlation = float(np.corrcoef(new_rank, old_rank)[0, 1])
    return {"ok": bool(correlation >= DRIFT_MIN_RANK_CORR), "rank_corr": correlation}
