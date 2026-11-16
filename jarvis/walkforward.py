"""Walk a recipe forward through history the way it would have run live.

Every `refit_every` sessions the model is retrained on rows whose labels were
final by then (see dataset.training_slice); in between the latest model keeps
predicting. Nothing from the future of a prediction day is ever used.
"""

from __future__ import annotations

import pandas as pd

from jarvis.config import MIN_TRAIN_ROWS, REFIT_EVERY
from jarvis.dataset import training_slice
from jarvis.features import VIX_PERCENTILE
from jarvis.model import Recipe, fit, predict

COLUMNS = ["date", "ticker", "prob", "fwd_excess", VIX_PERCENTILE]


def walk_forward(
    rows: pd.DataFrame,
    recipe: Recipe,
    calendar: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    refit_every: int = REFIT_EVERY,
) -> pd.DataFrame:
    """Out-of-sample predictions for eligible rows on every session in [start, end]."""
    sessions = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    eligible = rows[rows["eligible_pit"].astype(bool)]
    pieces = []
    for offset in range(0, len(sessions), refit_every):
        block = sessions[offset:offset + refit_every]
        train = training_slice(rows, block[0], calendar, train_years=recipe.train_years)
        if len(train) < MIN_TRAIN_ROWS or train["target"].nunique() < 2:
            continue
        model = fit(recipe, train)
        today = eligible[eligible["date"].isin(block)]
        if today.empty:
            continue
        piece = today[["date", "ticker", "fwd_excess", VIX_PERCENTILE]].copy()
        piece["prob"] = predict(model, recipe, today)
        pieces.append(piece[COLUMNS])
    if not pieces:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "ticker"], ignore_index=True)
