"""Prediction ledger: every day's predictions, later graded against reality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from jarvis.config import HORIZON, TOP_N
from jarvis.store import Store

LEDGER_TABLE = "ledger/predictions"
LEDGER_COLUMNS = [
    "date", "ticker", "prob", "rank", "recommended", "threshold", "model_id", "recipe_id",
    "entry_close", "matured", "exit_date", "exit_close", "fwd_return", "benchmark_return",
    "fwd_excess",
]


def empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "ticker": pd.Series(dtype=str),
            "prob": pd.Series(dtype=float),
            "rank": pd.Series(dtype="int64"),
            "recommended": pd.Series(dtype=bool),
            "threshold": pd.Series(dtype=float),
            "model_id": pd.Series(dtype=str),
            "recipe_id": pd.Series(dtype=str),
            "entry_close": pd.Series(dtype=float),
            "matured": pd.Series(dtype=bool),
            "exit_date": pd.Series(dtype="datetime64[ns]"),
            "exit_close": pd.Series(dtype=float),
            "fwd_return": pd.Series(dtype=float),
            "benchmark_return": pd.Series(dtype=float),
            "fwd_excess": pd.Series(dtype=float),
        }
    )[LEDGER_COLUMNS]


def read_ledger(store: Store, table: str = LEDGER_TABLE) -> pd.DataFrame:
    ledger = store.read_table(table)
    return empty_ledger() if ledger.empty else ledger[LEDGER_COLUMNS]


def build_entries(
    session: pd.Timestamp,
    predictions: pd.DataFrame,
    model_id: str,
    recipe_id: str,
    threshold: float,
    allow_recommendations: bool = True,
) -> pd.DataFrame:
    """Ledger rows for one day from `predictions` (ticker, prob, entry_close)."""
    ordered = predictions.sort_values(["prob", "ticker"], ascending=[False, True]).reset_index(drop=True)
    entries = empty_ledger().reindex(range(len(ordered)))
    entries["date"] = pd.Timestamp(session).normalize()
    entries["ticker"] = ordered["ticker"].astype(str)
    entries["prob"] = ordered["prob"].astype(float)
    entries["rank"] = np.arange(1, len(ordered) + 1, dtype="int64")
    recommended = (entries["rank"] <= TOP_N) & (entries["prob"] >= threshold)
    entries["recommended"] = recommended & allow_recommendations
    entries["threshold"] = float(threshold)
    entries["model_id"] = model_id
    entries["recipe_id"] = recipe_id
    entries["entry_close"] = ordered["entry_close"].astype(float)
    entries["matured"] = False
    entries["exit_date"] = pd.NaT
    return entries.astype(empty_ledger().dtypes.to_dict())


def record(
    store: Store,
    session: pd.Timestamp,
    predictions: pd.DataFrame,
    model_id: str,
    recipe_id: str,
    threshold: float,
    allow_recommendations: bool = True,
    table: str = LEDGER_TABLE,
) -> int:
    """Write one day's predictions; a second write for the same day replaces the first."""
    session = pd.Timestamp(session).normalize()
    entries = build_entries(session, predictions, model_id, recipe_id, threshold, allow_recommendations)
    ledger = read_ledger(store, table)
    kept = ledger[ledger["date"] != session]
    frames = [frame for frame in (kept, entries) if not frame.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else empty_ledger()
    combined = combined.sort_values(["date", "rank"], kind="stable", ignore_index=True)
    store.write_table(table, combined)
    return len(entries)


def mature(
    store: Store,
    price_store: Store,
    benchmark: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    session: pd.Timestamp,
    table: str = LEDGER_TABLE,
) -> int:
    """Grade predictions whose HORIZON sessions have passed. Returns rows graded.

    Returns are computed from the *current* adjusted series on both the entry
    and exit day, so they stay right after a split rescales stored history.
    A row stays unmatured when either price is missing.
    """
    session = pd.Timestamp(session).normalize()
    ledger = read_ledger(store, table)
    pending = ledger[~ledger["matured"]]
    if pending.empty:
        return 0
    positions = {day: i for i, day in enumerate(calendar)}
    benchmark_close = benchmark["Close"]
    graded = 0
    cache: dict[str, pd.Series] = {}
    for index, row in pending.iterrows():
        position = positions.get(row["date"])
        if position is None or position + HORIZON >= len(calendar):
            continue
        exit_day = calendar[position + HORIZON]
        if exit_day > session:
            continue
        ticker = row["ticker"]
        if ticker not in cache:
            cache[ticker] = price_store.read_prices(ticker)["Close"]
        close = cache[ticker]
        if row["date"] not in close.index or exit_day not in close.index:
            continue
        if row["date"] not in benchmark_close.index or exit_day not in benchmark_close.index:
            continue
        stock_return = float(close[exit_day] / close[row["date"]] - 1.0)
        index_return = float(benchmark_close[exit_day] / benchmark_close[row["date"]] - 1.0)
        ledger.loc[index, ["exit_date", "exit_close", "fwd_return", "benchmark_return",
                           "fwd_excess", "matured"]] = [
            exit_day, float(close[exit_day]), stock_return, index_return,
            stock_return - index_return, True,
        ]
        graded += 1
    if graded:
        store.write_table(table, ledger.astype(empty_ledger().dtypes.to_dict()))
    return graded
