"""(ticker, day) rows for training and prediction, built without lookahead.

Outcome: the stock's return over the next HORIZON sessions minus the BIST100
return over the same sessions (`fwd_excess`). A recipe picks one of three
yes/no targets derived from it (TARGET_COLUMNS):

- sign:          did the stock beat the index?
- cs_median:     did it beat that day's median eligible stock? This removes the
                 part every stock shares on a given day, which stock-level
                 features cannot predict.
- risk_adjusted: the same question after dividing by the stock's recent
                 volatility, so a volatile stock is not rewarded for moving more.

`fwd_excess_open` is the same outcome measured from the next session's open,
which is the first price anyone can trade at after the evening message. It is
reported next to `fwd_excess`; no target is built from it.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from jarvis.config import EMBARGO_SESSIONS, HORIZON, TRAIN_YEARS
from jarvis.eligibility import eligibility_history
from jarvis.features import (
    EXTRA_FEATURES, FEATURES, LOCAL_FEATURES, RANK_SUFFIX, RANKED_SOURCES, TURNOVER_RANK,
    VIX_PERCENTILE, extra_features, local_features, macro_features,
)
from jarvis.store import Store

TARGET_COLUMNS = {
    "sign": "target",
    "cs_median": "target_cs_median",
    "risk_adjusted": "target_risk_adjusted",
}
ROW_COLUMNS = [
    "ticker", "date", *FEATURES, *EXTRA_FEATURES,
    *(name + RANK_SUFFIX for name in RANKED_SOURCES), VIX_PERCENTILE, "close", "eligible_pit",
    "fwd_excess", "fwd_excess_open", *TARGET_COLUMNS.values(), "label_end",
]


def _above_daily_median(rows: pd.DataFrame, values: pd.Series) -> pd.Series:
    """1.0 where `values` beats that day's median over eligible rows; NaN where unknown."""
    eligible = rows["eligible_pit"].astype(bool) & values.notna()
    median = values[eligible].groupby(rows.loc[eligible, "date"]).median()
    reference = rows["date"].map(median)
    result = (values > reference).astype(float)
    return result.where(values.notna() & reference.notna())


def build_rows(
    store: Store,
    macro: Mapping[str, pd.DataFrame],
    tickers: list[str],
    session: pd.Timestamp,
) -> pd.DataFrame:
    """One row per (ticker, traded day <= session) with features and label."""
    session = pd.Timestamp(session).normalize()
    benchmark = macro["BIST100"]
    benchmark = benchmark[benchmark.index <= session]
    calendar = pd.DatetimeIndex(benchmark.index)
    macro_until = {name: frame[frame.index <= session] for name, frame in macro.items()}
    macro_frame = macro_features(macro_until, calendar)

    # Label pieces on the benchmark calendar: where each day's label ends and
    # the benchmark's forward return over the horizon.
    label_end = pd.Series(calendar, index=calendar).shift(-HORIZON)
    benchmark_forward = benchmark["Close"].shift(-HORIZON) / benchmark["Close"] - 1.0
    benchmark_forward_open = benchmark["Close"].shift(-HORIZON) / benchmark["Open"].shift(-1) - 1.0

    frames = []
    for ticker in tickers:
        equity = store.read_prices(ticker)
        equity = equity[(equity.index <= session) & equity.index.isin(calendar)]
        if equity.empty:
            continue
        features = local_features(equity, benchmark["Close"])
        eligibility = eligibility_history(equity)

        # Forward close HORIZON calendar steps ahead; NaN when the stock has no bar there.
        close_on_calendar = equity["Close"].reindex(calendar)
        stock_forward = close_on_calendar.shift(-HORIZON) / close_on_calendar - 1.0
        excess = (stock_forward - benchmark_forward).reindex(equity.index)

        frame = features.join(extra_features(equity, benchmark["Close"]))
        frame = frame.join(macro_frame.reindex(equity.index))
        frame["close"] = equity["Close"]
        frame["eligible_pit"] = eligibility["eligible_pit"].to_numpy()
        frame["fwd_excess"] = excess
        open_on_calendar = equity["Open"].reindex(calendar)
        stock_forward_open = close_on_calendar.shift(-HORIZON) / open_on_calendar.shift(-1) - 1.0
        frame["fwd_excess_open"] = (stock_forward_open - benchmark_forward_open).reindex(equity.index)
        frame["target"] = np.where(excess.isna(), np.nan, (excess > 0).astype(float))
        frame["label_end"] = label_end.reindex(equity.index).where(excess.notna())
        frame = frame.dropna(subset=list(LOCAL_FEATURES))
        frame.insert(0, "date", frame.index)
        frame.insert(0, "ticker", ticker)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=ROW_COLUMNS)
    rows = pd.concat(frames, ignore_index=True)
    eligible = rows["eligible_pit"].astype(bool)
    rows[TURNOVER_RANK] = rows["_turnover"].where(eligible).groupby(rows["date"]).rank(pct=True)
    ranked = rows[list(RANKED_SOURCES)].where(eligible).groupby(rows["date"]).rank(pct=True)
    rows = rows.join(ranked.add_suffix(RANK_SUFFIX))
    rows[TARGET_COLUMNS["cs_median"]] = _above_daily_median(rows, rows["fwd_excess"])
    volatility = rows["Volatility_20D"].where(rows["Volatility_20D"] > 0)
    rows[TARGET_COLUMNS["risk_adjusted"]] = _above_daily_median(rows, rows["fwd_excess"] / volatility)
    return rows[ROW_COLUMNS]


def training_slice(
    rows: pd.DataFrame,
    session: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    train_years: int = TRAIN_YEARS,
) -> pd.DataFrame:
    """Eligible, labelled rows whose label was final EMBARGO_SESSIONS before `session`."""
    session = pd.Timestamp(session).normalize()
    position = int(calendar.searchsorted(session, side="right")) - 1
    cutoff_position = position - EMBARGO_SESSIONS
    if cutoff_position < 0:
        return rows.iloc[0:0]
    cutoff = calendar[cutoff_position]
    start = session - pd.DateOffset(years=train_years)
    mask = (
        rows["eligible_pit"].astype(bool)
        & rows["target"].notna()
        & (rows["label_end"] <= cutoff)
        & (rows["date"] >= start)
    )
    return rows[mask]


def prediction_slice(
    rows: pd.DataFrame, session: pd.Timestamp, eligible_today: set[str]
) -> pd.DataFrame:
    session = pd.Timestamp(session).normalize()
    return rows[(rows["date"] == session) & rows["ticker"].isin(eligible_today)]
