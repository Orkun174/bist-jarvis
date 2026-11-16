"""Eligibility (manipulation) filter.

A ticker must have enough history, enough liquidity, few limit-move days
and no erratic price behaviour (extreme volatility or a parabolic run-up).
`eligibility_history` computes this for every past day using only data known
on that day, so training rows can be filtered without lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jarvis.config import (
    ERRATIC_COOLDOWN, LIMIT_MOVE, LIQUIDITY_WINDOW, MAX_LIMIT_DAYS, MAX_RUNUP_120,
    MAX_VOLATILITY_20, MIN_HISTORY_SESSIONS, MIN_MEDIAN_TURNOVER_TRY, RUNUP_WINDOW,
    VOLATILITY_WINDOW,
)
from jarvis.exclusions import excluded_on
from jarvis.store import Store

TODAY_COLUMNS = [
    "ticker", "session", "eligible", "reason", "median_turnover", "limit_days", "sessions",
]


def eligibility_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time eligibility for every row of one ticker's price history."""
    result = pd.DataFrame(index=frame.index)
    result["sessions"] = np.arange(1, len(frame) + 1)
    turnover = (frame["Close"] * frame["Volume"]).fillna(0.0)
    result["median_turnover"] = turnover.rolling(
        LIQUIDITY_WINDOW, min_periods=LIQUIDITY_WINDOW
    ).median()
    limit_day = (frame["Close"].pct_change().abs() >= LIMIT_MOVE).astype(int)
    result["limit_days"] = (
        limit_day.rolling(LIQUIDITY_WINDOW, min_periods=1).sum().astype(int)
    )
    returns = frame["Close"].pct_change()
    result["volatility_20"] = returns.rolling(VOLATILITY_WINDOW).std()
    result["runup_120"] = frame["Close"] / frame["Close"].shift(RUNUP_WINDOW) - 1.0
    triggered = (result["volatility_20"] > MAX_VOLATILITY_20) | (result["runup_120"] > MAX_RUNUP_120)
    # The exclusion lasts ERRATIC_COOLDOWN sessions after the last trigger.
    erratic = triggered.astype(int).rolling(ERRATIC_COOLDOWN + 1, min_periods=1).max() > 0

    short = result["sessions"] < MIN_HISTORY_SESSIONS
    illiquid = ~(result["median_turnover"] >= MIN_MEDIAN_TURNOVER_TRY)
    limits = result["limit_days"] > MAX_LIMIT_DAYS
    # First failing rule wins, in this order.
    result["reason_pit"] = np.select(
        [short, illiquid, limits, erratic],
        ["short_history", "low_liquidity", "limit_moves", "erratic"],
        default="",
    )
    result["eligible_pit"] = result["reason_pit"] == ""
    return result


def eligibility_today(
    store: Store,
    tickers: list[str],
    session: pd.Timestamp,
    exclusions: pd.DataFrame,
    quarantined: set[str],
) -> pd.DataFrame:
    """One row per ticker for `session`: quarantine > exclusion list > rules."""
    session = pd.Timestamp(session).normalize()
    excluded = excluded_on(exclusions, session)
    rows = []
    for ticker in tickers:
        frame = store.read_prices(ticker)
        frame = frame[frame.index <= session]
        if frame.empty:
            sessions, turnover, limit_days, rule = 0, 0.0, 0, "short_history"
        else:
            last = eligibility_history(frame).iloc[-1]
            sessions = int(last["sessions"])
            turnover = 0.0 if pd.isna(last["median_turnover"]) else float(last["median_turnover"])
            limit_days = int(last["limit_days"])
            rule = str(last["reason_pit"])
        if ticker in quarantined:
            reason = "quarantined"
        elif ticker in excluded:
            reason = f"excluded:{excluded[ticker]}"
        else:
            reason = rule
        rows.append((ticker, session, reason == "", reason, turnover, limit_days, sessions))
    return pd.DataFrame(rows, columns=TODAY_COLUMNS)
