"""Price sanity checks, the Is Yatirim close cross-check, and the daily report."""

from __future__ import annotations

import contextlib
import io
import logging
import math
from typing import Callable

import pandas as pd

from jarvis.config import (
    CROSS_CHECK_TOLERANCE, MAX_QUARANTINE_RATIO, QUALITY_WINDOW, SUSPECT_JUMP,
)
from jarvis.store import Store

LOG = logging.getLogger("jarvis.quality")
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]

ReferenceCloses = Callable[[list[str], pd.Timestamp], dict[str, float]]


def check_prices(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> list[str]:
    """Issue codes for one ticker; `sessions` are XIST sessions ending at the target."""
    if frame.empty:
        return ["stale"]
    issues: list[str] = []
    recent_sessions = sessions[-QUALITY_WINDOW:]
    recent = frame[frame.index >= recent_sessions[0]]
    if (recent[PRICE_COLUMNS] <= 0).any().any():
        issues.append("nonpositive_price")
    if frame.index[-1] != sessions[-1]:
        issues.append("stale")
    elif frame["Volume"].iloc[-1] <= 0:
        issues.append("zero_volume_last")
    first = max(frame.index[0], recent_sessions[0])
    expected = recent_sessions[(recent_sessions >= first) & (recent_sessions <= frame.index[-1])]
    if len(expected.difference(frame.index)) > 0:
        issues.append("missing_sessions")
    returns = frame["Close"].pct_change()
    if (returns[returns.index >= recent_sessions[0]].abs() > SUSPECT_JUMP).any():
        issues.append("suspect_jump")
    return issues


def cross_check(yahoo_close: float, reference_close: float | None) -> str:
    if reference_close is None or not math.isfinite(reference_close) or reference_close <= 0:
        return "unavailable"
    difference = abs(yahoo_close / reference_close - 1.0)
    return "mismatch" if difference > CROSS_CHECK_TOLERANCE else "ok"


def isyatirim_closes(tickers: list[str], session: pd.Timestamp) -> dict[str, float]:
    """Unadjusted closes for `session` from Is Yatirim; empty when unavailable."""
    try:
        from isyatirimhisse import fetch_stock_data

        day = pd.Timestamp(session).strftime("%d-%m-%Y")
        with contextlib.redirect_stdout(io.StringIO()):  # the library prints per symbol
            raw = fetch_stock_data(symbols=list(tickers), start_date=day, end_date=day)
        dates = pd.to_datetime(raw["HGDG_TARIH"]).dt.normalize()
        rows = raw[dates == pd.Timestamp(session).normalize()]
        return {
            str(code): float(close)
            for code, close in zip(rows["HGDG_HS_KODU"], rows["HGDG_KAPANIS"])
        }
    except Exception as error:  # the library raises ValueError when nothing is published
        LOG.warning("İş Yatırım kapanışları alınamadı: %s", error)
        return {}


def build_report(
    store: Store,
    tickers: list[str],
    session: pd.Timestamp,
    statuses: dict[str, str],
    sessions: pd.DatetimeIndex,
    reference: ReferenceCloses = isyatirim_closes,
) -> dict:
    session = pd.Timestamp(session).normalize()
    fetch_failed = sorted(t for t in tickers if statuses.get(t) == "fetch_failed")
    frames = {ticker: store.read_prices(ticker) for ticker in tickers}
    checkable = [t for t in tickers if not frames[t].empty and frames[t].index[-1] == session]
    try:
        closes = reference(checkable, session) if checkable else {}
    except Exception as error:
        LOG.warning("Referans kapanış kaynağı hata verdi: %s", error)
        closes = {}

    quarantined: dict[str, list[str]] = {}
    unavailable: list[str] = []
    for ticker in tickers:
        issues = check_prices(frames[ticker], sessions)
        if ticker in fetch_failed:
            issues.append("fetch_failed")
        if ticker in checkable:
            outcome = cross_check(float(frames[ticker]["Close"].iloc[-1]), closes.get(ticker))
            if outcome == "mismatch":
                issues.append("cross_check_mismatch")
            elif outcome == "unavailable":
                unavailable.append(ticker)
        if issues:
            quarantined[ticker] = issues

    ratio = round(len(quarantined) / len(tickers), 6) if tickers else 0.0
    return {
        "session": session.date().isoformat(),
        "tickers": len(tickers),
        "quarantined": quarantined,
        "cross_check_unavailable": sorted(unavailable),
        "fetch_failed": fetch_failed,
        "quarantine_ratio": ratio,
        "training_allowed": ratio <= MAX_QUARANTINE_RATIO,
    }
