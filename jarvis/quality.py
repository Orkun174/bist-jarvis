"""Price sanity checks, the Is Yatirim close cross-check, and the daily report."""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import pandas as pd
import requests

from jarvis.config import (
    CROSS_CHECK_TOLERANCE, MAX_QUARANTINE_RATIO, QUALITY_WINDOW, SUSPECT_JUMP,
)
from jarvis.http import USER_AGENT
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


ISYATIRIM_URL = (
    "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil"
)
REFERENCE_PROBE = 3        # tickers asked first to see whether the session is published
REFERENCE_WORKERS = 16
REFERENCE_BUDGET_SECONDS = 600.0


def isyatirim_close(ticker: str, session: pd.Timestamp) -> float | None:
    """One ticker's unadjusted close for `session`, or None when not published."""
    day = pd.Timestamp(session).strftime("%d-%m-%Y")
    response = requests.get(
        ISYATIRIM_URL,
        params={"hisse": ticker, "startdate": day, "enddate": day},
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    for row in response.json().get("value", []):
        if row.get("HGDG_TARIH") == day and row.get("HGDG_KAPANIS") is not None:
            return float(row["HGDG_KAPANIS"])
    return None


def isyatirim_closes(
    tickers: list[str],
    session: pd.Timestamp,
    fetch_one: Callable[[str, pd.Timestamp], float | None] = isyatirim_close,
    budget_seconds: float = REFERENCE_BUDGET_SECONDS,
) -> dict[str, float]:
    """Unadjusted closes for `session` from Is Yatirim; empty when unavailable.

    A few tickers are probed first: when none has the session yet (the source
    publishes late in the evening) the rest are not requested at all.
    """
    deadline = time.monotonic() + budget_seconds

    def safe(ticker: str) -> float | None:
        try:
            return fetch_one(ticker, session)
        except Exception as error:
            LOG.debug("İş Yatırım %s alınamadı: %s", ticker, error)
            return None

    closes: dict[str, float] = {}
    probe, rest = tickers[:REFERENCE_PROBE], tickers[REFERENCE_PROBE:]
    for ticker in probe:
        value = safe(ticker)
        if value is not None:
            closes[ticker] = value
    if not closes:
        LOG.warning("İş Yatırım %s seansını henüz yayınlamamış; karşılaştırma atlandı.",
                    pd.Timestamp(session).date())
        return {}

    def bounded(ticker: str) -> float | None:
        return None if time.monotonic() >= deadline else safe(ticker)

    with ThreadPoolExecutor(max_workers=REFERENCE_WORKERS) as pool:
        for ticker, value in zip(rest, pool.map(bounded, rest)):
            if value is not None:
                closes[ticker] = value
    LOG.info("İş Yatırım: %d / %d kapanış alındı.", len(closes), len(tickers))
    return closes


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
