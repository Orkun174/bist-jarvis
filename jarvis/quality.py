"""Price sanity checks, the Is Yatirim close cross-check, and the daily report."""

from __future__ import annotations

import collections
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
REFERENCE_RETRY_WORKERS = 4
REFERENCE_TIMEOUT = (5, 25)  # connect, read; the cloud runner is far from the source
REFERENCE_BUDGET_SECONDS = 600.0


def isyatirim_close(ticker: str, session: pd.Timestamp) -> float | None:
    """One ticker's unadjusted close for `session`, or None when not published."""
    day = pd.Timestamp(session).strftime("%d-%m-%Y")
    response = requests.get(
        ISYATIRIM_URL,
        params={"hisse": ticker, "startdate": day, "enddate": day},
        headers={"User-Agent": USER_AGENT},
        timeout=REFERENCE_TIMEOUT,
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
    publishes late in the evening) the rest are not requested at all. Requests
    that fail (timeouts, connection or HTTP errors, as opposed to "not
    published") are tried once more with fewer workers while the budget lasts:
    from the cloud runner a large share of the first pass used to time out.
    """
    deadline = time.monotonic() + budget_seconds
    errors: collections.Counter = collections.Counter()

    def attempt(ticker: str) -> tuple[float | None, bool]:
        """(close, failed): failed is True when the request itself went wrong."""
        if time.monotonic() >= deadline:
            return None, True
        try:
            return fetch_one(ticker, session), False
        except Exception as error:
            errors[type(error).__name__] += 1
            LOG.debug("İş Yatırım %s alınamadı: %s", ticker, error)
            return None, True

    def run(batch: list[str], workers: int) -> tuple[dict[str, float], list[str]]:
        found: dict[str, float] = {}
        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(batch) or 1))) as pool:
            for ticker, (value, bad) in zip(batch, pool.map(attempt, batch)):
                if value is not None:
                    found[ticker] = value
                elif bad:
                    failed.append(ticker)
        return found, failed

    probe, rest = tickers[:REFERENCE_PROBE], tickers[REFERENCE_PROBE:]
    closes, failed = run(probe, REFERENCE_PROBE)
    if not closes and failed:
        retried, failed = run(failed, REFERENCE_PROBE)
        closes.update(retried)
    if not closes:
        LOG.warning("İş Yatırım %s seansını henüz yayınlamamış ya da yanıt vermiyor; "
                    "karşılaştırma atlandı (hatalar: %s).", pd.Timestamp(session).date(), dict(errors))
        return {}

    found, more_failed = run(rest, REFERENCE_WORKERS)
    closes.update(found)
    failed += more_failed
    if failed:
        recovered, failed = run(failed, REFERENCE_RETRY_WORKERS)
        closes.update(recovered)
    LOG.info("İş Yatırım: %d / %d kapanış alındı; %d istek başarısız (hatalar: %s).",
             len(closes), len(tickers), len(failed), dict(errors))
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
