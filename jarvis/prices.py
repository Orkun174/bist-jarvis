"""Incremental daily price updates from Yahoo Finance (adjusted OHLCV)."""

from __future__ import annotations

import logging
import time
from typing import Callable

import pandas as pd

from jarvis.config import HISTORY_YEARS, OHLCV
from jarvis.store import Store

LOG = logging.getLogger("jarvis.prices")
BATCH_SIZE = 50
# Relative close difference on the overlap bar that signals a new adjustment basis.
ADJUSTMENT_TOLERANCE = 0.001

Downloader = Callable[[list[str], pd.Timestamp, pd.Timestamp], dict[str, pd.DataFrame]]


def yahoo_symbol(ticker: str) -> str:
    return f"{ticker}.IS"


def yahoo_download(
    symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """Download adjusted daily bars for `symbols`, `start`..`end` inclusive."""
    import yfinance as yf

    result: dict[str, pd.DataFrame] = {}
    for offset in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[offset:offset + BATCH_SIZE]
        for attempt in range(3):
            try:
                raw = yf.download(
                    batch,
                    start=start.date().isoformat(),
                    end=(end + pd.Timedelta(days=1)).date().isoformat(),
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
            except Exception as error:  # yfinance raises many unrelated types
                LOG.warning("Yahoo partisi başarısız (deneme %d): %s", attempt + 1, error)
                time.sleep(3 * (attempt + 1))
                continue
            if raw is None or raw.empty:
                time.sleep(3 * (attempt + 1))
                continue
            for symbol in batch:
                if isinstance(raw.columns, pd.MultiIndex):
                    if symbol not in raw.columns.get_level_values(0):
                        continue
                    frame = raw[symbol]
                else:
                    frame = raw
                frame = frame.dropna(how="all")
                if not frame.empty:
                    result[symbol] = frame
            break
    return result


def _clean(frame: pd.DataFrame, session: pd.Timestamp) -> pd.DataFrame:
    if frame is None or frame.empty or not set(OHLCV).issubset(frame.columns):
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([]))
    frame = frame[OHLCV].copy()
    frame.columns.name = None
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    frame = frame.dropna(subset=["Close"])
    frame["Volume"] = frame["Volume"].fillna(0.0)
    return frame[frame.index <= session]


def _download_group(download: Downloader, symbols: list[str], start, session) -> dict:
    try:
        return download(symbols, pd.Timestamp(start), session)
    except Exception as error:
        LOG.warning("İndirme başarısız (%d sembol): %s", len(symbols), error)
        return {}


def update_prices(
    store: Store,
    tickers: list[str],
    session: pd.Timestamp,
    download: Downloader = yahoo_download,
    symbol_of: Callable[[str], str] = yahoo_symbol,
) -> dict[str, str]:
    """Bring every ticker's stored history up to `session`.

    Returns a status per ticker: "updated", "current" or "fetch_failed".

    Yahoo's adjusted prices are rescaled retroactively after a split or
    dividend. The last stored bar is therefore requested again: when its close
    no longer matches, the whole history is downloaded afresh instead of
    appending bars on a different price basis.
    """
    session = pd.Timestamp(session).normalize()
    full_start = pd.Timestamp(session - pd.DateOffset(years=HISTORY_YEARS))
    statuses: dict[str, str] = {}
    last_bars: dict[str, tuple[pd.Timestamp, float]] = {}
    requests: dict[pd.Timestamp, list[str]] = {}
    for ticker in tickers:
        existing = store.read_prices(ticker)
        if existing.empty:
            start = full_start
        elif existing.index[-1] >= session:
            statuses[ticker] = "current"
            continue
        else:
            start = existing.index[-1]
            last_bars[ticker] = (start, float(existing["Close"].iloc[-1]))
        requests.setdefault(pd.Timestamp(start), []).append(ticker)

    rebased: list[str] = []
    for start, group in requests.items():
        frames = _download_group(download, [symbol_of(t) for t in group], start, session)
        for ticker in group:
            cleaned = _clean(frames.get(symbol_of(ticker)), session)
            if ticker in last_bars:
                last_day, last_close = last_bars[ticker]
                if last_day in cleaned.index:
                    fresh_close = float(cleaned.loc[last_day, "Close"])
                    if abs(fresh_close / last_close - 1.0) > ADJUSTMENT_TOLERANCE:
                        rebased.append(ticker)
                        continue
                cleaned = cleaned[cleaned.index > last_day]
            if cleaned.empty:
                statuses[ticker] = "fetch_failed"
                continue
            store.write_prices(ticker, cleaned)
            statuses[ticker] = "updated"

    if rebased:
        LOG.info("Düzeltme katsayısı değişmiş, geçmiş yeniden indiriliyor: %s", ", ".join(rebased))
        frames = _download_group(download, [symbol_of(t) for t in rebased], full_start, session)
        for ticker in rebased:
            cleaned = _clean(frames.get(symbol_of(ticker)), session)
            if cleaned.empty or cleaned.index[-1] <= last_bars[ticker][0]:
                statuses[ticker] = "fetch_failed"
                continue
            store.write_prices(ticker, cleaned, replace=True)
            statuses[ticker] = "updated"

    return {ticker: statuses[ticker] for ticker in tickers}
