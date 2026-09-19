# core/market_data.py
"""Daily market data, vectorized technical indicators, and bounded RSS retrieval."""

from __future__ import annotations

import calendar
import html
import logging
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import feedparser
import numpy as np
import pandas as pd
import requests
import yfinance as yf

LOGGER = logging.getLogger(__name__)

RSS_URL = "https://news.google.com/rss/search"
MAX_FEED_BYTES = 2_000_000
OBV_LOOKBACK = 5
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# CMF is checked separately because a zero-volume window is legitimately undefined.
REQUIRED_INDICATORS = [
    "RSI", "MACD", "MACD_Signal", "MACD_Histogram",
    "EMA_20", "EMA_50", "BB_Middle", "BB_Upper", "BB_Lower",
    "ATR_14", "OBV", "ADX_14", "Plus_DI_14", "Minus_DI_14",
    "StochRSI_Raw", "StochRSI_K", "StochRSI_D",
]


class MarketDataError(RuntimeError):
    """Market retrieval, validation, or indicator processing failed."""


class NewsError(RuntimeError):
    """RSS retrieval or parsing failed."""


def normalize_tickers(raw: str) -> list[str]:
    """Validate symbols, remove optional .IS suffixes, and retain unique order."""
    tickers: list[str] = []
    for part in raw.split(","):
        ticker = part.strip().upper()
        if not ticker:
            continue
        if ticker.endswith(".IS"):
            ticker = ticker[:-3]
        if not re.fullmatch(r"[A-Z][A-Z0-9]{2,9}", ticker):
            raise ValueError(
                f"Invalid ticker: {part.strip()!r}. Use symbols such as FROTO."
            )
        if ticker not in tickers:
            tickers.append(ticker)

    if not tickers:
        raise ValueError("Enter at least one BIST ticker.")
    if len(tickers) > 10:
        raise ValueError("Analyze at most 10 unique tickers per request.")
    return tickers


def _wilder_average(series: pd.Series, period: int) -> pd.Series:
    """
    Seed with the mean of the first period valid observations, then use EMA.

    Leading NaNs are allowed for derived indicators such as DX. Interior gaps
    are rejected rather than silently changing the effective smoothing window.
    """
    if period < 1:
        raise ValueError("The smoothing period must be positive.")

    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid_positions = np.flatnonzero(series.notna().to_numpy())
    if not len(valid_positions):
        return result

    start = int(valid_positions[0])
    tail = series.iloc[start:].astype(float)
    if not np.isfinite(tail.to_numpy()).all():
        raise ValueError("Wilder smoothing requires contiguous finite values.")
    if len(tail) < period:
        return result

    seed_position = start + period - 1
    seeded = pd.Series(np.nan, index=series.index, dtype=float)
    seeded.iloc[seed_position] = float(tail.iloc[:period].mean())
    seeded.iloc[seed_position + 1:] = series.iloc[
        seed_position + 1:
    ].to_numpy(dtype=float)

    # The explicit SMA seed prevents pandas from choosing a first-value seed.
    return seeded.ewm(alpha=1.0 / period, adjust=False).mean()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI; a fully flat initialized window receives neutral RSI 50."""
    if period < 1:
        raise ValueError("RSI period must be positive.")

    changes = close.diff()
    gains = _wilder_average(changes.clip(lower=0), period)
    losses = _wilder_average(-changes.clip(upper=0), period)
    relative_strength = gains / losses.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    rsi = rsi.mask((losses == 0) & (gains > 0), 100.0)
    rsi = rsi.mask((losses == 0) & (gains == 0), 50.0)
    return rsi.rename("RSI")


def calculate_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators without future values or external TA libraries."""
    result = frame.copy()
    close = result["Close"].astype(float)
    high = result["High"].astype(float)
    low = result["Low"].astype(float)
    volume = result["Volume"].astype(float)

    result["RSI"] = calculate_rsi(close, period=14)

    fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["MACD"] = fast - slow
    result["MACD_Signal"] = result["MACD"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    result["MACD_Histogram"] = result["MACD"] - result["MACD_Signal"]

    for period in (20, 50):
        result[f"EMA_{period}"] = close.ewm(
            span=period, adjust=False, min_periods=period
        ).mean()

    rolling_close = close.rolling(20, min_periods=20)
    result["BB_Middle"] = rolling_close.mean()
    deviation = rolling_close.std(ddof=0)
    result["BB_Upper"] = result["BB_Middle"] + 2.0 * deviation
    result["BB_Lower"] = result["BB_Middle"] - 2.0 * deviation

    # First OBV contribution is zero; unchanged closes contribute zero volume.
    result["OBV"] = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["ATR_14"] = _wilder_average(true_range, 14)

    # ADX: mutually exclusive directional movements; ties contribute neither.
    upward_move = high.diff()
    downward_move = -low.diff()
    plus_dm = upward_move.where(
        (upward_move > downward_move) & (upward_move > 0), 0.0
    )
    minus_dm = downward_move.where(
        (downward_move > upward_move) & (downward_move > 0), 0.0
    )

    # Directional movement needs a previous bar. Excluding the first bar makes
    # the initial DI use 14 actual transitions and initial ADX appear at bar 28.
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan
    directional_tr = true_range.copy()
    directional_tr.iloc[0] = np.nan

    smoothed_tr = _wilder_average(directional_tr, 14)
    smoothed_plus = _wilder_average(plus_dm, 14)
    smoothed_minus = _wilder_average(minus_dm, 14)

    plus_di = 100.0 * smoothed_plus / smoothed_tr.replace(0, np.nan)
    minus_di = 100.0 * smoothed_minus / smoothed_tr.replace(0, np.nan)

    # An initialized no-movement window has no directional strength.
    plus_di = plus_di.mask(smoothed_tr == 0, 0.0)
    minus_di = minus_di.mask(smoothed_tr == 0, 0.0)
    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    dx = dx.mask(di_sum == 0, 0.0)

    result["Plus_DI_14"] = plus_di
    result["Minus_DI_14"] = minus_di
    result["ADX_14"] = _wilder_average(dx, 14)

    # CMF: a zero-range candle has a neutral close-location multiplier.
    daily_range = high - low
    multiplier = (2.0 * close - high - low) / daily_range.replace(0, np.nan)
    multiplier = multiplier.mask(daily_range == 0, 0.0)
    money_flow_volume = multiplier * volume
    volume_sum = volume.rolling(20, min_periods=20).sum()
    result["CMF_Volume_20"] = volume_sum
    result["CMF_20"] = (
        money_flow_volume.rolling(20, min_periods=20).sum()
        / volume_sum.replace(0, np.nan)
    )

    # StochRSI uses RSI 14, a 14-bar stochastic window, and SMA smoothing 3/3.
    # Values use a 0–100 scale, consistently with the UI and model prompt.
    rsi = result["RSI"]
    rsi_min = rsi.rolling(14, min_periods=14).min()
    rsi_max = rsi.rolling(14, min_periods=14).max()
    rsi_range = rsi_max - rsi_min
    raw_stoch = 100.0 * (rsi - rsi_min) / rsi_range.replace(0, np.nan)

    # A flat RSI range has no relative location: explicitly use neutral 50.
    # Warm-up NaNs remain NaN because their range is not zero.
    raw_stoch = raw_stoch.mask(rsi_range == 0, 50.0)
    result["StochRSI_Raw"] = raw_stoch
    result["StochRSI_K"] = raw_stoch.rolling(3, min_periods=3).mean()
    result["StochRSI_D"] = result["StochRSI_K"].rolling(
        3, min_periods=3
    ).mean()
    return result


def fetch_market_data(ticker: str) -> pd.DataFrame:
    """Fetch initialization history, calculate indicators, and retain 3 months."""
    try:
        normalized = normalize_tickers(ticker)
    except ValueError as exc:
        raise MarketDataError(str(exc)) from exc
    if len(normalized) != 1:
        raise MarketDataError("Expected exactly one ticker.")

    symbol = f"{normalized[0]}.IS"
    try:
        frame = yf.Ticker(symbol).history(
            period="6mo",
            interval="1d",
            auto_adjust=False,
            actions=False,
            timeout=20,
        )
    except Exception as exc:
        LOGGER.exception("Yahoo request failed for %s", symbol)
        raise MarketDataError(
            f"{symbol}: Yahoo Finance request failed. Try refreshing later."
        ) from exc

    if frame is None or frame.empty:
        raise MarketDataError(
            f"{symbol}: no prices returned. Check the symbol or retry later."
        )

    try:
        if not set(REQUIRED_COLUMNS).issubset(frame.columns):
            raise MarketDataError(f"{symbol}: incomplete OHLCV response.")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise MarketDataError(f"{symbol}: invalid price timestamps.")

        frame = frame[REQUIRED_COLUMNS].copy()
        frame = frame.loc[~frame.index.isna()]
        frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
        for column in REQUIRED_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan)

        # Reject corrupt bars instead of silently introducing calculation gaps.
        if frame.isna().any().any():
            raise MarketDataError(f"{symbol}: missing or invalid OHLCV values.")

        invalid = (
            (frame[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
            | (frame["Volume"] < 0)
            | (frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1))
            | (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1))
        )
        if invalid.any():
            raise MarketDataError(f"{symbol}: inconsistent OHLCV values.")
        if len(frame) < 50:
            raise MarketDataError(
                f"{symbol}: {len(frame)} bars available; at least 50 are "
                "required for the full indicator suite."
            )

        history_bars = len(frame)
        frame = calculate_indicators(frame)
        if not np.isfinite(
            frame[REQUIRED_INDICATORS].iloc[-1].to_numpy(dtype=float)
        ).all():
            raise MarketDataError(
                f"{symbol}: the latest technical indicators are unavailable."
            )

        latest_volume_sum = float(frame["CMF_Volume_20"].iloc[-1])
        if not np.isfinite(latest_volume_sum):
            raise MarketDataError(f"{symbol}: invalid CMF volume window.")
        if latest_volume_sum > 0 and not np.isfinite(frame["CMF_20"].iloc[-1]):
            raise MarketDataError(f"{symbol}: invalid CMF calculation.")

        # Trimming after calculation preserves indicator initialization.
        cutoff = frame.index[-1] - pd.DateOffset(months=3)
        frame = frame.loc[frame.index >= cutoff].copy()
        if len(frame) < OBV_LOOKBACK + 1:
            raise MarketDataError(f"{symbol}: insufficient recent OBV history.")

        frame.attrs.update(
            fetched_at=datetime.now(timezone.utc).isoformat(),
            indicator_history_bars=history_bars,
        )
        return frame

    except MarketDataError:
        raise
    except Exception as exc:
        LOGGER.exception("Market processing failed for %s", symbol)
        raise MarketDataError(
            f"{symbol}: the market response could not be processed."
        ) from exc


def _bollinger_position(
    price: float, lower: float, upper: float
) -> tuple[str, float | None]:
    """Classify price location without clipping genuine band breakouts."""
    width = upper - lower
    if width <= max(abs(price), 1.0) * 1e-12:
        return "Flat bands", None

    percent_b = (price - lower) / width
    if percent_b > 1.0:
        label = "Above upper"
    elif percent_b < 0.0:
        label = "Below lower"
    elif percent_b >= 0.8:
        label = "Near upper"
    elif percent_b <= 0.2:
        label = "Near lower"
    else:
        label = "Mid-band"
    return label, percent_b


def market_snapshot(frame: pd.DataFrame) -> dict:
    """Create finite scalar evidence; undefined CMF is explicitly JSON null."""
    if len(frame) < OBV_LOOKBACK + 1:
        raise MarketDataError("Insufficient bars for the OBV trend snapshot.")

    latest = frame.iloc[-1]
    price = float(latest["Close"])
    previous_close = float(frame["Close"].iloc[-2])
    obv_change = float(
        latest["OBV"] - frame["OBV"].iloc[-OBV_LOOKBACK - 1]
    )

    values = {
        "price": price,
        "change_pct": (price / previous_close - 1.0) * 100.0,
        "rsi": float(latest["RSI"]),
        "macd": float(latest["MACD"]),
        "macd_signal": float(latest["MACD_Signal"]),
        "macd_histogram": float(latest["MACD_Histogram"]),
        "ema_20": float(latest["EMA_20"]),
        "ema_50": float(latest["EMA_50"]),
        "bb_middle": float(latest["BB_Middle"]),
        "bb_upper": float(latest["BB_Upper"]),
        "bb_lower": float(latest["BB_Lower"]),
        "atr_14": float(latest["ATR_14"]),
        "atr_pct": float(latest["ATR_14"]) / price * 100.0,
        "obv": float(latest["OBV"]),
        "obv_change": obv_change,
        "adx_14": float(latest["ADX_14"]),
        "plus_di_14": float(latest["Plus_DI_14"]),
        "minus_di_14": float(latest["Minus_DI_14"]),
        "stochrsi_raw": float(latest["StochRSI_Raw"]),
        "stochrsi_k": float(latest["StochRSI_K"]),
        "stochrsi_d": float(latest["StochRSI_D"]),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise MarketDataError("Latest technical indicators are not finite.")

    cmf_available = bool(latest["CMF_Volume_20"] > 0)
    cmf = float(latest["CMF_20"]) if cmf_available else None
    if cmf is not None and not np.isfinite(cmf):
        raise MarketDataError("Latest CMF is invalid.")

    position, percent_b = _bollinger_position(
        price, values["bb_lower"], values["bb_upper"]
    )
    adx = values["adx_14"]
    adx_regime = (
        "Trending" if adx >= 25
        else "Weak trend" if adx < 20
        else "Transitional"
    )

    return {
        **values,
        "currency": "TRY",
        "bar_time": frame.index[-1].isoformat(),
        "fetched_at": frame.attrs["fetched_at"],
        "price_basis": "unadjusted daily close; may be an unfinished bar",
        "indicator_history_bars": frame.attrs["indicator_history_bars"],
        "indicator_history": "up to 6 months; recursive EMA initialization",
        "bb_position": position,
        "bb_percent_b": percent_b,
        "bb_method": "SMA 20 plus/minus 2 population standard deviations",
        "atr_method": "14-bar Wilder-smoothed true range",
        "obv_trend": (
            "Rising" if obv_change > 0
            else "Falling" if obv_change < 0
            else "Flat"
        ),
        "obv_lookback_bars": OBV_LOOKBACK,
        "obv_method": "sign of OBV[t] - OBV[t-5]; not necessarily monotonic",
        "adx_regime": adx_regime,
        "adx_method": "Wilder 14; >=25 trending, <20 weak, 20–25 transitional",
        "cmf_20": cmf,
        "cmf_status": "available" if cmf_available else "zero_volume_window",
        "cmf_method": "20-bar flow/volume ratio; zero-range bars contribute zero",
        "stochrsi_method": (
            "RSI 14, stochastic window 14, SMA K=3, SMA D=3; scale 0–100; "
            "flat RSI range assigned neutral raw value 50"
        ),
    }


def _plain_text(value: object, limit: int = 600) -> str:
    """Remove feed markup and normalize whitespace."""
    stripped = re.sub(r"<[^>]*>", " ", str(value or ""))
    return " ".join(html.unescape(stripped).split())[:limit]


def fetch_news(ticker: str, limit: int = 5) -> list[dict]:
    """Return the latest unique RSS headlines with bounded network retrieval."""
    if not 1 <= limit <= 5:
        raise ValueError("News limit must be between 1 and 5.")
    try:
        normalized = normalize_tickers(ticker)
    except ValueError as exc:
        raise NewsError(str(exc)) from exc
    if len(normalized) != 1:
        raise NewsError("Expected exactly one ticker.")

    params = {
        "q": f"{normalized[0]} hisse kap",
        "hl": "tr",
        "gl": "TR",
        "ceid": "TR:tr",
    }
    try:
        with requests.get(
            RSS_URL,
            params=params,
            headers={"User-Agent": "BIST-AI-Radar/3.0"},
            timeout=(5, 20),
            stream=True,
        ) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_content(chunk_size=16_384):
                content.extend(chunk)
                if len(content) > MAX_FEED_BYTES:
                    raise NewsError("Google News RSS exceeded the size limit.")
    except requests.Timeout as exc:
        raise NewsError(
            "Google News RSS timed out; news is unavailable."
        ) from exc
    except requests.RequestException as exc:
        raise NewsError("Google News RSS could not be retrieved.") from exc

    try:
        feed = feedparser.parse(bytes(content))
    except Exception as exc:
        LOGGER.exception("RSS parsing failed")
        raise NewsError("Google News RSS could not be parsed.") from exc

    if feed.get("bozo") or not feed.get("version"):
        raise NewsError("Google News returned malformed RSS.")

    items: list[dict] = []
    for entry in feed.get("entries", []):
        title = _plain_text(entry.get("title"))
        if not title:
            continue

        parsed_time = (
            entry.get("published_parsed") or entry.get("updated_parsed")
        )
        timestamp = None
        published_at = None
        if parsed_time:
            try:
                timestamp = float(calendar.timegm(parsed_time))
                published_at = datetime.fromtimestamp(
                    timestamp, timezone.utc
                ).isoformat()
            except (TypeError, ValueError, OverflowError, OSError):
                pass

        url = str(entry.get("link", "")).strip()
        try:
            parsed_url = urlparse(url)
            if (
                parsed_url.scheme not in {"https", "http"}
                or not parsed_url.netloc
            ):
                url = ""
        except ValueError:
            url = ""

        items.append(
            {
                "title": title,
                "url": url,
                "published_at": published_at,
                "_timestamp": timestamp,
            }
        )

    # Undated articles sort last; equal dates retain their source order.
    items.sort(
        key=lambda item: (
            item["_timestamp"] is not None,
            item["_timestamp"] if item["_timestamp"] is not None else 0,
        ),
        reverse=True,
    )

    results: list[dict] = []
    seen: set[str] = set()
    for item in items:
        identity = item["title"].casefold()
        if identity in seen:
            continue
        seen.add(identity)
        results.append(
            {key: value for key, value in item.items() if key != "_timestamp"}
        )
        if len(results) == limit:
            break
    return results