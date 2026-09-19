# core/ai_analyzer.py
"""Shared feature engineering, dynamic inference, SHAP, and Qwen synthesis."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st
import yfinance as yf
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "bist_xgb_model.json"

FEATURE_CONTRACT_VERSION = "bist-dynamic-alpha-v3"
BENCHMARK_TICKER = "XU100.IS"
HORIZON = 5
HISTORY_PERIOD = "5y"
PRICE_BASIS = "yfinance auto_adjust=True"
TARGET_DEFINITION = (
    "Equity five-session total return exceeds XU100 five-session total return"
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"
INSUFFICIENT_SIGNAL = "Insufficient Signal / Neutral Regime"

# Both training and inference use this ordered, unpruned input universe.
# The deployed model metadata records the selected ordered subset.
MACRO_TICKERS: dict[str, str] = {
    "USDTRY": "USDTRY=X",
    "BRENT": "BZ=F",
    "GOLD": "GC=F",
    "SP500": "^GSPC",
    "VIX": "^VIX",
    "BIST100": BENCHMARK_TICKER,
    "BISTBANK": "XBANK.IS",
}

LOCAL_FEATURE_COLUMNS = (
    "EMA20_Distance",
    "EMA50_Distance",
    "Volume_Surge",
    "RSI_Momentum_5D",
    "Relative_Return_1D",
    "MACD_Normalized",
    "ATR_Normalized",
    "ADX_14",
    "CMF_20",
    "StochRSI_K",
)
MACRO_FEATURE_COLUMNS = tuple(
    name
    for macro in MACRO_TICKERS
    for name in (
        f"macro_{macro}_1d_lag",
        f"macro_{macro}_5d_mean_lag",
    )
)
FEATURE_COLUMNS = LOCAL_FEATURE_COLUMNS + MACRO_FEATURE_COLUMNS

# The percentile uses lagged VIX LEVELS, not VIX returns. A large VIX daily
# return and a high absolute VIX volatility regime are different quantities.
VIX_PERCENTILE_COLUMN = "VIX_Percentile"
VIX_PERCENTILE_WINDOW = 252
VIX_PERCENTILE_MIN_PERIODS = 126

SYSTEM_PROMPT = 'You are a quantitative technical synthesizer. Return ONLY a valid JSON object with exactly Sentiment, Technical_Context, News_Impact, Conviction_Score. Sentiment must be Bullish, Bearish, or Neutral; Conviction_Score must be an integer from 1 to 10. This score is directional: 1-4 bearish, 5 undecided, 6-10 bullish. It is not a calibrated probability or a confidence guarantee. Evaluate xgb_probability against dynamic_threshold_used and vix_percentile. Use the raw CMF_20, OBV_trend/OBV_Change_5D, RSI_14 and MACD in the payload for the technical tie-break. Discuss SHAP contributions as log-odds, not causal proof. CRITICAL INSTRUCTION: You are strictly forbidden from defaulting to a 5/10 conviction just because the XGBoost probability is near 50%. When the statistical signal is weak, you MUST act as a technical tie-breaker using the raw indicators provided in the payload. If indicators show distribution (e.g., CMF < 0, falling OBV, weak RSI), you MUST lower conviction to 3/10 or 4/10. If they show accumulation, you MUST raise it to 6/10 or 7/10. Force a directional bias based strictly on the technical context. Only use supplied evidence; null indicators are unavailable. Explain conflicting signals explicitly. If the XGBoost score is inside its ambiguity zone, state that the directional view is a technical lean, not a statistical edge established by XGBoost. Account for technical_data_quality, especially estimated volume affecting CMF and OBV. No headlines are supplied: News_Impact must say news was not evaluated, without inventing sentiment. Write concise analysis in Turkish.'

LLM_SCHEMA: dict[str, Any] = {'type': 'object', 'properties': {'Sentiment': {'type': 'string', 'enum': ['Bullish', 'Bearish', 'Neutral']}, 'Technical_Context': {'type': 'string', 'minLength': 1}, 'News_Impact': {'type': 'string', 'minLength': 1}, 'Conviction_Score': {'type': 'integer', 'minimum': 1, 'maximum': 10}}, 'required': ['Sentiment', 'Technical_Context', 'News_Impact', 'Conviction_Score'], 'additionalProperties': False}

_DOWNLOAD_LOCK = threading.Lock()


def calculate_dynamic_thresholds(
    vix_percentile: float,
) -> dict[str, float]:
    """Interpolate the ambiguity zone using a causal VIX-level percentile."""
    value = float(vix_percentile)
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError("VIX percentile must be finite and within [0, 100].")

    fraction = float(np.clip((value - 25.0) / 50.0, 0.0, 1.0))
    return {
        "lower": 0.48 - 0.13 * fraction,
        "upper": 0.52 + 0.13 * fraction,
    }


def trading_signal(probability: float, vix_percentile: float) -> int:
    """Return the relative-alpha direction; threshold boundaries are inclusive."""
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be finite and within [0, 1].")

    bounds = calculate_dynamic_thresholds(vix_percentile)
    if probability >= bounds["upper"]:
        return 1
    if probability <= bounds["lower"]:
        return -1
    return 0


def vectorized_signals(
    probabilities: np.ndarray,
    percentiles: np.ndarray,
) -> np.ndarray:
    """Use exactly the same threshold arithmetic during training and inference."""
    probabilities = np.asarray(probabilities, dtype=float)
    percentiles = np.asarray(percentiles, dtype=float)
    if probabilities.shape != percentiles.shape:
        raise ValueError("Probability and percentile shapes differ.")
    if (
        not np.isfinite(probabilities).all()
        or not np.isfinite(percentiles).all()
        or ((probabilities < 0.0) | (probabilities > 1.0)).any()
        or ((percentiles < 0.0) | (percentiles > 100.0)).any()
    ):
        raise ValueError("Invalid probability or VIX-percentile inputs.")

    fraction = np.clip((percentiles - 25.0) / 50.0, 0.0, 1.0)
    lower = 0.48 - 0.13 * fraction
    upper = 0.52 + 0.13 * fraction
    return np.where(
        probabilities >= upper,
        1,
        np.where(probabilities <= lower, -1, 0),
    ).astype(np.int8)


def normalize_ticker(ticker: str) -> str:
    value = ticker.strip().upper().removesuffix(".IS")
    if not re.fullmatch(r"[A-Z0-9]{3,12}", value):
        raise ValueError("Invalid BIST ticker.")
    return f"{value}.IS"


def download_history(
    symbol: str,
    period: str = HISTORY_PERIOD,
) -> pd.DataFrame:
    """Download adjusted closed bars, retrying transient Yahoo failures."""
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            with _DOWNLOAD_LOCK:
                frame = yf.Ticker(symbol).history(
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    actions=False,
                    timeout=25,
                    raise_errors=True,
                )
            if frame.empty:
                raise ValueError(f"No history returned for {symbol}.")

            columns = ["Open", "High", "Low", "Close", "Volume"]
            if not set(columns).issubset(frame.columns):
                raise ValueError(f"Missing OHLCV fields for {symbol}.")

            frame = frame.loc[:, columns].copy()
            index = pd.DatetimeIndex(frame.index)
            if index.tz is not None:
                index = index.tz_localize(None)
            frame.index = index.normalize()
            frame = frame.sort_index()

            if frame.index.has_duplicates:
                raise ValueError(f"Duplicate daily bars for {symbol}.")

            # Conservatively exclude today's candle even after the close.
            # This also avoids treating an unfinished overseas session as final.
            cutoff = pd.Timestamp.now(tz="Europe/Istanbul").tz_localize(None)
            frame = frame.loc[frame.index < cutoff.normalize()]
            frame = frame.apply(pd.to_numeric, errors="coerce")
            frame = frame.replace([np.inf, -np.inf], np.nan)

            prices = frame[["Open", "High", "Low", "Close"]]
            valid = (
                prices.notna().all(axis=1)
                & prices.gt(0).all(axis=1)
                & frame["High"].ge(prices[["Open", "Close"]].max(axis=1))
                & frame["Low"].le(prices[["Open", "Close"]].min(axis=1))
                & frame["High"].ge(frame["Low"])
            )
            frame = frame.loc[valid].copy()
            frame.loc[frame["Volume"] < 0, "Volume"] = np.nan

            if len(frame) < 180:
                raise ValueError(f"Insufficient valid history for {symbol}.")
            return frame
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (2**attempt))

    raise RuntimeError(f"Yahoo download failed for {symbol}: {last_error}")


@st.cache_data(ttl=900, max_entries=96, show_spinner=False)
def cached_history(symbol: str, as_of_date: str) -> pd.DataFrame:
    """Share macro downloads across tickers; failed calls are not cached."""
    del as_of_date
    return download_history(symbol)


def align_equity_to_benchmark(
    equity: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.Series]:
    """Forward-fill prices without manufacturing volume or prelisting data."""
    calendar = pd.DatetimeIndex(calendar).sort_values()
    if calendar.has_duplicates:
        raise ValueError("Benchmark calendar contains duplicate dates.")

    raw = equity.reindex(calendar)
    observed = raw[["Open", "High", "Low", "Close", "Volume"]].notna().all(
        axis=1
    )
    aligned = raw.ffill()
    synthetic = ~observed

    for name in ("Open", "High", "Low"):
        aligned.loc[synthetic, name] = aligned.loc[synthetic, "Close"]
    aligned.loc[synthetic, "Volume"] = 0.0
    aligned = aligned.loc[aligned["Close"].notna()].copy()

    return aligned, observed.reindex(aligned.index).astype(bool)


def _wilder(series: pd.Series, period: int = 14) -> pd.Series:
    """Causal Wilder-type exponential smoothing, identical in both pipelines."""
    return series.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def build_stationary_features(
    equity: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Construct dimensionless local features using only current/past bars."""
    close = equity["Close"]
    high = equity["High"]
    low = equity["Low"]
    volume = equity["Volume"]

    change = close.diff()
    gain = _wilder(change.clip(lower=0))
    loss = _wilder(-change.clip(upper=0))
    rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0, np.nan))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)

    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    macd = (
        close.ewm(span=12, adjust=False, min_periods=12).mean()
        - close.ewm(span=26, adjust=False, min_periods=26).mean()
    )

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = _wilder(true_range)

    upward = high.diff()
    downward = -low.diff()
    positive_dm = upward.where((upward > downward) & (upward > 0), 0.0)
    negative_dm = downward.where((downward > upward) & (downward > 0), 0.0)
    positive_di = 100.0 * _wilder(positive_dm) / atr.replace(0, np.nan)
    negative_di = 100.0 * _wilder(negative_dm) / atr.replace(0, np.nan)
    denominator = positive_di + negative_di
    dx = 100.0 * (positive_di - negative_di).abs()
    dx = dx / denominator.replace(0, np.nan)
    dx = dx.mask(denominator == 0, 0.0)
    adx = _wilder(dx)

    spread = high - low
    multiplier = ((2.0 * close - high - low) / spread.replace(0, np.nan))
    multiplier = multiplier.mask(spread == 0, 0.0)
    cmf = (multiplier * volume).rolling(20).sum()
    cmf = cmf / volume.rolling(20).sum().replace(0, np.nan)

    rsi_min = rsi.rolling(14).min()
    rsi_range = rsi.rolling(14).max() - rsi_min
    stochastic = 100.0 * (rsi - rsi_min) / rsi_range.replace(0, np.nan)
    stochastic = stochastic.mask(rsi_range == 0, 50.0)
    stochastic_k = stochastic.rolling(3).mean()

    benchmark_return = benchmark["Close"].pct_change(fill_method=None)
    frame = pd.DataFrame(
        {
            "EMA20_Distance": close / ema20 - 1.0,
            "EMA50_Distance": close / ema50 - 1.0,
            "Volume_Surge": (
                volume / volume.rolling(20).mean().replace(0, np.nan)
            ),
            "RSI_Momentum_5D": rsi - rsi.shift(5),
            "Relative_Return_1D": (
                close.pct_change(fill_method=None)
                - benchmark_return.reindex(equity.index)
            ),
            "MACD_Normalized": macd / close,
            "ATR_Normalized": atr / close,
            "ADX_14": adx,
            "CMF_20": cmf,
            "StochRSI_K": stochastic_k,
        },
        index=equity.index,
    )
    return frame.replace([np.inf, -np.inf], np.nan)


def _available_on_calendar(
    native: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Make a source-session value available no earlier than its next date."""
    available = native.copy()
    available["_source_date"] = available.index
    available.index = available.index + pd.Timedelta(days=1)

    union = available.index.union(calendar).sort_values()
    aligned = available.reindex(union).ffill().reindex(calendar)
    age = pd.Series(calendar, index=calendar) - aligned["_source_date"]
    stale = age > pd.Timedelta(days=7)

    return aligned.drop(columns="_source_date").mask(stale, axis=0)


def build_macro_features(
    histories: Mapping[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Return lagged macro returns and a causal lagged VIX-level percentile."""
    result = pd.DataFrame(index=calendar)
    for name in MACRO_TICKERS:
        close = histories[name]["Close"]
        returns = close.pct_change(fill_method=None)
        native = pd.DataFrame(
            {
                f"macro_{name}_1d_lag": returns,
                f"macro_{name}_5d_mean_lag": returns.rolling(5).mean(),
            }
        )

        if name == "VIX":
            native[VIX_PERCENTILE_COLUMN] = (
                close.rolling(
                    VIX_PERCENTILE_WINDOW,
                    min_periods=VIX_PERCENTILE_MIN_PERIODS,
                )
                .rank(pct=True)
                .mul(100.0)
            )

        result = result.join(_available_on_calendar(native, calendar))

    return result.replace([np.inf, -np.inf], np.nan)


def feature_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Enforce the persisted feature names, order, types, and finite values."""
    names = list(columns)
    if not names or len(names) != len(set(names)):
        raise ValueError("Feature contract is empty or contains duplicates.")
    if not set(names).issubset(FEATURE_COLUMNS):
        raise ValueError("Unknown model features.")
    if not set(names).issubset(frame.columns):
        raise ValueError("Required model features are missing.")

    matrix = frame.loc[:, names].astype(np.float32)
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Feature matrix contains missing or nonfinite values.")
    return matrix


def shap_values(
    model: XGBClassifier,
    matrix: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Explain raw margins; SHAP contributions are log-odds, not probabilities."""
    explanation = shap.TreeExplainer(model)(
        matrix,
        check_additivity=True,
    )
    values = np.asarray(explanation.values, dtype=float)
    if values.shape != matrix.shape:
        raise ValueError(f"Unexpected binary SHAP shape: {values.shape}")
    base = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    if base.size == 1:
        base = np.repeat(base, len(matrix))
    return values, base


@st.cache_resource(show_spinner=False)
def _load_model(
    path: str,
    modified_ns: int,
    size: int,
) -> XGBClassifier:
    del modified_ns, size
    model = XGBClassifier()
    model.load_model(path)
    booster = model.get_booster()

    if booster.attr("feature_contract_version") != FEATURE_CONTRACT_VERSION:
        raise ValueError("Model requires retraining with the dynamic-alpha code.")
    if booster.attr("target_definition") != TARGET_DEFINITION:
        raise ValueError("Model target contract is incompatible.")

    features = json.loads(booster.attr("selected_features") or "null")
    if (
        not isinstance(features, list)
        or not features
        or len(features) != len(set(features))
        or not set(features).issubset(FEATURE_COLUMNS)
        or booster.feature_names != features
    ):
        raise ValueError("Invalid persisted model feature subset.")

    expected_policy = {
        "percentile_source": "lagged_VIX_level",
        "window": VIX_PERCENTILE_WINDOW,
        "min_periods": VIX_PERCENTILE_MIN_PERIODS,
        "low_regime": [25.0, 0.48, 0.52],
        "high_regime": [75.0, 0.35, 0.65],
    }
    actual_policy = json.loads(booster.attr("threshold_policy") or "null")
    if actual_policy != expected_policy:
        raise ValueError("Training/inference threshold policies differ.")

    return model


def load_xgboost_model() -> XGBClassifier:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "bist_xgb_model.json is missing. Run train_xgboost.py first."
        )
    stat = MODEL_PATH.stat()
    return _load_model(str(MODEL_PATH), stat.st_mtime_ns, stat.st_size)


def threshold_policy() -> dict[str, Any]:
    return {
        "percentile_source": "lagged_VIX_level",
        "window": VIX_PERCENTILE_WINDOW,
        "min_periods": VIX_PERCENTILE_MIN_PERIODS,
        "low_regime": [25.0, 0.48, 0.52],
        "high_regime": [75.0, 0.35, 0.65],
    }


def prepare_latest_inference_bar(
    aligned: pd.DataFrame,
    equity: pd.DataFrame,
    observed: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill inference volume and restore latest prices without mutating caches.

    Historical training alignment and the v3 feature formulas remain unchanged.
    Missing volume is an estimate, never reported as an actual observation.
    """
    if aligned.empty:
        raise ValueError("No equity bars are available for inference.")
    repaired = aligned.copy(deep=True)
    date = repaired.index[-1]
    raw = equity.reindex([date]).iloc[0]

    # Alignment classifies a NaN-volume bar as synthetic and flattens its OHLC.
    # Restore genuine available OHLC prices before estimating its volume.
    prices = pd.to_numeric(raw[["Open", "High", "Low", "Close"]], errors="coerce")
    actual_prices = bool(
        np.isfinite(prices.to_numpy(dtype=float)).all()
        and prices.gt(0).all()
        and prices["High"] >= max(prices["Open"], prices["Close"])
        and prices["Low"] <= min(prices["Open"], prices["Close"])
    )
    if actual_prices:
        repaired.loc[date, prices.index] = prices.to_numpy()

    close = float(repaired.at[date, "Close"])
    if not math.isfinite(close) or close <= 0:
        raise ValueError("Latest available Close must be finite and positive.")

    # Volume must never block inference. Work on a copy, use only preceding
    # observations, and use 1 when no positive history exists.
    volumes = pd.to_numeric(repaired["Volume"], errors="coerce")
    valid = np.isfinite(volumes) & volumes.gt(0)
    estimated = not bool(valid.iloc[-1])
    repaired["Volume"] = volumes.where(valid).ffill().fillna(1.0)
    volume = float(repaired.at[date, "Volume"])
    previous = volumes.loc[valid & (volumes.index < date)]
    source_date = (
        previous.index[-1].date().isoformat()
        if estimated and not previous.empty else None
    )
    method = (
        ("forward_fill" if source_date else "constant_one")
        if estimated else None
    )

    return repaired, {
        "latest_bar_marked_synthetic": not bool(observed.loc[date]),
        "latest_prices_observed": actual_prices,
        "latest_volume_imputed": estimated,
        "volume_used": volume,
        "volume_source_date": source_date,
        "volume_fallback_method": method,
    }


def raw_technical_context(
    equity: pd.DataFrame,
    local: pd.DataFrame,
) -> dict[str, Any]:
    """Supply causal raw technical context without changing the ML feature set."""
    close = equity["Close"]
    change = close.diff()
    gain = _wilder(change.clip(lower=0))
    loss = _wilder(-change.clip(upper=0))
    rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0, np.nan))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)
    obv = (np.sign(change).fillna(0.0) * equity["Volume"]).cumsum()
    obv_change = float(obv.iloc[-1] - obv.iloc[-6])
    raw = {
        "CMF_20": float(local["CMF_20"].iloc[-1]),
        "OBV": float(obv.iloc[-1]),
        "OBV_Change_5D": obv_change,
        "OBV_trend": (
            "rising" if obv_change > 0 else "falling" if obv_change < 0
            else "flat"
        ),
        "RSI_14": float(rsi.iloc[-1]),
        "MACD": float(local["MACD_Normalized"].iloc[-1] * close.iloc[-1]),
    }
    # Missing indicators remain explicit nulls, never invented neutral values.
    return {
        name: None if isinstance(value, float) and not math.isfinite(value)
        else value for name, value in raw.items()
    }


def predict_uptrend(ticker: str) -> tuple[dict[str, Any], dict[str, float]]:
    """Return existing application evidence plus the structured LLM context."""
    model = load_xgboost_model()
    day = datetime.now().date().isoformat()
    symbol = normalize_ticker(ticker)

    histories = {
        name: cached_history(macro_ticker, day)
        for name, macro_ticker in MACRO_TICKERS.items()
    }
    benchmark = histories["BIST100"]
    equity = cached_history(symbol, day)
    aligned, observed = align_equity_to_benchmark(equity, benchmark.index)

    if aligned.empty or aligned.index[-1] != benchmark.index[-1]:
        raise ValueError("Equity does not reach the latest benchmark session.")
    aligned, data_quality = prepare_latest_inference_bar(
        aligned, equity, observed
    )

    local = build_stationary_features(aligned, benchmark)
    macro = build_macro_features(histories, benchmark.index)
    row = local.join(macro).tail(1)
    features = model.get_booster().feature_names
    matrix = feature_matrix(row, features)

    percentile = float(row[VIX_PERCENTILE_COLUMN].iloc[0])
    bounds = calculate_dynamic_thresholds(percentile)
    probability = float(model.predict_proba(matrix)[0, 1])
    signal = trading_signal(probability, percentile)
    contributions, base = shap_values(model, matrix)

    order = np.argsort(-np.abs(contributions[0]), kind="stable")[:3]
    top = [
        {
            "feature": features[index],
            "value": float(matrix.iloc[0, index]),
            "shap_log_odds": float(contributions[0, index]),
            "driver_type": (
                "technical"
                if features[index] in LOCAL_FEATURE_COLUMNS
                else "macro"
            ),
        }
        for index in order
    ]

    raw_technicals = raw_technical_context(aligned, local)
    context = {
        "xgb_probability": probability,
        "vix_percentile": percentile,
        "dynamic_threshold_used": bounds,
        "top_3_shap_contributors": top,
        **raw_technicals,
        "technical_data_quality": data_quality,
    }
    status = (
        INSUFFICIENT_SIGNAL
        if signal == 0
        else "Relative Outperformance" if signal > 0
        else "Relative Underperformance"
    )

    evidence: dict[str, Any] = {
        "XGBoost_Uptrend_Probability_Percent": probability * 100.0,
        "XGB_Prob": probability * 100.0,
        "probability": probability,
        "signal": signal,
        "Signal_Status": status,
        "abstain": signal == 0,
        "feature_count": len(features),
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "target_definition": TARGET_DEFINITION,
        "as_of": row.index[-1].date().isoformat(),
        "data_quality": data_quality,
        "vix_percentile": percentile,
        "dynamic_threshold_used": bounds,
        "top_3_shap_contributors": top,
        "shap_base_log_odds": float(base[0]),
        "LLM_Context": context,
        "validation": json.loads(
            model.get_booster().attr("validation_report") or "{}"
        ),
    }
    technical = {
        name: float(local.iloc[-1][name])
        for name in LOCAL_FEATURE_COLUMNS
        if pd.notna(local.iloc[-1][name])
    }
    return evidence, technical


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON field: {key}")
        output[key] = value
    return output


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def query_ollama(context: Mapping[str, Any]) -> dict[str, Any]:
    """Send only the strict quantitative context to the requested synthesizer."""
    # Persistent 0.49-0.51 probabilities across tickers can indicate collapse
    # toward the base margin; extreme Optuna regularization is a possible cause,
    # not a diagnosis established by the probability range alone.
    prompt_json = json.dumps(context, allow_nan=False, indent=2)
    debug_path = Path(__file__).resolve().parents[1] / "debug_llm_payload.json"
    with debug_path.open(mode="w", encoding="utf-8") as debug_file:
        debug_file.write(prompt_json)
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "system": SYSTEM_PROMPT,
            "prompt": prompt_json,
            "format": LLM_SCHEMA,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "seed": 42,
                "num_ctx": 4096,
                "num_predict": 600,
            },
        },
        timeout=(5, 180),
    )
    response.raise_for_status()
    envelope = response.json()
    raw = envelope.get("response")
    if not isinstance(raw, str):
        raise ValueError("Ollama response text is missing.")

    parsed = json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
    )
    required = {"Sentiment", "Technical_Context", "News_Impact", "Conviction_Score"}
    if not isinstance(parsed, dict) or set(parsed) != required:
        raise ValueError("Invalid four-field analysis JSON contract.")
    if parsed["Sentiment"] not in {"Bullish", "Bearish", "Neutral"}:
        raise ValueError("Invalid sentiment.")
    score = parsed["Conviction_Score"]
    if type(score) is not int or not 1 <= score <= 10:
        raise ValueError("Conviction_Score must be an integer from 1 to 10.")
    for field in ("Technical_Context", "News_Impact"):
        if (
            not isinstance(parsed[field], str)
            or not parsed[field].strip()
            or len(parsed[field]) > 4000
        ):
            raise ValueError(f"Invalid analysis text: {field}")

    return parsed


def analyze(
    ticker: str,
    snapshot: Mapping[str, Any] | None = None,
    headlines: Sequence[Any] | None = None,
    news_status: str | None = None,
) -> dict[str, Any]:
    """Keep the existing UI envelope while exposing the new LLM JSON separately."""
    # The new strict synthesizer deliberately consumes quantitative context only.
    # These arguments remain accepted so existing app.py calls continue to work.
    del snapshot, headlines, news_status

    result: dict[str, Any] = {
        "XGB_Prob": None,
        "ML_Evidence": None,
        "ML_Error": None,
        "Analysis": None,
        "AI_Error": None,
        "Quant_Synthesis": None,
        "Signal_Status": INSUFFICIENT_SIGNAL,
    }

    try:
        evidence, technical = predict_uptrend(ticker)
        result.update(
            XGB_Prob=evidence["XGB_Prob"],
            ML_Evidence=evidence,
            Technical_Snapshot=technical,
            Signal_Status=evidence["Signal_Status"],
        )
    except Exception as exc:
        result["ML_Error"] = str(exc)
        return result

    try:
        synthesis = query_ollama(evidence["LLM_Context"])
        # The validated LLM object is the sole source of displayed assessment.
        # Keep statistical abstention in ML_Evidence, separate from the LLM lean.
        result["Quant_Synthesis"] = synthesis
        result["Analysis"] = synthesis
    except (requests.RequestException, ValueError, TypeError, KeyError, OSError) as exc:
        result["AI_Error"] = str(exc)
        # No synthetic Neutral/5 fallback when inference or parsing fails.
    return result
