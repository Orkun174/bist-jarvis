"""Model features: dimensionless local indicators and lagged macro returns.

Ported from core/ai_analyzer.py (which cannot be imported in the cloud job
because it pulls in streamlit and shap). tests/jarvis/test_features.py checks
that both implementations produce identical numbers.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from jarvis.macro import MACRO_SYMBOLS

FEATURE_VERSION = "jarvis-features-v1"

LOCAL_FEATURES = (
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
MACRO_FEATURES = tuple(
    name
    for macro in MACRO_SYMBOLS
    for name in (f"macro_{macro}_1d_lag", f"macro_{macro}_5d_mean_lag")
)
FEATURES = LOCAL_FEATURES + MACRO_FEATURES

# Candidate stock-specific features. They are not part of the default recipe:
# a recipe that uses them has to win the weekly exam first.
TURNOVER_RANK = "Turnover_Rank"   # cross-sectional, filled in by dataset.build_rows
EXTRA_FEATURES = (
    "Rel_Return_5D",
    "Rel_Return_20D",
    "Rel_Return_60D",
    "Rel_Return_120D",
    "Volatility_20D",
    "High_252_Distance",
    "Volume_Trend",
    "Beta_60D",
    TURNOVER_RANK,
)
STOCK_FEATURES = LOCAL_FEATURES + EXTRA_FEATURES
FEATURES_V2 = STOCK_FEATURES + MACRO_FEATURES

# The same stock features as each day's rank among eligible stocks (0-1). An
# indicator's level drifts with the market's mood; its rank on the day does not.
RANK_SUFFIX = "_Rank"
RANKED_SOURCES = tuple(name for name in STOCK_FEATURES if name != TURNOVER_RANK)
RANKED_FEATURES = tuple(name + RANK_SUFFIX for name in RANKED_SOURCES) + (TURNOVER_RANK,)
FEATURES_V3 = RANKED_FEATURES + MACRO_FEATURES
FEATURES_V4 = RANKED_FEATURES            # no macro: pure "which stock today"
FEATURE_SETS = (FEATURES_V2, FEATURES_V3, FEATURES_V4)
FEATURE_SET_NAMES = {FEATURES: "v1", FEATURES_V2: "v2", FEATURES_V3: "v3", FEATURES_V4: "v4"}


def feature_set_name(features: tuple[str, ...]) -> str:
    """"v1".."v4" for the known feature sets, "custom-<n>" otherwise."""
    return FEATURE_SET_NAMES.get(tuple(features), f"custom-{len(features)}")

VIX_PERCENTILE = "VIX_Percentile"
VIX_PERCENTILE_WINDOW = 252
VIX_PERCENTILE_MIN_PERIODS = 126
MACRO_MAX_AGE = pd.Timedelta(days=7)


def _wilder(series: pd.Series, period: int = 14) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def local_features(equity: pd.DataFrame, benchmark_close: pd.Series) -> pd.DataFrame:
    """Local features for one ticker, using only current and past bars."""
    close, high, low, volume = equity["Close"], equity["High"], equity["Low"], equity["Volume"]

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
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
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
    multiplier = (2.0 * close - high - low) / spread.replace(0, np.nan)
    multiplier = multiplier.mask(spread == 0, 0.0)
    cmf = (multiplier * volume).rolling(20).sum()
    cmf = cmf / volume.rolling(20).sum().replace(0, np.nan)

    rsi_min = rsi.rolling(14).min()
    rsi_range = rsi.rolling(14).max() - rsi_min
    stochastic = 100.0 * (rsi - rsi_min) / rsi_range.replace(0, np.nan)
    stochastic = stochastic.mask(rsi_range == 0, 50.0)
    stochastic_k = stochastic.rolling(3).mean()

    benchmark_return = benchmark_close.pct_change(fill_method=None)
    frame = pd.DataFrame(
        {
            "EMA20_Distance": close / ema20 - 1.0,
            "EMA50_Distance": close / ema50 - 1.0,
            "Volume_Surge": volume / volume.rolling(20).mean().replace(0, np.nan),
            "RSI_Momentum_5D": rsi - rsi.shift(5),
            "Relative_Return_1D": (
                close.pct_change(fill_method=None) - benchmark_return.reindex(equity.index)
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


def extra_features(equity: pd.DataFrame, benchmark_close: pd.Series) -> pd.DataFrame:
    """Longer-horizon stock features, using only current and past bars.

    Returns every EXTRA_FEATURES column except the cross-sectional
    TURNOVER_RANK, plus the raw "_turnover" it is ranked from.
    """
    close, volume = equity["Close"], equity["Volume"]
    benchmark = benchmark_close.reindex(equity.index)
    returns = close.pct_change(fill_method=None)
    benchmark_returns = benchmark.pct_change(fill_method=None)

    columns = {}
    for span in (5, 20, 60, 120):
        columns[f"Rel_Return_{span}D"] = (
            close.pct_change(span, fill_method=None) - benchmark.pct_change(span, fill_method=None)
        )
    columns["Volatility_20D"] = returns.rolling(20).std()
    columns["High_252_Distance"] = close / close.rolling(252, min_periods=120).max() - 1.0
    columns["Volume_Trend"] = (
        volume.rolling(20).mean() / volume.rolling(120).mean().replace(0, np.nan)
    )
    variance = benchmark_returns.rolling(60).var()
    columns["Beta_60D"] = returns.rolling(60).cov(benchmark_returns) / variance.replace(0, np.nan)
    columns["_turnover"] = (close * volume).rolling(20).median()
    return pd.DataFrame(columns, index=equity.index).replace([np.inf, -np.inf], np.nan)


def _available_on_calendar(native: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Make a source-session value usable no earlier than the following date."""
    available = native.copy()
    available["_source_date"] = available.index
    available.index = available.index + pd.Timedelta(days=1)

    union = available.index.union(calendar).sort_values()
    aligned = available.reindex(union).ffill().reindex(calendar)
    age = pd.Series(calendar, index=calendar) - aligned["_source_date"]
    return aligned.drop(columns="_source_date").mask(age > MACRO_MAX_AGE, axis=0)


def macro_features(
    histories: Mapping[str, pd.DataFrame], calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    """Lagged macro returns plus a causal, lagged VIX-level percentile."""
    result = pd.DataFrame(index=calendar)
    for name in MACRO_SYMBOLS:
        history = histories.get(name)
        columns = [f"macro_{name}_1d_lag", f"macro_{name}_5d_mean_lag"]
        if name == "VIX":
            columns.append(VIX_PERCENTILE)
        if history is None or history.empty:
            for column in columns:
                result[column] = np.nan
            continue
        close = history["Close"]
        returns = close.pct_change(fill_method=None)
        native = pd.DataFrame({columns[0]: returns, columns[1]: returns.rolling(5).mean()})
        if name == "VIX":
            native[VIX_PERCENTILE] = (
                close.rolling(VIX_PERCENTILE_WINDOW, min_periods=VIX_PERCENTILE_MIN_PERIODS)
                .rank(pct=True)
                .mul(100.0)
            )
        result = result.join(_available_on_calendar(native, calendar))
    ordered = list(MACRO_FEATURES) + [VIX_PERCENTILE]
    return result[ordered].replace([np.inf, -np.inf], np.nan)
