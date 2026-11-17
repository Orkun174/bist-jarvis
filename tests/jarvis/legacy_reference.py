"""Frozen copy of the original feature functions from core/ai_analyzer.py.

core/ was removed on 2026-11-17. These four functions (and the constants they
use) are kept verbatim so tests/jarvis/test_features.py can keep checking that
jarvis.features produces the same numbers as the model it was ported from.
Do not edit; this is a reference, not production code.
"""
# ruff: noqa
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

BENCHMARK_TICKER = 'XU100.IS'
MACRO_TICKERS: dict[str, str] = {'USDTRY': 'USDTRY=X', 'BRENT': 'BZ=F', 'GOLD': 'GC=F', 'SP500': '^GSPC', 'VIX': '^VIX', 'BIST100': BENCHMARK_TICKER, 'BISTBANK': 'XBANK.IS'}
VIX_PERCENTILE_COLUMN = 'VIX_Percentile'
VIX_PERCENTILE_WINDOW = 252
VIX_PERCENTILE_MIN_PERIODS = 126

def _wilder(series: pd.Series, period: int=14) -> pd.Series:
    """Causal Wilder-type exponential smoothing, identical in both pipelines."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

def build_stationary_features(equity: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Construct dimensionless local features using only current/past bars."""
    close = equity['Close']
    high = equity['High']
    low = equity['Low']
    volume = equity['Volume']
    change = close.diff()
    gain = _wilder(change.clip(lower=0))
    loss = _wilder(-change.clip(upper=0))
    rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0, np.nan))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    macd = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(span=26, adjust=False, min_periods=26).mean()
    previous_close = close.shift(1)
    true_range = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
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
    benchmark_return = benchmark['Close'].pct_change(fill_method=None)
    frame = pd.DataFrame({'EMA20_Distance': close / ema20 - 1.0, 'EMA50_Distance': close / ema50 - 1.0, 'Volume_Surge': volume / volume.rolling(20).mean().replace(0, np.nan), 'RSI_Momentum_5D': rsi - rsi.shift(5), 'Relative_Return_1D': close.pct_change(fill_method=None) - benchmark_return.reindex(equity.index), 'MACD_Normalized': macd / close, 'ATR_Normalized': atr / close, 'ADX_14': adx, 'CMF_20': cmf, 'StochRSI_K': stochastic_k}, index=equity.index)
    return frame.replace([np.inf, -np.inf], np.nan)

def _available_on_calendar(native: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Make a source-session value available no earlier than its next date."""
    available = native.copy()
    available['_source_date'] = available.index
    available.index = available.index + pd.Timedelta(days=1)
    union = available.index.union(calendar).sort_values()
    aligned = available.reindex(union).ffill().reindex(calendar)
    age = pd.Series(calendar, index=calendar) - aligned['_source_date']
    stale = age > pd.Timedelta(days=7)
    return aligned.drop(columns='_source_date').mask(stale, axis=0)

def build_macro_features(histories: Mapping[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Return lagged macro returns and a causal lagged VIX-level percentile."""
    result = pd.DataFrame(index=calendar)
    for name in MACRO_TICKERS:
        close = histories[name]['Close']
        returns = close.pct_change(fill_method=None)
        native = pd.DataFrame({f'macro_{name}_1d_lag': returns, f'macro_{name}_5d_mean_lag': returns.rolling(5).mean()})
        if name == 'VIX':
            native[VIX_PERCENTILE_COLUMN] = close.rolling(VIX_PERCENTILE_WINDOW, min_periods=VIX_PERCENTILE_MIN_PERIODS).rank(pct=True).mul(100.0)
        result = result.join(_available_on_calendar(native, calendar))
    return result.replace([np.inf, -np.inf], np.nan)
