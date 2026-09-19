# train_xgboost.py
"""Yahoo data validation and causal local/macro feature generation.

Extracted from the original trainer. Raw local features remain unchanged here;
core.ml_contract applies explicit price normalization downstream in both training
and live inference. Nonconsensus dates are excluded and recorded, rather than
causing an otherwise valid ticker to be discarded.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

from core.ai_analyzer import (
    FEATURE_COLUMNS as LOCAL_FEATURE_COLUMNS,
    FEATURE_CONTRACT_VERSION as LOCAL_CONTRACT_VERSION,
    TARGET_DEFINITION,
    build_feature_matrix as build_local_feature_matrix,
)
from core.market_data import calculate_indicators


LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "bist_xgb_model.json"

# Curated major BIST equities; membership is not asserted for every historical date.
EQUITY_TICKERS = (
    "AKBNK.IS",
    "THYAO.IS",
    "KCHOL.IS",
    "TUPRS.IS",
    "FROTO.IS",
    "GARAN.IS",
    "ISCTR.IS",
    "YKBNK.IS",
    "EREGL.IS",
    "BIMAS.IS",
    "SISE.IS",
    "SAHOL.IS",
    "TOASO.IS",
    "PGSUS.IS",
    "ASELS.IS",
    "TCELL.IS",
    "ENKAI.IS",
    "KRDMD.IS",
    "PETKM.IS",
    "TTKOM.IS",
    "ARCLK.IS",
    "MGROS.IS",
    "ULKER.IS",
    "TAVHL.IS",
    "EKGYO.IS",
    "SASA.IS",
    "ASTOR.IS",
    "ALARK.IS",
)

# Insertion order is part of the persisted model feature contract.
MACRO_TICKERS = {
    "USDTRY": "USDTRY=X",
    "BRENT": "BZ=F",
    "GOLD": "GC=F",
    "SP500": "^GSPC",
    "VIX": "^VIX",
    "BIST100": "XU100.IS",
    "BISTBANK": "XBANK.IS",
}

MACRO_FEATURE_COLUMNS = tuple(
    feature
    for name in MACRO_TICKERS
    for feature in (
        f"{name}_Return_1D_Pct",
        f"{name}_Return_5D_Mean_Pct",
    )
)
FEATURE_COLUMNS = tuple(LOCAL_FEATURE_COLUMNS) + MACRO_FEATURE_COLUMNS
FEATURE_CONTRACT_VERSION = (
    f"{LOCAL_CONTRACT_VERSION}+macro-native-returns-lag1-v1"
)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
HORIZON = 3
RETURN_THRESHOLD = 0.02
MIN_HISTORY_BARS = 252
MIN_TICKER_ROWS = 150
MIN_SUCCESSFUL_TICKERS = 15
MACRO_MAX_AGE_DAYS = 7

MACRO_TIMING_POLICY = {
    "source": "yfinance daily Close, auto_adjust=False",
    "return_1d": "100 * Close.pct_change(fill_method=None)",
    "return_5d_mean": "rolling mean of 5 native-observation 1d returns",
    "availability": "source session date + 1 calendar day",
    "alignment": "forward-fill eligible macro signals onto BIST session dates",
    "maximum_source_age_calendar_days": MACRO_MAX_AGE_DAYS,
    "backfill": False,
    "holiday_behavior": "carry last eligible signal; do not insert zero returns",
}


def normalize_dates(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Preserve source-market session dates and exclude unfinished current bars."""
    if frame is None or frame.empty:
        raise ValueError(f"{symbol}: no data returned.")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{symbol}: invalid date index.")
    if frame.index.isna().any():
        raise ValueError(f"{symbol}: missing dates.")

    result = frame.copy()
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    result.index = result.index.normalize()

    if result.index.duplicated().any():
        raise ValueError(f"{symbol}: duplicate session dates.")

    result = result.sort_index()
    today = pd.Timestamp.now(tz="Europe/Istanbul").tz_localize(None).normalize()
    return result.loc[result.index < today]


def validate_equity(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Reject corrupt equity bars instead of silently changing target horizons."""
    result = normalize_dates(frame, symbol)
    if not set(OHLCV_COLUMNS).issubset(result.columns):
        raise ValueError(f"{symbol}: missing OHLCV columns.")

    result = result.loc[:, OHLCV_COLUMNS].copy()
    for column in OHLCV_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{symbol}: missing or nonfinite OHLCV values.")

    invalid = (
        (result[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
        | (result["Volume"] < 0)
        | (
            result["High"]
            < result[["Open", "Close", "Low"]].max(axis=1)
        )
        | (
            result["Low"]
            > result[["Open", "Close", "High"]].min(axis=1)
        )
    )
    if invalid.any():
        raise ValueError(f"{symbol}: inconsistent OHLCV bars.")
    if len(result) < MIN_HISTORY_BARS:
        raise ValueError(f"{symbol}: insufficient completed history.")
    return result


def validate_macro(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Validate macro closes; volume is not required for indices or FX."""
    result = normalize_dates(frame, symbol)
    if "Close" not in result:
        raise ValueError(f"{symbol}: missing Close column.")

    close = pd.to_numeric(result["Close"], errors="coerce")
    if not np.isfinite(close.to_numpy(dtype=float)).all():
        raise ValueError(f"{symbol}: missing or nonfinite closes.")
    if (close <= 0).any():
        raise ValueError(f"{symbol}: nonpositive close.")
    if len(close) < MIN_HISTORY_BARS:
        raise ValueError(f"{symbol}: insufficient macro history.")

    return close.to_frame("Close")


def fetch_history(
    symbol: str,
    *,
    macro: bool = False,
    attempts: int = 3,
) -> pd.DataFrame:
    """Fetch five years with bounded retries for network and provider failures."""
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            raw = yf.Ticker(symbol).history(
                period="5y",
                interval="1d",
                auto_adjust=False,
                actions=False,
                timeout=30,
            )
            validator = validate_macro if macro else validate_equity
            return validator(raw, symbol)
        except Exception as exc:
            last_error = exc
            LOGGER.warning(
                "%s: attempt %d/%d failed: %s",
                symbol,
                attempt,
                attempts,
                exc,
            )
            if attempt < attempts:
                time.sleep(min(2 ** attempt, 8))

    raise RuntimeError(
        f"{symbol}: retrieval failed after {attempts} attempts."
    ) from last_error


def infer_bist_sessions(
    histories: dict[str, pd.DataFrame],
) -> pd.DatetimeIndex:
    """
    Infer observed BIST sessions to detect ticker-specific missing dates.

    A session needs observations from at least 60% of histories spanning it.
    This cannot detect an outage affecting every instrument simultaneously.
    """
    dates = pd.DatetimeIndex(
        sorted(set().union(*(set(frame.index) for frame in histories.values())))
    )
    observations = np.zeros(len(dates), dtype=np.int32)
    active = np.zeros(len(dates), dtype=np.int32)

    for frame in histories.values():
        observations += dates.isin(frame.index).astype(np.int32)
        active += (
            (dates >= frame.index.min()) & (dates <= frame.index.max())
        ).astype(np.int32)

    threshold = np.maximum(2, np.ceil(active * 0.60).astype(int))
    sessions = dates[observations >= threshold]
    if len(sessions) < MIN_HISTORY_BARS:
        raise ValueError("Insufficient common BIST sessions.")
    return sessions


def build_macro_features(
    histories: dict[str, pd.DataFrame],
    bist_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Calculate native-market returns, lag availability, and forward-fill."""
    pieces = []

    for name, symbol in MACRO_TICKERS.items():
        history = histories[name]
        close = history["Close"].astype(float)

        # Calculate before alignment: holidays must not become artificial zeros.
        daily_return = close.pct_change(fill_method=None) * 100.0
        average_return = daily_return.rolling(5, min_periods=5).mean()

        native = pd.DataFrame(
            {
                f"{name}_Return_1D_Pct": daily_return,
                f"{name}_Return_5D_Mean_Pct": average_return,
                "_Source_Date": history.index,
            },
            index=history.index,
        )

        # Yesterday's date is conservatively available at today's BIST close.
        # Same-date US/futures/FX closes are deliberately excluded.
        native.index = native.index + pd.Timedelta(days=1)

        # Include native dates before ffill so observations on BIST holidays
        # are retained and can be carried into the next BIST trading session.
        alignment_index = native.index.union(bist_dates).sort_values()
        aligned = native.reindex(alignment_index).ffill().reindex(bist_dates)

        age = (
            pd.Series(bist_dates, index=bist_dates) - aligned["_Source_Date"]
        ).dt.days
        columns = [
            f"{name}_Return_1D_Pct",
            f"{name}_Return_5D_Mean_Pct",
        ]

        # Do not silently carry an outage forward indefinitely.
        invalid_age = age.isna() | age.lt(1) | age.gt(MACRO_MAX_AGE_DAYS)
        aligned.loc[invalid_age, columns] = np.nan
        pieces.append(aligned[columns])

        LOGGER.info(
            "%s: macro features available for %d/%d BIST dates",
            symbol,
            int(aligned[columns].notna().all(axis=1).sum()),
            len(bist_dates),
        )

    result = pd.concat(pieces, axis=1)
    result = result.loc[:, list(MACRO_FEATURE_COLUMNS)]
    return result.replace([np.inf, -np.inf], np.nan)


def split_contiguous_history(
    history: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> list[pd.DataFrame]:
    """Restart indicator warm-up after missing bars; never bridge target gaps."""
    # Exclude unconfirmed exchange dates, not an otherwise valid company.
    # Every excluded date is recorded by make_training_rows below.
    history = history.loc[history.index.isin(sessions)].copy()
    positions = sessions.get_indexer(history.index)

    boundaries = np.flatnonzero(np.diff(positions) != 1) + 1
    return [
        history.iloc[indexes].copy()
        for indexes in np.split(np.arange(len(history)), boundaries)
        if len(indexes)
    ]


def build_expanded_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce the combined local-plus-macro feature order and float32 dtype."""
    if not frame.columns.is_unique:
        raise ValueError("Duplicate feature column names.")
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")

    matrix = frame.loc[:, list(FEATURE_COLUMNS)].astype(np.float32)
    if matrix.empty or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("The training matrix contains missing or infinite values.")
    return matrix


def make_training_rows(
    symbol: str,
    history: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    macro_features: pd.DataFrame,
    minimum_turnover: float,
) -> tuple[pd.DataFrame, dict]:
    """Engineer local indicators per ticker and join causally available macros."""
    blocks = split_contiguous_history(history, sessions)
    batches = []
    missing_rows = 0
    liquidity_rows = 0

    for block in blocks:
        if len(block) < 50 + HORIZON:
            continue

        # Preserve the existing local indicator implementation and ordering.
        indicators = calculate_indicators(block)
        local = build_local_feature_matrix(indicators, allow_missing=True)
        rows = local.join(macro_features, how="left", validate="one_to_one")

        # Form labels before filtering any feature or liquidity observations.
        future_close = block["Close"].shift(-HORIZON)
        future_return = future_close / block["Close"] - 1.0

        target = pd.Series(np.nan, index=block.index, dtype=float)
        known = future_close.notna()
        target.loc[known] = (
            future_return.loc[known] >= RETURN_THRESHOLD
        ).astype(np.int32)

        rows["Close"] = block["Close"]
        rows["Target"] = target
        rows["Date"] = block.index
        rows["Target_Date"] = pd.Series(
            block.index, index=block.index
        ).shift(-HORIZON)
        rows["Ticker"] = symbol.removesuffix(".IS")

        # Historical liquidity filtering uses only information available at t.
        # Close×Volume is a nominal turnover proxy, not exact exchange turnover.
        median_turnover = (
            block["Close"].astype(float) * block["Volume"].astype(float)
        ).rolling(20, min_periods=20).median()
        liquid = median_turnover.ge(minimum_turnover) & block["Volume"].gt(0)

        numeric = rows.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        finite = pd.Series(np.isfinite(numeric).all(axis=1), index=rows.index)
        complete = (
            finite
            & rows["Target"].notna()
            & rows["Target_Date"].notna()
        )

        missing_rows += int((~complete).sum())
        liquidity_rows += int((complete & ~liquid).sum())

        selected = rows.loc[complete & liquid].copy()
        if not selected.empty:
            selected["Target"] = selected["Target"].astype(np.int32)
            batches.append(selected)

    if not batches:
        raise ValueError(f"{symbol}: no complete training observations.")

    result = pd.concat(batches, ignore_index=True)
    if len(result) < MIN_TICKER_ROWS:
        raise ValueError(
            f"{symbol}: only {len(result)} complete training observations."
        )

    build_expanded_matrix(result)
    return result, {
        "observed_bars": len(history),
        "excluded_nonconsensus_dates": [str(d.date()) for d in history.index.difference(sessions)],
        "contiguous_blocks": len(blocks),
        "missing_feature_or_target_rows": missing_rows,
        "liquidity_excluded_rows": liquidity_rows,
        "retained_rows": len(result),
    }

