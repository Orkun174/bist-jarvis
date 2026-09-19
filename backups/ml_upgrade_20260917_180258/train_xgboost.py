# train_xgboost.py
"""
Train a BIST equity classifier using local indicators and Yahoo Finance macros.

All market data comes exclusively from yfinance.

Target:
    Close[t+3] / Close[t] - 1 >= 0.02
    where t+3 means three BIST trading sessions.

Local indicators reuse core.market_data.calculate_indicators and the existing
core.ai_analyzer feature builder, preserving their calculation and column order.

IMPORTANT INFERENCE CONTRACT:
    This model adds 14 macro features. The existing local-only inference engine
    cannot supply them. The saved model therefore receives a new feature-contract
    version, causing the existing loader to reject it rather than silently make
    invalid predictions. Inference must reproduce the macro features and timing
    policy defined here before using this expanded model.

Temporal policy:
    Macro returns are calculated on each instrument's native daily observations.
    Each observation becomes eligible on the following calendar date. This
    conservative lag avoids using same-date US/futures/FX daily closes that may
    not be finalized when BIST closes. Eligible signals are forward-filled onto
    BIST dates; they are never backward-filled.

The curated equity list is not a point-in-time constituent database. Historical
training on a fixed present-day selection remains subject to survivorship bias.
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

BASE_DIR = Path(__file__).resolve().parent
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
    positions = sessions.get_indexer(history.index)
    if (positions < 0).any():
        raise ValueError("Equity has dates outside the observed BIST calendar.")

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
        "contiguous_blocks": len(blocks),
        "missing_feature_or_target_rows": missing_rows,
        "liquidity_excluded_rows": liquidity_rows,
        "retained_rows": len(result),
    }


def new_classifier() -> XGBClassifier:
    """Use fixed regularization without tuning against the holdout."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=450,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=15,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=6.0,
        tree_method="hist",
        scale_pos_weight=1.0,
        n_jobs=min(4, os.cpu_count() or 1),
        random_state=42,
    )


def validate_chronologically(rows: pd.DataFrame) -> dict:
    """Keep all companies on one temporal split and purge overlapping labels."""
    dates = np.sort(rows["Date"].unique())
    if len(dates) < MIN_HISTORY_BARS:
        raise ValueError("Insufficient dates for chronological validation.")

    split_date = pd.Timestamp(dates[int(len(dates) * 0.8)])
    before_split = rows["Date"] < split_date
    label_crosses_split = rows["Target_Date"] >= split_date

    training = rows.loc[before_split & ~label_crosses_split]
    validation = rows.loc[~before_split]

    if len(training) < 2_000 or len(validation) < 300:
        raise ValueError("Insufficient rows after chronological purging.")
    if training["Target"].nunique() != 2:
        raise ValueError("Training requires both target classes.")

    model = new_classifier()
    model.fit(build_expanded_matrix(training), training["Target"])
    probabilities = model.predict_proba(
        build_expanded_matrix(validation)
    )[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError("Invalid validation probabilities.")

    actual = validation["Target"].to_numpy()
    baseline = np.full(len(validation), float(training["Target"].mean()))

    return {
        "split_date": split_date.date().isoformat(),
        "training_rows": len(training),
        "validation_rows": len(validation),
        "purged_rows": int((before_split & label_crosses_split).sum()),
        "training_positive_rate": float(training["Target"].mean()),
        "validation_positive_rate": float(validation["Target"].mean()),
        "log_loss": float(log_loss(actual, probabilities, labels=[0, 1])),
        "baseline_log_loss": float(log_loss(actual, baseline, labels=[0, 1])),
        "brier_score": float(brier_score_loss(actual, probabilities)),
        "baseline_brier_score": float(brier_score_loss(actual, baseline)),
        "roc_auc": (
            float(roc_auc_score(actual, probabilities))
            if np.unique(actual).size == 2 else None
        ),
        "notes": (
            "Purged chronological development-model validation. Production "
            "weights are subsequently refitted on all labeled rows. Scores "
            "are uncalibrated. Overlapping labels and shared macro exposures "
            "make observations dependent. Fixed-universe survivorship bias remains."
        ),
    }


def save_model(
    model: XGBClassifier,
    rows: pd.DataFrame,
    validation: dict,
    quality: dict,
    skipped: dict,
    minimum_turnover: float,
) -> None:
    """Persist the expanded contract and atomically replace production weights."""
    model.get_booster().set_attr(
        feature_contract_version=FEATURE_CONTRACT_VERSION,
        local_feature_contract_version=LOCAL_CONTRACT_VERSION,
        feature_columns=json.dumps(list(FEATURE_COLUMNS)),
        local_feature_columns=json.dumps(list(LOCAL_FEATURE_COLUMNS)),
        macro_feature_columns=json.dumps(list(MACRO_FEATURE_COLUMNS)),
        macro_tickers=json.dumps(MACRO_TICKERS),
        macro_timing_policy=json.dumps(MACRO_TIMING_POLICY),
        target_definition=TARGET_DEFINITION,
        price_basis="unadjusted",
        trained_at=datetime.now(timezone.utc).isoformat(),
        training_tickers=json.dumps(sorted(rows["Ticker"].unique().tolist())),
        last_feature_date=rows["Date"].max().date().isoformat(),
        last_label_date=rows["Target_Date"].max().date().isoformat(),
        validation_report=json.dumps(validation, allow_nan=False),
        data_quality_report=json.dumps(quality, allow_nan=False),
        skipped_tickers=json.dumps(skipped),
        minimum_20d_median_turnover_try=str(minimum_turnover),
        inference_requirement=(
            "Supply all local and macro features in feature_columns order; "
            "reproduce macro_timing_policy. Local-only inference is incompatible."
        ),
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix="bist_macro_",
        suffix=".json",
        dir=MODEL_PATH.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        model.save_model(str(temporary_path))

        # Validate serialization before replacing an existing model.
        restored = XGBClassifier()
        restored.load_model(str(temporary_path))
        if restored.get_booster().feature_names != list(FEATURE_COLUMNS):
            raise ValueError("Serialized feature order does not match the contract.")

        sample = build_expanded_matrix(rows.tail(32))
        np.testing.assert_allclose(
            model.predict_proba(sample),
            restored.predict_proba(sample),
            rtol=1e-6,
            atol=1e-7,
        )
        os.replace(temporary_path, MODEL_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    """Fetch, synchronize, engineer, validate, refit, and save the macro model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-median-turnover-try",
        type=float,
        default=20_000_000.0,
        help="Minimum trailing 20-session median Close×Volume in nominal TRY.",
    )
    args = parser.parse_args()
    minimum_turnover = args.min_median_turnover_try

    if not np.isfinite(minimum_turnover) or minimum_turnover < 0:
        parser.error("The turnover threshold must be finite and nonnegative.")

    if TARGET_DEFINITION != (
        "Close[t+3] / Close[t] - 1 >= 0.02; 3 trading sessions"
    ):
        raise ValueError("The existing target contract has changed.")
    if len(set(FEATURE_COLUMNS)) != len(FEATURE_COLUMNS):
        raise ValueError("Local and macro feature names must be unique.")

    # All seven macros are mandatory; never silently train a different schema.
    macro_histories = {}
    for name, symbol in MACRO_TICKERS.items():
        LOGGER.info("Fetching macro %s (%s)", name, symbol)
        macro_histories[name] = fetch_history(symbol, macro=True)
        time.sleep(0.4)

    equity_histories = {}
    skipped = {}
    for symbol in EQUITY_TICKERS:
        LOGGER.info("Fetching equity %s", symbol)
        try:
            equity_histories[symbol] = fetch_history(symbol)
        except Exception as exc:
            skipped[symbol] = str(exc)
            LOGGER.error("Skipping %s: %s", symbol, exc)
        time.sleep(0.4)

    if len(equity_histories) < MIN_SUCCESSFUL_TICKERS:
        raise RuntimeError(
            f"Only {len(equity_histories)} equity histories were retrieved; "
            f"{MIN_SUCCESSFUL_TICKERS} are required."
        )

    sessions = infer_bist_sessions(equity_histories)
    macro_features = build_macro_features(macro_histories, sessions)

    batches = []
    quality = {}
    for symbol, history in equity_histories.items():
        try:
            rows, statistics = make_training_rows(
                symbol,
                history,
                sessions,
                macro_features,
                minimum_turnover,
            )
            batches.append(rows)
            quality[symbol] = statistics
            LOGGER.info("%s: retained %d rows", symbol, len(rows))
        except Exception as exc:
            skipped[symbol] = str(exc)
            LOGGER.error("Skipping %s during engineering: %s", symbol, exc)

    if len(batches) < MIN_SUCCESSFUL_TICKERS:
        raise RuntimeError(
            f"Only {len(batches)} tickers passed feature and quality checks; "
            f"{MIN_SUCCESSFUL_TICKERS} are required."
        )

    # Concatenate only after independent ticker indicator calculations.
    dataset = pd.concat(batches, ignore_index=True)
    dataset = dataset.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    if dataset.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate ticker/date observations.")
    if dataset["Target"].nunique() != 2:
        raise ValueError("The dataset must contain both target classes.")

    LOGGER.info(
        "Training dataset: %d rows, %d equities, %d local features, %d macros",
        len(dataset),
        dataset["Ticker"].nunique(),
        len(LOCAL_FEATURE_COLUMNS),
        len(MACRO_FEATURE_COLUMNS),
    )

    report = validate_chronologically(dataset)
    print(json.dumps(report, indent=2, allow_nan=False))
    if report["log_loss"] >= report["baseline_log_loss"]:
        LOGGER.warning("Validation did not outperform the frequency baseline.")

    final_model = new_classifier()
    final_model.fit(build_expanded_matrix(dataset), dataset["Target"])

    save_model(
        final_model,
        dataset,
        report,
        quality,
        skipped,
        minimum_turnover,
    )

    LOGGER.info("Saved expanded model to %s", MODEL_PATH)
    LOGGER.warning(
        "Inference must supply the expanded macro feature contract; "
        "the existing local-only loader will reject these weights."
    )
    if skipped:
        LOGGER.warning("Skipped equities: %s", ", ".join(sorted(skipped)))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        main()
    except Exception:
        LOGGER.exception(
            "Training failed; any previously installed model remains unchanged."
        )
        raise SystemExit(1)