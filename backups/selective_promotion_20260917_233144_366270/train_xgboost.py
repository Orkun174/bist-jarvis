# train_xgboost.py
"""Train and validate the bist-stationary-alpha-v2 relative-return classifier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# All feature definitions, adjustments, alignment, and macro timing are shared
# with live inference. Training must never implement a second feature pipeline.
from core.ai_analyzer import (
    BENCHMARK_TICKER,
    FEATURE_COLUMNS,
    FEATURE_CONTRACT_VERSION,
    HISTORY_PERIOD,
    HORIZON,
    MACRO_TICKERS,
    MODEL_PATH,
    PRICE_BASIS,
    TARGET_DEFINITION,
    align_equity_to_benchmark,
    build_macro_features,
    build_stationary_features,
    download_history,
    feature_matrix,
)


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache" / FEATURE_CONTRACT_VERSION
REPORT_DIR = ROOT / "training_reports"

# Curated liquid equities, not a point-in-time constituent database.
# Historical tests on this fixed universe remain subject to survivorship bias.
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

MIN_SUCCESSFUL_TICKERS = 15
MIN_TICKER_ROWS = 150


def cached_history(symbol: str, refresh: bool) -> pd.DataFrame:
    """Cache adjusted observations separately from all older raw-price caches."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = symbol.replace("^", "").replace("=", "_") + ".csv"
    path = CACHE_DIR / filename

    if path.exists() and not refresh:
        age_seconds = datetime.now().timestamp() - path.stat().st_mtime
        if 0 <= age_seconds < 20 * 3600:
            cached = pd.read_csv(path, index_col=0, parse_dates=True)
            required = {"Open", "High", "Low", "Close", "Volume"}
            if (
                isinstance(cached.index, pd.DatetimeIndex)
                and required.issubset(cached.columns)
                and not cached.index.duplicated().any()
            ):
                return cached.sort_index()

    frame = download_history(symbol, HISTORY_PERIOD)

    # Replace the cache only after a complete successful download.
    descriptor, temporary = tempfile.mkstemp(
        dir=CACHE_DIR, prefix="history_", suffix=".csv"
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)

    return frame


def make_target(
    aligned_equity: pd.DataFrame,
    benchmark: pd.DataFrame,
    observed: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Label terminal outperformance over five canonical benchmark sessions.

    Forward-filled prices are useful for feature continuity, but synthetic
    observations must not create artificial training outcomes. A label requires
    actual equity observations throughout t ... t+5.
    """
    equity_close = aligned_equity["Close"]
    benchmark_close = benchmark["Close"].reindex(aligned_equity.index)

    equity_growth = equity_close.shift(-HORIZON) / equity_close
    benchmark_growth = benchmark_close.shift(-HORIZON) / benchmark_close

    complete_horizon = (
        observed.astype(float)
        .rolling(HORIZON + 1, min_periods=HORIZON + 1)
        .min()
        .shift(-HORIZON)
        .eq(1.0)
    )

    known = (
        complete_horizon
        & equity_growth.notna()
        & benchmark_growth.notna()
        & np.isfinite(equity_growth)
        & np.isfinite(benchmark_growth)
    )

    target = pd.Series(np.nan, index=aligned_equity.index, dtype=float)
    target.loc[known] = (
        equity_growth.loc[known] > benchmark_growth.loc[known]
    ).astype(np.int32)

    target_dates = pd.Series(
        aligned_equity.index, index=aligned_equity.index
    ).shift(-HORIZON)

    return target, target_dates


def build_dataset(refresh: bool) -> tuple[pd.DataFrame, dict]:
    """Use XU100 trading dates as the sole calendar for every equity."""
    benchmark = cached_history(BENCHMARK_TICKER, refresh)
    benchmark = benchmark.loc[benchmark["Close"].notna()].copy()
    if len(benchmark) < 500:
        raise ValueError("Insufficient completed XU100 history.")

    macro_histories = {}
    for name, symbol in MACRO_TICKERS.items():
        macro_histories[name] = (
            benchmark
            if symbol == BENCHMARK_TICKER
            else cached_history(symbol, refresh)
        )

    macro_features = build_macro_features(macro_histories, benchmark.index)

    batches = []
    quality = {}
    skipped = {}

    for symbol in EQUITY_TICKERS:
        try:
            history = cached_history(symbol, refresh)
            aligned, observed = align_equity_to_benchmark(
                history, benchmark.index
            )

            local_features = build_stationary_features(aligned, benchmark)
            combined = local_features.join(macro_features, how="left")
            target, target_dates = make_target(aligned, benchmark, observed)

            rows = combined.copy()
            rows["Target"] = target
            rows["Date"] = aligned.index
            rows["Target_Date"] = target_dates
            rows["Ticker"] = symbol.removesuffix(".IS")
            rows["Target_Definition"] = TARGET_DEFINITION
            rows["Feature_Contract"] = FEATURE_CONTRACT_VERSION

            # Do not train at a synthetic current bar or a zero-volume session.
            eligible = observed & aligned["Volume"].gt(0)
            finite = np.isfinite(
                rows.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
            ).all(axis=1)

            rows = rows.loc[
                eligible
                & finite
                & rows["Target"].notna()
                & rows["Target_Date"].notna()
            ].copy()

            if len(rows) < MIN_TICKER_ROWS:
                raise ValueError(
                    f"Only {len(rows)} complete alpha observations available."
                )

            rows["Target"] = rows["Target"].astype(np.int32)
            batches.append(rows.reset_index(drop=True))

            quality[symbol] = {
                "downloaded_bars": len(history),
                "canonical_bars": len(aligned),
                "forward_filled_sessions": int((~observed).sum()),
                "retained_training_rows": len(rows),
            }
            LOGGER.info("%s: retained %d observations", symbol, len(rows))

        except Exception as exc:
            LOGGER.exception("Skipping %s", symbol)
            skipped[symbol] = str(exc)

    if len(batches) < MIN_SUCCESSFUL_TICKERS:
        raise RuntimeError(
            f"Only {len(batches)} equities passed validation; "
            f"at least {MIN_SUCCESSFUL_TICKERS} are required."
        )

    dataset = pd.concat(batches, ignore_index=True)
    dataset = dataset.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    if dataset.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate equity/session observations.")
    if dataset["Target"].nunique() != 2:
        raise ValueError("The alpha dataset must contain both classes.")

    feature_matrix(dataset)
    return dataset, {"equities": quality, "skipped": skipped}


def new_xgboost() -> XGBClassifier:
    """Regularized configuration fixed before inspecting the validation set."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=3,
        learning_rate=0.01,
        n_estimators=500,
        min_child_weight=60,
        reg_alpha=1.0,
        reg_lambda=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        tree_method="hist",
        n_jobs=min(4, os.cpu_count() or 1),
        random_state=42,
    )


def metrics(target: pd.Series, probability: np.ndarray) -> dict:
    """Report discrimination and probability quality without conflating them."""
    return {
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probability)),
        "roc_auc": (
            float(roc_auc_score(target, probability))
            if target.nunique() == 2 else None
        ),
    }


def validate_model(rows: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Fit XGBoost and logistic regression on exactly the same purged partition.

    The scaler is fitted only on training rows. Neither candidate sees validation
    labels during fitting, calibration, early stopping, or feature selection.
    """
    dates = np.sort(rows["Date"].unique())
    if len(dates) < 500:
        raise ValueError("Insufficient dates for chronological validation.")

    split_date = pd.Timestamp(dates[int(len(dates) * 0.8)])
    before_split = rows["Date"] < split_date
    crossing_labels = rows["Target_Date"] >= split_date

    training = rows.loc[before_split & ~crossing_labels]
    validation = rows.loc[~before_split]

    if len(training) < 2_000 or len(validation) < 300:
        raise ValueError("Insufficient observations after target purging.")
    if training["Target"].nunique() != 2 or validation["Target"].nunique() != 2:
        raise ValueError("Both temporal partitions must contain both classes.")

    x_train = feature_matrix(training)
    x_valid = feature_matrix(validation)
    y_train = training["Target"]
    y_valid = validation["Target"]

    xgb = new_xgboost()
    xgb.fit(x_train, y_train)
    xgb_probability = xgb.predict_proba(x_valid)[:, 1]

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=3_000,
            random_state=42,
        ),
    )
    logistic.fit(x_train, y_train)
    if int(logistic[-1].n_iter_.max()) >= logistic[-1].max_iter:
        raise RuntimeError("LogisticRegression did not converge.")

    logistic_probability = logistic.predict_proba(x_valid)[:, 1]
    naive_probability = np.full(len(validation), float(y_train.mean()))

    candidate = metrics(y_valid, xgb_probability)
    logistic_result = metrics(y_valid, logistic_probability)
    naive_result = metrics(y_valid, naive_probability)

    # These strict comparisons implement the requested baseline competition.
    # Nonconstant predictions are required so a prior-only solution cannot pass.
    accepted = bool(
        np.isfinite(xgb_probability).all()
        and np.std(xgb_probability) > 1e-6
        and candidate["log_loss"] < naive_result["log_loss"]
        and candidate["roc_auc"] > logistic_result["roc_auc"]
    )

    selected = (xgb_probability < 0.40) | (xgb_probability > 0.60)
    selected_count = int(selected.sum())

    report = {
        **candidate,
        "validation_auc": candidate["roc_auc"],
        "baseline_log_loss": naive_result["log_loss"],
        "baseline_brier_score": naive_result["brier_score"],
        "logistic_regression": logistic_result,
        "naive_frequency": naive_result,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "target_definition": TARGET_DEFINITION,
        "split_date": split_date.date().isoformat(),
        "training_rows": len(training),
        "validation_rows": len(validation),
        "purged_rows": int((before_split & crossing_labels).sum()),
        "training_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_valid.mean()),
        "prediction_std": float(np.std(xgb_probability)),
        "selective_coverage": selected_count / len(validation),
        "selective_rows": selected_count,
        "selective_direction_accuracy": (
            float(
                np.mean(
                    (xgb_probability[selected] > 0.5)
                    == y_valid.to_numpy()[selected]
                )
            )
            if selected_count else None
        ),
        "promotion_passed": accepted,
        "acceptance_rule": (
            "XGB log-loss < naive-frequency log-loss AND "
            "XGB AUC > LogisticRegression AUC; predictions must vary"
        ),
        "notes": (
            "Chronological validation with five-session label purging. "
            "Both fitted models use identical rows and features. "
            "No calibration or early stopping uses validation labels. "
            "Historical thresholds are not statistical significance tests. "
            "Repeated inspection of this historical period and fixed-universe "
            "survivorship bias limit claims of independent performance."
        ),
    }

    predictions = validation[["Date", "Ticker", "Target"]].copy()
    predictions["XGBoost_Probability"] = xgb_probability
    predictions["Logistic_Probability"] = logistic_probability
    predictions["Naive_Probability"] = naive_probability
    predictions["Signal_Status"] = np.where(
        selected,
        "Selective Signal",
        "Insufficient Signal / Neutral Regime",
    )
    return report, predictions


def save_candidate(
    model: XGBClassifier,
    rows: pd.DataFrame,
    report: dict,
    quality: dict,
    destination: Path,
) -> None:
    """Persist all semantics needed to reject incompatible inference inputs."""
    model.get_booster().set_attr(
        feature_contract_version=FEATURE_CONTRACT_VERSION,
        feature_columns=json.dumps(list(FEATURE_COLUMNS)),
        target_definition=TARGET_DEFINITION,
        target_horizon_sessions=str(HORIZON),
        benchmark_ticker=BENCHMARK_TICKER,
        price_basis=PRICE_BASIS,
        macro_availability="source_session_date + 1 calendar day",
        macro_tickers=json.dumps(MACRO_TICKERS),
        ambiguity_zone=json.dumps([0.40, 0.60]),
        calibration="uncalibrated",
        trained_at=datetime.now(timezone.utc).isoformat(),
        training_tickers=json.dumps(sorted(rows["Ticker"].unique().tolist())),
        last_feature_date=rows["Date"].max().date().isoformat(),
        last_label_date=rows["Target_Date"].max().date().isoformat(),
        validation_report=json.dumps(report, allow_nan=False),
        data_quality_report=json.dumps(quality, allow_nan=False),
    )

    model.save_model(str(destination))

    restored = XGBClassifier()
    restored.load_model(str(destination))
    if restored.get_booster().feature_names != list(FEATURE_COLUMNS):
        raise ValueError("Serialized model feature order is incorrect.")

    sample = feature_matrix(rows.tail(64))
    np.testing.assert_allclose(
        model.predict_proba(sample),
        restored.predict_proba(sample),
        rtol=1e-6,
        atol=1e-7,
    )


def promote_model(candidate: Path, experiment_dir: Path) -> None:
    """Back up existing weights and replace them only after validation passes."""
    if MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, experiment_dir / "previous_model.json")

    descriptor, temporary = tempfile.mkstemp(
        dir=MODEL_PATH.parent,
        prefix="bist_alpha_",
        suffix=".json",
    )
    os.close(descriptor)

    try:
        shutil.copyfile(candidate, temporary)
        os.replace(temporary, MODEL_PATH)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()

    rows, quality = build_dataset(args.refresh_data)
    report, predictions = validate_model(rows)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    experiment_dir = REPORT_DIR / f"alpha_v2_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=False)

    dataset_path = experiment_dir / "dataset.csv"
    rows.to_csv(dataset_path, index=False)
    report["dataset_sha256"] = hashlib.sha256(
        dataset_path.read_bytes()
    ).hexdigest()
    report["training_tickers"] = sorted(rows["Ticker"].unique().tolist())

    predictions.to_csv(experiment_dir / "validation_predictions.csv", index=False)
    (experiment_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    # Validation above belongs to the development model. This production
    # candidate is refitted on all currently known, fully labeled observations.
    final_model = new_xgboost()
    final_model.fit(feature_matrix(rows), rows["Target"])

    candidate = experiment_dir / "bist_xgb_candidate.json"
    save_candidate(final_model, rows, report, quality, candidate)

    if report["promotion_passed"] and not args.no_promote:
        promote_model(candidate, experiment_dir)
        LOGGER.info("Accepted alpha model installed: %s", MODEL_PATH)
    else:
        LOGGER.warning(
            "Production weights unchanged. Accepted=%s, no_promote=%s",
            report["promotion_passed"],
            args.no_promote,
        )

    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Experiment: {experiment_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()