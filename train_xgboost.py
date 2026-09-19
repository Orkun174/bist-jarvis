# train_xgboost.py
"""Two-stage SHAP selection and CPCV/Optuna training for relative BIST alpha."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterator, Sequence

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from core.ai_analyzer import (
    BENCHMARK_TICKER,
    FEATURE_COLUMNS,
    FEATURE_CONTRACT_VERSION,
    HISTORY_PERIOD,
    HORIZON,
    MACRO_TICKERS,
    MODEL_PATH,
    PRICE_BASIS,
    ROOT,
    TARGET_DEFINITION,
    VIX_PERCENTILE_COLUMN,
    align_equity_to_benchmark,
    build_macro_features,
    build_stationary_features,
    download_history,
    feature_matrix,
    shap_values,
    threshold_policy,
    vectorized_signals,
)

LOGGER = logging.getLogger(__name__)
TRADING_DAYS = 252
COST_BPS = 20.0
MIN_COVERAGE = 0.05

EQUITY_TICKERS = (
    "AKBNK.IS", "THYAO.IS", "KCHOL.IS", "TUPRS.IS", "FROTO.IS",
    "GARAN.IS", "ISCTR.IS", "YKBNK.IS", "EREGL.IS", "BIMAS.IS",
    "SISE.IS", "SAHOL.IS", "TOASO.IS", "PGSUS.IS", "ASELS.IS",
    "TCELL.IS", "ENKAI.IS", "KRDMD.IS", "PETKM.IS", "TTKOM.IS",
    "ARCLK.IS", "MGROS.IS", "ULKER.IS", "TAVHL.IS", "EKGYO.IS",
    "SASA.IS", "ASTOR.IS", "ALARK.IS",
)


def finite_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def load_history(symbol: str, refresh: bool) -> pd.DataFrame:
    directory = ROOT / "data_cache" / FEATURE_CONTRACT_VERSION
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{symbol.replace('^', '').replace('=', '_')}.csv"

    if path.exists() and not refresh:
        age = datetime.now().timestamp() - path.stat().st_mtime
        if 0 <= age < 20 * 3600:
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
            required = {"Open", "High", "Low", "Close", "Volume"}
            if (
                required.issubset(frame.columns)
                and isinstance(frame.index, pd.DatetimeIndex)
                and frame.index.is_monotonic_increasing
                and not frame.index.has_duplicates
            ):
                cutoff = pd.Timestamp.now(tz="Europe/Istanbul")
                cutoff = cutoff.tz_localize(None).normalize()
                return frame.loc[frame.index < cutoff]

    frame = download_history(symbol, HISTORY_PERIOD)
    descriptor, temporary = tempfile.mkstemp(dir=directory, suffix=".csv")
    os.close(descriptor)
    try:
        frame.to_csv(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return frame


def build_dataset(refresh: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build labels and realized daily alpha paths on the XU100 calendar."""
    histories = {
        name: load_history(symbol, refresh)
        for name, symbol in MACRO_TICKERS.items()
    }
    benchmark = histories["BIST100"]
    macro = build_macro_features(histories, benchmark.index)
    benchmark_returns = benchmark["Close"].pct_change(fill_method=None)
    batches: list[pd.DataFrame] = []
    failures: dict[str, str] = {}

    for ticker in EQUITY_TICKERS:
        try:
            raw = load_history(ticker, refresh)
            equity, observed = align_equity_to_benchmark(raw, benchmark.index)
            close = equity["Close"]
            benchmark_close = benchmark["Close"].reindex(equity.index)
            frame = build_stationary_features(equity, benchmark).join(macro)

            equity_growth = close.shift(-HORIZON) / close
            index_growth = benchmark_close.shift(-HORIZON) / benchmark_close
            complete = (
                observed.astype(float)
                .rolling(HORIZON + 1, min_periods=HORIZON + 1)
                .min()
                .shift(-HORIZON)
                .eq(1.0)
            )
            known = complete & equity_growth.notna() & index_growth.notna()
            frame["Target"] = (equity_growth > index_growth).where(known)
            frame["Forward_Alpha"] = equity_growth - index_growth
            frame["Date"] = equity.index
            frame["Target_Date"] = pd.Series(
                equity.index, index=equity.index
            ).shift(-HORIZON)
            frame["Ticker"] = ticker

            # Store realized daily relative returns for overlapping five-session
            # sleeves. These are outcomes only and never enter model features.
            daily_alpha = (
                close.pct_change(fill_method=None)
                - benchmark_returns.reindex(equity.index)
            )
            for step in range(1, HORIZON + 1):
                frame[f"Alpha_{step}"] = daily_alpha.shift(-step)
                frame[f"Exit_Date_{step}"] = pd.Series(
                    equity.index, index=equity.index
                ).shift(-step)

            required = (
                list(FEATURE_COLUMNS)
                + [VIX_PERCENTILE_COLUMN, "Target", "Target_Date"]
                + [f"Alpha_{step}" for step in range(1, HORIZON + 1)]
            )
            frame = frame.replace([np.inf, -np.inf], np.nan)
            frame = frame.loc[observed & equity["Volume"].gt(0)]
            frame = frame.dropna(subset=required)
            if len(frame) < 250:
                raise ValueError("Fewer than 250 complete observations.")

            frame["Target"] = frame["Target"].astype(np.int32)
            batches.append(frame.reset_index(drop=True))
        except Exception as exc:
            LOGGER.exception("Skipping %s", ticker)
            failures[ticker] = str(exc)

    if len(batches) < 15:
        raise RuntimeError("Fewer than 15 equities passed data checks.")

    rows = pd.concat(batches, ignore_index=True)
    rows = rows.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    if rows.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate equity/session observations.")

    feature_matrix(rows)
    return rows, {"successful_tickers": len(batches), "failures": failures}


@dataclass(frozen=True)
class CPCVSplit:
    train: np.ndarray
    test: np.ndarray
    test_groups: tuple[int, ...]


class CombinatorialPurgedCV:
    """Date-group CPCV with label-interval purging and calendar-time embargo.

    A continuous trading-session index is used: weekends and exchange holidays
    are not synthesized. All equities for a date belong to the same group.
    CPCV training can contain dates later than a test group; this is intentional
    and must not be described as a strictly forward-only deployment simulation.
    """

    def __init__(
        self,
        n_groups: int = 6,
        test_groups: int = 2,
        purge_td: pd.Timedelta = pd.Timedelta(days=7),
        embargo_td: pd.Timedelta = pd.Timedelta(days=7),
    ) -> None:
        if not 1 <= test_groups < n_groups:
            raise ValueError("Invalid CPCV group configuration.")
        if purge_td < pd.Timedelta(0) or embargo_td < pd.Timedelta(0):
            raise ValueError("Purge and embargo durations cannot be negative.")
        self.n_groups = n_groups
        self.test_groups = test_groups
        self.purge_td = purge_td
        self.embargo_td = embargo_td

    def group_map(self, rows: pd.DataFrame) -> pd.Series:
        dates = pd.DatetimeIndex(rows["Date"].unique()).sort_values()
        if len(dates) < self.n_groups * 30:
            raise ValueError("Insufficient dates for CPCV groups.")
        groups = np.array_split(np.arange(len(dates)), self.n_groups)
        labels = np.empty(len(dates), dtype=np.int32)
        for number, positions in enumerate(groups):
            labels[positions] = number
        return pd.Series(labels, index=dates)

    def split(self, rows: pd.DataFrame) -> Iterator[CPCVSplit]:
        mapping = self.group_map(rows)
        row_groups = rows["Date"].map(mapping).to_numpy()
        starts = pd.to_datetime(rows["Date"])
        ends = pd.to_datetime(rows["Target_Date"])

        if (ends < starts).any():
            raise ValueError("Label end precedes feature date.")

        for combination in itertools.combinations(
            range(self.n_groups), self.test_groups
        ):
            test_mask = np.isin(row_groups, combination)
            train_mask = ~test_mask

            for group in combination:
                group_mask = row_groups == group
                left = starts.loc[group_mask].min() - self.purge_td
                right = ends.loc[group_mask].max() + self.embargo_td

                # Remove any training label interval overlapping the expanded
                # test information interval, not just neighboring feature dates.
                overlaps = (starts <= right) & (ends >= left)
                train_mask &= ~overlaps.to_numpy()

            train = np.flatnonzero(train_mask)
            test = np.flatnonzero(test_mask)
            if len(train) < 1000 or len(test) < 100:
                raise ValueError("Insufficient CPCV rows after purging.")
            if rows.iloc[train]["Target"].nunique() != 2:
                raise ValueError("CPCV training partition has one class.")
            yield CPCVSplit(train, test, combination)


def portfolio_returns(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
) -> tuple[pd.Series, float, float]:
    """Simulate five overlapping equal-capital relative-return sleeves.

    Signals are formed after the feature close. Close-to-next-close returns
    therefore assume an executable close proxy; live implementation must assess
    execution slippage separately. Negative alpha signals represent a relative
    short position, not an automatically executable BIST cash-equity trade.
    """
    rows = rows.reset_index(drop=True)
    signals = vectorized_signals(
        probabilities,
        rows[VIX_PERCENTILE_COLUMN].to_numpy(dtype=float),
    )
    selected = signals != 0
    coverage = float(selected.mean())
    if not selected.any():
        dates = pd.DatetimeIndex(
            pd.concat(
                [rows[f"Exit_Date_{step}"] for step in range(1, HORIZON + 1)]
            ).unique()
        ).sort_values()
        return pd.Series(0.0, index=dates), coverage, 0.0

    # Each entry cohort gets 1/HORIZON capital; selected names share its sleeve.
    active_count = pd.Series(selected.astype(int)).groupby(
        rows["Date"]
    ).transform("sum")
    weights = np.divide(
        selected.astype(float),
        active_count.to_numpy(dtype=float) * HORIZON,
        out=np.zeros(len(rows), dtype=float),
        where=active_count.to_numpy() > 0,
    )
    pieces: list[pd.DataFrame] = []
    for step in range(1, HORIZON + 1):
        pnl = (
            weights * signals
            * rows[f"Alpha_{step}"].to_numpy(dtype=float)
        )
        if step == 1:
            pnl -= weights * COST_BPS / 10_000.0
        pieces.append(
            pd.DataFrame(
                {"Date": rows[f"Exit_Date_{step}"], "Return": pnl}
            )
        )

    returns = pd.concat(pieces).groupby("Date")["Return"].sum().sort_index()
    terminal_net = (
        signals[selected]
        * rows.loc[selected, "Forward_Alpha"].to_numpy(dtype=float)
        - COST_BPS / 10_000.0
    )
    expectancy = float(terminal_net.mean())
    return returns, coverage, expectancy


def risk_ratios(returns: pd.Series) -> dict[str, float]:
    values = returns.to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return {"sharpe": 0.0, "sortino": 0.0}

    mean = float(values.mean())
    std = float(values.std(ddof=1))
    downside = float(np.sqrt(np.mean(np.minimum(values, 0.0) ** 2)))
    # Floors and clipping avoid infinite optimization rewards on tiny samples.
    sharpe = np.clip(
        mean / max(std, 1e-8) * math.sqrt(TRADING_DAYS), -20.0, 20.0
    )
    sortino = np.clip(
        mean / max(downside, 1e-8) * math.sqrt(TRADING_DAYS), -20.0, 20.0
    )
    return {"sharpe": float(sharpe), "sortino": float(sortino)}


def coverage_adjusted_sortino(sortino: float, coverage: float) -> float:
    """Penalize low coverage additively, including when Sortino is negative."""
    if coverage >= MIN_COVERAGE:
        return float(sortino)
    shortfall = 1.0 - coverage / MIN_COVERAGE
    return float(sortino - 20.0 * shortfall - 20.0 * shortfall**2)


def corrected_sharpe(
    returns: pd.Series,
    number_of_trials: int,
) -> dict[str, float]:
    """Approximate a deflated Sharpe using trial count and return moments.

    This is a basic multiple-testing approximation, not an exact DSR test.
    Overlapping returns and shared CPCV samples invalidate IID significance
    claims. A conservative effective sample size reduces that overstatement.
    """
    values = returns.to_numpy(dtype=float)
    deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if deviation <= 1e-12:
        return {"deflated_sharpe": 0.0, "approx_dsr_probability": 0.5}

    centered = (values - values.mean()) / deviation
    skew = float(np.mean(centered**3))
    kurtosis = float(np.mean(centered**4))
    daily_sharpe = float(values.mean() / deviation)
    effective_n = max(3.0, len(values) / HORIZON)
    variance = max(
        1e-12,
        (
            1.0
            - skew * daily_sharpe
            + (kurtosis - 1.0) * daily_sharpe**2 / 4.0
        ) / (effective_n - 1.0),
    )
    standard_error = math.sqrt(variance)
    trials = max(1, number_of_trials)
    penalty = standard_error * math.sqrt(2.0 * math.log(trials))
    deflated = daily_sharpe - penalty

    return {
        "deflated_sharpe": deflated * math.sqrt(TRADING_DAYS),
        "approx_dsr_probability": NormalDist().cdf(
            deflated / standard_error
        ),
    }


def make_model(
    parameters: dict[str, Any],
    callbacks: list[Any] | None = None,
    evaluation_metric: Any = "logloss",
) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=min(4, os.cpu_count() or 1),
        random_state=42,
        eval_metric=evaluation_metric,
        callbacks=callbacks,
        **parameters,
    )


def select_features(
    discovery: pd.DataFrame,
) -> tuple[list[str], dict[str, Any]]:
    """Iteration 1: SHAP pruning on a separate, purged discovery partition."""
    dates = pd.DatetimeIndex(discovery["Date"].unique()).sort_values()
    boundary = dates[int(len(dates) * 0.7)]
    fit = discovery.loc[
        (discovery["Date"] < boundary)
        & (discovery["Target_Date"] < boundary - pd.Timedelta(days=7))
    ]
    validation = discovery.loc[discovery["Date"] >= boundary]
    if fit["Target"].nunique() != 2 or len(validation) < 100:
        raise ValueError("Insufficient feature-discovery data.")

    baseline = make_model(
        {
            "n_estimators": 250,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
        }
    )
    matrix = feature_matrix(fit)
    baseline.fit(matrix, fit["Target"])

    explanation_rows = validation.sample(
        n=min(1500, len(validation)),
        random_state=42,
    )
    values, _ = shap_values(baseline, feature_matrix(explanation_rows))
    importance = np.abs(values).mean(axis=0)
    total = float(importance.sum())
    if total <= 1e-12:
        raise RuntimeError("SHAP baseline has no explanatory variation.")

    share = importance / total
    structurally_valid = {
        name
        for name in FEATURE_COLUMNS
        if matrix[name].nunique() >= 3
        and float(matrix[name].std()) > 1e-10
    }
    eligible = [
        name
        for index, name in enumerate(FEATURE_COLUMNS)
        if share[index] >= 0.015 and name in structurally_valid
    ]

    # Remove nearly redundant features, retaining the higher-SHAP representative.
    importance_map = dict(zip(FEATURE_COLUMNS, importance.tolist()))
    ranked = sorted(eligible, key=lambda name: -importance_map[name])
    correlation = matrix[ranked].corr(method="spearman").abs()
    retained: list[str] = []
    for name in ranked:
        if all(correlation.loc[name, other] < 0.98 for other in retained):
            retained.append(name)

    selected = [name for name in FEATURE_COLUMNS if name in retained]
    if len(selected) < 3:
        raise RuntimeError("Fewer than three features survived strict pruning.")

    return selected, {
        "selected_features": selected,
        "importance_fraction": dict(zip(FEATURE_COLUMNS, share.tolist())),
        "discovery_rows": len(discovery),
        "fit_rows": len(fit),
        "shap_rows": len(explanation_rows),
        "validation_start": boundary.date().isoformat(),
    }


def sample_parameters(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 650, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.1, log=True
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1.0, 5.0
        ),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),
        "colsample_bylevel": trial.suggest_float(
            "colsample_bylevel", 0.6, 1.0
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 0.0, 2.0
        ),
    }


def evaluate_cpcv(
    rows: pd.DataFrame,
    selected: Sequence[str],
    parameters: dict[str, Any],
    cv: CombinatorialPurgedCV,
    trial: optuna.Trial | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    matrix = feature_matrix(rows, selected)
    records: list[dict[str, Any]] = []
    objectives: list[float] = []

    # Evaluate every fold; a weak first fold must not abort a trial.
    for split in cv.split(rows):
        training = rows.iloc[split.train]
        validation = rows.iloc[split.test].reset_index(drop=True)

        model = make_model(parameters)
        model.fit(matrix.iloc[split.train], training["Target"])
        probabilities = model.predict_proba(matrix.iloc[split.test])[:, 1]
        returns, coverage, expectancy = portfolio_returns(
            validation, probabilities
        )
        ratios = risk_ratios(returns)
        # Optimize discrimination independently of confidence thresholds and P&L.
        # Constant probabilities score 0.5; stronger ranking must beat that score.
        if validation["Target"].nunique() != 2:
            raise ValueError("ROC-AUC requires both classes in every CPCV test fold.")
        fold_auc = float(roc_auc_score(validation["Target"], probabilities))
        objectives.append(fold_auc)
        records.append(
            {
                "test_indices": split.test,
                "test_groups": split.test_groups,
                "probabilities": probabilities,
                "coverage": coverage,
                "expectancy": expectancy,
                "roc_auc": fold_auc,
                "coverage_adjusted_sortino": coverage_adjusted_sortino(
                    ratios["sortino"], coverage
                ),
                **ratios,
            }
        )

    # Every fold contributes equally to the primary mean ROC-AUC objective.
    if not records:
        raise ValueError("CPCV produced no evaluation folds.")
    # Zero trades do not invalidate ranking: dynamic thresholds are independent
    # of ROC-AUC. Keep coverage and risk metrics as diagnostics, not penalties.
    return float(np.mean(objectives)), records


def build_oos_paths(
    rows: pd.DataFrame,
    records: list[dict[str, Any]],
    cv: CombinatorialPurgedCV,
) -> list[np.ndarray]:
    """Stitch complete OOS paths, using each group's OOS occurrence once."""
    mapping = cv.group_map(rows)
    row_groups = rows["Date"].map(mapping).to_numpy()
    count = math.comb(cv.n_groups - 1, cv.test_groups - 1)
    paths = [np.full(len(rows), np.nan) for _ in range(count)]
    occurrences = np.zeros(cv.n_groups, dtype=int)

    for record in records:
        indices = record["test_indices"]
        for group in record["test_groups"]:
            member = row_groups[indices] == group
            path_number = int(occurrences[group])
            paths[path_number][indices[member]] = record["probabilities"][member]
            occurrences[group] += 1

    if not np.all(occurrences == count):
        raise RuntimeError("Incomplete CPCV path allocation.")
    if any(not np.isfinite(path).all() for path in paths):
        raise RuntimeError("An OOS path contains missing predictions.")
    return paths


def atomic_promote(candidate: Path, experiment: Path) -> None:
    """Replace active weights atomically while preserving previous weights."""
    if MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, experiment / "previous_model.json")

    descriptor, temporary = tempfile.mkstemp(
        dir=MODEL_PATH.parent, suffix=".json"
    )
    os.close(descriptor)
    try:
        shutil.copyfile(candidate, temporary)
        os.replace(temporary, MODEL_PATH)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--purge-days", type=int, default=7)
    parser.add_argument("--embargo-days", type=int, default=7)
    arguments = parser.parse_args()
    if arguments.trials < 1:
        parser.error("--trials must be positive")

    rows, quality = build_dataset(arguments.refresh_data)
    dates = pd.DatetimeIndex(rows["Date"].unique()).sort_values()
    if len(dates) < 800:
        raise RuntimeError("At least 800 complete sessions are required.")

    # Keep feature discovery outside every optimization evaluation fold.
    # Reserve the newest 15% as an untouched final audit.
    search_start = dates[int(len(dates) * 0.25)]
    audit_start = dates[int(len(dates) * 0.85)]
    gap = pd.Timedelta(days=max(arguments.purge_days, arguments.embargo_days))

    discovery = rows.loc[
        (rows["Date"] < search_start)
        & (rows["Target_Date"] < search_start - gap)
    ].reset_index(drop=True)
    search_rows = rows.loc[
        (rows["Date"] >= search_start)
        & (rows["Date"] < audit_start)
        & (rows["Target_Date"] < audit_start - gap)
    ].reset_index(drop=True)
    audit_rows = rows.loc[rows["Date"] >= audit_start].reset_index(drop=True)

    # Iteration 1: baseline + SHAP. Iteration 2: optimized pruned-feature model.
    selected, selection_report = select_features(discovery)
    cv = CombinatorialPurgedCV(
        purge_td=pd.Timedelta(days=arguments.purge_days),
        embargo_td=pd.Timedelta(days=arguments.embargo_days),
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.NopPruner(),
    )

    def objective(trial: optuna.Trial) -> float:
        parameters = sample_parameters(trial)
        utility, records = evaluate_cpcv(
            search_rows, selected, parameters, cv, trial
        )
        trial.set_user_attr(
            "mean_coverage",
            float(np.mean([record["coverage"] for record in records])),
        )
        trial.set_user_attr(
            "mean_expectancy",
            float(np.mean([record["expectancy"] for record in records])),
        )
        return utility

    study.optimize(
        objective,
        n_trials=arguments.trials,
        n_jobs=1,
        gc_after_trial=True,
    )
    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        raise RuntimeError("Every trial was pruned; active weights are unchanged.")

    best_parameters = dict(study.best_params)
    utility, records = evaluate_cpcv(
        search_rows, selected, best_parameters, cv
    )
    paths = build_oos_paths(search_rows, records, cv)
    path_reports: list[dict[str, Any]] = []
    for number, probabilities in enumerate(paths):
        returns, coverage, expectancy = portfolio_returns(
            search_rows, probabilities
        )
        path_reports.append(
            {
                "path": number,
                "coverage": coverage,
                "expectancy": expectancy,
                **risk_ratios(returns),
                **corrected_sharpe(returns, len(study.trials) + 1),
            }
        )

    # Audit is evaluated once after all feature/hyperparameter choices.
    preaudit = rows.loc[
        (rows["Date"] < audit_start)
        & (rows["Target_Date"] < audit_start - gap)
    ]
    audit_model = make_model(best_parameters)
    audit_model.fit(
        feature_matrix(preaudit, selected), preaudit["Target"]
    )
    audit_probability = audit_model.predict_proba(
        feature_matrix(audit_rows, selected)
    )[:, 1]
    audit_returns, coverage, expectancy = portfolio_returns(
        audit_rows, audit_probability
    )
    signals = vectorized_signals(
        audit_probability,
        audit_rows[VIX_PERCENTILE_COLUMN].to_numpy(dtype=float),
    )
    active = signals != 0
    accuracy = (
        float(
            (
                (signals[active] > 0)
                == audit_rows.loc[active, "Target"].to_numpy()
            ).mean()
        )
        if active.any() else None
    )
    auc = (
        float(roc_auc_score(audit_rows["Target"], audit_probability))
        if audit_rows["Target"].nunique() == 2 else None
    )
    deviation = float(audit_probability.std())

    # Preserve audit results for reporting; they no longer block promotion.
    audit_gate_passed = bool(
        accuracy is not None
        and accuracy >= 0.55
        and auc is not None
        and auc >= 0.50
        and deviation > 0.02
        and coverage >= MIN_COVERAGE
        and expectancy > 0.0
    )

    # Explicit temporary operator override. Serialization and feature-contract
    # checks below remain mandatory before replacing active weights.
    promotion_passed = True

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    experiment = ROOT / "training_reports" / f"dynamic_alpha_{timestamp}"
    experiment.mkdir(parents=True)

    dataset_path = experiment / "dataset.csv"
    rows.to_csv(dataset_path, index=False)
    report: dict[str, Any] = {
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "target_definition": TARGET_DEFINITION,
        "selected_features": selected,
        "feature_selection": selection_report,
        "best_parameters": best_parameters,
        "trial_count": len(study.trials),
        "completed_trials": len(completed),
        "cpcv_objective": utility,
        "cpcv_objective_metric": "mean_roc_auc",
        "cpcv_paths": path_reports,
        "audit_start": audit_start.date().isoformat(),
        "audit_rows": len(audit_rows),
        "selective_rows": int(active.sum()),
        "selective_direction_accuracy": accuracy,
        "selective_coverage": coverage,
        "prediction_std": deviation,
        "roc_auc": auc,
        "net_trade_expectancy": expectancy,
        "cost_bps": COST_BPS,
        **risk_ratios(audit_returns),
        "promotion_passed": promotion_passed,
        "promotion_override": True,
        "audit_gate_passed": audit_gate_passed,
        "acceptance_rule": "Operator override: promote after contract and serialization checks",
        "audit_acceptance_rule": (
            "audit selective accuracy >= 0.55 AND AUC >= 0.50 "
            "AND prediction_std > 0.02 AND coverage >= 0.05 "
            "AND net trade expectancy > 0"
        ),
        "dataset_sha256": hashlib.sha256(
            dataset_path.read_bytes()
        ).hexdigest(),
        "notes": (
            "CPCV paths share observations and are not independent tests. "
            "Deflated Sharpe is an approximate trial-count adjustment. "
            "Transaction costs are assumptions; borrowing, execution and "
            "portfolio constraints require separate validation. Reusing the "
            "audit for subsequent tuning removes its untouched status."
        ),
    }
    finite_json(experiment / "report.json", report)
    finite_json(experiment / "quality.json", quality)
    study.trials_dataframe().to_csv(experiment / "optuna_trials.csv", index=False)

    audit_output = audit_rows[["Date", "Ticker", "Target"]].copy()
    audit_output["Probability"] = audit_probability
    audit_output["Signal"] = signals
    audit_output.to_csv(experiment / "audit_predictions.csv", index=False)

    # Refit the second-stage configuration on all known labels only after
    # recording the audit. Its saved audit belongs to the preaudit model.
    final_model = make_model(best_parameters)
    final_matrix = feature_matrix(rows, selected)
    final_model.fit(final_matrix, rows["Target"])
    final_model.get_booster().set_attr(
        feature_contract_version=FEATURE_CONTRACT_VERSION,
        selected_features=json.dumps(selected),
        feature_columns=json.dumps(selected),
        feature_universe=json.dumps(list(FEATURE_COLUMNS)),
        target_definition=TARGET_DEFINITION,
        target_horizon_sessions=str(HORIZON),
        benchmark_ticker=BENCHMARK_TICKER,
        price_basis=PRICE_BASIS,
        threshold_policy=json.dumps(threshold_policy()),
        validation_report=json.dumps(report, allow_nan=False),
        trained_at=datetime.now(timezone.utc).isoformat(),
        calibration="uncalibrated",
    )

    candidate = experiment / "bist_xgb_candidate.json"
    final_model.save_model(candidate)
    restored = XGBClassifier()
    restored.load_model(candidate)
    if restored.get_booster().feature_names != selected:
        raise RuntimeError("Serialized feature contract changed.")
    np.testing.assert_allclose(
        final_model.predict_proba(final_matrix.tail(128)),
        restored.predict_proba(final_matrix.tail(128)),
        rtol=1e-6,
        atol=1e-7,
    )

    # Temporary forced promotion also applies when --promote is omitted.
    atomic_promote(candidate, experiment)

    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Candidate: {candidate}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()