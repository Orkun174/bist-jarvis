# backtest_engine.py
"""Cross-sectional BIST portfolio replay using the existing XGBoost model.

Signals formed at close[t] execute at open[t+1]. All assets share one cash pool.
Historical results may overlap model training and are not walk-forward results.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import vectorbt as vbt
from xgboost import XGBClassifier

from core.ai_analyzer import (
    BENCHMARK_TICKER,
    FEATURE_COLUMNS,
    FEATURE_CONTRACT_VERSION,
    MACRO_TICKERS,
    PRICE_BASIS,
    TARGET_DEFINITION,
    VIX_PERCENTILE_COLUMN,
    VIX_PERCENTILE_MIN_PERIODS,
    VIX_PERCENTILE_WINDOW,
    align_equity_to_benchmark,
    build_macro_features,
    build_stationary_features,
    download_history,
    feature_matrix,
)

ROOT = Path(__file__).resolve().parent
TRADING_SESSIONS = 252

DEFAULT_TICKERS = (
    "THYAO.IS",
    "BIMAS.IS",
    "KCHOL.IS",
    "FROTO.IS",
    "TUPRS.IS",
    "SAHOL.IS",
    "GARAN.IS",
    "AKBNK.IS",
    "ISCTR.IS",
    "YKBNK.IS",
    "EREGL.IS",
    "SISE.IS",
    "ASELS.IS",
    "TCELL.IS",
    "ENKAI.IS",
    "PGSUS.IS",
    "TOASO.IS",
    "MGROS.IS",
    "TTKOM.IS",
    "TAVHL.IS",
)


@dataclass(frozen=True)
class Config:
    tickers: tuple[str, ...]
    model_path: Path
    period: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    initial_cash: float
    top_n: int
    allocation_method: str
    gross_exposure: float
    base_fraction: float
    payoff_ratio: float
    commission: float
    impact_gamma: float
    output_directory: Path


@dataclass(frozen=True)
class ModelBundle:
    model: XGBClassifier
    features: tuple[str, ...]
    policy: dict[str, Any]
    trained_at: pd.Timestamp | None


@dataclass
class PortfolioInputs:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    open_present: pd.DataFrame
    probabilities: pd.DataFrame
    conviction: pd.DataFrame
    signals: pd.DataFrame
    scores: pd.DataFrame
    volatility: pd.DataFrame
    average_volume: pd.DataFrame


def model_json(
    model: XGBClassifier,
    name: str,
    required: bool = True,
) -> Any:
    value = model.get_booster().attr(name)
    if value is None:
        if required:
            raise ValueError(f"Missing model metadata: {name}")
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid model metadata: {name}") from exc


def validate_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("percentile_source") != "lagged_VIX_level":
        raise ValueError("Unsupported VIX percentile source.")
    if policy.get("window") != VIX_PERCENTILE_WINDOW:
        raise ValueError("VIX percentile window differs from the shared pipeline.")
    if policy.get("min_periods") != VIX_PERCENTILE_MIN_PERIODS:
        raise ValueError("VIX percentile warm-up differs from the shared pipeline.")

    low = np.asarray(policy.get("low_regime"), dtype=float)
    high = np.asarray(policy.get("high_regime"), dtype=float)
    if low.shape != (3,) or high.shape != (3,):
        raise ValueError("Threshold regimes require percentile/lower/upper.")
    if not np.isfinite(np.concatenate((low, high))).all():
        raise ValueError("Nonfinite threshold policy.")
    if not 0 <= low[0] < high[0] <= 100:
        raise ValueError("Invalid percentile anchors.")
    if not (
        0 <= low[1] < low[2] <= 1
        and 0 <= high[1] < high[2] <= 1
    ):
        raise ValueError("Invalid probability thresholds.")


def load_model(path: Path) -> ModelBundle:
    """Preserve the application's selected feature order and model semantics."""
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")

    model = XGBClassifier()
    model.load_model(str(path))
    booster = model.get_booster()

    expected = {
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "target_definition": TARGET_DEFINITION,
        "price_basis": PRICE_BASIS,
        "benchmark_ticker": BENCHMARK_TICKER,
    }
    for name, value in expected.items():
        if booster.attr(name) != value:
            raise ValueError(f"Incompatible model metadata: {name}")

    selected = model_json(model, "selected_features")
    if (
        not isinstance(selected, list)
        or not selected
        or not all(isinstance(name, str) for name in selected)
        or len(selected) != len(set(selected))
        or not set(selected).issubset(FEATURE_COLUMNS)
        or booster.feature_names != selected
    ):
        raise ValueError("Invalid selected feature contract.")
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Expected binary classes [0, 1].")

    stored_macros = model_json(model, "macro_tickers", required=False)
    if stored_macros is not None and stored_macros != MACRO_TICKERS:
        raise ValueError("Model macro mapping differs from the shared pipeline.")

    policy = model_json(model, "threshold_policy")
    if not isinstance(policy, dict):
        raise ValueError("Threshold policy must be an object.")
    validate_policy(policy)

    timestamp = booster.attr("trained_at")
    trained_at = pd.Timestamp(timestamp) if timestamp else None
    if trained_at is not None and trained_at.tzinfo is not None:
        trained_at = trained_at.tz_convert("Europe/Istanbul").tz_localize(None)

    return ModelBundle(
        model=model,
        features=tuple(selected),
        policy=policy,
        trained_at=trained_at,
    )


def dynamic_thresholds(
    percentile: pd.Series,
    policy: Mapping[str, Any],
) -> pd.DataFrame:
    low = np.asarray(policy["low_regime"], dtype=float)
    high = np.asarray(policy["high_regime"], dtype=float)
    values = percentile.to_numpy(dtype=float)
    known = np.isfinite(values)

    if ((values[known] < 0) | (values[known] > 100)).any():
        raise ValueError("VIX percentile must be between zero and 100.")

    weight = np.clip((values - low[0]) / (high[0] - low[0]), 0, 1)
    return pd.DataFrame(
        {
            "Lower": low[1] + weight * (high[1] - low[1]),
            "Upper": low[2] + weight * (high[2] - low[2]),
        },
        index=percentile.index,
    )


def fetch_histories(config: Config) -> dict[str, pd.DataFrame]:
    """Fetch each adjusted daily history once; fail explicitly on missing assets."""
    symbols = dict.fromkeys(
        (*config.tickers, *MACRO_TICKERS.values(), BENCHMARK_TICKER)
    )
    histories: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        print(f"Downloading {symbol}", flush=True)
        try:
            histories[symbol] = download_history(symbol, config.period)
        except Exception as exc:
            raise RuntimeError(f"Could not download {symbol}: {exc}") from exc

    return histories


def conviction_proxy(equity: pd.DataFrame, cmf: pd.Series) -> pd.Series:
    """Deterministic proxy: accumulation=7, distribution=3, otherwise=5."""
    obv = (
        np.sign(equity["Close"].diff()).fillna(0.0) * equity["Volume"]
    ).cumsum()
    change = obv.diff(5)

    return pd.Series(
        np.select(
            [
                (cmf.gt(0) & change.gt(0)).to_numpy(),
                (cmf.lt(0) & change.lt(0)).to_numpy(),
            ],
            [7, 3],
            default=5,
        ),
        index=equity.index,
        dtype=np.int8,
    )


def build_inputs(
    histories: Mapping[str, pd.DataFrame],
    bundle: ModelBundle,
    config: Config,
) -> tuple[PortfolioInputs, pd.DataFrame]:
    """Align all assets and collect batch model inference into common matrices."""
    benchmark = histories[BENCHMARK_TICKER]
    calendar = benchmark.index

    if not isinstance(calendar, pd.DatetimeIndex):
        raise ValueError("The benchmark requires a DatetimeIndex.")
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("The benchmark calendar must be unique and sorted.")

    macro_histories = {
        name: histories[symbol] for name, symbol in MACRO_TICKERS.items()
    }
    macros = build_macro_features(macro_histories, calendar)
    thresholds = dynamic_thresholds(
        macros[VIX_PERCENTILE_COLUMN], bundle.policy
    )

    open_prices: dict[str, pd.Series] = {}
    high_prices: dict[str, pd.Series] = {}
    low_prices: dict[str, pd.Series] = {}
    close_prices: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    open_presence: dict[str, pd.Series] = {}
    probabilities: dict[str, pd.Series] = {}
    convictions: dict[str, pd.Series] = {}
    signals: dict[str, pd.Series] = {}
    scores: dict[str, pd.Series] = {}
    volatility: dict[str, pd.Series] = {}
    average_volume: dict[str, pd.Series] = {}

    for symbol in config.tickers:
        native = histories[symbol]
        equity, observed = align_equity_to_benchmark(native, calendar)
        local = build_stationary_features(equity, benchmark)
        features = local.join(macros, how="left")

        selected_values = features.loc[:, list(bundle.features)].to_numpy(
            dtype=float
        )
        valid = pd.Series(
            np.isfinite(selected_values).all(axis=1),
            index=features.index,
        )
        valid &= np.isfinite(features[VIX_PERCENTILE_COLUMN])
        valid &= observed & equity["Volume"].gt(0)

        probability = pd.Series(np.nan, index=equity.index, dtype=float)
        if valid.any():
            matrix = feature_matrix(features.loc[valid], bundle.features)
            prediction = bundle.model.predict_proba(matrix)[:, 1]
            if not np.isfinite(prediction).all():
                raise ValueError(f"Nonfinite model output for {symbol}.")
            probability.loc[valid] = prediction

        bounds = thresholds.reindex(equity.index)
        signal = pd.Series(
            np.select(
                [
                    probability.ge(bounds["Upper"]),
                    probability.le(bounds["Lower"]),
                ],
                [1, -1],
                default=0,
            ),
            index=equity.index,
            dtype=np.int8,
        )
        conviction = conviction_proxy(equity, local["CMF_20"])
        kelly = (
            probability * config.payoff_ratio - (1 - probability)
        ) / config.payoff_ratio

        raw_score = (
            config.base_fraction * conviction.astype(float) / 10.0 * kelly
        )
        # Retain the existing long-only eligibility policy. Rank positive
        # eligible scores; do not invest in bearish/ambiguous names just to fill N.
        score = raw_score.clip(0.0, 1.0).where(signal.eq(1), 0.0).fillna(0.0)

        native_open = native["Open"].reindex(calendar)
        open_presence[symbol] = (
            native_open.notna() & np.isfinite(native_open) & native_open.gt(0)
        )

        open_prices[symbol] = equity["Open"].reindex(calendar)
        high_prices[symbol] = equity["High"].reindex(calendar)
        low_prices[symbol] = equity["Low"].reindex(calendar)
        close_prices[symbol] = equity["Close"].reindex(calendar)
        volumes[symbol] = equity["Volume"].reindex(calendar)
        probabilities[symbol] = probability.reindex(calendar)
        convictions[symbol] = conviction.reindex(calendar)
        signals[symbol] = signal.reindex(calendar).fillna(0).astype(np.int8)
        scores[symbol] = score.reindex(calendar).fillna(0.0)

        daily_return = equity["Close"].pct_change(fill_method=None)
        volatility[symbol] = daily_return.rolling(20).std().reindex(calendar)
        average_volume[symbol] = (
            equity["Volume"].rolling(20).mean().reindex(calendar)
        )

    def frame(series: Mapping[str, pd.Series]) -> pd.DataFrame:
        return pd.DataFrame(series, index=calendar).reindex(columns=config.tickers)

    inputs = PortfolioInputs(
        open=frame(open_prices),
        high=frame(high_prices),
        low=frame(low_prices),
        close=frame(close_prices),
        volume=frame(volumes),
        open_present=frame(open_presence).fillna(False).astype(bool),
        probabilities=frame(probabilities),
        conviction=frame(convictions),
        signals=frame(signals),
        scores=frame(scores),
        volatility=frame(volatility),
        average_volume=frame(average_volume),
    )
    return inputs, thresholds


def cross_sectional_weights(
    scores: pd.DataFrame,
    top_n: int,
    method: str,
    gross_exposure: float,
) -> pd.DataFrame:
    """Rank each date and normalize selected positive scores to the equity budget."""
    positive = scores.where(np.isfinite(scores) & scores.gt(0))
    # Stable column ordering gives deterministic tie resolution.
    ranks = positive.rank(axis=1, method="first", ascending=False)
    selected = ranks.le(top_n) & positive.notna()

    if method == "equal":
        raw_weights = selected.astype(float)
    else:
        raw_weights = scores.where(selected, 0.0)

    denominator = raw_weights.sum(axis=1).replace(0.0, np.nan)
    weights = raw_weights.div(denominator, axis=0).fillna(0.0)
    weights *= gross_exposure

    if (weights.sum(axis=1) > 1.0 + 1e-10).any():
        raise RuntimeError("Portfolio target exposure exceeds 100%.")
    if weights.lt(0).any().any():
        raise RuntimeError("Long-only target contains negative weights.")
    return weights


def prepare_execution(
    inputs: PortfolioInputs,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decision_weights = cross_sectional_weights(
        inputs.scores,
        config.top_n,
        config.allocation_method,
        config.gross_exposure,
    )

    # Shift once, after ranking, so today's close never selects today's open.
    execution_weights = decision_weights.shift(1).fillna(0.0)
    available_volatility = inputs.volatility.shift(1)
    available_volume = inputs.average_volume.shift(1)

    known_dates = inputs.probabilities.notna().any(axis=1)
    if not known_dates.any():
        raise RuntimeError("No valid model predictions for the universe.")
    first_decision = known_dates.index[known_dates][0]
    first_position = int(inputs.close.index.get_loc(first_decision)) + 1
    evaluation = inputs.close.index[first_position:]

    if config.start is not None:
        evaluation = evaluation[evaluation >= config.start]
    if config.end is not None:
        evaluation = evaluation[evaluation <= config.end]
    if len(evaluation) < 2:
        raise RuntimeError("Insufficient executable sessions in the chosen range.")

    open_prices = inputs.open.loc[evaluation].copy()
    close_prices = inputs.close.loc[evaluation].copy()
    weights = execution_weights.loc[evaluation].copy()

    # NaN orders mean "do not trade", not liquidation at a fabricated price.
    # A suspended holding can therefore temporarily prevent perfect rebalancing.
    weights = weights.where(inputs.open_present.loc[evaluation])

    intended_holdings = weights.ffill().fillna(0.0)
    previous_holdings = intended_holdings.shift(1).fillna(0.0)
    weight_change = (intended_holdings - previous_holdings).abs()

    # Preserve the previous square-root approximation based on initial capital
    # and lagged average volume. It is not an exact recursive impact model.
    mocked_order_size = weight_change * config.initial_cash / open_prices
    impact = (
        config.impact_gamma
        * available_volatility.loc[evaluation]
        * np.sqrt(
            mocked_order_size
            / available_volume.loc[evaluation].replace(0.0, np.nan)
        )
    )
    impact = impact.where(mocked_order_size.gt(0), 0.0)

    invalid = mocked_order_size.gt(0) & ~np.isfinite(impact)
    if invalid.any().any():
        raise RuntimeError("An order has missing lagged liquidity inputs.")

    slippage = impact.fillna(0.0).clip(0.0, 0.25)
    return open_prices, close_prices, weights, slippage, decision_weights


def simulate(
    open_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    weights: pd.DataFrame,
    slippage: pd.DataFrame,
    config: Config,
) -> Any:
    """Execute one cross-sectional portfolio with shared cash and sell-first routing."""
    return vbt.Portfolio.from_orders(
        close=close_prices,
        size=weights,
        size_type="targetpercent",
        direction="longonly",
        price=open_prices,
        val_price=open_prices,
        fees=config.commission,
        slippage=slippage,
        init_cash=config.initial_cash,
        cash_sharing=True,
        group_by=True,
        # Route reductions before additions to release shared cash.
        call_seq="auto",
        # Keep the row's valuation basis consistent while processing its orders.
        update_value=False,
        allow_partial=True,
        raise_reject=False,
        log=True,
        freq="1D",
    )


def grouped_series(value: Any, name: str) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.rename(name)
    if isinstance(value, pd.DataFrame) and value.shape[1] == 1:
        return value.iloc[:, 0].rename(name)
    raise ValueError(f"Expected one grouped portfolio series for {name}.")


def finite_number(value: Any) -> float | None:
    result = float(np.asarray(value).reshape(-1)[0])
    return result if math.isfinite(result) else None


def save_and_report(
    portfolio: Any,
    inputs: PortfolioInputs,
    thresholds: pd.DataFrame,
    weights: pd.DataFrame,
    slippage: pd.DataFrame,
    decision_weights: pd.DataFrame,
    bundle: ModelBundle,
    config: Config,
) -> None:
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    dates = weights.index

    equity = grouped_series(portfolio.value(), "Portfolio_Equity")
    returns = grouped_series(portfolio.returns(), "Portfolio_Return")
    cash = grouped_series(portfolio.cash(), "Cash")

    standard_deviation = float(returns.std(ddof=1))
    sharpe = (
        float(returns.mean() / standard_deviation * np.sqrt(TRADING_SESSIONS))
        if standard_deviation > 0 else None
    )
    cagr = (
        (float(equity.iloc[-1]) / config.initial_cash)
        ** (TRADING_SESSIONS / len(equity))
        - 1.0
    )
    maximum_drawdown = abs(float(finite_number(portfolio.max_drawdown()) or 0.0))
    closed_trades = portfolio.trades.closed
    closed_count = int(np.asarray(closed_trades.count()).sum())
    win_rate = (
        finite_number(closed_trades.win_rate()) if closed_count else None
    )
    mean_invested = float((1.0 - cash / equity).mean())

    trained_date = (
        bundle.trained_at.normalize() if bundle.trained_at is not None else None
    )
    post_training = bool(
        trained_date is not None and dates[0] > trained_date
    )

    summary = {
        "evaluation_type": "fixed_model_cross_sectional_replay",
        "strictly_post_training_dates": post_training,
        "model_trained_at": (
            bundle.trained_at.isoformat() if bundle.trained_at is not None else None
        ),
        "model_path": str(config.model_path),
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "selected_features": list(bundle.features),
        "tickers": list(config.tickers),
        "top_n": config.top_n,
        "allocation_method": config.allocation_method,
        "gross_target_exposure": config.gross_exposure,
        "initial_cash_try": config.initial_cash,
        "final_equity_try": float(equity.iloc[-1]),
        "start": dates[0].date().isoformat(),
        "end": dates[-1].date().isoformat(),
        "sharpe_252_sessions": sharpe,
        "cagr_252_sessions": float(cagr),
        "max_drawdown": maximum_drawdown,
        "closed_trade_win_rate": win_rate,
        "closed_trades": closed_count,
        "mean_actual_invested_fraction": mean_invested,
        "commission": config.commission,
        "threshold_policy": bundle.policy,
    }

    print("\nCROSS-SECTIONAL SHARED-CASH PORTFOLIO")
    if not post_training:
        print(
            "Training overlap is possible: this fixed-model replay is not "
            "established out-of-sample performance."
        )
    print(
        f"Universe={len(config.tickers)} | Top N={config.top_n} | "
        f"Allocation={config.allocation_method}"
    )
    print("\nVECTORBT TEAR SHEET")
    print(portfolio.stats().to_string())
    print("\n252-SESSION PORTFOLIO METRICS")
    print(f"Sharpe Ratio: {sharpe:.4f}" if sharpe is not None else "Sharpe Ratio: N/A")
    print(f"Max Drawdown: {maximum_drawdown:.2%}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Win Rate: {win_rate:.2%}" if win_rate is not None else "Win Rate: N/A")
    print(f"Average Invested Equity: {mean_invested:.2%}")
    print(f"Final Equity: TRY {equity.iloc[-1]:,.2f}")

    matrices = {
        "open": inputs.open.loc[dates],
        "high": inputs.high.loc[dates],
        "low": inputs.low.loc[dates],
        "close": inputs.close.loc[dates],
        "volume": inputs.volume.loc[dates],
        "probabilities": inputs.probabilities.loc[dates],
        "conviction_proxy": inputs.conviction.loc[dates],
        "signals": inputs.signals.loc[dates],
        "ranking_scores": inputs.scores.loc[dates],
        "decision_weights": decision_weights.loc[dates],
        "execution_targets": weights,
        "slippage": slippage,
        "thresholds": thresholds.loc[dates],
    }
    for name, matrix in matrices.items():
        matrix.to_csv(output / f"{name}.csv", index_label="Date")

    pd.concat([equity, returns, cash], axis=1).to_csv(
        output / "portfolio_equity.csv", index_label="Date"
    )
    portfolio.orders.records_readable.to_csv(output / "orders.csv", index=False)
    portfolio.trades.records_readable.to_csv(output / "trades.csv", index=False)
    # With raise_reject=False, retain all order diagnostics rather than silently
    # treating requested target weights as successfully executed holdings.
    portfolio.logs.records_readable.to_csv(
        output / "execution_logs.csv", index=False
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Results: {output}")


def normalize_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not symbol:
        raise ValueError("Ticker cannot be empty.")
    return symbol if symbol.endswith(".IS") else f"{symbol}.IS"


def parse_date(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Invalid date.")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--model", type=Path, default=ROOT / "bist_xgb_model.json")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--allocation", choices=("equal", "weighted"), default="equal")
    parser.add_argument("--gross-exposure", type=float, default=1.0)
    parser.add_argument("--base-fraction", type=float, default=0.25)
    parser.add_argument("--payoff-ratio", type=float, default=1.5)
    parser.add_argument("--impact-gamma", type=float, default=0.10)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "portfolio_backtest_results"
    )
    args = parser.parse_args()

    try:
        tickers = tuple(dict.fromkeys(map(normalize_symbol, args.tickers)))
        start = parse_date(args.start)
        end = parse_date(args.end)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    if len(tickers) < 2:
        parser.error("Specify at least two distinct equities.")
    if not 1 <= args.top_n <= len(tickers):
        parser.error("--top-n must be between one and the universe size.")
    if start is not None and end is not None and start > end:
        parser.error("--start must not follow --end.")
    if not math.isfinite(args.initial_cash) or args.initial_cash <= 0:
        parser.error("--initial-cash must be finite and positive.")
    if not 0 < args.gross_exposure <= 1:
        parser.error("--gross-exposure must be within (0, 1].")
    if not 0 < args.base_fraction <= 1:
        parser.error("--base-fraction must be within (0, 1].")
    if not math.isfinite(args.payoff_ratio) or args.payoff_ratio <= 0:
        parser.error("--payoff-ratio must be finite and positive.")
    if not math.isfinite(args.impact_gamma) or args.impact_gamma < 0:
        parser.error("--impact-gamma must be finite and nonnegative.")

    return Config(
        tickers=tickers,
        model_path=args.model.resolve(),
        period=args.period,
        start=start,
        end=end,
        initial_cash=args.initial_cash,
        top_n=args.top_n,
        allocation_method=args.allocation,
        gross_exposure=args.gross_exposure,
        base_fraction=args.base_fraction,
        payoff_ratio=args.payoff_ratio,
        commission=0.0004,
        impact_gamma=args.impact_gamma,
        output_directory=args.output.resolve(),
    )


def main() -> None:
    config = parse_args()
    bundle = load_model(config.model_path)
    histories = fetch_histories(config)
    inputs, thresholds = build_inputs(histories, bundle, config)

    open_prices, close_prices, weights, slippage, decisions = prepare_execution(
        inputs, config
    )
    portfolio = simulate(
        open_prices, close_prices, weights, slippage, config
    )
    save_and_report(
        portfolio,
        inputs,
        thresholds,
        weights,
        slippage,
        decisions,
        bundle,
        config,
    )


if __name__ == "__main__":
    main()