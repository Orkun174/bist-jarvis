# live_inference.py
"""Daily cross-sectional BIST scan and Telegram notification."""

from __future__ import annotations

# Configure timezone before importing market-data/application modules.
import os
import time

os.environ["TZ"] = "Europe/Istanbul"
if hasattr(time, "tzset"):
    time.tzset()

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier

from core.ai_analyzer import (
    BENCHMARK_TICKER,
    FEATURE_COLUMNS,
    FEATURE_CONTRACT_VERSION,
    HISTORY_PERIOD,
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
MODEL_PATH = ROOT / "bist_xgb_model.json"
ISTANBUL = ZoneInfo("Europe/Istanbul")
LOGGER = logging.getLogger("bist_daily_scan")

TOP_N = 3
BASE_FRACTION = 0.25
PAYOFF_RATIO = 1.5

TICKERS = (
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


class IstanbulFormatter(logging.Formatter):
    def formatTime(
        self,
        record: logging.LogRecord,
        datefmt: str | None = None,
    ) -> str:
        timestamp = datetime.fromtimestamp(record.created, ISTANBUL)
        return timestamp.strftime(datefmt or "%Y-%m-%d %H:%M:%S %Z")


@dataclass(frozen=True)
class ModelBundle:
    model: XGBClassifier
    features: tuple[str, ...]
    policy: dict[str, Any]


@dataclass(frozen=True)
class Signal:
    ticker: str
    probability: float
    score: float
    conviction_proxy: int
    close: float


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        IstanbulFormatter("%(asctime)s %(levelname)s %(message)s")
    )
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def read_credentials() -> tuple[str, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be configured."
        )
    return token, chat_id


def metadata(model: XGBClassifier, name: str) -> Any:
    raw = model.get_booster().attr(name)
    if raw is None:
        raise ValueError(f"Missing model metadata: {name}")
    return json.loads(raw)


def load_model() -> ModelBundle:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("bist_xgb_model.json is missing from the repository.")

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    booster = model.get_booster()

    expected = {
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "target_definition": TARGET_DEFINITION,
        "benchmark_ticker": BENCHMARK_TICKER,
        "price_basis": PRICE_BASIS,
    }
    for name, value in expected.items():
        if booster.attr(name) != value:
            raise ValueError(f"Model contract mismatch: {name}")

    selected = metadata(model, "selected_features")
    if (
        not isinstance(selected, list)
        or not selected
        or not all(isinstance(name, str) for name in selected)
        or len(selected) != len(set(selected))
        or not set(selected).issubset(FEATURE_COLUMNS)
        or selected != booster.feature_names
    ):
        raise ValueError("Invalid model feature names or ordering.")

    # The deployed model currently selects 18 features from the shared universe.
    # Persisted names/order are authoritative; never substitute mock features.
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("The model must have binary classes [0, 1].")

    policy = metadata(model, "threshold_policy")
    if not isinstance(policy, dict):
        raise ValueError("Invalid threshold policy.")
    if (
        policy.get("percentile_source") != "lagged_VIX_level"
        or policy.get("window") != VIX_PERCENTILE_WINDOW
        or policy.get("min_periods") != VIX_PERCENTILE_MIN_PERIODS
    ):
        raise ValueError("VIX policy differs from the shared feature pipeline.")

    low = np.asarray(policy.get("low_regime"), dtype=float)
    high = np.asarray(policy.get("high_regime"), dtype=float)
    if low.shape != (3,) or high.shape != (3,):
        raise ValueError("Invalid threshold regime dimensions.")
    if not np.isfinite(np.concatenate((low, high))).all():
        raise ValueError("Threshold policy contains nonfinite values.")
    if not (
        0 <= low[0] < high[0] <= 100
        and 0 <= low[1] < low[2] <= 1
        and 0 <= high[1] < high[2] <= 1
    ):
        raise ValueError("Invalid threshold regime bounds.")

    return ModelBundle(model, tuple(selected), policy)


def threshold_bounds(
    percentile: float,
    policy: Mapping[str, Any],
) -> tuple[float, float]:
    if not math.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("Current VIX percentile is unavailable.")

    low = np.asarray(policy["low_regime"], dtype=float)
    high = np.asarray(policy["high_regime"], dtype=float)
    fraction = float(np.clip(
        (percentile - low[0]) / (high[0] - low[0]), 0, 1
    ))
    lower = float(low[1] + fraction * (high[1] - low[1]))
    upper = float(low[2] + fraction * (high[2] - low[2]))
    return lower, upper


def deterministic_conviction(
    equity: pd.DataFrame,
    cmf: float,
) -> int:
    """Match the backtest's CMF/OBV proxy; this is not a Qwen inference."""
    obv = (
        np.sign(equity["Close"].diff()).fillna(0.0) * equity["Volume"]
    ).cumsum()
    if len(obv) < 6 or not math.isfinite(cmf):
        raise ValueError("Insufficient CMF/OBV history.")

    change = float(obv.iloc[-1] - obv.iloc[-6])
    if cmf > 0 and change > 0:
        return 7
    if cmf < 0 and change < 0:
        return 3
    return 5


def scan(
    bundle: ModelBundle,
) -> tuple[list[Signal], dict[str, Any]]:
    """Download macros once, construct shared features, and batch-score equities."""
    histories: dict[str, pd.DataFrame] = {}
    for symbol in dict.fromkeys(MACRO_TICKERS.values()):
        LOGGER.info("Downloading macro history: %s", symbol)
        histories[symbol] = download_history(symbol, HISTORY_PERIOD)

    benchmark = histories[BENCHMARK_TICKER]
    if benchmark.empty:
        raise ValueError("Benchmark history is empty.")
    as_of = benchmark.index[-1]

    now = datetime.now(ISTANBUL)
    age_days = (now.date() - as_of.date()).days
    if age_days < 0 or age_days > 7:
        raise ValueError("Benchmark data is stale or future-dated.")

    macro_histories = {
        name: histories[symbol] for name, symbol in MACRO_TICKERS.items()
    }
    macro_features = build_macro_features(
        macro_histories, benchmark.index
    )
    percentile = float(
        macro_features.loc[as_of, VIX_PERCENTILE_COLUMN]
    )
    lower, upper = threshold_bounds(percentile, bundle.policy)

    rows: list[pd.DataFrame] = []
    details: dict[str, tuple[int, float]] = {}
    failures: dict[str, str] = {}

    for ticker in TICKERS:
        try:
            LOGGER.info("Preparing equity: %s", ticker)
            history = download_history(ticker, HISTORY_PERIOD)
            aligned, observed = align_equity_to_benchmark(
                history, benchmark.index
            )
            if aligned.empty or aligned.index[-1] != as_of:
                raise ValueError("Missing the common benchmark session.")
            if not bool(observed.iloc[-1]) or aligned["Volume"].iloc[-1] <= 0:
                raise ValueError("Latest common-session OHLCV is incomplete.")

            local = build_stationary_features(aligned, benchmark)
            frame = local.join(macro_features, how="left").loc[[as_of]]
            matrix = feature_matrix(frame, bundle.features)
            matrix.index = pd.Index([ticker], name="Ticker")

            proxy = deterministic_conviction(
                aligned, float(local.loc[as_of, "CMF_20"])
            )
            close = float(aligned.loc[as_of, "Close"])
            if not math.isfinite(close) or close <= 0:
                raise ValueError("Invalid latest close.")

            rows.append(matrix)
            details[ticker] = (proxy, close)
        except Exception as exc:
            failures[ticker] = type(exc).__name__
            LOGGER.warning(
                "Equity unavailable: %s (%s)", ticker, type(exc).__name__
            )

    if not rows:
        raise RuntimeError("No equities have valid features.")

    matrix = pd.concat(rows, axis=0)
    probabilities = bundle.model.predict_proba(matrix)[:, 1]
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Invalid model probabilities.")

    eligible: list[Signal] = []
    for ticker, probability in zip(matrix.index, probabilities):
        probability = float(probability)
        proxy, close = details[str(ticker)]
        kelly = (
            probability * PAYOFF_RATIO - (1.0 - probability)
        ) / PAYOFF_RATIO
        score = float(np.clip(
            BASE_FRACTION * (proxy / 10.0) * kelly, 0.0, 1.0
        ))

        if probability >= upper and score > 0:
            eligible.append(
                Signal(str(ticker), probability, score, proxy, close)
            )

    # Resolve ties deterministically using the configured universe order.
    universe_order = {ticker: index for index, ticker in enumerate(TICKERS)}
    eligible.sort(key=lambda item: (-item.score, universe_order[item.ticker]))

    return eligible[:TOP_N], {
        "as_of": as_of.date().isoformat(),
        "scan_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "vix_percentile": percentile,
        "lower": lower,
        "upper": upper,
        "evaluated": len(matrix),
        "universe_size": len(TICKERS),
        "eligible_count": len(eligible),
        "failures": failures,
        "feature_count": len(bundle.features),
        "age_days": age_days,
    }


def format_message(
    selected: list[Signal],
    diagnostics: Mapping[str, Any],
) -> str:
    lines = [
        "📊 BIST AI Radar — Günlük Portföy Taraması",
        f"🕒 Tarama: {diagnostics['scan_time']} (Europe/Istanbul)",
        f"📅 Kullanılan fiyat seansı: {diagnostics['as_of']}",
        (
            f"🔎 Kapsama: {diagnostics['evaluated']}/"
            f"{diagnostics['universe_size']} hisse"
        ),
        f"🌡️ VIX yüzdelik dilimi: %{diagnostics['vix_percentile']:.1f}",
        (
            f"🎯 Dinamik eşikler: %{100 * diagnostics['lower']:.2f} / "
            f"%{100 * diagnostics['upper']:.2f}"
        ),
        "",
    ]

    failures = diagnostics["failures"]
    if failures:
        # A partial scan cannot establish the true top three of the universe.
        lines.extend([
            "⚠️ Evren taraması eksik; Top 3 portföy sinyali yayımlanmadı.",
            "Verisi doğrulanamayan hisseler:",
            ", ".join(ticker.removesuffix(".IS") for ticker in failures),
            "Mevcut portföyün değiştirilmesi için sinyal üretilmedi.",
        ])
    elif not selected:
        lines.extend([
            "🔴 CASH — Hedef portföy %100 nakit.",
            "Hiçbir hisse dinamik giriş eşiğini geçmedi.",
        ])
    else:
        weight = 100.0 / len(selected)
        lines.append("🟢 LONG — Eşit ağırlıklı hedef portföy")
        for number, signal in enumerate(selected, start=1):
            lines.extend([
                (
                    f"{number}. 🟢 {signal.ticker.removesuffix('.IS')} "
                    f"| Hedef ağırlık %{weight:.2f}"
                ),
                (
                    f"   XGB %{signal.probability * 100:.2f} "
                    f"| Sıralama skoru {signal.score:.4f}"
                ),
                (
                    f"   Teknik proxy {signal.conviction_proxy}/10 "
                    f"| Kapanış ₺{signal.close:,.2f}"
                ),
            ])
        lines.append(
            "Listeden çıkan hisselerin hedef ağırlığı %0'dır."
        )

    lines.extend([
        "",
        "ℹ️ Olasılık, 5 seansta BIST100'ü geçme model skorudur.",
        "Teknik proxy CMF + OBV kuralıdır; canlı Qwen yanıtı değildir.",
        "Bildirim hedef dağılımdır; emir gönderilmez.",
    ])

    if diagnostics["age_days"] > 0:
        lines.append(
            "Veri katmanı yalnızca tamamlanmış barları kullanır; "
            "bugünün mumu hariç tutulabilir."
        )

    message = "\n".join(lines)
    if len(message) > 4096:
        raise ValueError("Telegram message exceeds the supported length.")
    return message


def send_telegram(token: str, chat_id: str, message: str) -> None:
    """Send once; retry only an explicit Telegram rate-limit rejection."""
    endpoint = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": True,
    }

    for attempt in range(3):
        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=(10, 40),
                allow_redirects=False,
            )
        except requests.RequestException:
            # Do not print exception text: it may contain the token-bearing URL.
            # A timeout may occur after delivery, so avoid blind duplicate sends.
            raise RuntimeError(
                "Telegram connection failed; delivery status is unknown."
            ) from None

        try:
            result = response.json()
        except ValueError:
            raise RuntimeError(
                f"Telegram returned invalid JSON (HTTP {response.status_code})."
            ) from None

        if response.status_code == 429 and attempt < 2:
            retry_after = result.get("parameters", {}).get("retry_after", 5)
            delay = min(max(float(retry_after), 1.0), 60.0)
            time.sleep(delay)
            continue

        if response.status_code != 200 or not result.get("ok", False):
            raise RuntimeError(
                f"Telegram rejected the message (HTTP {response.status_code})."
            )

        LOGGER.info("Telegram message sent successfully.")
        return

    raise RuntimeError("Telegram rate-limit retries exhausted.")


def main() -> int:
    configure_logging()

    try:
        token, chat_id = read_credentials()
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1

    try:
        bundle = load_model()
        selected, diagnostics = scan(bundle)
        message = format_message(selected, diagnostics)
    except Exception as exc:
        LOGGER.error("Scan failed (%s).", type(exc).__name__)
        timestamp = datetime.now(ISTANBUL).strftime("%Y-%m-%d %H:%M:%S")
        message = (
            "⚠️ BIST AI Radar — Tarama tamamlanamadı\n"
            f"🕒 {timestamp} (Europe/Istanbul)\n"
            f"Hata sınıfı: {type(exc).__name__}\n"
            "Yeni portföy sinyali üretilmedi. GitHub Actions kayıtlarını kontrol edin."
        )
        try:
            send_telegram(token, chat_id, message)
        except RuntimeError as notification_error:
            LOGGER.error("%s", notification_error)
        return 1

    try:
        send_telegram(token, chat_id, message)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1

    return 1 if diagnostics["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())