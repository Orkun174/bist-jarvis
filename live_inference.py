"""T-0 ingestion and schema-safe inference; importing this module runs no jobs.

Dependencies: pandas numpy yfinance isyatirimhisse>=5.0.0 truststore xgboost
The existing feature builder and NLP provider are injected, not reimplemented.
History must use unadjusted TRY OHLC prices and share-count Volume.
"""
from __future__ import annotations

import importlib
import logging
from datetime import time
from importlib.metadata import version

import numpy as np
import pandas as pd

TRT = "Europe/Istanbul"
LOG = logging.getLogger(__name__)
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def resolve(spec):
    """Resolve an explicit existing integration function: package.module:name."""
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


def normalize_daily(frame):
    result = frame.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if result.index.tz is not None:
        result.index = result.index.tz_convert(TRT).tz_localize(None)
    result.index = result.index.normalize()
    if result.index.has_duplicates:
        raise ValueError("Duplicate daily bars.")
    return result.sort_index()


def flatten_yahoo(frame, ticker):
    if frame is None or frame.empty:
        raise ValueError("Yahoo returned no data.")
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        levels = [
            i for i in range(frame.columns.nlevels)
            if ticker in frame.columns.get_level_values(i)
        ]
        if len(levels) != 1:
            raise ValueError("Unexpected Yahoo column schema.")
        frame = frame.xs(ticker, axis=1, level=levels[0])
    return frame


def is_close(symbol, day):
    """The library returns raw HGDG_* columns, not Yahoo-style columns."""
    if int(version("isyatirimhisse").split(".")[0]) < 5:
        raise RuntimeError("isyatirimhisse >= 5.0.0 is required.")

    from isyatirimhisse import fetch_stock_data

    raw = fetch_stock_data(
        symbols=symbol,
        start_date=day.strftime("%d-%m-%Y"),
        end_date=day.strftime("%d-%m-%Y"),
        save_to_excel=False,
    )
    required = {"HGDG_TARIH", "HGDG_HS_KODU", "HGDG_KAPANIS"}
    if not isinstance(raw, pd.DataFrame) or not required.issubset(raw.columns):
        raise ValueError("Unexpected Is Yatirim response schema.")

    dates = pd.to_datetime(raw["HGDG_TARIH"], errors="raise").dt.date
    rows = raw.loc[
        (dates == day) & (raw["HGDG_HS_KODU"] == symbol)
    ]
    if len(rows) != 1:
        raise ValueError("Is Yatirim has not published exactly one T-0 row.")

    close = float(rows.iloc[0]["HGDG_KAPANIS"])
    if not np.isfinite(close) or close <= 0:
        raise ValueError("Invalid Is Yatirim closing price.")
    return close


def minute_bar(ticker, day):
    """A timestamped proxy, not a guaranteed official auction close."""
    import yfinance as yf

    start = pd.Timestamp(day, tz=TRT)
    raw = flatten_yahoo(
        yf.download(
            ticker,
            start=start.to_pydatetime(),
            end=(start + pd.Timedelta(days=1)).to_pydatetime(),
            interval="1m",
            auto_adjust=False,
            prepost=False,
            progress=False,
            threads=False,
            timeout=20,
        ),
        ticker,
    )
    if raw.index.tz is None:
        raise ValueError("Minute bars have no timezone; refusing to guess.")

    raw.index = raw.index.tz_convert(TRT)
    cutoff = start + pd.Timedelta(hours=18, minutes=10)

    # Strictly exclude the 18:10 bar and all later bars.
    raw = raw.loc[
        (raw.index >= start + pd.Timedelta(hours=9, minutes=55))
        & (raw.index < cutoff)
    ].sort_index()
    raw = raw.loc[~raw.index.duplicated(keep="last"), OHLCV]
    raw = raw.apply(pd.to_numeric, errors="raise")

    if raw.empty or raw.isna().any().any() or not np.isfinite(raw).all().all():
        raise ValueError("Incomplete or invalid T-0 minute data.")
    if (raw[OHLCV[:4]] <= 0).any().any() or (raw.Volume < 0).any():
        raise ValueError("Invalid minute price or volume.")

    # Reject stale/half-day data instead of treating it as a normal close.
    if cutoff - raw.index[-1] > pd.Timedelta(minutes=10):
        raise ValueError("Last minute is too old for the 18:10 cutoff.")
    if raw.index[0] > start + pd.Timedelta(hours=10, minutes=5):
        raise ValueError("Minute history misses the session opening.")

    return {
        "Open": float(raw.Open.iloc[0]),
        "High": float(raw.High.max()),
        "Low": float(raw.Low.min()),
        "Close": float(raw.Close.iloc[-1]),
        "Volume": float(raw.Volume.sum()),
    }, raw.index[-1].isoformat()


def fetch_t0(ticker, as_of, require_ohlcv=True):
    symbol = ticker.upper().removesuffix(".IS")
    ticker = symbol + ".IS"
    day = as_of.date()
    primary_error = None

    try:
        close = is_close(symbol, day)
    except Exception as exc:
        primary_error = str(exc)
        LOG.warning("Primary T-0 close failed: %s", exc)
        close = None

    # Do not assume the Is Yatirim close endpoint provides Open or share volume.
    # TRY turnover must never be substituted for share-count Volume.
    proxy, minute_time = None, None
    if close is None or require_ohlcv:
        try:
            proxy, minute_time = minute_bar(ticker, day)
        except Exception:
            if close is None or require_ohlcv:
                raise RuntimeError("No valid T-0 bar; inference aborted.")

    bar = proxy or {name: np.nan for name in OHLCV}
    if close is not None:
        bar["Close"] = close

        # Preserve OHLC consistency when the official close includes auction
        # trades absent from the minute feed. Other fields remain proxies.
        if proxy:
            bar["High"] = max(bar["High"], close)
            bar["Low"] = min(bar["Low"], close)

    return pd.DataFrame([bar], index=[pd.Timestamp(day)]), {
        "close_source": (
            "isyatirimhisse" if close is not None else "yahoo_1m_proxy"
        ),
        "ohlv_source": "yahoo_1m_proxy" if proxy else "unavailable",
        "last_minute": minute_time,
        "primary_error": primary_error,
    }


def align_and_predict(model, df):
    names = model.get_booster().feature_names
    if not names or len(set(names)) != len(names):
        raise ValueError("Deployed model has no valid named feature contract.")
    if df.columns.has_duplicates:
        raise ValueError("Duplicate input feature names.")

    missing = set(names) - set(df.columns)
    if missing:
        raise ValueError(
            f"Required model features are missing: {sorted(missing)}"
        )
    if not np.isfinite(df.loc[:, names].to_numpy(dtype=float)).all():
        raise ValueError("Required model features contain NaN or infinity.")

    # Old model: news_sentiment is dropped.
    # Retrained model: news_sentiment is retained in its trained position.
    # Do not disable XGBoost feature validation.
    df = df.reindex(columns=model.get_booster().feature_names)
    return model.predict(df)


def run_live(
    ticker,
    history,
    model,
    feature_builder,
    sentiment_provider,
    *,
    require_ohlcv=True,
    as_of=None,
):
    """Run one live inference.

    sentiment_provider(ticker, cutoff) -> finite scalar in [-1, 1].

    feature_builder receives unadjusted daily bars and must preserve their
    DatetimeIndex. Reuse the SAME causal builder in retraining.

    The NLP provider must implement the same model, label mapping, daily
    aggregation and forward-fill policy used during retraining.
    """
    now = (
        pd.Timestamp.now(tz=TRT)
        if as_of is None
        else pd.Timestamp(as_of)
    )
    if now.tzinfo is None:
        raise ValueError("as_of must be timezone-aware.")
    now = now.tz_convert(TRT)
    if now.time() < time(18, 30):
        raise ValueError("Run at or after 18:30 TRT.")

    # Late cron execution must not incorporate news published after 18:30.
    cutoff = now.normalize() + pd.Timedelta(hours=18, minutes=30)
    history = normalize_daily(history)
    today = pd.Timestamp(now.date())
    history = history.loc[history.index < today]
    if history.empty:
        raise ValueError("Historical warm-up bars are required.")

    bar, provenance = fetch_t0(ticker, cutoff, require_ohlcv)
    bars = pd.concat([history, bar]).sort_index()
    features = feature_builder(bars.copy())

    if not isinstance(features, pd.DataFrame) or not features.index.is_unique:
        raise ValueError(
            "Feature builder must return a uniquely indexed DataFrame."
        )
    if today not in features.index:
        raise ValueError("Feature builder did not produce T-0 features.")

    df = features.loc[[today]].copy()
    sentiment = float(sentiment_provider(ticker, cutoff))
    if not np.isfinite(sentiment) or not -1 <= sentiment <= 1:
        raise ValueError("Invalid news_sentiment; inference aborted.")

    df["news_sentiment"] = sentiment
    prediction = align_and_predict(model, df)

    return {
        "date": str(today.date()),
        "prediction": prediction.tolist(),
        "news_sentiment": sentiment,
        "provenance": provenance,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument(
        "--history", required=True, help="Daily CSV with a Date column"
    )
    parser.add_argument(
        "--features", required=True, help="Existing module:function"
    )
    parser.add_argument(
        "--sentiment-provider", required=True, help="nlp_engine:function"
    )
    parser.add_argument(
        "--model-factory", required=True, help="Existing module:function"
    )
    parser.add_argument("--close-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    model = resolve(args.model_factory)()
    model.load_model("model.json")
    history = pd.read_csv(
        args.history, index_col="Date", parse_dates=["Date"]
    )

    result = run_live(
        args.ticker,
        history,
        model,
        resolve(args.features),
        resolve(args.sentiment_provider),
        require_ohlcv=not args.close_only,
    )
    print(json.dumps(result, ensure_ascii=False))