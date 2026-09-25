"""
live_inference.py

Dependencies:
    pip install pandas numpy requests yfinance xgboost truststore tzdata
    pip install "isyatirimhisse>=5.0.0"
    pip install torch transformers huggingface_hub feedparser beautifulsoup4

Optional local .env support:
    pip install python-dotenv

Project integrations:
    core.ai_analyzer:build_pipeline_features
    core.ai_analyzer:get_model
    core.market_data.fetch_news
    retrain_model.score_corpus

Telegram environment variables:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

GitHub Actions must expose repository secrets to the Python step:
    env:
      TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
      TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}

Default historical price file:
    work/backfill/THYAO/THYAO.csv

Supported date columns:
    Date, date, datetime, timestamp, Seans Tarihi

The built-in sentiment adapter scores available same-day headlines published
at or before the inference cutoff. It returns 0.0 when no eligible headlines
exist. Network/model failures are not silently converted into neutral scores.

The existing fetch_news helper supplies at most five headlines. This live
headline sample is narrower than the historical KAP/RSS training corpus.

Importing this module does not execute inference or send notifications.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from datetime import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TRT = "Europe/Istanbul"
LOG = logging.getLogger("bist.live_inference")
OHLCV = ["Open", "High", "Low", "Close", "Volume"]

DEFAULT_SENTIMENT_MODEL = "savasy/bert-base-turkish-sentiment-cased"
DEFAULT_SENTIMENT_LABELS = {
    "LABEL_0": -1.0,
    "LABEL_1": 1.0,
}

# main() updates these from the deployed model's training metadata.
_SENTIMENT_MODEL = DEFAULT_SENTIMENT_MODEL
_SENTIMENT_LABELS = dict(DEFAULT_SENTIMENT_LABELS)


def resolve(spec):
    """Resolve a project function or this module's built-in adapter."""
    module_name, function_name = spec.split(":", 1)

    # Avoid importing a second copy of this script when executed directly.
    if module_name in {
        "__main__",
        __name__,
        Path(__file__).stem,
    }:
        function = globals().get(function_name)
    else:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)

    if not callable(function):
        raise TypeError(
            f"Integration is missing or not callable: {spec}"
        )

    return function


def load_local_environment():
    """Load .env beside this script without overriding existing CI secrets."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        LOG.warning(
            ".env exists, but python-dotenv is unavailable. "
            "Using existing environment variables."
        )
        return

    try:
        load_dotenv(dotenv_path=env_path, override=False)
    except Exception as exc:
        LOG.warning(
            "Could not load .env (%s). Using existing environment variables.",
            type(exc).__name__,
        )


def configure_sentiment_provider(model):
    """Reuse the NLP checkpoint and label map recorded during retraining."""
    global _SENTIMENT_MODEL, _SENTIMENT_LABELS

    booster = model.get_booster()
    saved_model = booster.attr("sentiment_model")
    saved_labels = booster.attr("sentiment_labels")

    if saved_model:
        _SENTIMENT_MODEL = saved_model.strip()
    else:
        _SENTIMENT_MODEL = DEFAULT_SENTIMENT_MODEL

    if saved_labels:
        try:
            parsed_labels = json.loads(saved_labels)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "The deployed model contains invalid sentiment_labels metadata."
            ) from exc

        if not isinstance(parsed_labels, dict) or not parsed_labels:
            raise ValueError(
                "sentiment_labels metadata must be a nonempty JSON object."
            )

        _SENTIMENT_LABELS = {
            str(label): float(polarity)
            for label, polarity in parsed_labels.items()
        }
    else:
        _SENTIMENT_LABELS = dict(DEFAULT_SENTIMENT_LABELS)

    if not _SENTIMENT_MODEL:
        raise ValueError("The sentiment model identifier is empty.")

    if not all(
        np.isfinite(value) and -1 <= value <= 1
        for value in _SENTIMENT_LABELS.values()
    ):
        raise ValueError(
            "Sentiment label polarities must be finite numbers in [-1, 1]."
        )

    LOG.info("Live sentiment checkpoint: %s", _SENTIMENT_MODEL)


def live_sentiment_provider(ticker, cutoff) -> float:
    """Return the mean sentiment of eligible same-day news in [-1, 1].

    Uses the project's verified functions:
        core.market_data.fetch_news(ticker, limit=5)
        retrain_model.score_corpus(news, model_path, labels)

    Only dated headlines from the cutoff's Turkish calendar day, published
    no later than the cutoff, are scored. No eligible news returns 0.0.
    Retrieval or NLP failures propagate instead of masquerading as no news.
    """
    cutoff = pd.Timestamp(cutoff)

    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError(
            "Sentiment cutoff must be a valid timezone-aware timestamp."
        )

    cutoff = cutoff.tz_convert(TRT)
    day_start = cutoff.normalize()
    symbol = ticker.strip().upper().removesuffix(".IS")

    from core.market_data import fetch_news

    # This project's helper supports limits from 1 through 5.
    try:
        articles = fetch_news(symbol, limit=5)
    except Exception as exc:
        raise RuntimeError(
            f"News retrieval failed for {symbol}; sentiment is unavailable."
        ) from exc

    if not isinstance(articles, list):
        raise TypeError("fetch_news must return a list of news dictionaries.")

    eligible = []
    seen = set()
    undated_count = 0

    for article in articles:
        if not isinstance(article, dict):
            raise TypeError("Each news item must be a dictionary.")

        title = " ".join(str(article.get("title") or "").split())
        if not title:
            continue

        published_raw = article.get("published_at")
        if not published_raw:
            undated_count += 1
            continue

        try:
            published = pd.Timestamp(published_raw)
            if pd.isna(published) or published.tzinfo is None:
                undated_count += 1
                continue
            published = published.tz_convert(TRT)
        except (TypeError, ValueError, OverflowError):
            undated_count += 1
            continue

        if not day_start <= published <= cutoff:
            continue

        identity = title.casefold()
        if identity in seen:
            continue
        seen.add(identity)

        eligible.append({
            "text": title,
            "published_at": published.isoformat(),
            "source": "live_rss",
            "url": str(article.get("url") or ""),
        })

    if undated_count:
        LOG.warning(
            "Excluded %d news item(s) without a valid timezone-aware "
            "publication timestamp.",
            undated_count,
        )

    if not eligible:
        LOG.info(
            "%s: no eligible same-day headlines at or before %s; "
            "news_sentiment=0.0.",
            symbol,
            cutoff.isoformat(),
        )
        return 0.0

    # Importing retrain_model does not execute its guarded main().
    # Its scorer keeps the same probability-weighted label calculation,
    # batch_size=32, truncation=True, and maximum token length as training.
    from retrain_model import score_corpus

    news = pd.DataFrame(eligible)

    try:
        scored = score_corpus(
            news,
            _SENTIMENT_MODEL,
            dict(_SENTIMENT_LABELS),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Local NLP scoring failed for {symbol}; sentiment is unavailable."
        ) from exc

    if (
        not isinstance(scored, pd.DataFrame)
        or "score" not in scored.columns
        or len(scored) != len(news)
    ):
        raise ValueError(
            "score_corpus returned an invalid sentiment result."
        )

    values = pd.to_numeric(
        scored["score"],
        errors="raise",
    ).to_numpy(dtype=float)

    if (
        values.size == 0
        or not np.isfinite(values).all()
        or (values < -1.000001).any()
        or (values > 1.000001).any()
    ):
        raise ValueError(
            "NLP sentiment scores must be finite and within [-1, 1]."
        )

    sentiment = float(np.clip(values.mean(), -1.0, 1.0))

    LOG.info(
        "%s: scored %d same-day headline(s); news_sentiment=%+.4f.",
        symbol,
        len(values),
        sentiment,
    )

    return sentiment


def load_history_csv(path, require_ohlcv=True):
    """Read daily prices and dynamically detect the date column."""
    history = pd.read_csv(path)
    history.columns = [
        column.strip() if isinstance(column, str) else column
        for column in history.columns
    ]

    date_col = next(
        (
            column
            for column in [
                "Date",
                "date",
                "datetime",
                "timestamp",
                "Seans Tarihi",
            ]
            if column in history.columns
        ),
        None,
    )

    if date_col:
        if pd.api.types.is_numeric_dtype(history[date_col]):
            raise ValueError(
                f"Date column {date_col!r} is numeric. Convert it to ISO "
                "date strings or explicitly define its timestamp unit."
            )

        history[date_col] = pd.to_datetime(
            history[date_col],
            format="mixed",
            dayfirst=(date_col == "Seans Tarihi"),
            errors="raise",
        )

        if history[date_col].isna().any():
            raise ValueError(
                f"History CSV contains missing dates in {date_col!r}."
            )

        history = history.set_index(date_col)
    else:
        raise ValueError(
            f"No supported date column found in {path!r}. "
            "Expected Date, date, datetime, timestamp, or Seans Tarihi. "
            "Supply an actual daily price CSV, not a news coverage report."
        )

    required = set(OHLCV) if require_ohlcv else {"Close"}
    missing = sorted(required - set(history.columns))

    if missing:
        raise ValueError(
            f"History CSV {path!r} is missing price columns: {missing}."
        )

    for column in required:
        history[column] = pd.to_numeric(
            history[column],
            errors="raise",
        )

    return normalize_daily(history)


def normalize_daily(frame):
    """Normalize daily bar dates without inventing trading sessions."""
    result = frame.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))

    if result.index.tz is not None:
        result.index = result.index.tz_convert(TRT).tz_localize(None)

    result.index = result.index.normalize()

    if result.index.isna().any():
        raise ValueError("Historical bars contain missing dates.")

    if result.index.has_duplicates:
        raise ValueError("Duplicate daily bars.")

    return result.sort_index()


def flatten_yahoo(frame, ticker):
    """Support ordinary and MultiIndex yfinance columns."""
    if frame is None or frame.empty:
        raise ValueError("Yahoo returned no data.")

    result = frame.copy()

    if isinstance(result.columns, pd.MultiIndex):
        levels = [
            level
            for level in range(result.columns.nlevels)
            if ticker in result.columns.get_level_values(level)
        ]

        if len(levels) != 1:
            raise ValueError("Unexpected Yahoo column schema.")

        result = result.xs(ticker, axis=1, level=levels[0])

    return result


def is_close(symbol, day):
    """Read a validated T-0 close from isyatirimhisse v5+."""
    if int(version("isyatirimhisse").split(".")[0]) < 5:
        raise RuntimeError("isyatirimhisse >= 5.0.0 is required.")

    from isyatirimhisse import fetch_stock_data

    raw = fetch_stock_data(
        symbols=symbol,
        start_date=day.strftime("%d-%m-%Y"),
        end_date=day.strftime("%d-%m-%Y"),
        save_to_excel=False,
    )

    required = {
        "HGDG_TARIH",
        "HGDG_HS_KODU",
        "HGDG_KAPANIS",
    }

    if not isinstance(raw, pd.DataFrame) or not required.issubset(raw.columns):
        raise ValueError("Unexpected Is Yatirim response schema.")

    dates = pd.to_datetime(
        raw["HGDG_TARIH"],
        errors="raise",
    ).dt.date

    rows = raw.loc[
        (dates == day)
        & (raw["HGDG_HS_KODU"] == symbol)
    ]

    if len(rows) != 1:
        raise ValueError(
            "Is Yatirim has not published exactly one T-0 row."
        )

    close = float(rows.iloc[0]["HGDG_KAPANIS"])

    if not np.isfinite(close) or close <= 0:
        raise ValueError("Invalid Is Yatirim closing price.")

    return close


def minute_bar(ticker, day):
    """Build a timestamped proxy from minutes strictly before 18:10 TRT."""
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
        raise ValueError(
            "Minute bars have no timezone; refusing to guess."
        )

    raw.index = raw.index.tz_convert(TRT)
    cutoff = start + pd.Timedelta(hours=18, minutes=10)

    raw = raw.loc[
        (raw.index >= start + pd.Timedelta(hours=9, minutes=55))
        & (raw.index < cutoff)
    ].sort_index()

    raw = raw.loc[
        ~raw.index.duplicated(keep="last"),
        OHLCV,
    ]
    raw = raw.apply(pd.to_numeric, errors="raise")

    if (
        raw.empty
        or raw.isna().any().any()
        or not np.isfinite(raw).all().all()
    ):
        raise ValueError("Incomplete or invalid T-0 minute data.")

    if (raw[OHLCV[:4]] <= 0).any().any() or (raw.Volume < 0).any():
        raise ValueError("Invalid minute price or volume.")

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
    """Prefer Is Yatirim close; supplement or fall back to minute data."""
    symbol = ticker.upper().removesuffix(".IS")
    yahoo_ticker = symbol + ".IS"
    day = as_of.date()
    primary_error = None

    try:
        close = is_close(symbol, day)
    except Exception as exc:
        primary_error = str(exc)
        LOG.warning("Primary T-0 close failed: %s", exc)
        close = None

    proxy = None
    minute_time = None

    # Do not substitute TRY turnover for share-count Volume.
    if close is None or require_ohlcv:
        try:
            proxy, minute_time = minute_bar(yahoo_ticker, day)
        except Exception as exc:
            raise RuntimeError(
                "No valid T-0 bar; inference aborted."
            ) from exc

    bar = (
        dict(proxy)
        if proxy is not None
        else {name: np.nan for name in OHLCV}
    )

    if close is not None:
        bar["Close"] = close

        if proxy is not None:
            # Official close may include auction trades absent from the feed.
            bar["High"] = max(bar["High"], close)
            bar["Low"] = min(bar["Low"], close)

    return pd.DataFrame(
        [bar],
        index=[pd.Timestamp(day)],
    ), {
        "close_source": (
            "isyatirimhisse"
            if close is not None
            else "yahoo_1m_proxy"
        ),
        "ohlv_source": (
            "yahoo_1m_proxy"
            if proxy is not None
            else "unavailable"
        ),
        "last_minute": minute_time,
        "primary_error": primary_error,
    }


def align_and_predict(model, df):
    """Align features to the deployed XGBoost schema."""
    names = model.get_booster().feature_names

    if not names or len(set(names)) != len(names):
        raise ValueError(
            "Deployed model has no valid named feature contract."
        )

    if df.columns.has_duplicates:
        raise ValueError("Duplicate input feature names.")

    missing = set(names) - set(df.columns)

    if missing:
        raise ValueError(
            f"Required model features are missing: {sorted(missing)}"
        )

    if not np.isfinite(
        df.loc[:, names].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Required model features contain NaN or infinity."
        )

    # Preserve exact training order and drop extras for legacy models.
    df = df.reindex(columns=model.get_booster().feature_names)
    return model.predict(df)


def prediction_probabilities(model, df):
    """Return actual classifier probabilities when supported."""
    predict_proba = getattr(model, "predict_proba", None)

    if not callable(predict_proba):
        return None

    try:
        aligned = df.reindex(
            columns=model.get_booster().feature_names
        )
        probabilities = np.asarray(
            predict_proba(aligned),
            dtype=float,
        )
        classes = np.asarray(model.classes_)

        if (
            probabilities.ndim != 2
            or probabilities.shape != (len(df), len(classes))
            or len(df) != 1
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.isclose(
                probabilities[0].sum(),
                1.0,
                atol=1e-5,
            )
        ):
            raise ValueError("Invalid class-probability output.")

        return {
            str(label): float(probability)
            for label, probability in zip(
                classes,
                probabilities[0],
            )
        }

    except Exception as exc:
        LOG.warning(
            "Class probabilities unavailable (%s). "
            "The prediction will still be reported.",
            type(exc).__name__,
        )
        return None


def read_holdout_score(model):
    """Read saved validation metadata, not a prediction probability."""
    try:
        raw = model.get_booster().attr("holdout_score")

        if raw is None or raw == "not_evaluated":
            return None

        score = float(raw)
        return score if np.isfinite(score) else None

    except (AttributeError, TypeError, ValueError):
        return None


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
    """Run inference using the existing feature and sentiment adapters."""
    now = (
        pd.Timestamp.now(tz=TRT)
        if as_of is None
        else pd.Timestamp(as_of)
    )

    if now.tzinfo is None:
        raise ValueError("as_of must be timezone-aware.")

    now = now.tz_convert(TRT)

    #if now.time() < time(18, 30):
     #   raise ValueError("Run at or after 18:30 TRT.")

    symbol = ticker.upper().removesuffix(".IS")
    cutoff = now.normalize() + pd.Timedelta(hours=18, minutes=30)
    today = pd.Timestamp(now.date())

    history = normalize_daily(history)
    history = history.loc[history.index < today]

    if history.empty:
        raise ValueError("Historical warm-up bars are required.")

    bar, provenance = fetch_t0(
        symbol,
        cutoff,
        require_ohlcv=require_ohlcv,
    )
    bars = pd.concat([history, bar]).sort_index()
    features = feature_builder(bars.copy())

    if (
        not isinstance(features, pd.DataFrame)
        or not features.index.is_unique
    ):
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
        "ticker": symbol,
        "date": str(today.date()),
        "prediction": np.asarray(prediction).tolist(),
        "probabilities": prediction_probabilities(model, df),
        "holdout_score": read_holdout_score(model),
        "news_sentiment": sentiment,
        "timestamp": pd.Timestamp.now(tz=TRT).isoformat(),
        "news_cutoff": cutoff.isoformat(),
        "provenance": provenance,
    }


def compact_text(value, limit=250):
    """Create bounded single-line text."""
    if value is None:
        return "N/A"

    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)

    return " ".join(text.split())[:limit]


def format_telegram_message(result):
    """Format predictions without inventing BUY/SELL class meanings."""
    prediction = result.get("prediction")

    if isinstance(prediction, list) and len(prediction) == 1:
        prediction = prediction[0]

    decision = result.get("decision")
    if decision is None:
        decision = prediction

    probabilities = result.get("probabilities")

    if probabilities:
        probability_text = "; ".join(
            f"{compact_text(label, 40)}: {float(value):.2%}"
            for label, value in probabilities.items()
        )[:800]
    else:
        probability_text = "N/A"

    holdout = result.get("holdout_score")
    holdout_text = (
        f"{float(holdout):.6f}"
        if holdout is not None
        else "N/A"
    )

    sentiment = result.get("news_sentiment")
    sentiment_text = (
        f"{float(sentiment):+.4f}"
        if sentiment is not None
        else "N/A"
    )

    provenance = result.get("provenance") or {}

    message = "\n".join([
        "BIST CRISIS JARVIS",
        "",
        f"Ticker: {compact_text(result.get('ticker'), 50)}",
        f"Decision/Prediction: {compact_text(decision)}",
        f"Class probabilities: {probability_text}",
        f"Holdout score: {holdout_text}",
        f"News sentiment: {sentiment_text}",
        f"Trading date: {compact_text(result.get('date'), 30)}",
        f"Timestamp (TRT): {compact_text(result.get('timestamp'), 80)}",
        f"News cutoff (TRT): {compact_text(result.get('news_cutoff'), 80)}",
        f"Close source: {compact_text(provenance.get('close_source'), 100)}",
    ])

    return message[:1900]


def notification_warning(message):
    """Log notification failures and annotate GitHub Actions runs."""
    LOG.warning("%s", message)

    if os.getenv("GITHUB_ACTIONS", "").lower() == "true":
        escaped = (
            message.replace("%", "%25")
            .replace("\r", "%0D")
            .replace("\n", "%0A")
        )

        print(
            f"::warning title=Telegram notification::{escaped}",
            flush=True,
        )


def send_telegram_notification(result):
    """Attempt delivery without crashing successful inference."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    missing = []

    if not token:
        missing.append("TELEGRAM_BOT_TOKEN")

    if not chat_id:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        notification_warning(
            "Telegram notification skipped: missing environment variables "
            + ", ".join(missing)
            + ". In GitHub Actions, map repository secrets into the step's env."
        )

        return {
            "status": "skipped",
            "reason": "missing_credentials",
            "missing": missing,
        }

    try:
        message = format_telegram_message(result)
    except Exception as exc:
        notification_warning(
            "Telegram message formatting failed "
            f"({type(exc).__name__}); inference completed."
        )

        return {
            "status": "failed",
            "reason": "formatting_error",
        }

    # Never log this URL because it contains the bot token.
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    try:
        with requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": message,
                "disable_web_page_preview": True,
            },
            timeout=(5, 20),
            allow_redirects=False,
        ) as response:
            status_code = response.status_code

            try:
                body = response.json()
            except ValueError:
                body = None

            if status_code != 200:
                hints = {
                    400: "Check chat ID and request parameters.",
                    401: "Check the bot token.",
                    403: (
                        "The bot may be blocked or lack access to the chat. "
                        "Start a private chat with the bot or check group permissions."
                    ),
                    429: "Telegram rate limit reached.",
                }

                hint = hints.get(
                    status_code,
                    "Telegram returned an unsuccessful HTTP response.",
                )

                notification_warning(
                    f"Telegram delivery failed: HTTP {status_code}. {hint}"
                )

                return {
                    "status": "failed",
                    "reason": "http_error",
                    "http_status": status_code,
                }

            if not isinstance(body, dict) or body.get("ok") is not True:
                notification_warning(
                    "Telegram did not confirm delivery: invalid response "
                    "or API ok=false."
                )

                return {
                    "status": "failed",
                    "reason": "api_rejected",
                }

            delivered_message = body.get("result")

            if (
                not isinstance(delivered_message, dict)
                or not isinstance(
                    delivered_message.get("message_id"),
                    int,
                )
            ):
                notification_warning(
                    "Telegram response did not contain a valid message ID."
                )

                return {
                    "status": "unknown",
                    "reason": "missing_delivery_receipt",
                }

            message_id = delivered_message["message_id"]

            LOG.info(
                "Telegram delivery confirmed; message_id=%s.",
                message_id,
            )

            return {
                "status": "sent",
                "message_id": message_id,
            }

    except requests.Timeout:
        # A timed-out POST may already have been delivered.
        notification_warning(
            "Telegram request timed out; delivery status is unknown. "
            "No automatic retry was made to avoid duplicate notifications."
        )

        return {
            "status": "unknown",
            "reason": "timeout",
        }

    except requests.RequestException as exc:
        # Exception text may contain the token-bearing request URL.
        notification_warning(
            "Telegram network request failed "
            f"({type(exc).__name__}); delivery was not confirmed."
        )

        return {
            "status": "unknown",
            "reason": "network_error",
        }

    except Exception as exc:
        notification_warning(
            "Telegram notification failed "
            f"({type(exc).__name__}); inference completed."
        )

        return {
            "status": "failed",
            "reason": "unexpected_notification_error",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--ticker",
        default="THYAO",
    )
    parser.add_argument(
        "--history",
        default="work/backfill/THYAO/THYAO.csv",
        help="Daily CSV with a Date column",
    )
    parser.add_argument(
        "--features",
        default="core.ai_analyzer:build_pipeline_features",
        help="Existing module:function",
    )
    parser.add_argument(
        "--sentiment-provider",
        default="__main__:live_sentiment_provider",
        help="Sentiment callable: module:function",
    )
    parser.add_argument(
        "--model-factory",
        default="core.ai_analyzer:get_model",
        help="Existing module:function",
    )
    parser.add_argument(
        "--close-only",
        action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    load_local_environment()

    history = load_history_csv(
        args.history,
        require_ohlcv=not args.close_only,
    )

    model = resolve(args.model_factory)()
    model.load_model("model.json")

    sentiment_provider = resolve(args.sentiment_provider)

    if sentiment_provider is live_sentiment_provider:
        configure_sentiment_provider(model)

    result = run_live(
        args.ticker,
        history,
        model,
        resolve(args.features),
        sentiment_provider,
        require_ohlcv=not args.close_only,
    )

    # Telegram delivery remains explicitly connected to successful inference.
    result["telegram"] = send_telegram_notification(result)

    print(
        json.dumps(result, ensure_ascii=False, default=str),
        flush=True,
    )


if __name__ == "__main__":
    main()