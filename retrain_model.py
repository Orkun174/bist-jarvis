"""
retrain_model.py

Manual-only XGBoost schema migration with historical news sentiment.

Dependencies:
    pip install pandas numpy requests beautifulsoup4 feedparser pykap
    pip install torch transformers huggingface_hub xgboost scikit-learn yfinance

Project adapters:
    --features module:function
        Receives daily OHLCV and returns a causal feature DataFrame.
    --targets module:function
        Receives daily OHLCV and returns columns: target, label_end.
    --model-factory module:function
        Returns an XGBoost sklearn estimator with the original objective.

Migration behavior:
- Constant sentiment logs a warning and does not block training.
- Fewer than 250 usable rows logs a warning and does not block training.
- Missing original features are retained as NaN columns, with a warning.
  XGBoost handles these as missing values; no feature values are fabricated.
- A constant feature is saved in the schema but cannot teach sentiment effects.
- Insufficient holdout data skips validation rather than blocking final fitting.
- Empty training data and incompatible estimator objectives still stop execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import random
import shutil
import tempfile
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import feedparser
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

TRT = "Europe/Istanbul"
SAVASY_MODEL = "savasy/bert-base-turkish-sentiment-cased"
LOG = logging.getLogger("bist.retrain")


def resolve(spec):
    """Resolve an existing project function using module:function."""
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"Integration is not callable: {spec}")
    return function


def normalize_daily(frame):
    """Normalize daily dates without synthesizing missing trading sessions."""
    result = frame.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))

    if result.index.tz is not None:
        result.index = result.index.tz_convert(TRT).tz_localize(None)

    result.index = result.index.normalize()
    if result.index.has_duplicates:
        raise ValueError("Duplicate dates in historical OHLCV.")

    return result.sort_index()


def flatten_yahoo(frame, ticker):
    """Handle ordinary and MultiIndex yfinance columns."""
    if frame is None or frame.empty:
        raise RuntimeError("Yahoo returned no historical OHLCV.")

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


class CachedHTTP:
    """Browser-style headers, persistent caching, and exponential retries."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/140.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Origin": "https://www.kap.org.tr",
            "Referer": "https://www.kap.org.tr/tr/",
        })
        self.last = 0.0

    @staticmethod
    def retry_after_seconds(response):
        """Parse numeric and HTTP-date Retry-After values."""
        value = response.headers.get("Retry-After", "").strip()
        if not value:
            return 0.0

        try:
            if value.isdigit():
                return float(value)
            return max(
                0.0,
                parsedate_to_datetime(value).timestamp() - time.time(),
            )
        except (TypeError, ValueError, OverflowError):
            LOG.warning("Malformed Retry-After header: %r", value)
            return 0.0

    def get(self, method, url, **kwargs):
        key = hashlib.sha256(
            json.dumps(
                [method, url, kwargs],
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        cache_path = self.directory / key

        if cache_path.exists():
            return cache_path.read_bytes()

        for attempt in range(6):
            time.sleep(max(0, 4.0 - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            server_delay = 0.0

            try:
                with self.session.request(
                    method,
                    url,
                    timeout=(10, 40),
                    **kwargs,
                ) as response:
                    server_delay = self.retry_after_seconds(response)

                    if response.status_code == 429:
                        delay = max(30 * (2 ** attempt), server_delay)
                        LOG.warning(
                            "HTTP 429 from %s; attempt %d/6. "
                            "Waiting %.1f seconds.",
                            url,
                            attempt + 1,
                            delay,
                        )
                        response.close()
                        time.sleep(delay)
                        continue

                    if response.status_code in (401, 403):
                        raise RuntimeError(
                            f"Access denied: HTTP {response.status_code} "
                            f"from {url}. Cached progress is preserved."
                        )

                    response.raise_for_status()
                    data = response.content

                if len(data) > 10_000_000:
                    raise RuntimeError(
                        f"Unexpectedly large response from {url}."
                    )

                temporary = cache_path.with_suffix(".tmp")
                temporary.write_bytes(data)
                temporary.replace(cache_path)
                return data

            except requests.RequestException as exc:
                if isinstance(exc, requests.HTTPError):
                    if (
                        exc.response is not None
                        and exc.response.status_code < 500
                    ):
                        raise

                if attempt == 5:
                    raise

                delay = max(
                    5 * (2 ** attempt) + random.uniform(0, 2),
                    server_delay,
                )
                LOG.warning(
                    "Request failed for %s; attempt %d/6: %s. "
                    "Retrying in %.1f seconds.",
                    url,
                    attempt + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise requests.exceptions.RetryError(
            f"HTTP 429 persisted through six attempts for {url}. "
            "Cached progress is preserved; resume the script later."
        )

    def close(self):
        self.session.close()


def backward_windows(start, end):
    """Yield half-open weekly windows, newest first."""
    while end > start:
        left = max(start, end - pd.Timedelta(days=7))
        yield left, end
        end = left


def parse_kap_time(value):
    """Parse KAP timestamps using the Turkish timezone."""
    text = str(value).strip()

    for fmt in (
        "%Y.%m.%d %H:%M:%S",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y.%m.%d",
        "%d.%m.%Y",
        "%Y-%m-%d",
    ):
        try:
            stamp = pd.Timestamp(datetime.strptime(text, fmt))
            break
        except ValueError:
            continue
    else:
        stamp = pd.Timestamp(value)

    if pd.isna(stamp):
        raise ValueError(f"Missing KAP publication timestamp: {value!r}")

    if len(text) <= 10:
        stamp = stamp.normalize() + pd.Timedelta(
            hours=23,
            minutes=59,
            seconds=59,
        )

    if stamp.tzinfo is None:
        return stamp.tz_localize(TRT)

    return stamp.tz_convert(TRT)


def kap_news(http, symbol, start, end):
    """Retrieve public ODA disclosures using pykap company-ID mapping."""
    import pykap

    companies = pykap.get_bist_companies(output_format="dict")
    matches = [
        company
        for company in companies
        if company.get("ticker") == symbol
    ]
    if len(matches) != 1 or not matches[0].get("company_id"):
        raise ValueError(
            f"No unique pykap company_id mapping for {symbol}."
        )

    company = matches[0]
    LOG.info("KAP mapping: %s -> %s", symbol, company["company_id"])

    def window(left, right):
        payload = {
            "fromDate": str(left.date()),
            "toDate": str((right - pd.Timedelta(days=1)).date()),
            "disclosureClass": "ODA",
            "subjectList": [],
            "mkkMemberOidList": [company["company_id"]],
            "inactiveMkkMemberOidList": [],
            "bdkMemberOidList": [],
            "fromSrc": False,
            "disclosureIndexList": [],
        }

        rows = json.loads(
            http.get(
                "POST",
                "https://www.kap.org.tr/tr/api/disclosure/members/byCriteria",
                json=payload,
            )
        )
        if not isinstance(rows, list):
            raise ValueError("Unexpected KAP disclosure-list schema.")

        if len(rows) >= 2000:
            if (right - left).days <= 1:
                raise RuntimeError(
                    "KAP daily result cap reached; archive is incomplete."
                )
            middle = left + pd.Timedelta(
                days=(right - left).days // 2
            )
            yield from window(left, middle)
            yield from window(middle, right)
            return

        for row in rows:
            index = int(row["disclosureIndex"])
            detail = json.loads(
                http.get(
                    "GET",
                    "https://www.kap.org.tr/tr/api/notification/"
                    f"attachment-detail/{index}",
                    headers={
                        "Referer": (
                            f"https://www.kap.org.tr/tr/Bildirim/{index}"
                        ),
                    },
                )
            )
            if not isinstance(detail, list) or len(detail) != 1:
                raise ValueError(
                    f"Unexpected KAP detail schema for {index}."
                )

            basic = detail[0]["disclosure"]["disclosureBasic"]
            published = parse_kap_time(basic["publishDate"])
            body = detail[0].get("disclosureBody")

            if not isinstance(body, list) or not all(
                isinstance(part, str) for part in body
            ):
                raise ValueError(
                    f"Missing KAP disclosure body: {index}."
                )

            soup = BeautifulSoup(" ".join(body), "html.parser")
            for element in soup.select("script, style"):
                element.decompose()

            text = soup.get_text(" ", strip=True)
            if not text:
                raise ValueError(
                    f"Empty KAP disclosure body: {index}."
                )

            if left.date() <= published.date() < right.date():
                yield {
                    "id": f"kap:{index}",
                    "source": "kap",
                    "text": text,
                    "published_at": published.isoformat(),
                }

    for left, right in backward_windows(start, end):
        LOG.info("KAP window: %s -> %s", left.date(), right.date())
        yield from window(left, right)


def rss_news(http, query, start, end):
    """Retrieve RSS entries, accepting capped windows with a warning."""
    for left, right in backward_windows(start, end):
        age_days = max(
            1,
            (pd.Timestamp.now(tz=TRT).date() - left.date()).days + 2,
        )
        search = (
            f"{query} "
            f"after:{(left - pd.Timedelta(days=1)).date()} "
            f"before:{right.date()} "
            f"when:{age_days}d"
        )

        raw = http.get(
            "GET",
            "https://news.google.com/rss/search",
            params={
                "q": search,
                "hl": "tr",
                "gl": "TR",
                "ceid": "TR:tr",
            },
            headers={
                "Accept": (
                    "application/rss+xml, application/xml;q=0.9, */*;q=0.8"
                ),
                "Origin": "https://news.google.com",
                "Referer": "https://news.google.com/",
            },
        )

        feed = feedparser.parse(raw)
        if feed.bozo or not feed.version:
            raise ValueError("Invalid or blocked Google News RSS response.")

        if len(feed.entries) >= 100:
            LOG.warning(
                "RSS window %s -> %s returned %d entries. "
                "Continuing with available entries; coverage may be incomplete.",
                left.date(),
                right.date(),
                len(feed.entries),
            )
            # Continue processing available results. This does not remove
            # Google's server-side cap or guarantee complete archive coverage.
            pass

        for item in feed.entries:
            published = pd.Timestamp(
                parsedate_to_datetime(item["published"])
            )
            if published.tzinfo is None:
                raise ValueError(
                    "RSS publication timestamp lacks a timezone."
                )
            published = published.tz_convert(TRT)

            if not left.date() <= published.date() < right.date():
                continue

            text = BeautifulSoup(
                item.get("title", "") + " " + item.get("summary", ""),
                "html.parser",
            ).get_text(" ", strip=True)

            if text:
                yield {
                    "id": "rss:" + item["link"],
                    "source": "rss",
                    "text": text,
                    "published_at": published.isoformat(),
                }


def normalize_label_map(model_path, config, supplied):
    """Accept semantic aliases for the documented savasy binary checkpoint."""
    if not isinstance(supplied, dict) or not supplied:
        raise ValueError("--label-map must be a nonempty JSON object.")

    expected = set(config.id2label.values())
    expected_by_case = {
        str(label).casefold(): label
        for label in expected
    }

    identifiers = {
        str(model_path).strip().rstrip("/").casefold(),
        str(getattr(config, "_name_or_path", "")).strip().rstrip("/").casefold(),
    }
    is_savasy = SAVASY_MODEL.casefold() in identifiers

    aliases = {}
    if is_savasy and expected == {"LABEL_0", "LABEL_1"}:
        aliases = {
            "negative": "LABEL_0",
            "positive": "LABEL_1",
        }

    normalized = {}
    for supplied_name, supplied_value in supplied.items():
        key = str(supplied_name).strip().casefold()
        actual_name = aliases.get(key, expected_by_case.get(key))

        if actual_name is None:
            raise ValueError(
                f"Unknown sentiment label {supplied_name!r}. "
                f"Checkpoint labels: {sorted(expected)}. "
                f"Accepted semantic aliases: {sorted(aliases)}."
            )

        value = float(supplied_value)
        if not np.isfinite(value) or not -1 <= value <= 1:
            raise ValueError(
                "Label polarities must be finite numbers in [-1, 1]."
            )

        if actual_name in normalized and normalized[actual_name] != value:
            raise ValueError(
                f"Conflicting polarities supplied for {actual_name}."
            )
        normalized[actual_name] = value

    if set(normalized) != expected:
        raise ValueError(
            "Label map does not cover all checkpoint labels. "
            f"Expected: {sorted(expected)}; "
            f"received after normalization: {sorted(normalized)}."
        )

    LOG.info("Effective sentiment label mapping: %s", normalized)
    return normalized


def score_corpus(news, model_path, labels):
    """Download missing model files and run bounded local CPU inference."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        pipeline,
    )

    torch.set_num_threads(2)

    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=False,
    )
    effective_labels = normalize_label_map(
        model_path,
        config,
        labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=False,
    )
    encoder = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        local_files_only=False,
    )

    nlp = pipeline(
        "text-classification",
        model=encoder,
        tokenizer=tokenizer,
        device=-1,
        batch_size=32,
        truncation=True,
    )

    scored = news.copy()
    scores = []

    for offset in range(0, len(news), 32):
        batch = news.text.iloc[offset:offset + 32].tolist()
        predictions = nlp(
            batch,
            batch_size=32,
            truncation=True,
            max_length=256,
            padding=True,
            top_k=None,
        )

        if len(predictions) != len(batch):
            raise RuntimeError(
                "NLP output count does not match input count."
            )

        for prediction in predictions:
            score = sum(
                effective_labels[item["label"]] * float(item["score"])
                for item in prediction
            )
            if not np.isfinite(score):
                raise RuntimeError(
                    "NLP produced a nonfinite sentiment score."
                )
            scores.append(score)

        if offset % 1024 == 0:
            LOG.info(
                "Sentiment scored: %d/%d",
                min(offset + 32, len(news)),
                len(news),
            )

    scored["score"] = scores
    scored.attrs["effective_label_map"] = effective_labels
    return scored


def sentiment_for_bars(news, index):
    """Forward-fill calendar sentiment subject to the 18:30 TRT cutoff."""
    stamps = pd.to_datetime(
        news.published_at,
        utc=True,
    ).dt.tz_convert(TRT)

    dates = stamps.dt.tz_localize(None).dt.normalize()
    cutoff = stamps.dt.normalize() + pd.Timedelta(
        hours=18,
        minutes=30,
    )
    dates = dates + pd.to_timedelta(
        (stamps > cutoff).astype(int),
        unit="D",
    )

    daily = pd.Series(
        news.score.to_numpy(),
        index=dates,
    ).groupby(level=0).mean()

    calendar = pd.date_range(
        min(daily.index.min(), index.min()),
        index.max(),
        freq="D",
    )

    aligned = daily.reindex(calendar).ffill().reindex(index)
    initial_missing = int(aligned.isna().sum())
    if initial_missing:
        LOG.warning(
            "%d trading rows precede available sentiment; initializing "
            "their news_sentiment to 0.0.",
            initial_missing,
        )

    return aligned.fillna(0.0)


def save_open_model(model, factory, expected_old_hash):
    """Validate staged JSON, preserve a backup, and publish atomically."""
    destination = Path("model.json").resolve()
    old_cwd = Path.cwd()

    with tempfile.TemporaryDirectory(
        dir=destination.parent
    ) as directory:
        try:
            os.chdir(directory)
            model.save_model("model.json")
        finally:
            os.chdir(old_cwd)

        staged = Path(directory) / "model.json"
        check = factory()
        check.load_model(str(staged))

        if (
            check.get_booster().feature_names
            != model.get_booster().feature_names
        ):
            raise RuntimeError(
                "Serialized feature contract verification failed."
            )

        current_hash = hashlib.sha256(
            destination.read_bytes()
        ).hexdigest()
        if current_hash != expected_old_hash:
            raise RuntimeError(
                "Deployed model changed during retraining; refusing overwrite."
            )

        shutil.copy2(
            destination,
            destination.with_name("model.previous.json"),
        )
        os.replace(staged, destination)


def chronological_validation(factory, X, y, label_end):
    """Evaluate when possible; small holdouts do not block final training."""
    from sklearn.base import is_classifier

    if len(X) < 10:
        LOG.warning(
            "Only %d usable rows; skipping chronological validation.",
            len(X),
        )
        return None

    split_position = min(len(X) - 1, max(1, int(len(X) * 0.8)))
    split_date = X.index[split_position]

    train = (X.index < split_date) & (label_end < split_date)
    test = X.index >= split_date

    if train.sum() < 2 or test.sum() < 2:
        LOG.warning(
            "Insufficient holdout data after label purging: "
            "train=%d, test=%d. Skipping validation.",
            int(train.sum()),
            int(test.sum()),
        )
        return None

    candidate = factory()
    candidate.set_params(n_jobs=2)

    if is_classifier(candidate):
        training_classes = np.unique(y.loc[train])
        all_classes = np.unique(y)
        if (
            len(training_classes) < 2
            or not np.array_equal(training_classes, all_classes)
        ):
            LOG.warning(
                "Chronological training split does not contain all target "
                "classes. Skipping validation; final training uses all rows."
            )
            return None

    # Unexpected estimator errors remain visible rather than being hidden.
    candidate.fit(X.loc[train], y.loc[train])
    score = float(candidate.score(X.loc[test], y.loc[test]))

    if not np.isfinite(score):
        LOG.warning(
            "Chronological holdout score is nonfinite. "
            "Continuing without a validation score."
        )
        return None

    LOG.info("Chronological holdout score: %.6f", score)
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manual",
        action="store_true",
        required=True,
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--model-factory", required=True)
    parser.add_argument(
        "--nlp-model",
        required=True,
        help="Hugging Face model ID or local checkpoint directory.",
    )
    parser.add_argument(
        "--label-map",
        required=True,
        help=(
            'JSON mapping. For savasy, either '
            '{"negative":-1.0,"positive":1.0} or '
            '{"LABEL_0":-1.0,"LABEL_1":1.0}.'
        ),
    )
    args = parser.parse_args()

    if (
        os.getenv("GITHUB_ACTIONS", "").lower() == "true"
        and os.getenv("GITHUB_EVENT_NAME") != "workflow_dispatch"
    ):
        raise RuntimeError(
            "GitHub Actions retraining requires workflow_dispatch."
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    LOG.info("Running script: %s", Path(__file__).resolve())
    LOG.info("Working directory: %s", Path.cwd())

    try:
        labels = json.loads(args.label_map)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--label-map must contain valid JSON. "
            "Escape embedded double quotes when using Windows CMD."
        ) from exc

    if not isinstance(labels, dict) or not labels:
        raise ValueError("--label-map must be a nonempty JSON object.")

    # Semantic aliases are normalized against model configuration in
    # score_corpus. There is no early LABEL_0/LABEL_1-only assertion.
    offline_values = {"1", "ON", "YES", "TRUE"}
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.getenv(name, "").upper() in offline_values:
            raise RuntimeError(
                f"{name} enables offline mode. In Windows CMD run "
                f"'set {name}=' before an initial model download."
            )

    symbol = args.ticker.upper().removesuffix(".IS")
    build_features = resolve(args.features)
    build_targets = resolve(args.targets)
    factory = resolve(args.model_factory)

    deployed_path = Path("model.json")
    deployed_hash = hashlib.sha256(
        deployed_path.read_bytes()
    ).hexdigest()

    legacy = factory()
    legacy.load_model(str(deployed_path))
    old_names = legacy.get_booster().feature_names

    if not old_names or len(set(old_names)) != len(old_names):
        raise RuntimeError(
            "Existing model must contain a unique named feature schema."
        )

    feature_names = [
        name for name in old_names
        if name != "news_sentiment"
    ] + ["news_sentiment"]

    end = pd.Timestamp.now(tz=TRT).tz_localize(None).normalize()
    start = end - pd.DateOffset(years=3)

    work = Path("work") / "backfill" / symbol
    work.mkdir(parents=True, exist_ok=True)

    http = CachedHTTP(work / "http")
    try:
        records = list(kap_news(http, symbol, start, end))
        records.extend(rss_news(http, args.query, start, end))
    finally:
        http.close()

    if not records:
        raise RuntimeError(
            "No news retrieved. The deployed model remains unchanged."
        )

    news = pd.DataFrame(records).drop_duplicates("id")
    news["text_hash"] = news.text.map(
        lambda text: hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()
    )
    news["_timestamp"] = pd.to_datetime(
        news.published_at,
        utc=True,
    )
    news = (
        news.sort_values("_timestamp")
        .drop_duplicates("text_hash")
        .drop(columns="_timestamp")
        .reset_index(drop=True)
    )

    if set(news.source) != {"kap", "rss"}:
        LOG.warning(
            "Available corpus sources: %s. Continuing with retrieved data.",
            sorted(set(news.source)),
        )

    news.to_json(
        work / "corpus.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    coverage = news.groupby(
        ["source", news.published_at.str[:7]]
    ).size()
    coverage.to_csv(work / "coverage.csv")

    LOG.warning(
        "Historical news coverage is best-effort. Coverage report: %s",
        (work / "coverage.csv").resolve(),
    )

    scored = score_corpus(news, args.nlp_model, labels)
    effective_labels = scored.attrs["effective_label_map"]

    scored.to_json(
        work / "sentiment.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    import yfinance as yf

    ticker = symbol + ".IS"
    bars = normalize_daily(
        flatten_yahoo(
            yf.download(
                ticker,
                start=(start - pd.Timedelta(days=120)).date(),
                end=end.date(),
                interval="1d",
                auto_adjust=False,
                threads=False,
                progress=False,
                timeout=30,
            ),
            ticker,
        )
    )

    required_ohlcv = {"Open", "High", "Low", "Close", "Volume"}
    if not required_ohlcv.issubset(bars.columns):
        raise RuntimeError("Historical data is missing OHLCV columns.")

    if bars.empty:
        raise RuntimeError("No historical OHLCV rows are available.")

    if bars.index.min() > start:
        LOG.warning(
            "Historical bars start at %s, later than requested %s. "
            "Continuing with available history.",
            bars.index.min().date(),
            start.date(),
        )

    if bars.index.max() < end - pd.Timedelta(days=7):
        LOG.warning(
            "Most recent historical bar is %s. Continuing with available data.",
            bars.index.max().date(),
        )

    X = build_features(bars.copy())
    targets = build_targets(bars.copy())

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Feature builder must return a DataFrame.")
    if not isinstance(targets, pd.DataFrame):
        raise TypeError("Target builder must return a DataFrame.")
    if not {"target", "label_end"}.issubset(targets.columns):
        raise RuntimeError(
            "Target builder must return target and label_end columns."
        )

    if (
        not X.index.is_unique
        or not targets.index.is_unique
        or X.columns.has_duplicates
    ):
        raise RuntimeError(
            "Feature/target indices and feature names must be unique."
        )

    if (
        not isinstance(X.index, pd.DatetimeIndex)
        or X.index.tz is not None
        or not X.index.is_monotonic_increasing
        or not X.index.isin(bars.index).all()
    ):
        raise RuntimeError(
            "Feature builder must preserve sorted, timezone-naive trading dates."
        )

    if X.empty:
        raise RuntimeError("Feature builder returned no training rows.")

    X = X.copy()
    X["news_sentiment"] = sentiment_for_bars(scored, X.index)

    # Warning only: preserve the old schema using NaN for absent features.
    missing = sorted(set(feature_names) - set(X.columns))
    if missing:
        LOG.warning(
            "Original model features are missing: %s. "
            "Adding NaN columns and proceeding with XGBoost missing-value "
            "handling. These features cannot learn effects in this fit.",
            missing,
        )

    X = X.reindex(columns=feature_names)
    X = X.apply(pd.to_numeric, errors="raise").astype(float)

    targets = targets.reindex(X.index)
    y = pd.to_numeric(
        targets["target"],
        errors="raise",
    ).astype(float)
    label_end = pd.to_datetime(
        targets["label_end"],
        errors="raise",
    )

    if label_end.dt.tz is not None:
        label_end = (
            label_end.dt.tz_convert(TRT)
            .dt.tz_localize(None)
        )

    # NaN features are supported by XGBoost. Do not drop every row merely
    # because an original feature was unavailable and added as NaN.
    # Infinity, unknown targets, and incomplete future labels are excluded.
    finite_or_missing_features = ~np.isinf(
        X.to_numpy(dtype=float)
    ).any(axis=1)

    valid = (
        (X.index >= start)
        & (X.index < end)
        & finite_or_missing_features
        & np.isfinite(y)
        & label_end.notna()
        & (label_end < end)
    )

    discarded = int((~valid).sum())
    if discarded:
        LOG.warning(
            "Excluded %d rows with out-of-range dates, infinite features, "
            "missing targets, or incomplete label horizons.",
            discarded,
        )

    X = X.loc[valid].sort_index()
    y = y.reindex(X.index)
    label_end = label_end.reindex(X.index)

    if X.empty:
        raise RuntimeError(
            "No usable training rows remain. Check target generation, "
            "label_end, and date alignment."
        )

    # Requested migration behavior: warning, not a blocking assertion.
    if len(X) < 250:
        LOG.warning(
            "Only %d usable training rows are available, below 250. "
            "Proceeding with schema migration.",
            len(X),
        )

    sentiment_unique = int(
        X["news_sentiment"].nunique(dropna=True)
    )
    if sentiment_unique < 2:
        LOG.warning(
            "news_sentiment has %d distinct nonmissing value(s). "
            "Proceeding with training and export. The feature will be "
            "stored in the schema but cannot learn sentiment effects "
            "from this dataset.",
            sentiment_unique,
        )

    all_missing_columns = X.columns[X.isna().all()].tolist()
    if all_missing_columns:
        LOG.warning(
            "Entirely missing feature columns retained in schema: %s",
            all_missing_columns,
        )

    candidate = factory()
    old_objective = json.loads(
        legacy.get_booster().save_config()
    )["learner"]["objective"]["name"]

    if candidate.get_xgb_params().get("objective") != old_objective:
        raise RuntimeError(
            "Model factory does not preserve the deployed objective."
        )

    # Small-data checks above are nonblocking. Estimator label requirements
    # remain necessary for a valid classifier fit.
    from sklearn.base import is_classifier

    if is_classifier(candidate):
        classes = np.unique(y)
        if not np.array_equal(classes, np.arange(len(classes))):
            raise RuntimeError(
                "XGBoost classifier targets must use contiguous class IDs "
                "starting at 0. Preserve the original target encoding."
            )
        if len(classes) < 2:
            raise RuntimeError(
                "Classifier training requires at least two target classes. "
                "This is a target-data issue, not constant news_sentiment."
            )

    holdout_score = chronological_validation(
        factory,
        X,
        y,
        label_end,
    )

    # Fresh fit: passing named columns binds news_sentiment into the schema.
    # Do not continue old trees against a changed feature matrix.
    model = factory()
    model.set_params(n_jobs=2)
    model.fit(X, y)

    model.get_booster().set_attr(
        sentiment_model=args.nlp_model,
        sentiment_labels=json.dumps(
            effective_labels,
            sort_keys=True,
        ),
        sentiment_policy=(
            "probability_weighted;max_length=256;"
            "calendar_mean;18:30_TRT;ffill;initial=0"
        ),
        sentiment_distinct_values=str(sentiment_unique),
        missing_original_features=json.dumps(missing),
        all_missing_features=json.dumps(all_missing_columns),
        training_rows=str(len(X)),
        holdout_score=(
            str(holdout_score)
            if holdout_score is not None
            else "not_evaluated"
        ),
        trained_through=str(X.index.max().date()),
    )

    if model.get_booster().feature_names != feature_names:
        raise RuntimeError(
            "Retraining did not preserve the expected feature schema."
        )

    report = {
        "ticker": symbol,
        "training_rows": len(X),
        "trained_from": str(X.index.min().date()),
        "trained_through": str(X.index.max().date()),
        "feature_names": feature_names,
        "missing_original_features": missing,
        "all_missing_features": all_missing_columns,
        "sentiment_distinct_values": sentiment_unique,
        "sentiment_model": args.nlp_model,
        "sentiment_labels": effective_labels,
        "holdout_score": holdout_score,
    }
    (work / "training_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    save_open_model(model, factory, deployed_hash)

    LOG.info(
        "Published %s with news_sentiment. Backup: model.previous.json",
        Path("model.json").resolve(),
    )
    LOG.info(
        "Training report: %s",
        (work / "training_report.json").resolve(),
    )


if __name__ == "__main__":
    main()