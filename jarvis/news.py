"""Headlines from Google News and their sentiment from a Turkish BERT model.

Used by the PC application (any stock) and, when sentiment is enabled, by the
daily job (the five recommendations). Sentiment is switched off for now (see
SENTIMENT_ENABLED); without `transformers` and `torch` it reports "unavailable".

Usage (after jarvis.train):
    python -m jarvis.news --data-dir data      # adds "news" to daily/<day>.json
"""

from __future__ import annotations

import argparse
import calendar
import html
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

LOG = logging.getLogger("jarvis.news")

RSS_URL = "https://news.google.com/rss/search"
MAX_FEED_BYTES = 2_000_000
MAX_HEADLINES = 5
TICKER_PATTERN = re.compile(r"[A-Z][A-Z0-9]{2,9}")
SENTIMENT_MODEL = "savasy/bert-base-turkish-sentiment-cased"
# Off: checked on 2026-11-17, this checkpoint (trained on product reviews) is
# wrong on finance headlines: "net kârı %80 arttı, rekor temettü" -> negative,
# "THY yolcu sayısında rekor kırdı" -> negative, routine KAP titles -> strongly
# positive. Turn on only with a model that passes such checks.
SENTIMENT_ENABLED = False
SENTIMENT_SIGN = {"LABEL_0": -1.0, "LABEL_1": 1.0, "negative": -1.0, "positive": 1.0}
NEUTRAL_BAND = 0.2                                    # |score| at or below this reads "nötr"
REQUEST_PAUSE_SECONDS = 1.0                           # between tickers in the daily job

_classifier = None


class NewsError(RuntimeError):
    """RSS retrieval or parsing failed."""


def _plain_text(value: object, limit: int = 600) -> str:
    """Remove feed markup and normalize whitespace."""
    stripped = re.sub(r"<[^>]*>", " ", str(value or ""))
    return " ".join(html.unescape(stripped).split())[:limit]


def _download_feed(ticker: str) -> bytes:
    params = {"q": f"{ticker} hisse kap", "hl": "tr", "gl": "TR", "ceid": "TR:tr"}
    try:
        with requests.get(RSS_URL, params=params, headers={"User-Agent": "BIST-Jarvis/1.0"},
                          timeout=(5, 20), stream=True) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_content(chunk_size=16_384):
                content.extend(chunk)
                if len(content) > MAX_FEED_BYTES:
                    raise NewsError("Google News RSS exceeded the size limit.")
            return bytes(content)
    except requests.Timeout as error:
        raise NewsError("Google News RSS timed out.") from error
    except requests.RequestException as error:
        raise NewsError("Google News RSS could not be retrieved.") from error


def fetch_news(ticker: str, limit: int = MAX_HEADLINES,
               download: Callable[[str], bytes] = _download_feed) -> list[dict]:
    """The latest unique headlines for `ticker`: [{title, url, published_at}], newest first."""
    import feedparser

    ticker = ticker.strip().upper().removesuffix(".IS")
    if not TICKER_PATTERN.fullmatch(ticker):
        raise NewsError(f"Invalid ticker: {ticker!r}")
    if not 1 <= limit <= MAX_HEADLINES:
        raise ValueError(f"limit must be between 1 and {MAX_HEADLINES}")
    try:
        feed = feedparser.parse(download(ticker))
    except NewsError:
        raise
    except Exception as error:
        raise NewsError("Google News RSS could not be parsed.") from error
    if feed.get("bozo") or not feed.get("version"):
        raise NewsError("Google News returned malformed RSS.")

    items = []
    for position, entry in enumerate(feed.get("entries", [])):
        title = _plain_text(entry.get("title"))
        if not title:
            continue
        timestamp = published_at = None
        parsed_time = entry.get("published_parsed") or entry.get("updated_parsed")
        if parsed_time:
            try:
                timestamp = float(calendar.timegm(parsed_time))
                published_at = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
            except (TypeError, ValueError, OverflowError, OSError):
                pass
        url = str(entry.get("link", "")).strip()
        try:
            parsed = urlparse(url)
            if parsed.scheme not in {"https", "http"} or not parsed.netloc:
                url = ""
        except ValueError:
            url = ""
        items.append((timestamp, position, {"title": title, "url": url, "published_at": published_at}))

    # Newest first; undated last; equal dates keep the feed's order.
    items.sort(key=lambda item: (item[0] is None, -(item[0] or 0), item[1]))
    results, seen = [], set()
    for _, _, item in items:
        if item["title"].casefold() in seen:
            continue
        seen.add(item["title"].casefold())
        results.append(item)
        if len(results) == limit:
            break
    return results


def _default_classifier():
    """The local Turkish BERT classifier, or None when transformers/torch are missing."""
    global _classifier
    if _classifier is None:
        try:
            from transformers import pipeline
        except ImportError:
            return None
        try:
            _classifier = pipeline("text-classification", model=SENTIMENT_MODEL,
                                   device=-1, truncation=True)
        except Exception as error:
            LOG.warning("Duygu modeli yüklenemedi: %s", type(error).__name__)
            return None
    return _classifier


def news_sentiment(headlines: list[str], classifier: Callable | None = None) -> dict:
    """Mean headline sentiment in [-1, 1], or why there is none.

    Returns {"status": "scored", "score", "count"} or {"status": "unavailable" |
    "no_news" | "failed" | "disabled"}. Headlines are matched to a stock by search only, so
    the score is a hint about the news flow, not about the stock's prospects.
    """
    texts = [" ".join(str(text).split())[:600] for text in headlines if str(text).strip()]
    if not texts:
        return {"status": "no_news"}
    if classifier is None and not SENTIMENT_ENABLED:
        return {"status": "disabled"}
    classifier = classifier or _default_classifier()
    if classifier is None:
        return {"status": "unavailable"}
    try:
        outputs = classifier(texts)
        values = [SENTIMENT_SIGN[item["label"]] * float(item["score"]) for item in outputs]
    except Exception:
        return {"status": "failed"}
    if len(values) != len(texts) or not all(np.isfinite(values)):
        return {"status": "failed"}
    return {"status": "scored", "score": float(np.clip(np.mean(values), -1.0, 1.0)), "count": len(values)}


def sentiment_word(score: float) -> str:
    if score > NEUTRAL_BAND:
        return "olumlu"
    return "olumsuz" if score < -NEUTRAL_BAND else "nötr"


def ticker_news(ticker: str, fetch: Callable[[str], list[dict]] = fetch_news,
                classifier: Callable | None = None) -> dict:
    """Headlines plus sentiment for one ticker, never raising."""
    try:
        headlines = fetch(ticker)
    except Exception as error:
        return {"status": "fetch_failed", "error": type(error).__name__, "headlines": []}
    result = news_sentiment([item["title"] for item in headlines], classifier=classifier)
    result["headlines"] = [{"title": item["title"], "published_at": item.get("published_at")}
                           for item in headlines]
    return result


# -- daily job ----------------------------------------------------------------

def add_news_to_daily(data_dir: Path | str, day: str | None = None,
                      fetch: Callable[[str], list[dict]] = fetch_news,
                      classifier: Callable | None = None,
                      pause: float = REQUEST_PAUSE_SECONDS) -> dict:
    """Write headline sentiment for the day's recommendations into daily/<day>.json."""
    folder = Path(data_dir) / "daily"
    days = sorted(path.stem for path in folder.glob("*.json"))
    day = day or (days[-1] if days else None)
    if day is None or not (folder / f"{day}.json").exists():
        return {"status": "skipped", "reason": "no_daily_result"}
    path = folder / f"{day}.json"
    daily = json.loads(path.read_text(encoding="utf-8"))
    tickers = [item["ticker"] for item in daily.get("recommendations") or []]
    if not tickers:
        return {"status": "skipped", "reason": "no_recommendations", "session": day}
    news = {}
    for number, ticker in enumerate(tickers):
        if number and pause:
            time.sleep(pause)
        news[ticker] = ticker_news(ticker, fetch=fetch, classifier=classifier)
    daily["news"] = news
    path.write_text(json.dumps(daily, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = pd.Series([item["status"] for item in news.values()]).value_counts().to_dict()
    return {"status": "ok", "session": day, "tickers": len(tickers), "outcomes": counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--day", default=None, help="YYYY-MM-DD; the newest daily result by default")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = add_news_to_daily(arguments.data_dir, arguments.day)
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
