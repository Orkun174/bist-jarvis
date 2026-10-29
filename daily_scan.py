"""Daily 20-stock, next-session BIST watchlist and Telegram report.

Usage:
    python daily_scan.py
    python daily_scan.py --dry-run --no-news
    python daily_scan.py --as-of "2026-10-28 18:30:00+03:00" --dry-run

Fixed technical rules produce watchlist scores, never success probabilities.
No old classifier is relabeled as a next-day prediction. HF news scoring is
batched once per scan; unavailable news is explicitly distinct from neutral.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from core.daily_radar import (
    OHLCV, RULE_VERSION, UNIVERSE, format_report, historical_diagnostics,
    select_candidates, technical_frame, validate_prices,
)
from live_inference import (
    DEFAULT_SENTIMENT_MODEL, TRT, fetch_t0, flatten_yahoo,
    last_completed_bist_session, load_local_environment,
)

LOG = logging.getLogger("bist.daily_radar")
MACROS = {"VIX": "^VIX", "USD/TRY": "USDTRY=X", "Brent": "BZ=F", "Altın": "GC=F"}


def next_bist_session(day):
    import exchange_calendars as xcals

    start = pd.Timestamp(day).normalize() + pd.Timedelta(days=1)
    calendar = xcals.get_calendar(
        "XIST", start=start.date().isoformat(),
        end=(start+pd.Timedelta(days=60)).date().isoformat(),
    )
    if calendar.sessions.empty:
        raise ValueError("Takvimde sonraki işlem seansı bulunamadı.")
    return pd.Timestamp(calendar.sessions[0]).tz_localize(None).normalize()


def download_prices(symbol, day, *, session_close=None, lookback_days=900):
    """Fetch one consistent adjusted price history and preserve raw-price scale.

    A missing selected stock bar can use the existing Is/minute/daily fallback
    only when the latest known adjustment factor is one. Otherwise the scale
    of a proxy bar cannot be verified and this ticker is excluded explicitly.
    """
    import yfinance as yf

    day = pd.Timestamp(day).normalize()
    raw = None
    for attempt in range(2):
        try:
            raw = flatten_yahoo(yf.download(
                symbol,
                start=(day-pd.Timedelta(days=lookback_days)).date().isoformat(),
                end=(day+pd.Timedelta(days=1)).date().isoformat(),
                interval="1d", auto_adjust=False, actions=True,
                progress=False, threads=False, timeout=25,
            ), symbol)
            break
        except Exception:
            if attempt == 1:
                raise
            time.sleep(2)

    raw = raw.copy()
    index = pd.DatetimeIndex(pd.to_datetime(raw.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    raw.index = index.normalize()
    raw = raw.loc[raw.index <= day].sort_index()
    if day in raw.index and raw.loc[[day], OHLCV].isna().any().any():
        raw = raw.drop(index=day)
    raw = validate_prices(raw)
    if len(raw) < 100:
        raise ValueError(f"{symbol}: en az 100 doğrulanmış günlük bar gerekli.")
    if "Adj Close" not in raw:
        raise ValueError(f"{symbol}: fiyat düzeltme bilgisi eksik.")
    factor = pd.to_numeric(raw["Adj Close"], errors="raise")/raw.Close
    if not np.isfinite(factor).all() or factor.le(0).any():
        raise ValueError(f"{symbol}: fiyat düzeltme katsayısı geçersiz.")
    result = raw.loc[:, OHLCV].copy()
    result[OHLCV[:4]] = result[OHLCV[:4]].mul(factor, axis=0)
    result["AdjustmentFactor"] = factor
    result["TradedValue"] = raw.Close * raw.Volume
    result["PriceSource"] = "yahoo_adjusted_daily"
    if session_close is not None and day not in result.index:
        if not np.isclose(float(factor.iloc[-1]), 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{symbol}: vekil barın düzeltme katsayısı doğrulanamadı.")
        bar, provenance = fetch_t0(
            symbol, session_close, session_close=session_close, require_ohlcv=True
        )
        bar = validate_prices(bar)
        if not bar.index.equals(pd.DatetimeIndex([day])):
            raise ValueError(f"{symbol}: yedek kaynak farklı seans döndürdü.")
        bar["AdjustmentFactor"] = 1.0
        bar["TradedValue"] = bar.Close*bar.Volume
        bar["PriceSource"] = provenance["ohlv_source"]
        result = pd.concat([result, bar]).sort_index()
    if session_close is not None and day not in result.index:
        raise ValueError(f"{symbol}: seçilen seans verisi yok.")
    return validate_prices(result)


def score_news(tickers, cutoff, since, *, enabled=True):
    """Retrieve bounded headlines, then call the project's BERT scorer once."""
    results = {ticker: {"status": "unavailable", "sentiment": None,
                        "count": 0, "headlines": []} for ticker in tickers}
    if not enabled:
        return results
    from core.market_data import fetch_news

    corpus = []
    for ticker in tickers:
        try:
            articles = fetch_news(ticker, limit=5)
            if not isinstance(articles, list):
                raise ValueError("Haber yanıtı liste değil.")
            seen = set()
            for article in articles:
                if not isinstance(article, dict) or not article.get("published_at"):
                    continue
                try:
                    published = pd.Timestamp(article["published_at"])
                except (TypeError, ValueError, OverflowError):
                    continue
                if pd.isna(published) or published.tzinfo is None:
                    continue
                published = published.tz_convert(TRT)
                title = " ".join(str(article.get("title") or "").split())[:600]
                if not title or not since <= published <= cutoff or title.casefold() in seen:
                    continue
                seen.add(title.casefold())
                corpus.append({"ticker": ticker, "text": title,
                               "published_at": published.isoformat(),
                               "url": str(article.get("url") or "")})
            results[ticker]["status"] = "no_news" if not seen else "unavailable"
            results[ticker]["count"] = len(seen)
            results[ticker]["headlines"] = [item["text"] for item in corpus if item["ticker"] == ticker]
        except Exception as exc:
            LOG.warning("%s news unavailable (%s)", ticker, type(exc).__name__)
        time.sleep(.3)
    if not corpus:
        return results
    try:
        from retrain_model import score_corpus

        texts = pd.DataFrame(corpus)
        scored = score_corpus(
            texts, DEFAULT_SENTIMENT_MODEL, {"negative": -1.0, "positive": 1.0}
        )
        if len(scored) != len(texts):
            raise ValueError("NLP yanıt sayısı tutarsız.")
        values = pd.to_numeric(scored["score"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (np.abs(values) > 1.000001).any():
            raise ValueError("NLP skorları geçersiz.")
        texts["score"] = np.clip(values, -1, 1)
        for ticker, group in texts.groupby("ticker"):
            results[ticker].update(status="scored", sentiment=float(group.score.mean()))
    except Exception as exc:
        LOG.warning("Batch NLP unavailable (%s); sentiment remains unknown.", type(exc).__name__)
    return results


def run_scan(as_of=None, *, with_news=True):
    now = pd.Timestamp.now(tz=TRT) if as_of is None else pd.Timestamp(as_of)
    if pd.isna(now) or now.tzinfo is None:
        raise ValueError("--as-of saat dilimi içermeli.")
    now = now.tz_convert(TRT)
    session, close = last_completed_bist_session(now)
    target = next_bist_session(session)
    report = {
        "rule_version": RULE_VERSION,
        "session": str(session.date()), "target_session": str(target.date()),
        "scanned_at": now.isoformat(), "session_close": close.isoformat(),
        "universe": list(UNIVERSE), "universe_size": len(UNIVERSE),
        "coverage": 0, "failures": {}, "warnings": [], "macros": {},
        "price_sources": {},
    }
    try:
        benchmark = download_prices("XU100.IS", session, session_close=close)
    except Exception as exc:
        LOG.error("Benchmark retrieval failed (%s)", type(exc).__name__)
        report.update(status="unavailable", error="Seçilen BIST100 seansı doğrulanamadı.")
        return report

    frames = {}
    for ticker in UNIVERSE:
        try:
            LOG.info("Preparing %s for %s", ticker, session.date())
            prices = download_prices(ticker+".IS", session, session_close=close)
            frame = technical_frame(prices, benchmark)
            if session not in frame.index or not bool(frame.loc[session, "ready"]):
                raise ValueError("Son seans teknik göstergeleri eksik.")
            frames[ticker] = frame
            source = str(prices.loc[session, "PriceSource"])
            report["price_sources"][ticker] = source
        except Exception as exc:
            report["failures"][ticker] = type(exc).__name__
            LOG.warning("%s excluded (%s)", ticker, type(exc).__name__)
        time.sleep(.3)
    report["coverage"] = len(frames)
    for source, count in Counter(report["price_sources"].values()).items():
        if source != "yahoo_adjusted_daily":
            report["warnings"].append(f"{count} hissede {source} seans verisi kullanıldı")
    if not frames:
        report.update(status="unavailable", error="20 hissede doğrulanmış teknik veri elde edilemedi.")
        return report

    # Overseas markets may still be open at the BIST close. Use only native
    # daily sessions strictly preceding the selected BIST session date.
    for name, symbol in MACROS.items():
        try:
            macro = download_prices(symbol, session-pd.Timedelta(days=1))
            if (session-macro.index[-1]).days > 7:
                raise ValueError("Makro kaynak yedi günden eski.")
            report["macros"][name] = {
                "date": str(macro.index[-1].date()),
                "close": float(macro.Close.iloc[-1]),
                "return1_pct": float(100*(macro.Close.iloc[-1]/macro.Close.iloc[-2]-1)),
            }
            if name == "VIX" and len(macro) >= 126:
                levels = macro.Close.tail(252)
                report["macros"][name]["level_percentile"] = float(
                    levels.rank(pct=True).iloc[-1]*100
                )
        except Exception as exc:
            LOG.warning("Macro %s unavailable (%s)", name, type(exc).__name__)
            report["warnings"].append(f"{name} kaynağı değerlendirilemedi")

    news = score_news(list(frames), now, session.tz_localize(TRT), enabled=with_news)
    candidates, context = select_candidates(frames, benchmark, session, news)
    history = historical_diagnostics(frames, benchmark)
    # Preserve evidence for every covered stock in the downloadable JSON,
    # while keeping the Telegram watchlist bounded to three candidates.
    report["stocks"] = {
        ticker: {
            key: float(frame.loc[session, key]) for key in [
                "score", "rsi", "atr_pct", "cmf", "obv5", "volume_ratio",
                "relative5", "extension_atr", "value20",
            ]
        } for ticker, frame in frames.items()
    }
    report.update(status="ok" if not report["failures"] else "partial",
                  market=context, candidates=candidates, news=news, history=history)
    if any(item["status"] == "unavailable" for item in news.values()):
        report["warnings"].append("Bazı haber/NLP sonuçları bilinmiyor")
    return report


def send_report(message):
    """Bounded delivery, with token-safe diagnostics and no blind retries."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        LOG.warning("Telegram secrets missing; report was saved locally.")
        return {"status": "skipped", "reason": "missing_credentials"}
    try:
        with requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "disable_web_page_preview": True},
            timeout=(5, 25), allow_redirects=False,
        ) as response:
            try:
                body = response.json()
            except ValueError:
                body = {}
            if (response.status_code == 200 and isinstance(body, dict)
                    and body.get("ok") is True and isinstance(body.get("result"), dict)
                    and isinstance(body["result"].get("message_id"), int)):
                return {"status": "sent", "message_id": body["result"]["message_id"]}
            LOG.warning("Telegram delivery unconfirmed; HTTP %s.", response.status_code)
            return {"status": "failed", "http_status": response.status_code}
    except requests.RequestException as exc:
        # Never print exception text, which can contain the bot token.
        LOG.warning("Telegram network failure (%s); delivery unknown.", type(exc).__name__)
        return {"status": "unknown", "reason": type(exc).__name__}


def main():
    # Windows CMD may otherwise default to cp1252 and crash on Turkish text.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print/save; do not send Telegram")
    parser.add_argument("--no-news", action="store_true", help="Diagnostic run without RSS/BERT")
    parser.add_argument("--output", default="work/daily/report.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_local_environment()
    try:
        report = run_scan(args.as_of, with_news=not args.no_news)
    except Exception as exc:
        LOG.error("Daily analysis failed (%s)", type(exc).__name__)
        report = {
            "status": "unavailable", "session": "doğrulanamadı",
            "target_session": "doğrulanamadı", "coverage": 0,
            "universe_size": len(UNIVERSE),
            "scanned_at": pd.Timestamp.now(tz=TRT).isoformat(),
            "error": f"Analiz hatası: {type(exc).__name__}. Actions kayıtlarını inceleyin.",
        }
    message = format_report(report)
    report["telegram"] = {"status": "dry_run"} if args.dry_run else send_report(message)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    output.with_suffix(".txt").write_text(message, encoding="utf-8")
    print(message, flush=True)
    if report["status"] == "unavailable":
        return 1
    if os.getenv("GITHUB_ACTIONS", "").lower() == "true" and not args.dry_run:
        return 0 if report["telegram"]["status"] == "sent" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
