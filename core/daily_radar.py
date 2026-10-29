"""Causal, explainable next-session watchlists; scores are not probabilities.

This module deliberately does not reuse the old five-session classifier as a
next-day forecast. Historical diagnostics use next-open to next-close returns,
fixed technical rules, and an explicit round-trip cost assumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNIVERSE = (
    "THYAO", "BIMAS", "KCHOL", "FROTO", "TUPRS", "SAHOL", "GARAN",
    "AKBNK", "ISCTR", "YKBNK", "EREGL", "SISE", "ASELS", "TCELL",
    "ENKAI", "PGSUS", "TOASO", "MGROS", "TTKOM", "TAVHL",
)
RULE_VERSION = "next-session-watchlist-v1"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def validate_prices(frame):
    """Keep actual, sorted daily observations; never invent missing bars."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Fiyat geçmişi boş.")
    if frame.columns.has_duplicates or not set(OHLCV).issubset(frame.columns):
        raise ValueError("OHLCV sütunları eksik veya tekrarlı.")
    result = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    result.index = index.normalize()
    if result.index.isna().any() or result.index.has_duplicates:
        raise ValueError("Fiyat tarihleri geçersiz veya tekrarlı.")
    result = result.sort_index()
    result[OHLCV] = result[OHLCV].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(result[OHLCV].to_numpy(dtype=float)).all():
        raise ValueError("OHLCV içinde eksik veya sonsuz değer var.")
    prices = result[OHLCV[:4]]
    if (
        prices.le(0).any().any()
        or result.Volume.lt(0).any()
        or result.High.lt(prices.max(axis=1) - 1e-8).any()
        or result.Low.gt(prices.min(axis=1) + 1e-8).any()
    ):
        raise ValueError("OHLCV değerleri tutarsız.")
    return result


def technical_frame(equity, benchmark):
    """Calculate fixed indicators using only data at or before each row."""
    frame = validate_prices(equity)
    benchmark = validate_prices(benchmark)
    c, h, low, v = (frame[name] for name in ["Close", "High", "Low", "Volume"])
    out = frame.copy()
    out["ema20"] = c.ewm(span=20, adjust=False, min_periods=20).mean()
    out["ema50"] = c.ewm(span=50, adjust=False, min_periods=50).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    out["rsi"] = rsi.mask((loss == 0) & (gain > 0), 100).mask(
        (loss == 0) & (gain == 0), 50
    )
    tr = pd.concat([h-low, (h-c.shift()).abs(), (low-c.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    out["atr_pct"] = 100 * out.atr / c
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    spread = h - low
    multiplier = ((2*c-h-low) / spread.replace(0, np.nan)).mask(spread == 0, 0)
    out["cmf"] = (multiplier*v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    obv = (np.sign(delta).fillna(0) * v).cumsum()
    out["obv5"] = obv.diff(5)
    out["volume_ratio"] = v / v.shift().rolling(20).mean().replace(0, np.nan)
    traded_value = frame.get("TradedValue", c*v)
    out["value20"] = traded_value.shift().rolling(20).mean()
    out["return1"] = c.pct_change(fill_method=None)
    benchmark_return = benchmark.Close.pct_change(5, fill_method=None)
    out["relative5"] = c.pct_change(5, fill_method=None) - benchmark_return.reindex(frame.index)
    out["resistance"] = h.shift().rolling(20).max()
    out["support"] = low.rolling(10).min()
    out["close_location"] = ((c-low)/spread.replace(0, np.nan)).mask(spread == 0, .5)
    out["extension_atr"] = (c-out.ema20) / out.atr.replace(0, np.nan)

    # The weights are public, fixed rules, not learned success probabilities.
    score = (
        15 * c.gt(out.ema20)
        + 10 * c.gt(out.ema50)
        + 5 * out.ema20.gt(out.ema20.shift(3))
        + 15 * out.relative5.gt(0)
        + np.select([out.volume_ratio.ge(1.2), out.volume_ratio.ge(1.0)], [15, 8], 0)
        + 10 * out.cmf.gt(0)
        + 10 * out.obv5.gt(0)
        + 5 * out.rsi.between(50, 70)
        + 5 * out.macd_hist.gt(0)
        + 5 * c.ge(.98*out.resistance)
        + 5 * out.close_location.ge(.65)
        - 20 * out.rsi.gt(75)
        - 15 * out.extension_atr.gt(2.5)
        - 10 * out.atr_pct.gt(8)
    )
    out["score"] = score.clip(0, 100).astype(float)
    required = [
        "ema20", "ema50", "rsi", "atr", "cmf", "obv5", "volume_ratio",
        "value20", "relative5", "resistance", "support", "extension_atr",
    ]
    out["ready"] = np.isfinite(out[required].to_numpy(dtype=float)).all(axis=1) & v.gt(0)
    # Do not span missing exchange sessions in relative momentum calculations.
    consecutive = pd.Series(frame.index.isin(benchmark.index), index=frame.index)
    positions = pd.Series(benchmark.index.get_indexer(frame.index), index=frame.index)
    out["ready"] &= consecutive & positions.diff().eq(1).rolling(21).sum().eq(21)
    return out


def qualifies(row, threshold=70):
    return bool(
        row["ready"] and row["score"] >= threshold
        and row["Close"] > row["ema20"] and row["relative5"] > 0
        and row["cmf"] > 0 and row["volume_ratio"] >= 1.0
        and 45 <= row["rsi"] <= 75 and row["atr_pct"] <= 8
        and row["extension_atr"] <= 2.5 and row["value20"] >= 30_000_000
    )


def market_context(benchmark, frames, day):
    """Breadth is explicitly the configured universe, not the whole BIST."""
    prices = benchmark.loc[:day].Close
    if len(prices) < 50 or prices.index[-1] != day:
        raise ValueError("Endeksin seçilen seansı veya 50 günlük geçmişi eksik.")
    ema20 = prices.ewm(span=20, adjust=False).mean().iloc[-1]
    above = sum(bool(f.loc[day, "Close"] > f.loc[day, "ema20"]) for f in frames.values() if day in f.index)
    available = sum(day in f.index for f in frames.values())
    breadth = above / available if available else 0.0
    weak = prices.iloc[-1] < ema20 or breadth < .4
    return {
        "index_close": float(prices.iloc[-1]),
        "index_return1": float(prices.iloc[-1]/prices.iloc[-2]-1),
        "index_above_ema20": bool(prices.iloc[-1] >= ema20),
        "breadth_above_ema20": above,
        "breadth_available": available,
        "regime": "temkinli" if weak else "destekleyici",
        "threshold": 80 if weak else 70,
    }


def select_candidates(frames, benchmark, day, news=None, limit=3):
    context = market_context(benchmark, frames, day)
    selected = []
    for ticker, frame in frames.items():
        if day not in frame.index:
            continue
        row = frame.loc[day]
        item_news = (news or {}).get(ticker, {})
        sentiment = item_news.get("sentiment")
        if sentiment is not None and sentiment <= -.35:
            continue
        if not qualifies(row, context["threshold"]):
            continue
        evidence = []
        if row.Close > row.ema20 and row.Close > row.ema50:
            evidence.append("20 ve 50 günlük ortalamaların üzerinde")
        if row.relative5 > 0:
            evidence.append(f"5 seansta endekse göre +%{100*row.relative5:.1f} güçlü")
        if row.cmf > 0 and row.obv5 > 0:
            evidence.append("CMF ve OBV para girişini destekliyor")
        evidence.append(f"hacim {row.volume_ratio:.2f}x")
        # All displayed reference prices use the latest raw-price scale.
        factor = float(row.get("AdjustmentFactor", 1.0))
        selected.append({
            "ticker": ticker,
            "technical_score": float(row.score),
            "ranking_score": float(row.score + (5*sentiment if sentiment is not None else 0)),
            "close": float(row.Close/factor),
            "support": float(row.support/factor),
            "resistance": float(row.resistance/factor),
            "atr_pct": float(row.atr_pct),
            "rsi": float(row.rsi),
            "volume_ratio": float(row.volume_ratio),
            "relative5_pct": float(100*row.relative5),
            "price_source": str(row.get("PriceSource", "daily")),
            "reasons": evidence,
            "risks": ["Destek altı kapanış teknik görünümü zayıflatır."]
                     + (["Geniş günlük fiyat aralığı."] if row.atr_pct > 5 else [])
                     + (["Haber analizi doğrulanamadı."] if item_news.get("status") == "unavailable" else []),
            "news": item_news,
        })
    selected.sort(key=lambda item: (-item["ranking_score"], item["ticker"]))
    return selected[:limit], context


def historical_diagnostics(frames, benchmark, lookback=252, cost_bps=20):
    """Measure fixed technical rules, excluding news, without future features.

    Enter at next observed session's open and exit at that session's close.
    Missing next-session bars do not turn into multi-day returns. These
    current-universe retrospective results are not independent validation.
    """
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("Geçerli bir pozitif maliyet varsayımı gerekli.")
    days = benchmark.index[-(lookback+1):]
    trades, portfolios = [], []
    skipped_outcomes = 0
    for signal_day, outcome_day in zip(days[:-1], days[1:]):
        if len(benchmark.loc[:signal_day]) < 50:
            continue
        selected, _ = select_candidates(frames, benchmark, signal_day)
        if not selected:
            continue
        daily_returns = []
        for item in selected:
            frame = frames[item["ticker"]]
            if outcome_day not in frame.index:
                skipped_outcomes += 1
                continue
            future = frame.loc[outcome_day]
            value = float(future.Close/future.Open - 1 - cost_bps/10_000)
            daily_returns.append(value)
            trades.append({"ticker": item["ticker"], "signal_date": str(signal_day.date()),
                           "outcome_date": str(outcome_day.date()), "net_return": value})
        # Do not drop a missing losing leg and report a survivor-only portfolio.
        if len(daily_returns) == len(selected):
            index_bar = benchmark.loc[outcome_day]
            baseline = float(index_bar.Close/index_bar.Open-1-cost_bps/10_000)
            portfolios.append({"date": str(outcome_day.date()),
                               "net_return": float(np.mean(daily_returns)),
                               "benchmark_net_return": baseline})
    returns = np.array([item["net_return"] for item in portfolios])
    mean_net = float(returns.mean()) if len(returns) else None
    mean_excess = float(np.mean([
        p["net_return"]-p["benchmark_net_return"] for p in portfolios
    ])) if portfolios else None
    evidence_status = (
        "insufficient_samples" if len(returns) < 30
        else "no_positive_net_return" if mean_net <= 0
        else "no_index_advantage" if mean_excess <= 0
        else "retrospective_only"
    )
    return {
        "method": "fixed technical rules; next-open to next-close; news excluded",
        "scope": "current 20-stock universe; retrospective, not independent validation",
        "cost_bps": float(cost_bps),
        "observed_sessions": max(0, len(days)-1),
        "signal_sessions": len(portfolios),
        "observed_trades": len(trades),
        "missing_outcomes": skipped_outcomes,
        "positive_session_rate": float(np.mean(returns > 0)) if len(returns) else None,
        "mean_net_return": mean_net,
        "mean_benchmark_excess": mean_excess,
        "evidence_status": evidence_status,
        "trades": trades,
        "portfolios": portfolios,
    }


def format_report(report):
    lines = [
        "BIST JARVIS — Ertesi Seans Radarı",
        f"Fiyat seansı: {report['session']} | Hedef seans: {report['target_session']}",
        f"Tarama: {report['scanned_at']}",
        f"Kapsama: {report['coverage']}/{report['universe_size']} hisse",
    ]
    if report.get("status") == "unavailable":
        return "\n".join(lines + ["Analiz tamamlanamadı; aday üretilmedi.", report["error"]])[:3500]
    context = report["market"]
    lines += [
        "",
        f"Piyasa: {context['regime']} | BIST100 %{100*context['index_return1']:+.2f}",
        f"20 günlük ortalama üstünde: {context['breadth_above_ema20']}/{context['breadth_available']}",
        f"Aday teknik puan eşiği: {context['threshold']}/100",
    ]
    if report.get("macros"):
        lines.append("Makro (son tamamlanan kaynak seansı):")
        for name, item in report["macros"].items():
            lines.append(f"{name}: {item['close']:.2f}, %{item['return1_pct']:+.2f} ({item['date']})")
            if item.get("level_percentile") is not None:
                lines.append(f"VIX yüzdelik dilimi: %{item['level_percentile']:.1f}")
    lines += ["", "İZLEME LİSTESİ — sonraki işlem seansı"]
    if report["coverage"] != report["universe_size"]:
        lines.append("Eksik kapsama: sıralama yalnızca doğrulanan hisseler arasında.")
    if not report["candidates"]:
        lines.append("Bugün ölçütleri sağlayan aday yok; liste zorlanmadı.")
    for number, item in enumerate(report["candidates"], 1):
        lines += [
            f"{number}) {item['ticker']} | Teknik puan {item['technical_score']:.0f}/100",
            f"Kapanış ₺{item['close']:.2f} | RSI {item['rsi']:.0f} | ATR %{item['atr_pct']:.1f}",
            f"Destek ₺{item['support']:.2f} | Önceki 20 seans direnç ₺{item['resistance']:.2f}",
            f"Fiyat kaynağı: {item.get('price_source', 'daily')}",
            "; ".join(item["reasons"]),
        ]
        news = item["news"]
        if news.get("sentiment") is not None:
            lines.append(f"Haber duygu: {news['sentiment']:+.2f} ({news['count']} başlık)")
        else:
            lines.append("Haber: uygun başlık yok." if news.get("status") == "no_news"
                         else "Haber/NLP: değerlendirilemedi.")
        if news.get("headlines"):
            lines.append("Başlık: " + " ".join(news["headlines"][0].split())[:100])
        lines.append("Risk: " + " ".join(item["risks"]))
        lines.append("")
    diagnostics = report["history"]
    if diagnostics["signal_sessions"]:
        lines += [
            f"Geçmiş teknik tarama: {diagnostics['signal_sessions']} sinyal seansı, "
            f"net pozitif oran %{100*diagnostics['positive_session_rate']:.1f}",
            f"Sonraki açılış→kapanış ort. net %{100*diagnostics['mean_net_return']:+.2f}; "
            f"maliyet {diagnostics['cost_bps']:.0f} baz puan varsayımı.",
        ]
    else:
        lines.append("Geçmiş ölçüm: yeterli gerçekleşmiş sinyal yok.")
    evidence_status = diagnostics.get("evidence_status", "insufficient_samples")
    if evidence_status == "no_positive_net_return":
        lines.append("Geçmiş ortalama net getiri pozitif değil; kanıtlanmış kazanç avantajı yok.")
    elif evidence_status == "no_index_advantage":
        lines.append("Geçmiş ölçüm endekse karşı üstünlük göstermiyor.")
    elif evidence_status == "insufficient_samples":
        lines.append("Başarı değerlendirmesi için en az 30 gerçekleşmiş sinyal seansı gerekli.")
    lines += [
        "Geçmiş ölçüm teknik kurallara aittir; haber filtresi dahil değildir.",
        "Geriye dönük incelemedir; bağımsız doğrulama değildir.",
        "Puan başarı olasılığı değildir. Eski 5 seans modeli bu sıralamada kullanılmadı.",
    ]
    if report.get("failures"):
        lines.append("Eksik: " + ", ".join(report["failures"]))
    if report.get("warnings"):
        lines.append("Veri notu: " + "; ".join(report["warnings"])[:300])
    text = "\n".join(lines)
    # Telegram counts UTF-16 code units for text-related limits.
    if len(text.encode("utf-16-le")) // 2 > 4000:
        raise ValueError("Rapor Telegram mesaj sınırını aşıyor.")
    return text
