"""The daily Telegram message: what the model ranked, and how far to trust it."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import requests

TELEGRAM_LIMIT = 4096
MONTHS = [
    "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
    "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
]
FEATURE_LABELS = {
    "EMA20_Distance": "20 günlük ortalamaya uzaklık",
    "EMA50_Distance": "50 günlük ortalamaya uzaklık",
    "Volume_Surge": "hacim artışı",
    "RSI_Momentum_5D": "RSI ivmesi",
    "Relative_Return_1D": "endekse göre günlük getiri",
    "MACD_Normalized": "MACD",
    "ATR_Normalized": "oynaklık (ATR)",
    "ADX_14": "trend gücü (ADX)",
    "CMF_20": "para akışı (CMF)",
    "StochRSI_K": "StochRSI",
    "Rel_Return_5D": "endekse göre 5 günlük getiri",
    "Rel_Return_20D": "endekse göre 1 aylık getiri",
    "Rel_Return_60D": "endekse göre 3 aylık getiri",
    "Rel_Return_120D": "endekse göre 6 aylık getiri",
    "Volatility_20D": "oynaklık (20 gün)",
    "High_252_Distance": "52 hafta zirvesine uzaklık",
    "Volume_Trend": "hacim eğilimi",
    "Beta_60D": "beta",
    "Turnover_Rank": "işlem hacmi sırası",
}
MACRO_LABELS = {
    "USDTRY": "dolar/TL", "BRENT": "Brent petrol", "GOLD": "altın", "SP500": "S&P 500",
    "VIX": "VIX", "BIST100": "BIST 100", "BISTBANK": "banka endeksi",
}
DISCLAIMER = "Yatırım tavsiyesi değildir."


class TelegramError(RuntimeError):
    """The message could not be delivered. Never contains the bot token."""


def _date(session: str) -> str:
    day = pd.Timestamp(session)
    return f"{day.day} {MONTHS[day.month - 1]} {day.year}"


def _percent(value: float, decimals: int = 2, signed: bool = True) -> str:
    text = f"{value * 100:+.{decimals}f}" if signed else f"{value * 100:.{decimals}f}"
    return "%" + text.replace(".", ",")


def _thousands(value: int) -> str:
    return f"{int(value):,}".replace(",", ".")


def _feature_label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.endswith("_Rank"):  # the day's rank of a known feature reads the same
        return _feature_label(feature[: -len("_Rank")])
    if feature.startswith("macro_"):
        return MACRO_LABELS.get(feature.split("_")[1], feature)
    return feature


def _reasons(item: dict) -> str:
    """"lehine: ..." are factors that raised this stock's rank, "aleyhine: ..." lowered it."""
    reasons = item.get("reasons") or []
    supporting = [_feature_label(r["feature"]) for r in reasons if r["direction"] == "up"]
    opposing = [_feature_label(r["feature"]) for r in reasons if r["direction"] != "up"]
    parts = []
    if supporting:
        parts.append("lehine: " + ", ".join(supporting))
    if opposing:
        parts.append("aleyhine: " + ", ".join(opposing))
    return " · ".join(parts)


def _mark(check: dict | None) -> str:
    if check is None:
        return "–"
    return "✓" if check.get("ok") else "✗"


def _unproven_line(simulated: dict | None) -> str:
    net = (simulated or {}).get("top5_net_of_cost") or {}
    if net.get("days") and net.get("mean_top_excess") is not None:
        text = (
            "⚠️ Model henüz kanıtlanmadı. Simülasyonda ilk 5 seçim "
            f"{net['days']} seans boyunca 5 günde endekse göre ortalama "
            f"{_percent(net['mean_top_excess'])} getirdi (maliyet düşülmüş); "
            f"kazanan gün oranı {_percent(net['hit_rate'], 0, signed=False)}."
        )
        if net.get("mean_quintile_spread") is not None:
            text += (
                " Sıralamanın geneli daha tutarlı: en iyi beşte birlik dilim, en kötü "
                f"beşte birliği ortalama {_percent(net['mean_quintile_spread'])} geçti."
            )
        return text
    return "⚠️ Model henüz kanıtlanmadı; canlı karne birikene kadar bu sıralama deneme amaçlıdır."


def _live_line(live: dict | None) -> str:
    window = ((live or {}).get("windows") or {}).get("60") or {}
    if not window.get("recommendations"):
        return "Canlı karne: henüz notlanan tahmin yok (tahminler 5 işlem günü sonra notlanır)."
    return (
        f"Canlı karne (son 60 seans): {window['recommendations']} öneri · "
        f"isabet {_percent(window['hit_rate'], 0, signed=False)} · "
        f"ort. fazla getiri {_percent(window['mean_excess'])}"
    )


def _model_lines(model: dict) -> list[str]:
    checks = model.get("checks") or {}
    rows = (checks.get("rows") or {}).get("train_rows")
    if model.get("weights_changed"):
        status = "bugün yeniden eğitildi" + (f" ({_thousands(rows)} satır)" if rows else "")
    elif model.get("model_id"):
        status = "dünkü ağırlıklar korundu"
    else:
        status = "henüz eğitilmiş model yok"
    marks = (
        f"veri {_mark(checks.get('data'))} · sağlık {_mark(checks.get('health'))} · "
        f"kayma {_mark(checks.get('drift'))}"
    )
    return [f"Model: {status}", f"Kontroller: {marks}"]


def _data_line(quality: dict | None) -> list[str]:
    if not quality:
        return ["Veri: kalite raporu bulunamadı."]
    reasons = quality.get("reasons") or {}
    lines = [
        f"Veri: {quality.get('tickers', 0)} hisse · {reasons.get('eligible', 0)} uygun · "
        f"{len(quality.get('quarantined') or {})} karantina · "
        f"{reasons.get('erratic', 0)} düzensiz · {reasons.get('low_liquidity', 0)} düşük hacim"
    ]
    sources = quality.get("exclusion_sources") or {}
    if sources.get("measures") == "failed":
        lines.append("Uyarı: tedbir listesi alınamadı, son başarılı liste kullanıldı.")
    if sources.get("kap_markets") == "failed":
        lines.append("Uyarı: KAP pazar bilgisi alınamadı, son başarılı liste kullanıldı.")
    if quality.get("training_allowed") is False:
        lines.append("Uyarı: veri sorunları yüzünden bugün eğitim yapılmadı.")
    return lines


def format_message(
    daily: dict | None,
    quality: dict | None,
    live: dict | None,
    simulated: dict | None,
    session: str | None = None,
) -> str:
    """Plain-text message. `daily` is None when the training job produced nothing."""
    session = (daily or {}).get("session") or (quality or {}).get("session") or session
    lines = [f"📡 BIST Jarvis — {_date(session)} kapanışı", ""]

    if daily is None:
        lines += ["Bugün için model sonucu üretilemedi (veri ya da eğitim adımı tamamlanmadı).", ""]
        lines += _data_line(quality) if quality else []
        lines += ["", DISCLAIMER]
        return "\n".join(lines)[:TELEGRAM_LIMIT]

    model = daily.get("model") or {}
    health = (model.get("checks") or {}).get("health") or {}
    recommendations = daily.get("recommendations") or []
    if recommendations:
        lines.append("Model sıralaması (ilk 5):")
        for item in recommendations:
            reasons = _reasons(item)
            lines.append(f"{item['rank']}. {item['ticker']}" + (f" — {reasons}" if reasons else ""))
        lines.append("(lehine/aleyhine: modelin sıralamasını en çok etkileyen etkenler)")
        bottom = [item["ticker"] for item in daily.get("bottom") or []]
        if bottom:
            lines += ["", "Sıralamanın sonu (modelin en zayıf gördükleri): " + ", ".join(bottom)]
    elif health.get("ok") is False:
        lines.append("Bugün öneri yok: model sağlık kontrolünü geçemedi.")
    else:
        lines.append("Bugün öneri yok.")

    lines += ["", _unproven_line(simulated), _live_line(live), ""]
    lines += _model_lines(model)
    lines += _data_line(quality)
    lines += ["", DISCLAIMER]
    return "\n".join(lines)[:TELEGRAM_LIMIT]


def send(token: str, chat_id: str, text: str, post: Callable = requests.post) -> int:
    """Send `text`; returns the Telegram message id or raises TelegramError."""
    if not token or not chat_id:
        raise TelegramError("TELEGRAM_BOT_TOKEN ya da TELEGRAM_CHAT_ID tanımlı değil.")
    try:
        response = post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=(5, 25),
        )
    except Exception as error:
        # The exception text can contain the URL and therefore the token.
        raise TelegramError(f"Telegram'a ulaşılamadı ({type(error).__name__}).") from None
    try:
        body = response.json()
    except ValueError:
        body = {}
    if response.status_code == 200 and isinstance(body, dict) and body.get("ok") is True:
        return int(body["result"]["message_id"])
    raise TelegramError(f"Telegram mesajı kabul etmedi (HTTP {response.status_code}).")
