import json

import pandas as pd
import pytest

from jarvis.notify import run as notify
from jarvis.telegram import TelegramError, format_message, send

DAILY = {
    "session": "2026-11-16",
    "eligible": 89,
    "recommendations": [
        {"ticker": "ARCLK", "prob": 0.4899, "rank": 1,
         "reasons": [{"feature": "CMF_20", "direction": "up"},
                     {"feature": "ATR_Normalized", "direction": "up"}]},
        {"ticker": "TSPOR", "prob": 0.4832, "rank": 2, "reasons": []},
    ],
    "top": [],
    "model": {
        "model_id": "2026-11-16-001bde800d",
        "retrained": True,
        "weights_changed": True,
        "checks": {
            "data": {"ok": True}, "rows": {"ok": True, "train_rows": 126481},
            "health": {"ok": True, "mean_rank_ic": 0.084, "days": 120},
            "drift": {"ok": True, "rank_corr": 0.9},
        },
    },
}
QUALITY = {
    "session": "2026-11-16", "tickers": 629, "eligible": 89,
    "quarantined": {"AAA": ["stale"], "BBB": ["stale"], "CCC": ["stale"]},
    "reasons": {"eligible": 89, "erratic": 61, "low_liquidity": 374, "quarantined": 3,
                "excluded:watchlist_market": 39, "excluded:vbts": 1, "limit_moves": 23,
                "short_history": 39},
    "exclusion_sources": {"kap_markets": "ok", "measures": "ok"},
    "training_allowed": True,
}
EMPTY_WINDOW = {"recommendations": 0, "hit_rate": None, "mean_excess": None, "mean_rank_ic": None, "days": 0}
LIVE_EMPTY = {"kind": "live", "windows": {"20": EMPTY_WINDOW, "60": EMPTY_WINDOW, "all": EMPTY_WINDOW}, "pending": 89}
LIVE_FILLED = {
    "kind": "live", "pending": 445,
    "windows": {
        "20": EMPTY_WINDOW, "all": EMPTY_WINDOW,
        "60": {"recommendations": 40, "hit_rate": 0.58, "mean_excess": 0.0081, "mean_rank_ic": 0.05, "days": 8},
    },
}
SIMULATED = {
    "kind": "simulated",
    "top5_net_of_cost": {"days": 245, "mean_top_excess": 0.0004, "mean_rank_ic": 0.078, "hit_rate": 0.469},
}


def test_message_lists_recommendations_with_reasons():
    text = format_message(DAILY, QUALITY, LIVE_EMPTY, SIMULATED)
    lines = text.splitlines()
    assert lines[0] == "📡 BIST Jarvis — 16 Kasım 2026 kapanışı"
    assert "1. ARCLK — lehine: para akışı (CMF), oynaklık (ATR)" in text
    assert "2. TSPOR" in text and "2. TSPOR —" not in text
    assert lines[-1] == "Yatırım tavsiyesi değildir."
    assert len(text) < 4096


def test_reasons_separate_supporting_and_opposing_factors():
    daily = json.loads(json.dumps(DAILY))
    daily["recommendations"][0]["reasons"] = [
        {"feature": "CMF_20", "direction": "up"}, {"feature": "macro_VIX_1d_lag", "direction": "down"},
    ]
    text = format_message(daily, QUALITY, LIVE_EMPTY, SIMULATED)
    assert "1. ARCLK — lehine: para akışı (CMF) · aleyhine: VIX" in text
    assert "↑" not in text and "↓" not in text


def test_message_states_that_the_model_is_unproven():
    text = format_message(DAILY, QUALITY, LIVE_EMPTY, SIMULATED)
    assert "Model henüz kanıtlanmadı" in text
    assert "245 seans" in text and "%+0,04" in text and "kazanan gün oranı %47" in text
    assert "'inde" not in text


def test_message_live_scorecard_empty_and_filled():
    empty = format_message(DAILY, QUALITY, LIVE_EMPTY, SIMULATED)
    assert "Canlı karne: henüz notlanan tahmin yok" in empty
    filled = format_message(DAILY, QUALITY, LIVE_FILLED, SIMULATED)
    assert "Canlı karne (son 60 seans): 40 öneri · isabet %58 · ort. fazla getiri %+0,81" in filled


def test_message_data_and_model_lines():
    text = format_message(DAILY, QUALITY, LIVE_EMPTY, SIMULATED)
    assert "Veri: 629 hisse · 89 uygun · 3 karantina · 61 düzensiz · 374 düşük hacim" in text
    assert "Model: bugün yeniden eğitildi (126.481 satır)" in text
    assert "veri ✓ · sağlık ✓ · kayma ✓" in text


def test_message_when_health_check_fails():
    daily = json.loads(json.dumps(DAILY))
    daily["recommendations"] = []
    daily["model"]["checks"]["health"] = {"ok": False, "mean_rank_ic": -0.02, "days": 120}
    text = format_message(daily, QUALITY, LIVE_EMPTY, SIMULATED)
    assert "Bugün öneri yok: model sağlık kontrolünü geçemedi" in text
    assert "sağlık ✗" in text
    assert "1. " not in text


def test_message_when_weights_were_kept():
    daily = json.loads(json.dumps(DAILY))
    daily["model"]["weights_changed"] = False
    daily["model"]["checks"]["drift"] = {"ok": False, "rank_corr": 0.1}
    text = format_message(daily, QUALITY, LIVE_EMPTY, SIMULATED)
    assert "Model: dünkü ağırlıklar korundu" in text and "kayma ✗" in text


def test_message_warns_about_failed_exclusion_source():
    quality = dict(QUALITY, exclusion_sources={"kap_markets": "ok", "measures": "failed"})
    text = format_message(DAILY, quality, LIVE_EMPTY, SIMULATED)
    assert "tedbir listesi alınamadı" in text


def test_message_without_daily_result():
    text = format_message(None, QUALITY, LIVE_EMPTY, None, session="2026-11-16")
    assert "Bugün için model sonucu üretilemedi" in text
    assert text.splitlines()[-1] == "Yatırım tavsiyesi değildir."


def test_message_without_simulation():
    text = format_message(DAILY, QUALITY, LIVE_EMPTY, None)
    assert "Model henüz kanıtlanmadı" in text and "seans boyunca" not in text


class Response:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = {"ok": True, "result": {"message_id": 7}} if body is None else body

    def json(self):
        return self._body


def test_send_posts_plain_text_to_the_bot_api():
    calls = []

    def post(url, json, timeout):
        calls.append((url, json))
        return Response()

    assert send("TOKEN", "42", "merhaba", post=post) == 7
    url, payload = calls[0]
    assert url == "https://api.telegram.org/botTOKEN/sendMessage"
    assert payload == {"chat_id": "42", "text": "merhaba", "disable_web_page_preview": True}


def test_send_raises_without_leaking_the_token():
    def rejected(url, json, timeout):
        return Response(status=401, body={"ok": False})

    with pytest.raises(TelegramError) as error:
        send("SECRET-TOKEN", "42", "x", post=rejected)
    assert "SECRET-TOKEN" not in str(error.value) and "401" in str(error.value)

    def broken(url, json, timeout):
        raise OSError("https://api.telegram.org/botSECRET-TOKEN/sendMessage unreachable")

    with pytest.raises(TelegramError) as error:
        send("SECRET-TOKEN", "42", "x", post=broken)
    assert "SECRET-TOKEN" not in str(error.value)


def test_send_requires_credentials():
    with pytest.raises(TelegramError):
        send("", "42", "x", post=lambda *a, **k: Response())


def write(root, name, payload):
    path = root / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


AS_OF = pd.Timestamp("2026-11-16 22:30", tz="Europe/Istanbul")


def test_notify_dry_run_reads_data_folder_and_sends_nothing(tmp_path):
    write(tmp_path, "daily/2026-11-16", DAILY)
    write(tmp_path, "quality/2026-11-16", QUALITY)
    write(tmp_path, "scorecard/scorecard", LIVE_EMPTY)
    write(tmp_path, "scorecard/simulated", SIMULATED)
    sent = []
    text = notify(tmp_path, AS_OF, dry_run=True, sender=lambda *a: sent.append(a))
    assert "1. ARCLK" in text and sent == []


def test_notify_sends_with_environment_credentials(tmp_path, monkeypatch):
    write(tmp_path, "daily/2026-11-16", DAILY)
    write(tmp_path, "quality/2026-11-16", QUALITY)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "T")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "C")
    sent = []
    text = notify(tmp_path, AS_OF, sender=lambda token, chat, message: sent.append((token, chat, message)))
    assert sent == [("T", "C", text)]


def test_notify_still_reports_when_nothing_was_produced(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "T")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "C")
    sent = []
    text = notify(tmp_path, AS_OF, sender=lambda token, chat, message: sent.append(message))
    assert "Bugün için model sonucu üretilemedi" in text and "16 Kasım 2026" in text
    assert len(sent) == 1


def test_ranked_feature_reads_like_the_feature_it_ranks():
    from jarvis.telegram import _feature_label

    assert _feature_label("CMF_20_Rank") == _feature_label("CMF_20")
    assert _feature_label("Rel_Return_60D_Rank") == "endekse göre 3 aylık getiri"
    assert _feature_label("Turnover_Rank") == "işlem hacmi sırası"


def test_message_lists_the_bottom_of_the_ranking_when_present():
    daily = {**DAILY, "bottom": [{"ticker": "ZZZ", "rank": 89}, {"ticker": "YYY", "rank": 88}]}
    text = format_message(daily, QUALITY, LIVE_EMPTY, SIMULATED)
    assert "Sıralamanın sonu (modelin en zayıf gördükleri): ZZZ, YYY" in text
    assert "Sıralamanın sonu" not in format_message(DAILY, QUALITY, LIVE_EMPTY, SIMULATED)
