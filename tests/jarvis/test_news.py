import json

import pytest

from jarvis import news

RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>t</title>
<item><title>Eski &amp; haber</title><link>https://example.com/a</link>
  <pubDate>Mon, 09 Nov 2026 08:00:00 GMT</pubDate></item>
<item><title>Yeni &lt;b&gt;haber&lt;/b&gt;</title><link>https://example.com/b</link>
  <pubDate>Mon, 16 Nov 2026 08:00:00 GMT</pubDate></item>
<item><title>Yeni haber</title><link>https://example.com/c</link>
  <pubDate>Sun, 15 Nov 2026 08:00:00 GMT</pubDate></item>
<item><title>Tarihsiz</title><link>javascript:alert(1)</link></item>
</channel></rss>"""


def test_fetch_news_parses_sorts_and_deduplicates():
    asked = []

    def download(ticker):
        asked.append(ticker)
        return RSS

    items = news.fetch_news("arclk.is", download=download)
    assert asked == ["ARCLK"]
    assert [item["title"] for item in items] == ["Yeni haber", "Eski & haber", "Tarihsiz"]
    assert items[0]["published_at"].startswith("2026-11-16")
    assert items[-1]["url"] == "" and items[-1]["published_at"] is None


def test_fetch_news_rejects_bad_input():
    with pytest.raises(news.NewsError):
        news.fetch_news("x!", download=lambda t: RSS)
    with pytest.raises(news.NewsError):
        news.fetch_news("ARCLK", download=lambda t: b"not rss")


def test_news_sentiment_with_a_classifier():
    def fake(texts):
        return [{"label": "positive", "score": 0.9}, {"label": "negative", "score": 0.5}][: len(texts)]

    result = news.news_sentiment(["İyi haber", "  ", "Kötü haber"], classifier=fake)
    assert result["status"] == "scored" and result["count"] == 2
    assert result["score"] == pytest.approx((0.9 - 0.5) / 2)


def test_news_sentiment_is_off_by_default():
    assert news.SENTIMENT_ENABLED is False
    assert news.news_sentiment(["Rekor kâr"]) == {"status": "disabled"}


def test_news_sentiment_without_news_or_model(monkeypatch):
    monkeypatch.setattr(news, "SENTIMENT_ENABLED", True)
    assert news.news_sentiment([], classifier=lambda texts: []) == {"status": "no_news"}
    assert news.news_sentiment(["x"], classifier=lambda texts: [{"label": "?", "score": 1}]) == {"status": "failed"}
    monkeypatch.setattr(news, "_default_classifier", lambda: None)
    assert news.news_sentiment(["x"]) == {"status": "unavailable"}


def test_sentiment_words():
    assert [news.sentiment_word(v) for v in (0.5, 0.2, 0.0, -0.2, -0.5)] == [
        "olumlu", "nötr", "nötr", "nötr", "olumsuz"]


def test_daily_job_adds_news_for_the_recommendations_only(tmp_path):
    folder = tmp_path / "daily"
    folder.mkdir()
    daily = {"session": "2026-11-16", "recommendations": [{"ticker": "AAA", "rank": 1},
                                                          {"ticker": "BBB", "rank": 2}],
             "top": [{"ticker": "CCC"}]}
    (folder / "2026-11-16.json").write_text(json.dumps(daily), encoding="utf-8")

    def fetch(ticker):
        if ticker == "BBB":
            raise news.NewsError("429")
        return [{"title": f"{ticker} kâr açıkladı", "published_at": None}]

    classifier = lambda texts: [{"label": "positive", "score": 0.8} for _ in texts]
    summary = news.add_news_to_daily(tmp_path, fetch=fetch, classifier=classifier, pause=0)
    assert summary == {"status": "ok", "session": "2026-11-16", "tickers": 2,
                       "outcomes": {"scored": 1, "fetch_failed": 1}}
    saved = json.loads((folder / "2026-11-16.json").read_text(encoding="utf-8"))
    assert set(saved["news"]) == {"AAA", "BBB"}
    assert saved["news"]["AAA"]["score"] == pytest.approx(0.8)
    assert saved["news"]["BBB"] == {"status": "fetch_failed", "error": "NewsError", "headlines": []}
    assert saved["top"] == daily["top"]


def test_daily_job_without_recommendations_does_nothing(tmp_path):
    assert news.add_news_to_daily(tmp_path)["reason"] == "no_daily_result"
    (tmp_path / "daily").mkdir()
    (tmp_path / "daily" / "2026-11-16.json").write_text(json.dumps({"recommendations": []}), encoding="utf-8")
    assert news.add_news_to_daily(tmp_path)["reason"] == "no_recommendations"
