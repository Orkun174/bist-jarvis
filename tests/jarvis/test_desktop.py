import io
import json
import zipfile

import pandas as pd
import pytest

from jarvis import desktop
from jarvis.eligibility import eligibility_today
from jarvis.exclusions import empty_exclusions
from jarvis.ledger import record
from tests.jarvis.synthetic import SESSION, build_market, random_ohlcv

DAY = SESSION.date().isoformat()


@pytest.fixture()
def data(tmp_path):
    root = tmp_path / "data"
    store, names, macros = build_market(root, tickers=6)
    store.write_prices("THIN", random_ohlcv(50, volume=1_000.0))
    today = eligibility_today(store, names + ["THIN"], SESSION, empty_exclusions(), set())
    store.write_table("universe/universe", today)
    predictions = pd.DataFrame({
        "ticker": names, "prob": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1], "entry_close": [10.0] * 6,
    })
    record(store, SESSION, predictions, "m1", "r1", allow_recommendations=True)
    store.write_json(f"daily/{DAY}", {
        "session": DAY, "eligible": 6, "model": {"model_id": "m1", "checks": {}},
        "recommendations": [{"ticker": names[0], "rank": 1, "prob": 0.6, "reasons": []}],
    })
    (root / "models").mkdir(exist_ok=True)
    (root / "models" / "history.jsonl").write_text(
        json.dumps({"session": DAY, "recipe_id": "r1", "retrained": True, "weights_changed": True,
                    "checks": {"health": {"ok": True, "mean_rank_ic": 0.05},
                               "rows": {"train_rows": 1234}}}) + "\n", encoding="utf-8")
    return root, names, macros


def archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
    return buffer.getvalue()


def test_sync_replaces_the_local_copy(tmp_path):
    target = tmp_path / "data"
    (target / "daily").mkdir(parents=True)
    (target / "daily" / "old.json").write_text("{}", encoding="utf-8")
    payload = archive({
        "bist-jarvis-data/universe/universe.parquet": b"x",
        "bist-jarvis-data/daily/2026-11-16.json": b"{}",
    })
    result = desktop.sync_data(target, fetch=lambda url: payload)
    assert result["ok"] and result["has_local"]
    assert (target / "daily" / "2026-11-16.json").exists()
    assert not (target / "daily" / "old.json").exists()
    assert desktop.last_synced(target) == result["synced_at"]
    assert not (tmp_path / "data.new").exists() and not (tmp_path / "data.old").exists()


def test_failed_sync_keeps_the_local_copy(tmp_path):
    target = tmp_path / "data"
    (target / "universe").mkdir(parents=True)
    (target / "universe" / "universe.parquet").write_bytes(b"keep")

    def offline(url):
        raise ConnectionError("no network")

    result = desktop.sync_data(target, fetch=offline)
    assert result == {"ok": False, "error": "ConnectionError", "synced_at": None, "has_local": True}
    assert (target / "universe" / "universe.parquet").read_bytes() == b"keep"


def test_sync_rejects_archives_that_escape_or_lack_data(tmp_path):
    target = tmp_path / "data"
    escaping = archive({"bist-jarvis-data/../../evil.txt": b"x",
                        "bist-jarvis-data/universe/universe.parquet": b"x"})
    assert desktop.sync_data(target, fetch=lambda url: escaping)["ok"] is False
    assert not (tmp_path.parent / "evil.txt").exists() and not (tmp_path / "evil.txt").exists()
    empty = archive({"bist-jarvis-data/readme.txt": b"x"})
    result = desktop.sync_data(target, fetch=lambda url: empty)
    assert result["ok"] is False and result["has_local"] is False


def test_day_view_carries_the_telegram_message(data):
    root, names, _ = data
    view = desktop.day_view(root)
    assert view["session"] == DAY and desktop.sessions(root) == [DAY]
    assert f"1. {names[0]}" in view["message"] and "Yatırım tavsiyesi değildir." in view["message"]
    assert desktop.day_view(root / "missing") is None


def test_stock_status_for_ranked_and_excluded_stocks(data):
    root, names, _ = data
    top = desktop.stock_status(root, names[0])
    assert top["eligible"] and top["rank"] == 1 and top["of"] == 6 and top["recommended"]
    assert top["prob"] == pytest.approx(0.6) and top["reason_text"].startswith("Uygun")
    thin = desktop.stock_status(root, "THIN")
    assert thin["eligible"] is False and thin["reason"] == "low_liquidity"
    assert "Düşük hacim" in thin["reason_text"] and thin["rank"] is None and thin["of"] == 6
    assert desktop.stock_status(root, "NOPE") == {"ticker": "NOPE", "known": False}


def test_rankings_are_ordered(data):
    root, names, _ = data
    ranked = desktop.rankings(root)
    assert list(ranked["ticker"]) == names and list(ranked["rank"]) == [1, 2, 3, 4, 5, 6]


def test_indicator_readings_describe_the_last_bar(data):
    root, names, macros = data
    readings = desktop.indicator_readings(desktop.prices(root, names[0]), desktop.benchmark_close(root))
    by_name = {item["name"]: item for item in readings}
    assert {"RSI (14)", "MACD", "Para akışı (CMF)", "52 hafta zirvesi"} <= set(by_name)
    assert by_name["RSI (14)"]["reading"] in {"aşırı alım", "aşırı satım", "nötr bölge"}
    assert all(item["tone"] in {"up", "down", "flat"} for item in readings)
    assert by_name["20 günlük ortalama"]["value"].startswith("%")
    assert desktop.indicator_readings(desktop.prices(root, names[0]).head(10), macros["BIST100"]["Close"]) == []


def test_rsi_extremes():
    index = pd.bdate_range("2026-01-01", periods=60)
    rising = pd.Series(range(100, 160), index=index, dtype=float)
    assert desktop._rsi(rising).iloc[-1] == 100.0
    assert desktop._rsi(rising[::-1].set_axis(index)).iloc[-1] == pytest.approx(0.0)


def test_model_history_and_exams(data):
    root, _, _ = data
    history = desktop.model_history(root)
    assert history.iloc[0]["session"] == DAY and history.iloc[0]["train_rows"] == 1234
    assert bool(history.iloc[0]["health_ok"]) is True
    assert desktop.recipe_exams(root).empty
    (root / "models" / "search").mkdir()
    (root / "models" / "search" / f"{DAY}.json").write_text(json.dumps({
        "session": DAY, "candidates": 16, "promoted": False, "champion_recipe_id": "a",
        "challenger_recipe_id": "b", "exam": {"top_excess_diff": 0.001, "p_top_excess": 0.6},
    }), encoding="utf-8")
    exams = desktop.recipe_exams(root)
    assert exams.iloc[0]["challenger"] == "b" and not exams.iloc[0]["promoted"]
    assert desktop.graded_recommendations(root).empty
