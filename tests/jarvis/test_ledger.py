import numpy as np
import pandas as pd
import pytest

from jarvis.config import OHLCV
from jarvis.ledger import LEDGER_COLUMNS, mature, read_ledger, record
from jarvis.scorecard import scorecard
from jarvis.store import Store

CALENDAR = pd.bdate_range("2026-10-01", periods=30)
DAY = CALENDAR[10]
EXIT = CALENDAR[15]


def prices(close_by_day: dict, index=CALENDAR):
    close = pd.Series(100.0, index=index)
    for day, value in close_by_day.items():
        close[day] = value
    frame = pd.DataFrame({name: close for name in OHLCV})
    frame["Volume"] = 1_000_000.0
    return frame


def predictions(probs: dict):
    return pd.DataFrame(
        {"ticker": list(probs), "prob": list(probs.values()), "entry_close": 100.0}
    )


def benchmark(close_by_day=None):
    return prices(close_by_day or {})


def test_record_ranks_and_flags_recommendations(tmp_path):
    store = Store(tmp_path)
    probs = {"AAA": 0.70, "BBB": 0.90, "CCC": 0.60, "DDD": 0.80, "EEE": 0.65, "FFF": 0.61, "GGG": 0.40}
    written = record(store, DAY, predictions(probs), "m1", "r1", threshold=0.55)
    ledger = read_ledger(store)
    assert written == 7 and list(ledger.columns) == LEDGER_COLUMNS
    ranks = dict(zip(ledger["ticker"], ledger["rank"]))
    assert ranks == {"BBB": 1, "DDD": 2, "AAA": 3, "EEE": 4, "FFF": 5, "CCC": 6, "GGG": 7}
    recommended = set(ledger[ledger["recommended"]]["ticker"])
    assert recommended == {"BBB", "DDD", "AAA", "EEE", "FFF"}  # top 5 only
    assert not ledger["matured"].any()
    assert (ledger["model_id"] == "m1").all() and (ledger["threshold"] == 0.55).all()


def test_record_below_threshold_not_recommended(tmp_path):
    store = Store(tmp_path)
    record(store, DAY, predictions({"AAA": 0.58, "BBB": 0.54, "CCC": 0.50}), "m1", "r1", threshold=0.55)
    ledger = read_ledger(store)
    assert set(ledger[ledger["recommended"]]["ticker"]) == {"AAA"}


def test_record_ties_break_alphabetically(tmp_path):
    store = Store(tmp_path)
    record(store, DAY, predictions({"ZZZ": 0.6, "AAA": 0.6}), "m1", "r1", threshold=0.5)
    ledger = read_ledger(store)
    assert dict(zip(ledger["ticker"], ledger["rank"])) == {"AAA": 1, "ZZZ": 2}


def test_record_no_recommendations_flag(tmp_path):
    store = Store(tmp_path)
    record(store, DAY, predictions({"AAA": 0.9}), "m1", "r1", threshold=0.5, allow_recommendations=False)
    assert not read_ledger(store)["recommended"].any()


def test_record_same_day_twice_replaces(tmp_path):
    store = Store(tmp_path)
    record(store, DAY, predictions({"AAA": 0.7, "BBB": 0.6}), "m1", "r1", threshold=0.5)
    record(store, CALENDAR[11], predictions({"AAA": 0.7}), "m1", "r1", threshold=0.5)
    record(store, DAY, predictions({"AAA": 0.9}), "m2", "r1", threshold=0.5)
    ledger = read_ledger(store)
    assert len(ledger) == 2
    first = ledger[ledger["date"] == DAY].iloc[0]
    assert first["model_id"] == "m2" and first["prob"] == 0.9


def test_mature_fills_after_five_sessions(tmp_path):
    store = Store(tmp_path)
    store.write_prices("AAA", prices({EXIT: 110.0}))
    record(store, DAY, predictions({"AAA": 0.7}), "m1", "r1", threshold=0.5)
    assert mature(store, store, benchmark({EXIT: 102.0}), CALENDAR, CALENDAR[14]) == 0
    assert mature(store, store, benchmark({EXIT: 102.0}), CALENDAR, EXIT) == 1
    row = read_ledger(store).iloc[0]
    assert bool(row["matured"]) is True and row["exit_date"] == EXIT
    assert row["fwd_return"] == pytest.approx(0.10)
    assert row["benchmark_return"] == pytest.approx(0.02)
    assert row["fwd_excess"] == pytest.approx(0.08)
    assert mature(store, store, benchmark({EXIT: 102.0}), CALENDAR, EXIT) == 0  # nothing left


def test_mature_skips_when_exit_price_missing(tmp_path):
    store = Store(tmp_path)
    store.write_prices("AAA", prices({}).drop(EXIT))
    record(store, DAY, predictions({"AAA": 0.7, "GONE": 0.6}), "m1", "r1", threshold=0.5)
    assert mature(store, store, benchmark(), CALENDAR, CALENDAR[-1]) == 0
    assert not read_ledger(store)["matured"].any()


def test_mature_uses_current_adjusted_series(tmp_path):
    """After a split the stored series is rescaled; the return must not use entry_close."""
    store = Store(tmp_path)
    rescaled = prices({EXIT: 110.0}) / 2.0  # both entry and exit halved
    store.write_prices("AAA", rescaled)
    record(store, DAY, predictions({"AAA": 0.7}), "m1", "r1", threshold=0.5)  # entry_close 100
    mature(store, store, benchmark(), CALENDAR, EXIT)
    assert read_ledger(store).iloc[0]["fwd_return"] == pytest.approx(0.10)


def filled_ledger(tmp_path):
    store = Store(tmp_path)
    closes = {"AAA": 110.0, "BBB": 95.0, "CCC": 101.0}
    for ticker, exit_close in closes.items():
        store.write_prices(ticker, prices({EXIT: exit_close}))
    record(store, DAY, predictions({"AAA": 0.62, "BBB": 0.57, "CCC": 0.40}), "m1", "r1", threshold=0.55)
    mature(store, store, benchmark(), CALENDAR, EXIT)
    return read_ledger(store)


def test_scorecard_windows_and_hit_rate(tmp_path):
    card = scorecard(filled_ledger(tmp_path), EXIT, "live")
    assert card["kind"] == "live" and card["as_of"] == EXIT.date().isoformat()
    window = card["windows"]["all"]
    assert window["recommendations"] == 2
    assert window["hit_rate"] == pytest.approx(0.5)
    assert window["mean_excess"] == pytest.approx((0.10 - 0.05) / 2)
    assert window["days"] == 1
    assert window["mean_rank_ic"] == pytest.approx(0.5)
    assert set(card["windows"]) == {"20", "60", "all"}
    buckets = {b["bucket"]: b for b in card["calibration"]}
    assert buckets["0.60+"]["count"] == 1 and buckets["0.60+"]["hit_rate"] == pytest.approx(1.0)
    assert buckets["0.55-0.60"]["count"] == 1 and buckets["0.55-0.60"]["hit_rate"] == pytest.approx(0.0)


def test_scorecard_window_excludes_old_days(tmp_path):
    ledger = filled_ledger(tmp_path)
    later = CALENDAR[-1] + pd.Timedelta(days=200)
    calendar_after = pd.bdate_range(CALENDAR[0], later)
    card = scorecard(ledger, later, "live", calendar=calendar_after)
    assert card["windows"]["20"]["recommendations"] == 0
    assert card["windows"]["all"]["recommendations"] == 2


def test_scorecard_empty_ledger():
    empty = pd.DataFrame(columns=LEDGER_COLUMNS)
    card = scorecard(empty, EXIT, "live")
    for window in card["windows"].values():
        assert window == {
            "recommendations": 0, "hit_rate": None, "mean_excess": None,
            "mean_rank_ic": None, "days": 0,
        }
    assert all(b["count"] == 0 and b["hit_rate"] is None for b in card["calibration"])
    assert not np.isnan(card["pending"])
