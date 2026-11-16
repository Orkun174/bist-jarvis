import numpy as np
import pandas as pd

from jarvis.config import OHLCV
from jarvis.quality import build_report, check_prices, cross_check
from jarvis.store import Store

SESSIONS = pd.bdate_range("2026-09-01", "2026-11-13")
SESSION = SESSIONS[-1]


def clean(sessions=SESSIONS, close=100.0, volume=1_000_000.0):
    frame = pd.DataFrame({name: np.full(len(sessions), close) for name in OHLCV}, index=sessions)
    frame["Volume"] = volume
    return frame


def test_clean_frame_has_no_issues():
    assert check_prices(clean(), SESSIONS) == []


def test_empty_frame_is_stale():
    assert check_prices(clean().iloc[0:0], SESSIONS) == ["stale"]


def test_nonpositive_price():
    frame = clean()
    frame.iloc[-3, frame.columns.get_loc("Low")] = 0.0
    assert check_prices(frame, SESSIONS) == ["nonpositive_price"]


def test_zero_volume_last():
    frame = clean()
    frame.iloc[-1, frame.columns.get_loc("Volume")] = 0.0
    assert check_prices(frame, SESSIONS) == ["zero_volume_last"]


def test_missing_sessions():
    frame = clean().drop(SESSIONS[-5])
    assert check_prices(frame, SESSIONS) == ["missing_sessions"]


def test_old_gaps_outside_last_twenty_sessions_are_ignored():
    frame = clean().drop(SESSIONS[3])
    assert check_prices(frame, SESSIONS) == []


def test_suspect_jump():
    frame = clean()
    frame.iloc[-2:, :4] = 130.0
    assert check_prices(frame, SESSIONS) == ["suspect_jump"]


def test_stale():
    frame = clean().iloc[:-1]
    assert "stale" in check_prices(frame, SESSIONS)


def test_cross_check_boundaries():
    assert cross_check(100.9, 100.0) == "ok"
    assert cross_check(101.1, 100.0) == "mismatch"
    assert cross_check(98.9, 100.0) == "mismatch"
    assert cross_check(100.0, None) == "unavailable"
    assert cross_check(100.0, float("nan")) == "unavailable"


def filled_store(tmp_path, tickers, bad=()):
    store = Store(tmp_path)
    for ticker in tickers:
        frame = clean()
        if ticker in bad:
            frame = frame.iloc[:-1]
        store.write_prices(ticker, frame)
    return store


def reference_ok(tickers, session):
    return {ticker: 100.0 for ticker in tickers}


def test_report_blocks_training_above_twenty_percent(tmp_path):
    tickers = [f"T{i}" for i in range(10)]
    statuses = {ticker: "updated" for ticker in tickers}
    store = filled_store(tmp_path / "a", tickers, bad=tickers[:3])
    report = build_report(store, tickers, SESSION, statuses, SESSIONS, reference=reference_ok)
    assert sorted(report["quarantined"]) == tickers[:3]
    assert report["quarantine_ratio"] == 0.3
    assert report["training_allowed"] is False

    store = filled_store(tmp_path / "b", tickers, bad=tickers[:2])
    report = build_report(store, tickers, SESSION, statuses, SESSIONS, reference=reference_ok)
    assert report["quarantine_ratio"] == 0.2
    assert report["training_allowed"] is True


def test_report_shape_and_fetch_failed(tmp_path):
    tickers = ["AAA", "BBB"]
    store = filled_store(tmp_path, ["AAA"])
    statuses = {"AAA": "current", "BBB": "fetch_failed"}
    report = build_report(store, tickers, SESSION, statuses, SESSIONS, reference=reference_ok)
    assert report["session"] == "2026-11-13"
    assert report["tickers"] == 2
    assert report["fetch_failed"] == ["BBB"]
    assert "fetch_failed" in report["quarantined"]["BBB"]
    assert "AAA" not in report["quarantined"]


def test_cross_check_mismatch_quarantines(tmp_path):
    store = filled_store(tmp_path, ["AAA", "BBB"])
    statuses = {"AAA": "updated", "BBB": "updated"}
    reference = lambda tickers, session: {"AAA": 100.0, "BBB": 120.0}
    report = build_report(store, ["AAA", "BBB"], SESSION, statuses, SESSIONS, reference=reference)
    assert report["quarantined"] == {"BBB": ["cross_check_mismatch"]}


def test_reference_outage_does_not_quarantine(tmp_path):
    store = filled_store(tmp_path, ["AAA", "BBB"])
    statuses = {"AAA": "updated", "BBB": "updated"}
    report = build_report(store, ["AAA", "BBB"], SESSION, statuses, SESSIONS, reference=lambda t, s: {})
    assert report["quarantined"] == {}
    assert report["cross_check_unavailable"] == ["AAA", "BBB"]
    assert report["training_allowed"] is True


# -- Is Yatirim reference closes ------------------------------------------

from jarvis.quality import isyatirim_closes  # noqa: E402


def test_reference_skips_everything_when_probe_finds_nothing():
    asked = []

    def fetch_one(ticker, session):
        asked.append(ticker)
        return None

    tickers = [f"T{i:03d}" for i in range(100)]
    assert isyatirim_closes(tickers, SESSION, fetch_one=fetch_one) == {}
    assert len(asked) == 3  # only the probe, not all 100


def test_reference_fetches_all_when_probe_succeeds():
    def fetch_one(ticker, session):
        return None if ticker == "T007" else 100.0

    tickers = [f"T{i:03d}" for i in range(40)]
    closes = isyatirim_closes(tickers, SESSION, fetch_one=fetch_one)
    assert len(closes) == 39
    assert "T007" not in closes
    assert closes["T000"] == 100.0


def test_reference_survives_fetch_errors():
    def fetch_one(ticker, session):
        if ticker == "T001":
            raise OSError("timeout")
        return 100.0

    closes = isyatirim_closes(["T000", "T001", "T002", "T003"], SESSION, fetch_one=fetch_one)
    assert sorted(closes) == ["T000", "T002", "T003"]


def test_reference_stops_at_deadline():
    def fetch_one(ticker, session):
        return 100.0

    tickers = [f"T{i:03d}" for i in range(50)]
    closes = isyatirim_closes(tickers, SESSION, fetch_one=fetch_one, budget_seconds=0.0)
    assert len(closes) < 50
