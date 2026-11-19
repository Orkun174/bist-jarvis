import json

import numpy as np
import pandas as pd
import pytest

from jarvis.calendar import sessions_until
from jarvis.config import OHLCV
from jarvis.exclusions import COLUMNS, empty_exclusions
from jarvis.pipeline import LAG_RETRIES, LAG_WAIT_SECONDS, run
from jarvis.store import Store
from jarvis.universe import UniverseError

AS_OF = pd.Timestamp("2026-11-13 19:00", tz="Europe/Istanbul")
SESSION = pd.Timestamp("2026-11-13")
HISTORY = sessions_until(SESSION, 300)


def bars(volume=1_000_000.0, sessions=HISTORY):
    frame = pd.DataFrame({name: np.full(len(sessions), 100.0) for name in OHLCV}, index=sessions)
    frame["Volume"] = volume
    return frame


FRAMES = {
    "GOOD.IS": bars(),
    "THIN.IS": bars(volume=10.0),
    "WATCH.IS": bars(),
    "LATE.IS": bars(sessions=HISTORY[:-1]),
}
TICKERS = ["GOOD", "LATE", "THIN", "WATCH"]


def download(symbols, start, end):
    return {s: FRAMES[s][(FRAMES[s].index >= start) & (FRAMES[s].index <= end)] for s in symbols}


def reference(tickers, session):
    return {ticker: 100.0 for ticker in tickers}


def exclusions(tickers, session):
    rows = pd.DataFrame(
        [{"ticker": "WATCH", "reason": "watchlist_market", "start": session, "end": session, "source": "kap:test"}],
        columns=COLUMNS,
    )
    return rows, {"kap_markets": "ok", "measures": "ok"}


def no_wait(seconds):
    raise AssertionError("one late ticker out of four is not a source lag")


def execute(tmp_path, **overrides):
    arguments = dict(
        symbols=lambda: list(TICKERS), download=download, reference=reference,
        exclusions=exclusions, exclusions_path=tmp_path / "manual.csv", wait=no_wait,
    )
    arguments.update(overrides)
    return run(tmp_path / "data", AS_OF, **arguments)


def test_summary_counts(tmp_path):
    summary = execute(tmp_path)
    assert summary == {
        "session": "2026-11-13", "tickers": 4, "eligible": 1, "quarantined": 1,
        "excluded": 1, "training_allowed": False,
        "exclusion_sources": {"kap_markets": "ok", "measures": "ok"},
    }


def test_run_writes_expected_files(tmp_path):
    execute(tmp_path)
    data = tmp_path / "data"
    assert (data / "prices" / "GOOD.parquet").exists()
    report = json.loads((data / "quality" / "2026-11-13.json").read_text(encoding="utf-8"))
    assert list(report["quarantined"]) == ["LATE"]
    assert report["exclusion_sources"] == {"kap_markets": "ok", "measures": "ok"}
    universe = Store(data).read_table("universe/universe")
    assert dict(zip(universe["ticker"], universe["reason"])) == {
        "GOOD": "", "LATE": "quarantined", "THIN": "low_liquidity",
        "WATCH": "excluded:watchlist_market",
    }
    measures = Store(data).read_table("universe/measures")
    assert list(measures["ticker"]) == ["WATCH"]
    assert (measures["session"] == SESSION).all()


def mtimes(folder):
    return {path.name: path.stat().st_mtime_ns for path in folder.rglob("*.parquet")}


def test_run_twice_same_session_is_idempotent(tmp_path):
    execute(tmp_path)
    before = mtimes(tmp_path / "data")
    execute(tmp_path)
    assert mtimes(tmp_path / "data") == before
    assert len(Store(tmp_path / "data").read_table("universe/universe")) == 4


def test_next_session_appends_universe_rows(tmp_path):
    execute(tmp_path)
    run(
        tmp_path / "data", pd.Timestamp("2026-11-12 19:00", tz="Europe/Istanbul"),
        symbols=lambda: list(TICKERS), download=download, reference=reference,
        exclusions=exclusions, exclusions_path=tmp_path / "manual.csv",
    )
    universe = Store(tmp_path / "data").read_table("universe/universe")
    assert len(universe) == 8
    assert universe["session"].is_monotonic_increasing


def test_universe_error_leaves_data_untouched(tmp_path):
    def broken():
        raise UniverseError("sayfa bozuk")

    with pytest.raises(UniverseError):
        execute(tmp_path, symbols=broken)
    assert not (tmp_path / "data").exists()


def test_failed_source_reuses_last_snapshot(tmp_path):
    execute(tmp_path)

    extended = bars(sessions=sessions_until(pd.Timestamp("2026-11-16"), 301))

    def later(symbols, start, end):
        return {s: extended[(extended.index >= start) & (extended.index <= end)] for s in symbols}

    def failing(tickers, session):
        return empty_exclusions(), {"kap_markets": "failed", "measures": "failed"}

    summary = run(
        tmp_path / "data", pd.Timestamp("2026-11-16 19:00", tz="Europe/Istanbul"),
        symbols=lambda: list(TICKERS), download=later, reference=reference,
        exclusions=failing, exclusions_path=tmp_path / "manual.csv",
    )
    assert summary["exclusion_sources"] == {"kap_markets": "failed", "measures": "failed"}
    universe = Store(tmp_path / "data").read_table("universe/universe")
    today = universe[universe["session"] == pd.Timestamp("2026-11-16")]
    assert dict(zip(today["ticker"], today["reason"]))["WATCH"] == "excluded:watchlist_market"
    measures = Store(tmp_path / "data").read_table("universe/measures")
    assert len(measures) == 1  # nothing new recorded for the failed day


def test_manual_csv_is_applied(tmp_path):
    (tmp_path / "manual.csv").write_text(
        "ticker,reason,start,end,source\nGOOD,manual,2026-11-01,,elle\n", encoding="utf-8"
    )
    execute(tmp_path)
    universe = Store(tmp_path / "data").read_table("universe/universe")
    assert dict(zip(universe["ticker"], universe["reason"]))["GOOD"] == "excluded:manual"


def test_limit_processes_only_first_symbols(tmp_path):
    summary = execute(tmp_path, limit=2)
    assert summary["tickers"] == 2


def lagging_download(published_after_calls):
    """Yahoo without the session bar for every ticker until `published_after_calls` calls."""
    calls = []

    def fetch(symbols, start, end):
        calls.append(symbols)
        last = end if len(calls) > published_after_calls else end - pd.Timedelta(days=1)
        return download(symbols, start, last)

    return fetch


def read_quality(tmp_path):
    return json.loads((tmp_path / "data" / "quality" / "2026-11-13.json").read_text(encoding="utf-8"))


def test_source_lag_waits_and_retries_until_the_session_arrives(tmp_path):
    waits = []
    execute(tmp_path, download=lagging_download(1), wait=waits.append)
    assert waits == [LAG_WAIT_SECONDS]
    report = read_quality(tmp_path)
    assert list(report["quarantined"]) == ["LATE"]
    assert report["source_lag"] is False


def test_source_lag_that_never_clears_is_reported(tmp_path):
    waits = []
    execute(tmp_path, download=lagging_download(99), wait=waits.append)
    assert waits == [LAG_WAIT_SECONDS] * LAG_RETRIES
    report = read_quality(tmp_path)
    assert report["source_lag"] is True
    assert len(report["quarantined"]) == 4
