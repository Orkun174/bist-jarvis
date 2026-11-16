import numpy as np
import pandas as pd

from jarvis.config import OHLCV
from jarvis.eligibility import eligibility_history, eligibility_today
from jarvis.store import Store

SESSIONS = pd.bdate_range(end="2026-11-13", periods=300)
SESSION = SESSIONS[-1]


def bars(n=300, close=100.0, volume=1_000_000.0):
    """Flat prices: turnover = close * volume = 100M TRY by default."""
    index = SESSIONS[-n:]
    frame = pd.DataFrame({name: np.full(n, close) for name in OHLCV}, index=index)
    frame["Volume"] = volume
    return frame


def with_limit_days(count):
    """Alternate +10% / back-to-base closes to create `count` limit days near the end."""
    frame = bars()
    closes = frame["Close"].to_numpy().copy()
    position = len(closes) - 1
    for _ in range(count):
        closes[position:] = closes[position:] * 1.10
        position -= 3
    frame["Close"] = closes
    return frame


def last(frame):
    return eligibility_history(frame).iloc[-1]


def test_clean_liquid_stock_is_eligible():
    row = last(bars())
    assert bool(row["eligible_pit"]) is True
    assert row["reason_pit"] == ""
    assert row["sessions"] == 300
    assert row["median_turnover"] == 100_000_000.0


def test_short_history_ineligible():
    row = last(bars(n=100))
    assert bool(row["eligible_pit"]) is False
    assert row["reason_pit"] == "short_history"


def test_history_boundary_252_sessions():
    assert last(bars(n=251))["reason_pit"] == "short_history"
    assert last(bars(n=252))["reason_pit"] == ""


def test_liquidity_threshold_boundary():
    assert last(bars(volume=499_000.0))["reason_pit"] == "low_liquidity"
    assert last(bars(volume=500_000.0))["reason_pit"] == ""


def test_zero_volume_days_do_not_produce_nan():
    frame = bars()
    frame.iloc[-12:, frame.columns.get_loc("Volume")] = 0.0
    row = last(frame)
    assert row["median_turnover"] == 0.0
    assert row["reason_pit"] == "low_liquidity"


def test_limit_days_boundary():
    five = last(with_limit_days(5))
    assert five["limit_days"] == 5
    assert five["reason_pit"] == ""
    six = last(with_limit_days(6))
    assert six["limit_days"] == 6
    assert six["reason_pit"] == "limit_moves"


def test_point_in_time_no_lookahead():
    frame = bars()
    before = eligibility_history(frame)
    changed = frame.copy()
    changed.iloc[-30:, changed.columns.get_loc("Volume")] = 1.0
    changed.iloc[-30:, changed.columns.get_loc("Close")] = 500.0
    after = eligibility_history(changed)
    pd.testing.assert_frame_equal(before.iloc[:-30], after.iloc[:-30])
    assert not before.iloc[-1].equals(after.iloc[-1])


def no_exclusions():
    return pd.DataFrame(columns=["ticker", "reason", "start", "end", "source"])


def test_priority_quarantine_over_exclusion_over_rules(tmp_path):
    store = Store(tmp_path)
    for ticker in ["QQQ", "XXX", "LOW", "GOOD"]:
        store.write_prices(ticker, bars(volume=1.0 if ticker in ("QQQ", "XXX", "LOW") else 1_000_000.0))
    exclusions = pd.DataFrame(
        {
            "ticker": ["QQQ", "XXX"],
            "reason": ["vbts", "vbts"],
            "start": pd.to_datetime(["2026-11-01", "2026-11-01"]),
            "end": pd.to_datetime(["2026-11-30", "2026-11-30"]),
            "source": ["test", "test"],
        }
    )
    table = eligibility_today(store, ["QQQ", "XXX", "LOW", "GOOD"], SESSION, exclusions, {"QQQ"})
    reasons = dict(zip(table["ticker"], table["reason"]))
    assert reasons == {"QQQ": "quarantined", "XXX": "excluded:vbts", "LOW": "low_liquidity", "GOOD": ""}
    assert dict(zip(table["ticker"], table["eligible"])) == {
        "QQQ": False, "XXX": False, "LOW": False, "GOOD": True,
    }
    assert list(table.columns) == [
        "ticker", "session", "eligible", "reason", "median_turnover", "limit_days", "sessions",
    ]
    assert (table["session"] == SESSION).all()


def test_ticker_without_prices_is_short_history(tmp_path):
    table = eligibility_today(Store(tmp_path), ["NEW"], SESSION, no_exclusions(), set())
    row = table.iloc[0]
    assert bool(row["eligible"]) is False
    assert row["reason"] == "short_history"
    assert row["sessions"] == 0


def test_stale_ticker_uses_only_rows_up_to_session(tmp_path):
    store = Store(tmp_path)
    store.write_prices("OLD", bars().iloc[:-40])
    table = eligibility_today(store, ["OLD"], SESSION, no_exclusions(), set())
    assert table.iloc[0]["sessions"] == 260
