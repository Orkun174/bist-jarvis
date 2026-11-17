import numpy as np
import pandas as pd
import pytest

from jarvis.dataset import TARGET_COLUMNS, build_rows, prediction_slice, training_slice
from jarvis.features import (
    EXTRA_FEATURES, FEATURES, LOCAL_FEATURES, RANKED_FEATURES, VIX_PERCENTILE,
)
from jarvis.store import Store
from tests.jarvis.synthetic import CALENDAR, SESSION, build_market, random_ohlcv


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    root = tmp_path_factory.mktemp("market")
    store, names, macros = build_market(root, tickers=4)
    rows = build_rows(store, macros, names, SESSION)
    return store, names, macros, rows


def test_columns(market):
    rows = market[3]
    ranked = [name for name in RANKED_FEATURES if name not in EXTRA_FEATURES]
    expected = ["ticker", "date", *FEATURES, *EXTRA_FEATURES, *ranked, VIX_PERCENTILE, "close",
                "eligible_pit", "fwd_excess", "fwd_excess_open", "target", "target_cs_median",
                "target_risk_adjusted", "label_end"]
    assert list(rows.columns) == expected
    assert rows[list(LOCAL_FEATURES)].notna().all().all()
    assert rows["date"].max() == SESSION


def test_target_uses_five_sessions_ahead_excess(market):
    store, names, macros, rows = market
    day, later = CALENDAR[300], CALENDAR[305]
    stock = store.read_prices("S01")["Close"]
    index = macros["BIST100"]["Close"]
    expected = (stock[later] / stock[day] - 1.0) - (index[later] / index[day] - 1.0)
    row = rows[(rows["ticker"] == "S01") & (rows["date"] == day)].iloc[0]
    assert row["fwd_excess"] == pytest.approx(expected)
    assert row["target"] == float(expected > 0)
    assert row["label_end"] == later


def test_last_five_sessions_have_no_label(market):
    rows = market[3]
    tail = rows[rows["date"] > CALENDAR[-6]]
    assert tail["fwd_excess"].isna().all() and tail["target"].isna().all()
    assert tail["label_end"].isna().all()
    assert rows[rows["date"] == CALENDAR[-6]]["target"].notna().all()


def test_training_slice_respects_embargo(market):
    rows = market[3]
    train = training_slice(rows, SESSION, CALENDAR)
    assert not train.empty
    assert train["label_end"].max() <= CALENDAR[-8]  # 7 sessions before the session
    assert train["target"].notna().all()


def test_training_slice_excludes_ineligible_rows(market):
    rows = market[3].copy()
    rows.loc[rows["ticker"] == "S02", "eligible_pit"] = False
    train = training_slice(rows, SESSION, CALENDAR)
    assert "S02" not in set(train["ticker"])
    assert train["eligible_pit"].all()


def test_training_slice_limits_window_to_three_years(market):
    rows = market[3]
    early = training_slice(rows, SESSION, CALENDAR, train_years=1)
    assert early["date"].min() >= SESSION - pd.DateOffset(years=1)


def test_no_lookahead_in_features(tmp_path, market):
    store, names, macros, rows = market
    other = Store(tmp_path)
    for name in names:
        frame = store.read_prices(name).copy()
        frame.iloc[-30:] = frame.iloc[-30:] * 1.7
        other.write_prices(name, frame)
    changed = build_rows(other, macros, names, SESSION)
    cutoff = CALENDAR[-31]
    left = rows[rows["date"] <= cutoff].reset_index(drop=True)
    right = changed[changed["date"] <= cutoff].reset_index(drop=True)
    columns = ["ticker", "date", *FEATURES, *EXTRA_FEATURES, *RANKED_FEATURES]
    pd.testing.assert_frame_equal(left[columns], right[columns])


def test_ticker_missing_bar_at_horizon_has_nan_label(tmp_path):
    store, names, macros = build_market(tmp_path, tickers=2)
    day, later = CALENDAR[300], CALENDAR[305]
    frame = store.read_prices("S00").drop(later)
    store.write_prices("S00", frame, replace=True)
    rows = build_rows(store, macros, names, SESSION)
    row = rows[(rows["ticker"] == "S00") & (rows["date"] == day)].iloc[0]
    assert np.isnan(row["fwd_excess"]) and np.isnan(row["target"])
    assert rows[(rows["ticker"] == "S00") & (rows["date"] == later)].empty  # no bar, no row


def test_rows_stop_at_session(tmp_path):
    store, names, macros = build_market(tmp_path, tickers=2)
    rows = build_rows(store, macros, names, CALENDAR[-10])
    assert rows["date"].max() == CALENDAR[-10]
    assert rows[rows["date"] > CALENDAR[-15]]["target"].isna().all()
    assert rows[rows["date"] == CALENDAR[-15]]["target"].notna().all()


def test_ticker_without_prices_is_skipped(tmp_path):
    store, names, macros = build_market(tmp_path, tickers=2)
    rows = build_rows(store, macros, names + ["GHOST"], SESSION)
    assert "GHOST" not in set(rows["ticker"])


def test_prediction_slice_only_todays_eligible(market):
    rows = market[3]
    today = prediction_slice(rows, SESSION, {"S00", "S03", "NOPE"})
    assert sorted(today["ticker"]) == ["S00", "S03"]
    assert (today["date"] == SESSION).all()


def test_cs_median_target_compares_with_the_days_median_eligible_stock(market):
    rows = market[3]
    day = rows[(rows["date"] == CALENDAR[300]) & rows["eligible_pit"].astype(bool)]
    assert len(day) >= 3
    median = day["fwd_excess"].median()
    expected = (day["fwd_excess"] > median).astype(float)
    assert (day[TARGET_COLUMNS["cs_median"]] == expected).all()


def test_risk_adjusted_target_divides_by_recent_volatility(market):
    rows = market[3]
    day = rows[(rows["date"] == CALENDAR[300]) & rows["eligible_pit"].astype(bool)]
    scaled = day["fwd_excess"] / day["Volatility_20D"]
    expected = (scaled > scaled.median()).astype(float)
    assert (day[TARGET_COLUMNS["risk_adjusted"]] == expected).all()


def test_alternative_targets_are_unknown_where_the_outcome_is(market):
    rows = market[3]
    tail = rows[rows["date"] > CALENDAR[-6]]
    for column in TARGET_COLUMNS.values():
        assert tail[column].isna().all()


def test_turnover_rank_is_a_share_within_the_day(market):
    rows = market[3]
    day = rows[rows["date"] == CALENDAR[300]]
    assert day["Turnover_Rank"].between(0, 1).all() and day["Turnover_Rank"].max() == 1.0


def test_ranked_features_rank_eligible_stocks_within_the_day(market):
    rows = market[3]
    day = rows[rows["date"] == CALENDAR[300]]
    eligible = day[day["eligible_pit"].astype(bool)]
    expected = eligible["CMF_20"].rank(pct=True)
    assert (eligible["CMF_20_Rank"] == expected).all()
    assert day[~day["eligible_pit"].astype(bool)]["CMF_20_Rank"].isna().all()


def test_outcome_from_next_open_is_what_a_buyer_after_the_message_gets(market):
    store, names, macros, rows = market
    day, entry, later = CALENDAR[300], CALENDAR[301], CALENDAR[305]
    stock, index = store.read_prices("S01"), macros["BIST100"]
    expected = (stock["Close"][later] / stock["Open"][entry] - 1.0) - (
        index["Close"][later] / index["Open"][entry] - 1.0)
    row = rows[(rows["ticker"] == "S01") & (rows["date"] == day)].iloc[0]
    assert row["fwd_excess_open"] == pytest.approx(expected)
    assert rows[rows["date"] > CALENDAR[-6]]["fwd_excess_open"].isna().all()
