import numpy as np
import pandas as pd
import pytest

from jarvis.features import (
    FEATURES, LOCAL_FEATURES, MACRO_FEATURES, VIX_PERCENTILE, local_features, macro_features,
)
from jarvis.macro import MACRO_SYMBOLS
from tests.jarvis.legacy_features import legacy_functions

CALENDAR = pd.bdate_range("2024-01-01", periods=400)


def random_ohlcv(seed, index=CALENDAR):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(index))))
    high = close * (1 + rng.uniform(0, 0.02, len(index)))
    low = close * (1 - rng.uniform(0, 0.02, len(index)))
    open_ = low + (high - low) * rng.uniform(0, 1, len(index))
    volume = rng.uniform(1e5, 1e6, len(index))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=index
    )


def histories():
    return {name: random_ohlcv(10 + i) for i, name in enumerate(MACRO_SYMBOLS)}


def test_feature_names_and_order():
    assert len(LOCAL_FEATURES) == 10 and len(MACRO_FEATURES) == 14
    assert FEATURES == LOCAL_FEATURES + MACRO_FEATURES
    assert LOCAL_FEATURES[0] == "EMA20_Distance" and LOCAL_FEATURES[-1] == "StochRSI_K"
    assert MACRO_FEATURES[:2] == ("macro_USDTRY_1d_lag", "macro_USDTRY_5d_mean_lag")


def test_parity_with_core_ai_analyzer():
    legacy = legacy_functions()
    equity, benchmark = random_ohlcv(0), random_ohlcv(1)
    ours = local_features(equity, benchmark["Close"])
    theirs = legacy["build_stationary_features"](equity, benchmark)
    pd.testing.assert_frame_equal(ours, theirs[list(LOCAL_FEATURES)], rtol=1e-9)
    data = histories()
    ours = macro_features(data, CALENDAR)
    theirs = legacy["build_macro_features"](data, CALENDAR)
    pd.testing.assert_frame_equal(ours, theirs[list(ours.columns)], rtol=1e-9)


def test_local_features_shape_and_columns():
    result = local_features(random_ohlcv(0), random_ohlcv(1)["Close"])
    assert list(result.columns) == list(LOCAL_FEATURES)
    assert result.index.equals(CALENDAR)
    assert result.iloc[-1].notna().all()


def test_local_features_are_causal():
    equity, benchmark = random_ohlcv(0), random_ohlcv(1)
    before = local_features(equity, benchmark["Close"])
    changed = equity.copy()
    changed.iloc[-30:] = changed.iloc[-30:] * 1.5
    after = local_features(changed, benchmark["Close"])
    pd.testing.assert_frame_equal(before.iloc[:-30], after.iloc[:-30])


def test_macro_columns():
    result = macro_features(histories(), CALENDAR)
    assert list(result.columns) == list(MACRO_FEATURES) + [VIX_PERCENTILE]


def test_macro_value_is_lagged_one_day():
    data = histories()
    close = data["GOLD"]["Close"]
    day = CALENDAR[200]
    result = macro_features(data, CALENDAR)
    expected = close.loc[CALENDAR[199]] / close.loc[CALENDAR[198]] - 1.0
    assert result.loc[day, "macro_GOLD_1d_lag"] == pytest.approx(expected)
    # Changing the same-day source value must not move the same-day feature.
    data["GOLD"] = data["GOLD"].copy()
    data["GOLD"].loc[day, "Close"] *= 2.0
    assert macro_features(data, CALENDAR).loc[day, "macro_GOLD_1d_lag"] == pytest.approx(expected)


def test_stale_macro_becomes_nan():
    data = histories()
    data["BRENT"] = data["BRENT"].iloc[:300]
    result = macro_features(data, CALENDAR)
    assert np.isnan(result.loc[CALENDAR[-1], "macro_BRENT_1d_lag"])
    assert not np.isnan(result.loc[CALENDAR[300], "macro_BRENT_1d_lag"])


def test_missing_macro_series_gives_nan_columns():
    data = histories()
    data["SP500"] = data["SP500"].iloc[0:0]
    result = macro_features(data, CALENDAR)
    assert result["macro_SP500_1d_lag"].isna().all()
    assert result["macro_GOLD_1d_lag"].notna().any()


def test_zero_volume_and_flat_prices_produce_no_inf():
    flat = pd.DataFrame(
        {"Open": 10.0, "High": 10.0, "Low": 10.0, "Close": 10.0, "Volume": 0.0}, index=CALENDAR
    )
    result = local_features(flat, random_ohlcv(1)["Close"])
    assert not np.isinf(result.to_numpy(dtype=float)).any()
