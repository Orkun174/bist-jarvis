import numpy as np
import pandas as pd

from jarvis.config import OHLCV
from jarvis.macro import MACRO_SYMBOLS, REQUIRED_MACRO, load_macro, macro_store, update_macro
from jarvis.prices import update_prices
from jarvis.store import Store

SESSION = pd.Timestamp("2026-11-13")


def bars(dates, close=10.0):
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    return pd.DataFrame({name: np.full(len(index), close) for name in OHLCV}, index=index)


def test_macro_symbols_and_required():
    assert list(MACRO_SYMBOLS) == ["USDTRY", "BRENT", "GOLD", "SP500", "VIX", "BIST100", "BISTBANK"]
    assert MACRO_SYMBOLS["BIST100"] == "XU100.IS" and MACRO_SYMBOLS["VIX"] == "^VIX"
    assert REQUIRED_MACRO == ("BIST100", "VIX")


def test_symbol_of_maps_names_to_yahoo_symbols(tmp_path):
    asked = []

    def download(symbols, start, end):
        asked.extend(symbols)
        return {"^VIX": bars(["2026-11-13"])}

    statuses = update_prices(Store(tmp_path), ["VIX"], SESSION, download=download,
                             symbol_of=MACRO_SYMBOLS.__getitem__)
    assert asked == ["^VIX"]
    assert statuses == {"VIX": "updated"}


def test_update_macro_writes_under_macro_folder(tmp_path):
    def download(symbols, start, end):
        return {symbol: bars(["2026-11-12", "2026-11-13"]) for symbol in symbols}

    statuses = update_macro(tmp_path, SESSION, download=download)
    assert set(statuses.values()) == {"updated"}
    assert (tmp_path / "macro" / "prices" / "BIST100.parquet").exists()
    assert not (tmp_path / "prices").exists()


def test_load_macro_returns_empty_for_missing(tmp_path):
    macro_store(tmp_path).write_prices("VIX", bars(["2026-11-13"]))
    loaded = load_macro(tmp_path)
    assert list(loaded) == list(MACRO_SYMBOLS)
    assert len(loaded["VIX"]) == 1
    assert loaded["GOLD"].empty


def test_nan_volume_becomes_zero(tmp_path):
    frame = bars(["2026-11-13"])
    frame["Volume"] = np.nan

    update_macro(tmp_path, SESSION, download=lambda s, a, b: {x: frame for x in s})
    assert load_macro(tmp_path)["USDTRY"]["Volume"].iloc[0] == 0.0
