import os

import numpy as np
import pandas as pd
import pytest

from jarvis.config import OHLCV
from jarvis.store import Store


def frame(dates, close):
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    data = {name: np.asarray(close, dtype=float) for name in OHLCV}
    return pd.DataFrame(data, index=index)


def test_read_missing_returns_empty(tmp_path):
    result = Store(tmp_path).read_prices("THYAO")
    assert list(result.columns) == OHLCV
    assert len(result) == 0


def test_write_then_read_roundtrip(tmp_path):
    store = Store(tmp_path)
    original = frame(["2026-11-11", "2026-11-12", "2026-11-13"], [1.0, 2.0, 3.0])
    assert store.write_prices("THYAO", original) is True
    result = store.read_prices("THYAO")
    pd.testing.assert_frame_equal(result, original, check_freq=False, check_names=False)


def test_merge_overwrites_same_date_and_sorts(tmp_path):
    store = Store(tmp_path)
    store.write_prices("THYAO", frame(["2026-11-12", "2026-11-13"], [2.0, 3.0]))
    store.write_prices("THYAO", frame(["2026-11-13", "2026-11-11"], [9.0, 1.0]))
    result = store.read_prices("THYAO")
    assert list(result.index) == list(pd.to_datetime(["2026-11-11", "2026-11-12", "2026-11-13"]))
    assert result.loc["2026-11-13", "Close"] == 9.0
    assert result.index.is_unique


def test_identical_write_returns_false_and_keeps_mtime(tmp_path):
    store = Store(tmp_path)
    data = frame(["2026-11-12", "2026-11-13"], [2.0, 3.0])
    store.write_prices("THYAO", data)
    path = tmp_path / "prices" / "THYAO.parquet"
    os.utime(path, (1_000_000_000, 1_000_000_000))
    assert store.write_prices("THYAO", data) is False
    assert path.stat().st_mtime == 1_000_000_000


def test_table_roundtrip_and_identical_write(tmp_path):
    store = Store(tmp_path)
    table = pd.DataFrame({"ticker": ["A", "B"], "eligible": [True, False]})
    assert store.write_table("universe/universe", table) is True
    assert store.write_table("universe/universe", table) is False
    pd.testing.assert_frame_equal(store.read_table("universe/universe"), table)
    assert store.read_table("universe/missing").empty


def test_write_json_roundtrip_keeps_turkish(tmp_path):
    store = Store(tmp_path)
    store.write_json("quality/2026-11-13", {"not": "ığüşöç"})
    text = (tmp_path / "quality" / "2026-11-13.json").read_text(encoding="utf-8")
    assert "ığüşöç" in text


def test_write_json_rejects_nan(tmp_path):
    with pytest.raises(ValueError):
        Store(tmp_path).write_json("quality/x", {"ratio": float("nan")})


def test_replace_discards_rows_missing_from_new_frame(tmp_path):
    store = Store(tmp_path)
    store.write_prices("THYAO", frame(["2026-11-11", "2026-11-12"], [1.0, 2.0]))
    assert store.write_prices("THYAO", frame(["2026-11-12", "2026-11-13"], [5.0, 6.0]), replace=True) is True
    result = store.read_prices("THYAO")
    assert list(result.index) == list(pd.to_datetime(["2026-11-12", "2026-11-13"]))
    assert list(result["Close"]) == [5.0, 6.0]
