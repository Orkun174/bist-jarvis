import numpy as np
import pandas as pd

from jarvis.config import OHLCV
from jarvis.prices import update_prices
from jarvis.store import Store

SESSION = pd.Timestamp("2026-11-13")


def bars(dates, close=10.0):
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    return pd.DataFrame({name: np.full(len(index), close) for name in OHLCV}, index=index)


class Recorder:
    """Fake downloader that records each request and replays canned frames."""

    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def __call__(self, symbols, start, end):
        self.calls.append((list(symbols), pd.Timestamp(start), pd.Timestamp(end)))
        return {s: self.frames[s] for s in symbols if s in self.frames}


def requested(recorder, symbol):
    return [(start, end) for symbols, start, end in recorder.calls if symbol in symbols]


def test_new_ticker_requests_five_years(tmp_path):
    recorder = Recorder({"THYAO.IS": bars(["2026-11-12", "2026-11-13"])})
    statuses = update_prices(Store(tmp_path), ["THYAO"], SESSION, download=recorder)
    assert requested(recorder, "THYAO.IS") == [(pd.Timestamp("2021-11-13"), SESSION)]
    assert statuses == {"THYAO": "updated"}
    assert len(Store(tmp_path).read_prices("THYAO")) == 2


def test_existing_ticker_requests_only_missing_days(tmp_path):
    store = Store(tmp_path)
    store.write_prices("THYAO", bars(["2026-11-10", "2026-11-11"]))
    recorder = Recorder({"THYAO.IS": bars(["2026-11-12", "2026-11-13"])})
    update_prices(store, ["THYAO"], SESSION, download=recorder)
    assert requested(recorder, "THYAO.IS") == [(pd.Timestamp("2026-11-12"), SESSION)]
    assert len(store.read_prices("THYAO")) == 4


def test_current_ticker_is_not_downloaded(tmp_path):
    store = Store(tmp_path)
    store.write_prices("THYAO", bars(["2026-11-12", "2026-11-13"]))
    recorder = Recorder({})
    statuses = update_prices(store, ["THYAO"], SESSION, download=recorder)
    assert requested(recorder, "THYAO.IS") == []
    assert statuses == {"THYAO": "current"}


def test_empty_download_marks_fetch_failed_and_continues(tmp_path):
    store = Store(tmp_path)
    recorder = Recorder({"GARAN.IS": bars(["2026-11-13"]), "THYAO.IS": bars([])})
    statuses = update_prices(store, ["THYAO", "GARAN", "AKBNK"], SESSION, download=recorder)
    assert statuses == {"THYAO": "fetch_failed", "GARAN": "updated", "AKBNK": "fetch_failed"}
    assert len(store.read_prices("GARAN")) == 1


def test_downloader_exception_marks_batch_failed(tmp_path):
    def broken(symbols, start, end):
        raise RuntimeError("yahoo down")

    statuses = update_prices(Store(tmp_path), ["THYAO"], SESSION, download=broken)
    assert statuses == {"THYAO": "fetch_failed"}


def test_rows_with_nan_close_are_dropped(tmp_path):
    frame = bars(["2026-11-12", "2026-11-13"])
    frame.loc["2026-11-12", "Close"] = np.nan
    store = Store(tmp_path)
    update_prices(store, ["THYAO"], SESSION, download=Recorder({"THYAO.IS": frame}))
    assert list(store.read_prices("THYAO").index) == [SESSION]


def test_rows_after_session_are_dropped(tmp_path):
    store = Store(tmp_path)
    recorder = Recorder({"THYAO.IS": bars(["2026-11-13", "2026-11-16"])})
    update_prices(store, ["THYAO"], SESSION, download=recorder)
    assert list(store.read_prices("THYAO").index) == [SESSION]
