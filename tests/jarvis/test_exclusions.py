import pandas as pd

from jarvis.exclusions import excluded_on, load_exclusions

CSV = """ticker,reason,start,end,source
KARTN,vbts,2026-11-10,2026-12-09,test
ATEKS,watchlist_market,2026-01-01,,test
OLDIE,manual,2026-01-01,2026-02-01,test
"""


def load(tmp_path, text=CSV):
    path = tmp_path / "exclusions.csv"
    path.write_text(text, encoding="utf-8")
    return load_exclusions(path)


def test_exclusion_active_window(tmp_path):
    table = load(tmp_path)
    assert excluded_on(table, pd.Timestamp("2026-11-13")) == {"KARTN": "vbts", "ATEKS": "watchlist_market"}
    assert excluded_on(table, pd.Timestamp("2026-11-10")) == {"KARTN": "vbts", "ATEKS": "watchlist_market"}
    assert excluded_on(table, pd.Timestamp("2026-12-09"))["KARTN"] == "vbts"
    assert "KARTN" not in excluded_on(table, pd.Timestamp("2026-12-10"))
    assert "KARTN" not in excluded_on(table, pd.Timestamp("2026-11-09"))


def test_header_only_file_excludes_nobody(tmp_path):
    table = load(tmp_path, "ticker,reason,start,end,source\n")
    assert excluded_on(table, pd.Timestamp("2026-11-13")) == {}


def test_missing_file_excludes_nobody(tmp_path):
    table = load_exclusions(tmp_path / "nope.csv")
    assert list(table.columns) == ["ticker", "reason", "start", "end", "source"]
    assert excluded_on(table, pd.Timestamp("2026-11-13")) == {}
