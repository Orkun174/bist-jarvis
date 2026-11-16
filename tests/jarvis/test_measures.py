from pathlib import Path

import pandas as pd

from jarvis.measures import (
    GCM_URL, KAP_URL, fetch_exclusions, parse_gcm_measures, parse_kap_markets,
)

FIXTURES = Path(__file__).parent / "fixtures"
KAP = (FIXTURES / "kap_pazarlar_sample.html").read_text(encoding="utf-8")
GCM = (FIXTURES / "gcm_tedbir_sample.html").read_text(encoding="utf-8")
SESSION = pd.Timestamp("2026-11-13")


def test_parse_kap_markets():
    assert parse_kap_markets(KAP) == {
        "THYAO": "YILDIZ PAZAR",
        "GARAN": "YILDIZ PAZAR",
        "ADEL": "ANA PAZAR",
        "ISATR": "ANA PAZAR",
        "ISBTR": "ANA PAZAR",
        "ATEKS": "YAKIN İZLEME PAZARI",
        "USAK": "YAKIN İZLEME PAZARI",
    }


def test_parse_gcm_measures():
    table = parse_gcm_measures(GCM)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["ticker"] == "KARTN"
    assert row["start"] == pd.Timestamp("2026-11-10")
    assert row["end"] == pd.Timestamp("2026-12-09")
    assert row["measures"] == "AÇIĞA SATIŞ YASAĞI"


def big_kap(extra=""):
    """A KAP page with enough codes to pass the sanity floor."""
    codes = "".join(
        '{\\"stockCode\\":\\"T%03d\\",\\"title\\":\\"x\\"},' % i for i in range(300)
    )
    return (
        '\\"marketName\\":\\"ANA PAZAR\\",\\"marketDetailContentList\\":['
        + codes
        + '{\\"stockCode\\":\\"GOOD\\"}]}'
        + '{\\"marketName\\":\\"YAKIN İZLEME PAZARI\\",\\"marketDetailContentList\\":[{\\"stockCode\\":\\"WATCH\\"}]}'
        + extra
    )


def pages(kap, gcm):
    def get(url):
        page = {KAP_URL: kap, GCM_URL: gcm}[url]
        if isinstance(page, Exception):
            raise page
        return page

    return get


def test_fetch_marks_watchlist_and_unknown_market():
    rows, status = fetch_exclusions(["GOOD", "WATCH", "GHOST", "KARTN"], SESSION, get=pages(big_kap(), GCM))
    assert status == {"kap_markets": "ok", "measures": "ok"}
    reasons = {(r.ticker, r.reason) for r in rows.itertuples()}
    assert ("WATCH", "watchlist_market") in reasons
    assert ("GHOST", "unknown_market") in reasons
    assert ("KARTN", "vbts") in reasons
    assert "GOOD" not in set(rows["ticker"])
    assert list(rows.columns) == ["ticker", "reason", "start", "end", "source"]
    market_rows = rows[rows["reason"] != "vbts"]
    assert (market_rows["start"] == SESSION).all() and (market_rows["end"] == SESSION).all()
    vbts = rows[rows["reason"] == "vbts"].iloc[0]
    assert vbts["start"] == pd.Timestamp("2026-11-10") and vbts["end"] == pd.Timestamp("2026-12-09")


def test_fetch_source_failure_is_reported_not_raised():
    rows, status = fetch_exclusions(["GOOD"], SESSION, get=pages(OSError("down"), OSError("down")))
    assert status == {"kap_markets": "failed", "measures": "failed"}
    assert rows.empty
    assert list(rows.columns) == ["ticker", "reason", "start", "end", "source"]


def test_kap_too_few_codes_is_failed():
    rows, status = fetch_exclusions(["THYAO", "GHOST"], SESSION, get=pages(KAP, GCM))
    assert status["kap_markets"] == "failed"
    assert set(rows["reason"]) == {"vbts"}


def test_measures_page_without_table_is_failed():
    rows, status = fetch_exclusions(["GOOD"], SESSION, get=pages(big_kap(), "<html>bakım</html>"))
    assert status == {"kap_markets": "ok", "measures": "failed"}
