"""Automatic exclusion sources.

- KAP "Pazarlar" page (official): which market every ticker trades on.
- A brokerage page listing shares under Borsa Istanbul measures (VBTS). Its
  completeness is not guaranteed, so it is one protection among several.

Neither source may break the daily run: a failure is reported, never raised.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

import pandas as pd
from bs4 import BeautifulSoup

from jarvis.config import MIN_UNIVERSE_SIZE
from jarvis.exclusions import COLUMNS, empty_exclusions
from jarvis.http import http_get

LOG = logging.getLogger("jarvis.measures")

KAP_URL = "https://www.kap.org.tr/tr/Pazarlar"
GCM_URL = "https://www.gcmyatirim.com.tr/borsa/tedbir-kapsamindaki-pay-senetleri"
ALLOWED_MARKETS = ("YILDIZ PAZAR", "ANA PAZAR", "ALT PAZAR")

_MARKET = re.compile(r'"marketName":"([^"]+)","marketDetailContentList":\[(.*?)\]', re.S)
_STOCK_CODE = re.compile(r'"stockCode":"([^"]*)"')


def parse_kap_markets(html: str) -> dict[str, str]:
    """Ticker -> market name from the JSON embedded in the KAP page."""
    text = html.replace('\\"', '"')
    markets: dict[str, str] = {}
    for name, body in _MARKET.findall(text):
        for raw in _STOCK_CODE.findall(body):
            for code in raw.split(","):
                code = code.strip().upper()
                if code:
                    markets.setdefault(code, name)
    return markets


def parse_gcm_measures(html: str) -> pd.DataFrame:
    """Rows of the measures table: ticker, start, end, measures (';'-joined)."""
    table = BeautifulSoup(html, "html.parser").find("table", class_="gcmTable")
    if table is None:
        raise ValueError("Tedbir tablosu bulunamadı.")
    rows = [
        [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
        for row in table.find_all("tr")
    ]
    if not rows or len(rows[0]) < 4:
        raise ValueError("Tedbir tablosunun başlığı beklenen biçimde değil.")
    header = rows[0]
    records = []
    for cells in rows[1:]:
        if len(cells) != len(header) or not cells[0]:
            continue
        measures = [name for name, value in zip(header[3:], cells[3:]) if value]
        records.append(
            {
                "ticker": cells[0].upper(),
                "start": pd.to_datetime(cells[1], format="%d.%m.%Y"),
                "end": pd.to_datetime(cells[2], format="%d.%m.%Y"),
                "measures": ";".join(measures),
            }
        )
    return pd.DataFrame(records, columns=["ticker", "start", "end", "measures"])


def fetch_exclusions(
    tickers: list[str],
    session: pd.Timestamp,
    get: Callable[[str], str] = http_get,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Exclusion rows for `session` plus a per-source status ("ok" / "failed")."""
    session = pd.Timestamp(session).normalize()
    status = {"kap_markets": "ok", "measures": "ok"}
    records: list[dict] = []

    try:
        markets = parse_kap_markets(get(KAP_URL))
        if len(markets) < MIN_UNIVERSE_SIZE:
            raise ValueError(f"KAP sayfasından yalnızca {len(markets)} kod çıktı.")
        for ticker in tickers:
            market = markets.get(ticker)
            if market in ALLOWED_MARKETS:
                continue
            records.append(
                {
                    "ticker": ticker,
                    "reason": "unknown_market" if market is None else "watchlist_market",
                    "start": session,
                    "end": session,
                    "source": "kap:" + (market or "listede yok"),
                }
            )
    except Exception as error:
        LOG.warning("KAP pazar bilgisi alınamadı: %s", error)
        status["kap_markets"] = "failed"

    try:
        measures = parse_gcm_measures(get(GCM_URL))
        for row in measures.itertuples():
            records.append(
                {
                    "ticker": row.ticker,
                    "reason": "vbts",
                    "start": row.start,
                    "end": row.end,
                    "source": "gcm:" + row.measures,
                }
            )
    except Exception as error:
        LOG.warning("Tedbir listesi alınamadı: %s", error)
        status["measures"] = "failed"

    if not records:
        return empty_exclusions(), status
    return pd.DataFrame(records, columns=COLUMNS), status
