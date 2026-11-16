"""The full list of BIST equity symbols, read from Is Yatirim's public page."""

from __future__ import annotations

import re
from typing import Callable

from jarvis.config import MIN_UNIVERSE_SIZE
from jarvis.http import http_get

ISYATIRIM_URL = (
    "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/"
    "Temel-Degerler-Ve-Oranlar.aspx"
)
_SYMBOL = re.compile(r"sirket-karti\.aspx\?hisse=([A-Z0-9]{3,6})\b")


class UniverseError(RuntimeError):
    """The symbol list could not be fetched or looks broken."""


def parse_symbols(html: str) -> list[str]:
    return sorted(set(_SYMBOL.findall(html)))


def fetch_symbols(get: Callable[[str], str] = http_get) -> list[str]:
    try:
        html = get(ISYATIRIM_URL)
    except Exception as error:
        raise UniverseError(f"Hisse listesi alınamadı: {error}") from error
    symbols = parse_symbols(html)
    if len(symbols) < MIN_UNIVERSE_SIZE:
        raise UniverseError(
            f"Hisse listesi şüpheli: {len(symbols)} sembol bulundu, "
            f"en az {MIN_UNIVERSE_SIZE} bekleniyordu."
        )
    return symbols
