from pathlib import Path

import pytest

from jarvis.universe import UniverseError, fetch_symbols, parse_symbols

SAMPLE = (Path(__file__).parent / "fixtures" / "isyatirim_sample.html").read_text(encoding="utf-8")


def test_parse_symbols_unique_sorted():
    assert parse_symbols(SAMPLE) == ["A1CAP", "AAGYO", "ACSEL", "ADEL", "THYAO"]


def test_fetch_symbols_raises_when_too_few():
    with pytest.raises(UniverseError):
        fetch_symbols(get=lambda url: SAMPLE)


def test_fetch_symbols_ok():
    html = "".join(f'<a href="sirket-karti.aspx?hisse=T{i:03d}">x</a>' for i in range(300))
    result = fetch_symbols(get=lambda url: html)
    assert len(result) == 300
    assert result[0] == "T000"


def test_fetch_symbols_wraps_network_errors():
    def broken(url):
        raise OSError("connection reset")

    with pytest.raises(UniverseError):
        fetch_symbols(get=broken)
