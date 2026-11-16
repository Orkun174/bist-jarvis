"""One shared HTTP GET with a browser User-Agent, timeout and retries."""

from __future__ import annotations

import time

import requests

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)


def http_get(url: str, *, timeout: float = 40.0, attempts: int = 3) -> str:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text
        except requests.RequestException as error:
            last_error = error
            time.sleep(2 * (attempt + 1))
    raise last_error
