"""Send the daily Telegram message from the files the data and training jobs wrote.

Usage:
    python -m jarvis.notify --data-dir data --dry-run    # print, send nothing
    python -m jarvis.notify --data-dir data              # needs TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from jarvis.calendar import last_completed_session
from jarvis.telegram import TelegramError, format_message, send

LOG = logging.getLogger("jarvis.notify")


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run(
    data_dir: Path | str,
    as_of: pd.Timestamp | None = None,
    *,
    dry_run: bool = False,
    sender: Callable[[str, str, str], object] = send,
) -> str:
    data_dir = Path(data_dir)
    day = last_completed_session(as_of).date().isoformat()
    text = format_message(
        _read(data_dir / "daily" / f"{day}.json"),
        _read(data_dir / "quality" / f"{day}.json"),
        _read(data_dir / "scorecard" / "scorecard.json"),
        _read(data_dir / "scorecard" / "simulated.json"),
        session=day,
    )
    if not dry_run:
        sender(
            os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            text,
        )
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    as_of = pd.Timestamp(arguments.as_of) if arguments.as_of else None
    try:
        text = run(arguments.data_dir, as_of, dry_run=arguments.dry_run)
    except TelegramError as error:
        LOG.error("%s", error)
        return 1
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
