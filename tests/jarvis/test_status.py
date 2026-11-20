import json

import pandas as pd

from jarvis.status import daily_done, main

SESSION = pd.Timestamp("2026-11-19")


def write(root, payload):
    (root / "daily").mkdir(exist_ok=True)
    (root / "daily" / "2026-11-19.json").write_text(json.dumps(payload), encoding="utf-8")


def test_not_done_without_a_daily_result(tmp_path):
    assert daily_done(tmp_path, SESSION) is False


def test_not_done_when_no_model_produced_a_result(tmp_path):
    write(tmp_path, {"session": "2026-11-19", "model": {"model_id": None}})
    assert daily_done(tmp_path, SESSION) is False


def test_not_done_when_stale_data_left_nothing_to_rank(tmp_path):
    """The job still writes a result with yesterday's model when no stock is eligible."""
    write(tmp_path, {"session": "2026-11-19", "eligible": 0, "model": {"model_id": "2026-11-18-abc"}})
    assert daily_done(tmp_path, SESSION) is False


def test_done_when_a_model_ranked_the_session(tmp_path):
    write(tmp_path, {"session": "2026-11-19", "eligible": 89, "model": {"model_id": "2026-11-19-abc"}})
    assert daily_done(tmp_path, SESSION) is True


def test_cli_reports_the_previous_session_before_the_open(tmp_path, capsys):
    write(tmp_path, {"session": "2026-11-19", "eligible": 89, "model": {"model_id": "m"}})
    # 09:17 TRT on Friday 2026-11-20, before the market opens: Thursday is the last session.
    main(["--data-dir", str(tmp_path), "--as-of", "2026-11-20T09:17:00+03:00"])
    assert capsys.readouterr().out == "session=2026-11-19\ndone=true\n"
    # 04:20 TRT the same morning (where GitHub's delayed evening run lands): also Thursday.
    main(["--data-dir", str(tmp_path), "--as-of", "2026-11-20T04:20:00+03:00"])
    assert capsys.readouterr().out == "session=2026-11-19\ndone=true\n"
