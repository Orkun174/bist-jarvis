from jarvis import config


def test_thresholds_match_spec():
    assert config.TRT == "Europe/Istanbul"
    assert config.CALENDAR == "XIST"
    assert config.HISTORY_YEARS == 5
    assert config.CROSS_CHECK_TOLERANCE == 0.01
    assert config.MAX_QUARANTINE_RATIO == 0.20
    assert config.MIN_HISTORY_SESSIONS == 252
    assert config.LIQUIDITY_WINDOW == 20
    assert config.MIN_MEDIAN_TURNOVER_TRY == 50_000_000
    assert config.LIMIT_MOVE == 0.095
    assert config.MAX_LIMIT_DAYS == 5
    assert config.SUSPECT_JUMP == 0.25
    assert config.MIN_UNIVERSE_SIZE == 300
    assert config.OHLCV == ["Open", "High", "Low", "Close", "Volume"]
