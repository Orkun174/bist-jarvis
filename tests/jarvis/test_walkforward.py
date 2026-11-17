import pandas as pd
import pytest

import jarvis.walkforward as walkforward
from jarvis.dataset import build_rows
from jarvis.features import FEATURES
from jarvis.model import Recipe
from jarvis.walkforward import walk_forward
from tests.jarvis.synthetic import CALENDAR, SESSION, build_market

RECIPE = Recipe(params={"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1}, features=FEATURES)


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    store, names, macros = build_market(tmp_path_factory.mktemp("wf"), tickers=8)
    return build_rows(store, macros, names, SESSION)


@pytest.fixture(autouse=True)
def small_minimum(monkeypatch):
    monkeypatch.setattr(walkforward, "MIN_TRAIN_ROWS", 200)


def test_walk_forward_predicts_every_eligible_row_in_range(rows):
    start, end = CALENDAR[-25], CALENDAR[-6]
    result = walk_forward(rows, RECIPE, CALENDAR, start, end)
    assert list(result.columns) == ["date", "ticker", "prob", "fwd_excess", "VIX_Percentile", "fwd_excess_open"]
    assert result["date"].min() == start and result["date"].max() == end
    expected = rows[(rows["date"] >= start) & (rows["date"] <= end) & rows["eligible_pit"]]
    assert len(result) == len(expected)
    assert result["prob"].between(0, 1).all()


def test_walk_forward_never_trains_on_future(rows, monkeypatch):
    seen = []
    original = walkforward.fit

    def spy(recipe, train):
        seen.append(train["label_end"].max())
        return original(recipe, train)

    monkeypatch.setattr(walkforward, "fit", spy)
    start, end = CALENDAR[-25], CALENDAR[-6]
    walk_forward(rows, RECIPE, CALENDAR, start, end, refit_every=5)
    blocks = [CALENDAR[-25 + 5 * i] for i in range(4)]
    assert len(seen) == 4
    for block_start, latest_label in zip(blocks, seen):
        position = CALENDAR.get_loc(block_start)
        assert latest_label <= CALENDAR[position - 7]


def test_walk_forward_refits_on_schedule(rows, monkeypatch):
    calls = []
    original = walkforward.fit
    monkeypatch.setattr(walkforward, "fit", lambda r, t: calls.append(1) or original(r, t))
    walk_forward(rows, RECIPE, CALENDAR, CALENDAR[-25], CALENDAR[-6], refit_every=5)
    assert len(calls) == 4
    calls.clear()
    walk_forward(rows, RECIPE, CALENDAR, CALENDAR[-25], CALENDAR[-6], refit_every=20)
    assert len(calls) == 1


def test_walk_forward_skips_when_too_few_rows(rows, monkeypatch):
    monkeypatch.setattr(walkforward, "MIN_TRAIN_ROWS", 10**9)
    result = walk_forward(rows, RECIPE, CALENDAR, CALENDAR[-25], CALENDAR[-6])
    assert result.empty
    assert list(result.columns) == ["date", "ticker", "prob", "fwd_excess", "VIX_Percentile", "fwd_excess_open"]


def test_walk_forward_is_deterministic(rows):
    first = walk_forward(rows, RECIPE, CALENDAR, CALENDAR[-15], CALENDAR[-6])
    second = walk_forward(rows, RECIPE, CALENDAR, CALENDAR[-15], CALENDAR[-6])
    pd.testing.assert_frame_equal(first, second)
