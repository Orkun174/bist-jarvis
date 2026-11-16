import numpy as np
import pandas as pd

from jarvis.features import FEATURES
from jarvis.model import DEFAULT_RECIPE, Recipe, fit, load_model, predict, save_model

SMALL = Recipe(
    params={"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1},
    features=FEATURES,
)


def synthetic_rows(n=2000, seed=0):
    """Target depends on one feature only, so a working model separates it well."""
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=list(FEATURES))
    data["target"] = (data["CMF_20"] + rng.normal(0, 0.1, n) > 0).astype(float)
    return data


def auc(labels, scores):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = labels == 1
    n_pos, n_neg = positives.sum(), (~positives).sum()
    return (ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def test_fit_predict_learns_separable_signal():
    train, test = synthetic_rows(seed=0), synthetic_rows(seed=1)
    model = fit(SMALL, train)
    probabilities = predict(model, SMALL, test)
    assert probabilities.index.equals(test.index)
    assert probabilities.between(0, 1).all()
    assert auc(test["target"].to_numpy(), probabilities.to_numpy()) > 0.9


def test_predict_is_deterministic():
    train, test = synthetic_rows(seed=0), synthetic_rows(seed=1)
    first = predict(fit(SMALL, train), SMALL, test)
    second = predict(fit(SMALL, train), SMALL, test)
    pd.testing.assert_series_equal(first, second)


def test_missing_macro_values_are_accepted():
    train, test = synthetic_rows(seed=0), synthetic_rows(seed=1)
    train.loc[::3, "macro_GOLD_1d_lag"] = np.nan
    test["macro_GOLD_1d_lag"] = np.nan
    assert predict(fit(SMALL, train), SMALL, test).notna().all()


def test_save_load_roundtrip_gives_same_predictions(tmp_path):
    train, test = synthetic_rows(seed=0), synthetic_rows(seed=1)
    model = fit(SMALL, train)
    save_model(tmp_path, model, SMALL, {"trained_through": "2026-11-13", "train_rows": 2000})
    loaded, recipe, meta = load_model(tmp_path)
    assert recipe == SMALL
    assert meta["trained_through"] == "2026-11-13" and meta["train_rows"] == 2000
    assert meta["recipe_id"] == SMALL.recipe_id
    pd.testing.assert_series_equal(predict(model, SMALL, test), predict(loaded, recipe, test))


def test_save_keeps_previous(tmp_path):
    train = synthetic_rows(seed=0)
    save_model(tmp_path, fit(SMALL, train), SMALL, {"trained_through": "2026-11-12"})
    save_model(tmp_path, fit(SMALL, train), SMALL, {"trained_through": "2026-11-13"})
    assert (tmp_path / "models" / "previous.json").exists()
    assert (tmp_path / "models" / "previous.meta.json").read_text(encoding="utf-8").count("2026-11-12") == 1
    assert load_model(tmp_path)[2]["trained_through"] == "2026-11-13"


def test_load_missing_returns_none(tmp_path):
    assert load_model(tmp_path) is None


def test_recipe_roundtrip_and_id():
    other = Recipe(params={**SMALL.params, "max_depth": 5}, features=FEATURES)
    assert other.recipe_id != SMALL.recipe_id
    assert Recipe.from_dict(SMALL.to_dict()) == SMALL
    assert len(SMALL.recipe_id) == 10
    assert DEFAULT_RECIPE.features == FEATURES and DEFAULT_RECIPE.train_years == 3


def test_explain_names_the_driving_feature_first():
    from jarvis.model import explain

    train, test = synthetic_rows(seed=0), synthetic_rows(seed=1)
    model = fit(SMALL, train)
    probabilities = predict(model, SMALL, test)
    strongest = test.loc[[probabilities.idxmax(), probabilities.idxmin()]]
    reasons = explain(model, SMALL, strongest, top=2)
    assert len(reasons) == 2 and all(len(item) == 2 for item in reasons)
    assert reasons[0][0] == {"feature": "CMF_20", "direction": "up"}
    assert reasons[1][0] == {"feature": "CMF_20", "direction": "down"}


def test_explain_empty_rows():
    from jarvis.model import explain

    model = fit(SMALL, synthetic_rows(seed=0))
    assert explain(model, SMALL, synthetic_rows(seed=1).iloc[0:0]) == []


def two_factor_rows(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=list(FEATURES))
    signal = data["CMF_20"] + 0.6 * data["ADX_14"] + 0.6 * data["macro_VIX_1d_lag"]
    data["target"] = (signal + rng.normal(0, 0.1, n) > 0).astype(float)
    return data


def test_explain_can_be_limited_to_stock_specific_features():
    from jarvis.features import LOCAL_FEATURES
    from jarvis.model import explain

    model = fit(SMALL, two_factor_rows(seed=0))
    rows = two_factor_rows(seed=1).head(50)
    for reasons in explain(model, SMALL, rows, top=3, only=LOCAL_FEATURES):
        assert all(reason["feature"] in LOCAL_FEATURES for reason in reasons)


def test_explain_relative_to_peers_names_what_sets_the_stock_apart():
    """Everyone shares the same strong CMF; only ADX separates the pick from its peers."""
    from jarvis.features import LOCAL_FEATURES
    from jarvis.model import explain

    model = fit(SMALL, two_factor_rows(seed=0))
    peers = two_factor_rows(seed=1).head(60).copy()
    peers["CMF_20"] = 2.0
    peers["ADX_14"] = np.linspace(-2.0, 2.0, len(peers))
    pick = peers.tail(1)
    alone = explain(model, SMALL, pick, top=1, only=LOCAL_FEATURES)
    relative = explain(model, SMALL, pick, top=1, only=LOCAL_FEATURES, reference=peers)
    assert alone[0][0]["feature"] == "CMF_20"
    assert relative[0][0] == {"feature": "ADX_14", "direction": "up"}
