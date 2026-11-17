import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.jarvis.test_desktop import data  # noqa: F401  (fixture)

APP = str(Path(__file__).resolve().parents[2] / "jarvis_app.py")


def test_three_screens_render_and_a_pick_opens_its_stock(data, monkeypatch):  # noqa: F811
    root, names, _ = data
    monkeypatch.setenv("JARVIS_DATA_DIR", str(root))
    monkeypatch.setattr("jarvis_app.cached_news", lambda ticker: [], raising=False)
    app = AppTest.from_file(APP, default_timeout=120).run()
    assert not app.exception
    assert f"1. {names[0]}" in app.text[0].value

    app.button[0].click().run()
    assert not app.exception
    assert app.radio[0].value == "Hisse araştır" and app.selectbox[0].value == names[0]
    assert any(metric.value == "1 / 6" for metric in app.metric)

    app.selectbox[0].set_value("THIN").run()
    assert not app.exception and "Düşük hacim" in app.warning[0].value

    app.radio[0].set_value("Karne ve model").run()
    assert not app.exception and len(app.dataframe) >= 1


def test_without_any_data_the_app_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path / "nothing"))
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception and "Veri yok" in app.error[0].value
