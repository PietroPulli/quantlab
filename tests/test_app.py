"""Smoke tests for the Streamlit app: it runs end to end on fake prices (no internet)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import quantlab.data  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture
def fake_prices(monkeypatch):
    """Replace the download with a seeded random walk of ~6 years."""

    def fake_load(tickers, start, end, cache_dir="data/"):
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2015-01-01", periods=1500)
        values = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))
        return pd.DataFrame({tickers[0]: values}, index=idx)

    monkeypatch.setattr(quantlab.data, "load_prices", fake_load)


def test_app_waits_for_the_button(fake_prices):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Metti alla prova" in box.value for box in at.info)
    assert not at.table  # no results before the click


@pytest.mark.parametrize("strategy", ["Breakout", "Momentum 12-1", "Incrocio medie mobili", "Mean reversion"])
def test_app_gives_a_verdict_for_every_strategy(fake_prices, strategy):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox[0].select(strategy).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.table, "the metrics tables should be shown"


def test_app_with_optimised_parameters_shows_walk_forward(fake_prices):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.toggle[0].set_value(True).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("Walk-forward" in s.value for s in at.subheader)


def test_app_rejects_invalid_parameters(fake_prices):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox[0].select("Incrocio medie mobili").run()
    at.sidebar.slider[0].set_value(100).run()  # fast 100
    at.sidebar.slider[1].set_value(50).run()  # slow 50 < fast
    at.sidebar.button[0].click().run()
    assert any("Parametri non validi" in e.value for e in at.error)


def test_app_custom_rule_gives_a_verdict(fake_prices):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio(key="mode").set_value("Crea la tua regola").run()
    at.toggle(key="has_exit").set_value(True).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.table
    assert any("Prezzo > Media mobile 100g" in m.value for m in at.markdown)


def test_app_custom_rule_against_a_number(fake_prices):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio(key="mode").set_value("Crea la tua regola").run()
    at.selectbox(key="entry-left-name").set_value("return").run()
    at.radio(key="entry-kind").set_value("un numero").run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("Rendimento 20g > 0" in m.value for m in at.markdown)


def test_app_custom_rule_with_two_conditions(fake_prices):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio(key="mode").set_value("Crea la tua regola").run()
    at.toggle(key="entry_and").set_value(True).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("Prezzo > Media mobile 100g E z-score (distanza dalla media) 10g < -1" in m.value
               for m in at.markdown)


def test_app_warns_after_several_attempts_on_the_same_ticker(fake_prices):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.button[0].click().run()
    assert not any("in questa sessione" in w.value for w in at.warning)  # first try: no warning
    at.sidebar.selectbox[0].select("Momentum 12-1").run()
    assert any("hai provato 2 strategie diverse" in w.value for w in at.warning)
    at.sidebar.toggle[0].set_value(True).run()  # grid of 4 combinations
    assert any("3 strategie diverse (6 combinazioni" in w.value for w in at.warning)
