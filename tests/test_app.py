"""Smoke tests for the Streamlit app: it runs end to end on fake prices (no internet)."""

import html
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import quantlab.data  # noqa: E402
import quantlab.earnings  # noqa: E402
import quantlab.macro  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _advanced() -> AppTest:
    """Open the app and switch to the advanced view (the simple one is the default)."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    return at.radio(key="view").set_value("Approfondita").run()


@pytest.fixture
def fake_prices(monkeypatch):
    """Replace the download with a seeded random walk of ~6 years."""

    def fake_load(tickers, start, end, cache_dir="data/"):
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2015-01-01", periods=1500)
        values = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))
        return pd.DataFrame({tickers[0]: values}, index=idx)

    monkeypatch.setattr(quantlab.data, "load_prices", fake_load)

    def fake_ohlcv(ticker, start, end, cache_dir="data/"):
        close = fake_load([ticker], start, end)[ticker]
        return pd.DataFrame({"Open": close * 0.999, "High": close * 1.01, "Low": close * 0.99,
                             "Close": close, "Volume": 1_000_000.0})

    monkeypatch.setattr(quantlab.data, "load_ohlcv", fake_ohlcv)
    fake_news = [{"title": "Notizia di prova su Apple e S&P 500", "publisher": "Test", "link": "https://example.com",
                  "published": pd.Timestamp("2026-01-01 10:00", tz="UTC")}]
    monkeypatch.setattr(quantlab.data, "latest_news", lambda query, count=5: fake_news)
    # macro data and earnings surprises: a calm, positive world (curve > 0, beats > 0)
    idx = pd.bdate_range("2014-01-01", periods=3000)
    monkeypatch.setattr(quantlab.macro, "load_macro", lambda name, cache_dir=None: pd.Series(1.5, index=idx))
    monkeypatch.setattr(quantlab.earnings, "load_earnings_surprises",
                        lambda ticker, cache_dir=None: pd.Series(4.0, index=idx[::63]))


def test_app_waits_for_the_button(fake_prices):
    at = _advanced()
    assert not at.exception
    assert any("Metti alla prova" in box.value for box in at.info)
    assert not at.table  # no results before the click


@pytest.mark.parametrize("strategy", ["Breakout", "Momentum 12-1", "Incrocio medie mobili", "Mean reversion"])
def test_app_gives_a_verdict_for_every_strategy(fake_prices, strategy):
    at = _advanced()
    at.sidebar.selectbox[0].select(strategy).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.table, "the metrics tables should be shown"
    assert any("operazioni in" in c.value for c in at.caption), "the trade chart caption should be shown"


def test_app_with_optimised_parameters_shows_walk_forward(fake_prices):
    at = _advanced()
    at.sidebar.toggle[0].set_value(True).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("Walk-forward" in m.value for m in at.markdown)


def test_app_rejects_invalid_parameters(fake_prices):
    at = _advanced()
    at.sidebar.selectbox[0].select("Incrocio medie mobili").run()
    at.sidebar.slider[0].set_value(100).run()  # fast 100
    at.sidebar.slider[1].set_value(50).run()  # slow 50 < fast
    at.sidebar.button[0].click().run()
    assert any("Parametri non validi" in e.value for e in at.error)


def test_app_custom_rule_gives_a_verdict(fake_prices):
    at = _advanced()
    at.radio(key="mode").set_value("Crea la tua regola").run()
    at.toggle(key="has_exit").set_value(True).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.table
    assert any("Prezzo > Media mobile 100g" in html.unescape(m.value) for m in at.markdown)


def test_app_custom_rule_against_a_number(fake_prices):
    at = _advanced()
    at.radio(key="mode").set_value("Crea la tua regola").run()
    at.selectbox(key="entry-left-name").set_value("return").run()
    at.radio(key="entry-kind").set_value("un numero").run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("Rendimento 20g > 0" in html.unescape(m.value) for m in at.markdown)


def test_app_custom_rule_with_two_conditions(fake_prices):
    at = _advanced()
    at.radio(key="mode").set_value("Crea la tua regola").run()
    at.toggle(key="entry_and").set_value(True).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("Prezzo > Media mobile 100g E z-score (distanza dalla media) 10g < -1" in html.unescape(m.value)
               for m in at.markdown)


def test_app_warns_after_several_attempts_on_the_same_ticker(fake_prices):
    at = _advanced()
    at.sidebar.button[0].click().run()
    assert not any("in questa sessione" in w.value for w in at.warning)  # first try: no warning
    at.sidebar.selectbox[0].select("Momentum 12-1").run()
    assert any("hai provato 2 strategie diverse" in w.value for w in at.warning)
    at.sidebar.toggle[0].set_value(True).run()  # grid of 4 combinations
    assert any("3 strategie diverse (6 combinazioni" in w.value for w in at.warning)


def test_app_cash_rate_changes_the_result(fake_prices):
    at = _advanced()
    at.number_input(key="cash_rate").set_value(0.0).run()
    at.sidebar.button[0].click().run()
    kpis = lambda: next(m.value for m in at.markdown if "Rendimento totale" in m.value)  # noqa: E731
    zero = kpis()
    at.number_input(key="cash_rate").set_value(5.0).run()
    assert not at.exception
    assert kpis() != zero


def test_app_shows_no_emoji(fake_prices):
    at = _advanced()
    at.sidebar.button[0].click().run()
    text = " ".join(m.value for m in at.markdown)
    assert not any(ch in text for ch in "✅❌⚖")


def test_app_shows_what_the_rule_says_today(fake_prices):
    at = _advanced()
    at.sidebar.button[0].click().run()
    assert any("Segnale alla chiusura" in m.value for m in at.markdown)
    assert any("DENTRO" in m.value or "FUORI" in m.value for m in at.markdown)
    assert any("non un consiglio di investimento" in c.value for c in at.caption)


def test_simple_view_is_the_default_and_answers_in_euros(fake_prices):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.sidebar.button  # no sidebar controls in the simple view
    at.button(key="simple_run").click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "1.000 € sarebbero diventati" in text
    assert "L'idea dice di" in text


@pytest.mark.parametrize("idea", ["Segui la tendenza", "Compra quando sfonda verso l'alto",
                                  "Compra dopo un forte calo", "Compra ciò che è salito nell'ultimo anno",
                                  "Esci quando il mercato ha paura", "Compra quando il mercato ha paura",
                                  "Esci quando la curva dei tassi si inverte",
                                  "Compra dopo trimestrali sopra le attese"])
def test_simple_view_works_for_every_idea(fake_prices, idea):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox(key="simple_idea").set_value(idea).run()
    at.button(key="simple_run").click().run()
    assert not at.exception
    assert any("sarebbero diventati" in m.value for m in at.markdown)


def test_simple_view_accepts_any_ticker(fake_prices):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox(key="simple_asset").set_value("Altro titolo...").run()
    at.text_input(key="simple_ticker").set_value("msft").run()
    at.button(key="simple_run").click().run()
    assert not at.exception
    assert any("<b>MSFT</b>" in m.value for m in at.markdown)


def test_news_are_shown_as_context_only(fake_prices):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="simple_run").click().run()
    assert any("Notizia di prova" in m.value for m in at.markdown)
    assert any("le notizie non entrano nei calcoli" in c.value for c in at.caption)


def test_advanced_rule_on_a_market_factor(fake_prices):
    at = _advanced()
    at.radio(key="mode").set_value("Crea la tua regola").run()
    at.selectbox(key="entry-left-source").set_value("VIX (paura del mercato)").run()
    at.selectbox(key="entry-left-name").set_value("price").run()
    at.selectbox(key="entry-op").set_value("<").run()
    at.radio(key="entry-kind").set_value("un numero").run()
    at.number_input(key="entry-number").set_value(120.0).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("VIX < 120" in html.unescape(m.value) for m in at.markdown)


@pytest.mark.parametrize("factor, label", [("Inflazione USA (% annuo)", "Inflazione"),
                                           ("Sorpresa dell'ultima trimestrale (%)", "Sorpresa utili")])
def test_advanced_rule_on_macro_and_earnings(fake_prices, factor, label):
    at = _advanced()
    at.radio(key="mode").set_value("Crea la tua regola").run()
    at.selectbox(key="entry-left-source").set_value(factor).run()
    assert any("giorni dopo" in c.value or "dopo l'annuncio" in c.value for c in at.caption)
    at.selectbox(key="entry-left-name").set_value("price").run()
    at.radio(key="entry-kind").set_value("un numero").run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any(f"{label} > 0" in html.unescape(m.value) for m in at.markdown)


def test_simple_view_explains_when_an_idea_does_not_apply(fake_prices, monkeypatch):
    def no_earnings(ticker, cache_dir=None):
        raise ValueError("No earnings history for SPY")

    monkeypatch.setattr(quantlab.earnings, "load_earnings_surprises", no_earnings)
    import streamlit as st
    st.cache_data.clear()  # forget the (fake) earnings cached by earlier tests
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox(key="simple_idea").set_value("Compra dopo trimestrali sopra le attese").run()
    at.button(key="simple_run").click().run()
    assert not at.exception
    assert any("non si può applicare" in e.value for e in at.error)


def test_scanner_view_lists_every_asset_with_a_corrected_verdict(fake_prices):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio(key="view").set_value("Scanner").run()
    at.multiselect(key="scan_assets").set_value(["Apple", "Bitcoin"]).run()
    at.text_input(key="scan_extra").set_value("msft").run()
    at.button(key="scan_run").click().run()
    assert not at.exception
    summary = at.table[0].value
    assert list(summary.index) == ["Apple", "Bitcoin", "MSFT"]
    assert set(summary["Idee provate"]) <= {7, 8}  # earnings may not apply
    assert any("intervallo richiesto" in m.value for m in at.markdown)


def test_demo_account_before_the_first_run_explains_when_it_starts(fake_prices, tmp_path, monkeypatch):
    import shutil

    shutil.copy(Path(APP).parents[1] / "paper" / "config.json", tmp_path / "config.json")
    monkeypatch.setenv("QUANTLAB_PAPER_DIR", str(tmp_path))
    import sys
    sys.modules.pop("paper_view", None)  # re-read the folder setting
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio(key="view").set_value("Conto demo").run()
    assert not at.exception
    assert any("prima esecuzione notturna" in i.value for i in at.info)


def test_demo_account_shows_value_slots_and_trades(fake_prices, tmp_path, monkeypatch):
    import json
    import sys

    config = {"start_cash": 1000, "commission": 0.001, "slippage": 0.0005, "cash_rate": 0.02,
              "slots": [{"ticker": "SPY", "idea": "Segui la tendenza"}]}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "state.json").write_text(json.dumps({"start": "2026-10-07", "slots": []}), encoding="utf-8")
    pd.DataFrame([{"date": "2026-10-07", "ticker": "SPY", "idea": "Segui la tendenza", "position": 1,
                   "value": 1000.0, "bench_value": 1000.0},
                  {"date": "2026-10-08", "ticker": "SPY", "idea": "Segui la tendenza", "position": 1,
                   "value": 1012.0, "bench_value": 1010.0}]).to_csv(tmp_path / "values.csv", index=False)
    pd.DataFrame([{"date": "2026-10-07", "ticker": "SPY", "idea": "Segui la tendenza", "price": 670.0,
                   "action": "buy", "shares": 1.49, "value": 1000.0, "cost": 1.5}]).to_csv(
        tmp_path / "trades.csv", index=False)
    monkeypatch.setenv("QUANTLAB_PAPER_DIR", str(tmp_path))
    sys.modules.pop("paper_view", None)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio(key="view").set_value("Conto demo").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "1.012 €" in text and "+1,2%" in text
    assert at.table[0].value.loc["SPY", "Posizione"] == "DENTRO"


def test_scanner_period_controls_move_the_test_start(fake_prices):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio(key="view").set_value("Scanner").run()
    at.multiselect(key="scan_assets").set_value(["Apple"]).run()
    at.slider(key="scan_share").set_value(50).run()
    assert any("Giudizio sul periodo" in c.value for c in at.caption)
    at.button(key="scan_run").click().run()
    assert not at.exception
    assert at.table[0].value.loc["Apple", "Periodo di prova"]  # dates shown for the chosen period


def test_scanner_refuses_a_test_period_without_one_year_of_history(fake_prices):
    from datetime import date

    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio(key="view").set_value("Scanner").run()
    today = date.today()
    at.date_input(key="scan_start").set_value(date(today.year - 4, today.month, 1)).run()
    at.slider(key="scan_share").set_value(90).run()  # 4 years x 10% = less than a year of history
    assert any("meno di 1 anno di storia" in w.value for w in at.warning)
    assert not at.table  # nothing analysed


def test_demo_replay_runs_the_account_over_a_past_period(fake_prices):
    import sys

    sys.modules.pop("paper_view", None)  # default paper folder (an earlier test may have changed it)
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.radio(key="view").set_value("Conto demo").run()
    from datetime import date
    at.date_input(key="replay_start").set_value(date(2019, 1, 2)).run()  # inside the fake prices (2015-2020)
    at.date_input(key="replay_end").set_value(date(2020, 6, 30)).run()
    at.button(key="replay_run").click().run()
    assert not at.exception
    assert any("Controllo backtest" in str(t.value.columns.tolist()) for t in at.table)
    assert any("Rendimento" in m.value for m in at.markdown)
