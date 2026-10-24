"""Tests for quantlab.agent: real calculations on fake prices, and a fake model that asks for tools."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantlab import agent
from quantlab.ideas import IDEAS

TODAY = date(2024, 12, 31)


def _prices(ticker: str, start: str, end: str) -> pd.Series:
    """Fake closes from 2010: 'UP' grows steadily, 'WAVY' goes up and down, 'NEW' starts only in 2020."""
    days = pd.bdate_range("2010-01-01", "2024-12-31")
    rng = np.random.default_rng(len(ticker))
    drift = {"UP": 0.0004, "WAVY": 0.0001, "NEW": 0.0003}.get(ticker)
    if drift is None:
        raise ValueError(f"ticker sconosciuto: {ticker}")
    prices = pd.Series(100 * np.cumprod(1 + drift + rng.normal(0, 0.01, len(days))), index=days)
    if ticker == "NEW":
        prices = prices[prices.index >= "2020-01-01"]
    return prices[(prices.index >= start) & (prices.index < end)]


def _toolbox(**kwargs) -> agent.Toolbox:
    return agent.Toolbox(_prices, factor=lambda name, ticker="": pd.Series(dtype=float), today=TODAY, **kwargs)


def test_pct_writes_italian_numbers_with_a_sign():
    assert agent.pct(0.0981) == "+9,8%"
    assert agent.pct(-0.5) == "-50,0%"


def test_compare_uses_the_same_period_for_every_ticker():
    result = _toolbox().run("confronta_titoli", {"tickers": ["UP", "NEW"], "dal": "2010-01-01"})
    assert result["periodo_comune"].startswith("2020-01")  # NEW has no prices before 2020
    assert set(result["titoli"]) == {"UP", "NEW"}
    assert result["titoli"]["UP"]["perdita_massima"].startswith("-")


def test_compare_numbers_match_the_metrics_library():
    from quantlab import metrics

    result = _toolbox().run("confronta_titoli", {"tickers": ["UP"], "dal": "2015-01-01", "al": "2020-01-01"})
    returns = _prices("UP", "2015-01-01", "2020-01-01").pct_change().dropna()
    assert result["titoli"]["UP"]["rendimento_annuo"] == agent.pct(metrics.annualized_return(returns))


def test_an_unknown_ticker_is_reported_not_raised():
    result = _toolbox().run("confronta_titoli", {"tickers": ["UP", "NONESISTE"]})
    assert "UP" in result["titoli"]
    assert "NONESISTE" in result["titoli_senza_dati"]
    assert "errore" in _toolbox().run("rendimenti_per_anno", {"ticker": "NONESISTE"})
    assert "errore" in _toolbox().run("strumento_inventato", {})


def test_a_bad_date_falls_back_to_the_default():
    result = _toolbox().run("confronta_titoli", {"tickers": ["UP"], "dal": "l'anno scorso"})
    assert result["periodo_comune"].startswith("2010")  # 20 years back, limited by the data


def test_yearly_returns_one_line_per_year():
    out = _toolbox().run("rendimenti_per_anno", {"ticker": "UP", "dal": "2020-01-01"})["rendimenti"]
    assert list(out) == ["2020", "2021", "2022", "2023", "2024"]


def test_ideas_test_gives_a_verdict_for_every_idea_that_applies():
    ideas = {name: IDEAS[name] for name in ["Segui la tendenza", "Compra ciò che è salito nell'ultimo anno"]}
    result = _toolbox(ideas=ideas, n_bootstrap=50).run("prova_le_idee", {"ticker": "UP"})
    assert set(result["idee"]) == set(ideas)
    assert all(idea["verdetto"] in agent.VERDICTS.values() for idea in result["idee"].values())


def test_results_are_remembered():
    calls = []
    box = agent.Toolbox(lambda *a: calls.append(a) or _prices(*a), factor=None, today=TODAY)
    box.run("rendimenti_per_anno", {"ticker": "UP"})
    box.run("rendimenti_per_anno", {"ticker": "UP"})
    assert len(calls) == 1


def _model(replies: list[dict]):
    """A fake model that answers with `replies` in order and keeps what it was sent."""
    seen = []

    def ask(messages, tools):
        seen.append((list(messages), tools))
        return replies[len(seen) - 1]
    return ask, seen


def _call(name: str, **args) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def test_the_model_gets_the_tool_result_before_answering():
    ask, seen = _model([_call("rendimenti_per_anno", ticker="UP"), {"role": "assistant", "content": "Fatto."}])
    events = list(agent.run([{"role": "user", "content": "q"}], _toolbox(), ask))
    assert [e["type"] for e in events] == ["tool", "answer"]
    assert events[-1]["content"] == "Fatto."
    last_sent = seen[-1][0]
    assert last_sent[-1]["role"] == "tool" and '"2024"' in last_sent[-1]["content"]


def test_arguments_written_as_a_json_string_are_read_too():
    call = {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "rendimenti_per_anno", "arguments": '{"ticker": "UP"}'}}]}
    ask, _ = _model([call, {"role": "assistant", "content": "ok"}])
    events = list(agent.run([], _toolbox(), ask))
    assert events[0]["args"] == {"ticker": "UP"} and "errore" not in events[0]["result"]


def test_after_max_rounds_the_model_must_answer_without_tools():
    ask, seen = _model([_call("rendimenti_per_anno", ticker="UP")] * 2 + [{"role": "assistant", "content": "Basta."}])
    events = list(agent.run([], _toolbox(), ask, max_rounds=2))
    assert events[-1] == {"type": "answer", "content": "Basta."}
    assert seen[-1][1] is None  # no tools offered on the last call


@pytest.mark.parametrize("name", [t["function"]["name"] for t in agent.TOOLS])
def test_every_tool_has_a_description_for_the_user(name):
    assert "sconosciuto" not in agent.describe(name, {"ticker": "SPY", "tickers": ["SPY"]})


def test_numbers_without_any_analysis_are_sent_back_once():
    ask, seen = _model([{"role": "assistant", "content": "Nel 2008 ha perso il 50%."},
                        _call("rendimenti_per_anno", ticker="UP"),
                        {"role": "assistant", "content": "Nel 2024: +12%."}])
    events = list(agent.run([], _toolbox(), ask))
    assert seen[1][0][-1]["content"] == agent.RECHECK
    assert [e["type"] for e in events] == ["tool", "answer"]


def test_a_theory_answer_without_numbers_is_not_sent_back():
    ask, seen = _model([{"role": "assistant", "content": "Lo Sharpe misura il rendimento per unità di rischio."}])
    assert list(agent.run([], _toolbox(), ask))[-1]["content"].startswith("Lo Sharpe")
    assert len(seen) == 1
