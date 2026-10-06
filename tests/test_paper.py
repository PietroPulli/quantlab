"""Tests for quantlab.paper (hand-built data, no internet)."""

import pandas as pd
import pytest

from quantlab.paper import (
    Slot,
    advance,
    completed_closes,
    new_slots,
    portfolio_history,
    slots_from_records,
    slots_to_records,
    step,
)

COST = 0.001


def _days(n: int, start: str = "2026-10-05") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def test_capital_is_split_equally_and_the_benchmark_gets_the_same():
    slots = new_slots([{"ticker": "SPY", "idea": "a"}, {"ticker": "QQQ", "idea": "b"}], 10_000)
    assert [s.cash for s in slots] == [5000, 5000] and [s.bench_cash for s in slots] == [5000, 5000]


def test_buy_then_sell_pays_costs_on_both_legs():
    slot = Slot("SPY", "a", cash=1000.0, bench_cash=1000.0)
    idx = _days(3)
    slot, buy = step(slot, idx[0], 100.0, 1.0, COST, 0.0)
    assert buy["action"] == "buy" and slot.shares == pytest.approx(1000 * (1 - COST) / 100)
    slot, none = step(slot, idx[1], 110.0, 1.0, COST, 0.0)
    assert none is None  # signal unchanged: no trade
    slot, sell = step(slot, idx[2], 120.0, 0.0, COST, 0.0)
    assert sell["action"] == "sell"
    assert slot.cash == pytest.approx(1000 * (1 - COST) / 100 * 120 * (1 - COST))
    assert slot.shares == 0


def test_benchmark_buys_on_the_first_close_and_never_trades():
    slot = Slot("SPY", "a", cash=1000.0, bench_cash=1000.0)
    idx = _days(3)
    for day, price, signal in zip(idx, [100.0, 50.0, 200.0], [0.0, 1.0, 0.0]):
        slot, _ = step(slot, day, price, signal, COST, 0.0)
    assert slot.bench_value(200.0) == pytest.approx(1000 * (1 - COST) * 2)


def test_idle_cash_earns_interest_but_not_on_the_first_day():
    slot = Slot("SPY", "a", cash=1000.0, bench_cash=1000.0)
    idx = _days(3)
    for day in idx:
        slot, _ = step(slot, day, 100.0, 0.0, COST, 0.001)
    assert slot.cash == pytest.approx(1000 * 1.001 ** 2)  # two intervals between three closes


def test_advance_starts_at_start_and_never_repeats_a_day():
    idx = _days(6)
    prices = pd.Series([100.0, 101, 102, 103, 104, 105], index=idx)
    signals = pd.Series([1.0, 1, 0, 0, 1, 1], index=idx)
    slot = Slot("SPY", "a", cash=1000.0, bench_cash=1000.0)
    slot, trades, values = advance(slot, prices, signals, start=str(idx[2].date()), cost=0, daily_cash=0)
    assert [v["date"] for v in values] == [str(d.date()) for d in idx[2:]]  # nothing before start
    assert [t["action"] for t in trades] == ["buy"]  # signal 0 at start, buys at idx[4]
    again, more_trades, more_values = advance(slot, prices, signals, str(idx[2].date()), 0, 0)
    assert more_trades == [] and more_values == []  # same data twice: nothing new


def test_completed_closes_never_uses_todays_bar():
    # Run in the morning, Milan is open: today's "close" would be an intraday price.
    idx = pd.bdate_range("2026-10-02", periods=3)  # Fri, Mon, Tue
    prices = pd.Series([1.0, 2, 3], index=idx)
    kept = completed_closes(prices, pd.Timestamp("2026-10-06"))
    assert kept.index[-1] == pd.Timestamp("2026-10-05")


def test_portfolio_history_carries_values_over_other_calendars():
    values = pd.DataFrame([
        {"date": "2026-10-09", "ticker": "SPY", "idea": "a", "value": 100.0, "bench_value": 100.0},
        {"date": "2026-10-09", "ticker": "BTC", "idea": "b", "value": 50.0, "bench_value": 60.0},
        {"date": "2026-10-10", "ticker": "BTC", "idea": "b", "value": 55.0, "bench_value": 70.0},  # Saturday
    ])
    history = portfolio_history(values)
    assert history.loc["2026-10-10", "value"] == pytest.approx(155.0)  # SPY's Friday value carried
    assert history.loc["2026-10-10", "bench_value"] == pytest.approx(170.0)


def test_slots_survive_a_round_trip_to_json_records():
    slots = new_slots([{"ticker": "SPY", "idea": "a"}], 1000)
    assert slots_from_records(slots_to_records(slots)) == slots


def test_run_day_creates_the_account_then_only_adds_new_closes():
    from quantlab.paper import run_day

    idx = pd.bdate_range("2023-01-02", "2026-10-09")
    trend = pd.Series(range(100, 100 + len(idx)), index=idx, dtype=float)  # always rising
    config = {"start_cash": 10_000, "commission": 0.001, "slippage": 0.0005, "cash_rate": 0.02,
              "slots": [{"ticker": "SPY", "idea": "Segui la tendenza"}, {"ticker": "XXX", "idea": "Segui la tendenza"}]}

    def get_prices(ticker, start):
        if ticker == "XXX":
            raise ValueError("no data")
        return trend[trend.index <= today]

    today = pd.Timestamp("2026-10-08")  # night run: the last completed close is 7 October
    state, trades, values, problems = run_day(config, None, today, get_prices, factor=None)
    assert state["start"] == "2026-10-07"  # the account starts with the first completed close
    assert [t["action"] for t in trades] == ["buy"]  # rising market: above its 200-day average
    assert len(values) == 1 and problems == ["XXX / Segui la tendenza: no data"]

    today = pd.Timestamp("2026-10-10")
    state, trades, values, _ = run_day(config, state, today, get_prices, factor=None)
    assert trades == []  # already invested
    assert [v["date"] for v in values] == ["2026-10-08", "2026-10-09"]
    assert state["slots"][1]["last_date"] is None  # the failing slot is untouched, retried next time


def test_the_real_account_config_is_valid():
    import json
    from pathlib import Path

    from quantlab.ideas import IDEAS

    config = json.loads((Path(__file__).resolve().parents[1] / "paper" / "config.json").read_text(encoding="utf-8"))
    assert config["start_cash"] > 0 and config["commission"] > 0 and config["slippage"] > 0  # costs never zero
    assert all(slot["idea"] in IDEAS for slot in config["slots"])
