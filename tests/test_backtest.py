"""Tests for quantlab.backtest (hand-built data, no internet)."""

import inspect

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from quantlab.backtest import (
    current_signal,
    trade_log,
    DEFAULT_COMMISSION,
    DEFAULT_SLIPPAGE,
    run_backtest,
    signals_to_positions,
    turnover_per_year,
)


def _zigzag(n: int = 20) -> pd.Series:
    """Prices that go +1%, -1%, +1%, ... so every day's direction flips."""
    idx = pd.bdate_range("2024-01-01", periods=n)
    moves = np.where(np.arange(n) % 2 == 1, 1.01, 1 / 1.01)
    moves[0] = 1.0
    return pd.Series(100 * np.cumprod(moves), index=idx)


def test_positions_are_signals_shifted_by_one_day():
    idx = pd.bdate_range("2024-01-01", periods=4)
    signals = pd.Series([0.0, 1.0, 1.0, 0.0], index=idx)
    assert signals_to_positions(signals).tolist() == [0.0, 0.0, 1.0, 1.0]


def test_no_look_ahead_same_day_signal_cannot_profit():
    # "Cheating" signal: be long on day t if day t's own return is positive.
    # With look-ahead it would win every up day; shifted, on a zigzag it always loses.
    prices = _zigzag()
    same_day_up = (prices.pct_change() > 0).astype(float)
    result = run_backtest(prices, same_day_up, commission=0.0, slippage=0.0)
    assert (result["gross_return"] <= 0).all()
    assert result["equity"].iloc[-1] < 1


def test_no_look_ahead_future_prices_do_not_change_the_past():
    prices = _zigzag(30)
    signals = (prices > prices.rolling(3).mean()).astype(float)
    full = run_backtest(prices, signals)
    cut = 15
    short = run_backtest(prices.iloc[:cut], signals.iloc[:cut])
    pdt.assert_frame_equal(short, full.iloc[:cut])


def test_costs_default_to_non_zero():
    assert DEFAULT_COMMISSION > 0 and DEFAULT_SLIPPAGE > 0
    params = inspect.signature(run_backtest).parameters
    assert params["commission"].default > 0
    assert params["slippage"].default > 0


def test_buy_and_hold_without_costs_matches_price_growth():
    prices = _zigzag()
    signals = pd.Series(1.0, index=prices.index)
    result = run_backtest(prices, signals, commission=0.0, slippage=0.0)
    # Position starts the day after the first signal: we earn from the first close on.
    assert np.isclose(result["equity"].iloc[-1], prices.iloc[-1] / prices.iloc[0])


def test_entry_trade_pays_commission_plus_slippage_once():
    prices = _zigzag()
    signals = pd.Series(1.0, index=prices.index)
    result = run_backtest(prices, signals, commission=0.001, slippage=0.0005)
    assert np.isclose(result["cost"].sum(), 0.0015)
    assert np.isclose(result["cost"].iloc[1], 0.0015)  # the day the position opens


def test_flipping_long_to_short_trades_twice_the_capital():
    idx = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.Series([100.0, 101.0, 102.0, 103.0], index=idx)
    signals = pd.Series([1.0, -1.0, -1.0, -1.0], index=idx)
    result = run_backtest(prices, signals, commission=0.01, slippage=0.0)
    assert result["cost"].tolist() == pytest.approx([0.0, 0.01, 0.02, 0.0])


@pytest.mark.parametrize(
    "kwargs",
    [{"commission": -0.001}, {"slippage": -0.001}],
)
def test_negative_costs_are_rejected(kwargs):
    prices = _zigzag()
    with pytest.raises(ValueError):
        run_backtest(prices, pd.Series(1.0, index=prices.index), **kwargs)


def test_leverage_and_missing_signals_are_rejected():
    prices = _zigzag()
    with pytest.raises(ValueError):
        run_backtest(prices, pd.Series(2.0, index=prices.index))
    bad = pd.Series(1.0, index=prices.index)
    bad.iloc[3] = np.nan
    with pytest.raises(ValueError):
        run_backtest(prices, bad)


def test_misaligned_index_is_rejected():
    prices = _zigzag()
    with pytest.raises(ValueError):
        run_backtest(prices, pd.Series(1.0, index=prices.index[:-1]))


def test_turnover_per_year():
    positions = pd.Series([0.0, 1.0, 1.0, 0.0])  # buy once, sell once = 2x capital traded
    assert turnover_per_year(positions, periods_per_year=4) == pytest.approx(2.0)


def test_trade_log_dates_trades_at_the_signal_close():
    idx = pd.bdate_range("2020-01-01", periods=6)
    prices = pd.Series([10.0, 11, 12, 13, 14, 15], index=idx)
    signals = pd.Series([0.0, 1, 1, 0, 0, 1], index=idx)  # buy signal at close of day 1
    log = trade_log(prices, run_backtest(prices, signals)["position"])
    assert log["action"].tolist() == ["buy", "sell"]
    assert log.index.tolist() == [idx[1], idx[3]]  # executed when the signal changed
    assert log["price"].tolist() == [11.0, 13.0]  # at that day's close, never a later price
    # the last buy signal (day 5) has no next day to trade on, so it is not in the log


def test_trade_log_long_to_short_is_one_sell_of_size_two():
    idx = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.Series([10.0, 11, 12, 13], index=idx)
    log = trade_log(prices, run_backtest(prices, pd.Series([1.0, -1, -1, -1], index=idx))["position"])
    assert log["action"].tolist() == ["buy", "sell"]
    assert log["size"].tolist() == [1.0, 2.0]


def test_trade_log_is_empty_without_trades():
    idx = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.Series([10.0, 11, 12, 13], index=idx)
    assert trade_log(prices, pd.Series(0.0, index=idx)).empty


def test_idle_cash_earns_the_cash_rate():
    idx = pd.bdate_range("2020-01-01", periods=253)
    prices = pd.Series(100.0, index=idx)  # flat market: only cash can earn anything
    flat = run_backtest(prices, pd.Series(0.0, index=idx), cash_rate=0.04)
    assert flat["equity"].iloc[-1] == pytest.approx(1.04)  # 252 daily returns = one year at 4%


def test_invested_capital_earns_no_cash_interest():
    idx = pd.bdate_range("2020-01-01", periods=10)
    prices = pd.Series(100.0, index=idx)
    half = run_backtest(prices, pd.Series(0.5, index=idx), commission=0, slippage=0, cash_rate=0.04)
    daily = 1.04 ** (1 / 252) - 1
    assert half["cash_return"].iloc[0] == 0.0  # day 0: starting point, no interest yet
    assert np.allclose(half["cash_return"].iloc[1:], 0.5 * daily)  # half invested from day 1


def test_cash_rate_zero_changes_nothing():
    prices = _zigzag()
    signals = (prices.pct_change() > 0).astype(float)
    pdt.assert_frame_equal(
        run_backtest(prices, signals).drop(columns="cash_return"),
        run_backtest(prices, signals, cash_rate=0.0).drop(columns="cash_return"),
    )
    assert (run_backtest(prices, signals)["cash_return"] == 0).all()


def test_cash_rate_in_percent_is_rejected():
    prices = _zigzag()
    with pytest.raises(ValueError):
        run_backtest(prices, pd.Series(0.0, index=prices.index), cash_rate=4.0)  # 400%: a typo for 0.04


def test_current_signal_reports_the_last_value_and_when_it_started():
    idx = pd.bdate_range("2024-01-01", periods=6)
    signals = pd.Series([0.0, 1, 1, 0, 1, 1], index=idx)
    day, value, since = current_signal(signals)
    assert (day, value, since) == (idx[5], 1.0, idx[4])  # long since day 4, not since day 1


def test_current_signal_without_changes_starts_at_the_beginning():
    idx = pd.bdate_range("2024-01-01", periods=3)
    assert current_signal(pd.Series(0.0, index=idx)) == (idx[2], 0.0, idx[0])


def test_current_signal_of_empty_series_is_an_error():
    with pytest.raises(ValueError):
        current_signal(pd.Series([], dtype=float))
