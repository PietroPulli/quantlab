"""Tests for quantlab.backtest (hand-built data, no internet)."""

import inspect

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from quantlab.backtest import (
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
