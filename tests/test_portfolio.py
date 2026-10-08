import numpy as np
import pandas as pd
import pytest

from quantlab.portfolio import daily_cash_from_yield, run_portfolio_backtest

DAYS = pd.bdate_range("2024-01-01", periods=6)


def _prices() -> pd.DataFrame:
    return pd.DataFrame({"A": [100.0, 110, 121, 121, 60.5, 60.5], "B": [50.0, 50, 50, 100, 100, 100]}, index=DAYS)


def test_full_weight_from_day_zero_is_buy_and_hold_without_costs():
    prices = _prices()
    targets = pd.DataFrame({"A": [1.0]}, index=DAYS[:1])
    result = run_portfolio_backtest(prices, targets, commission=0, slippage=0)
    expected = prices["A"].pct_change().fillna(0.0)
    expected.iloc[0] = 0.0  # day 0: decision taken at the close, nothing held yet
    np.testing.assert_allclose(result["net_return"], expected)


def test_new_weights_earn_only_from_the_next_day():
    """No look-ahead: a decision on day t does not earn day t's move."""
    prices = _prices()
    targets = pd.DataFrame({"B": [1.0]}, index=[DAYS[2]])  # decided at day 2's close, B doubles on day 3
    result = run_portfolio_backtest(prices, targets, commission=0, slippage=0)
    assert result["net_return"].iloc[2] == 0.0
    assert result["net_return"].iloc[3] == pytest.approx(1.0)


def test_weights_drift_between_decisions():
    """50/50 never rebalanced: the account is half in each asset's growth."""
    prices = _prices()
    targets = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=DAYS[:1])
    result = run_portfolio_backtest(prices, targets, commission=0, slippage=0)
    final = (1 + result["net_return"]).prod()
    expected = 0.5 * prices["A"].iloc[-1] / 100 + 0.5 * prices["B"].iloc[-1] / 50
    assert final == pytest.approx(expected)


def test_costs_are_paid_on_the_traded_amount():
    prices = _prices()
    targets = pd.DataFrame({"A": [1.0, 0.25]}, index=[DAYS[0], DAYS[3]])
    result = run_portfolio_backtest(prices, targets, commission=0.001, slippage=0.0005)
    assert result["cost"].iloc[1] == pytest.approx(1.0 * 0.0015)  # cash -> 100% A
    assert result["cost"].iloc[4] == pytest.approx(0.75 * 0.0015)  # 100% A -> 25% A
    assert result["cost"].drop(result.index[[1, 4]]).eq(0).all()


def test_uninvested_capital_earns_cash():
    prices = _prices()
    targets = pd.DataFrame({"A": [0.0]}, index=DAYS[:1])
    result = run_portfolio_backtest(prices, targets, commission=0, slippage=0, cash_return=0.0001)
    np.testing.assert_allclose(result["net_return"], 0.0001)


def test_invalid_weights_are_rejected():
    prices = _prices()
    for bad in ({"A": [0.7], "B": [0.6]}, {"A": [-0.1]}):
        with pytest.raises(ValueError):
            run_portfolio_backtest(prices, pd.DataFrame(bad, index=DAYS[:1]))
    prices.loc[DAYS[0], "B"] = np.nan
    with pytest.raises(ValueError):
        run_portfolio_backtest(prices, pd.DataFrame({"B": [1.0]}, index=DAYS[:1]))


def test_daily_cash_from_yield():
    cash = daily_cash_from_yield(pd.Series([5.0], index=DAYS[:1]), DAYS)
    assert ((1 + cash.iloc[-1]) ** 252 - 1) == pytest.approx(0.05)
    assert cash.nunique() == 1  # carried forward
