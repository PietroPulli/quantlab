"""Tests for quantlab.report (hand-built data, no internet)."""

import numpy as np
import pandas as pd
import pytest

from quantlab.report import evaluate_strategy, format_report
from quantlab.strategies import buy_and_hold, momentum
from quantlab.validation import param_grid


def _random_walk(n: int = 400, drift: float = 0.0, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))), index=idx)


def _zigzag(n: int = 400) -> pd.Series:
    """+2%, -1%, +2%, -1%, ...: tomorrow's direction is predictable from today's."""
    idx = pd.bdate_range("2015-01-01", periods=n)
    moves = np.where(np.arange(n) % 2 == 1, 1.02, 0.99)
    moves[0] = 1.0
    return pd.Series(100 * np.cumprod(moves), index=idx)


def always_short(prices: pd.Series) -> pd.Series:
    return pd.Series(-1.0, index=prices.index)


def buy_after_down_day(prices: pd.Series) -> pd.Series:
    """Legitimate (no look-ahead) rule that exploits the zigzag pattern."""
    return (prices.pct_change() < 0).astype(float)


def test_buy_and_hold_against_itself_shows_no_edge():
    report = evaluate_strategy(_random_walk(), buy_and_hold, n_bootstrap=100)
    assert report.sharpe_diff == pytest.approx(0)
    assert report.verdict.startswith("NO EVIDENCE")
    assert report.share_beating == 0  # never strictly better than itself


def test_shorting_a_strong_uptrend_is_judged_worse():
    report = evaluate_strategy(_random_walk(drift=0.003), always_short, n_bootstrap=100)
    assert report.verdict.startswith("WORSE")


def test_a_real_edge_is_detected():
    report = evaluate_strategy(_zigzag(), buy_after_down_day, n_bootstrap=100)
    assert report.verdict.startswith("BEATS")
    assert report.sharpe_diff_ci[0] > 0


def test_metrics_tables_compare_with_buy_and_hold():
    report = evaluate_strategy(_random_walk(), momentum, params={"lookback": 60, "skip": 5}, n_bootstrap=50)
    for table in (report.in_sample, report.out_of_sample):
        assert list(table.columns) == ["strategy", "buy_and_hold"]
    assert (report.returns.index >= report.split).sum() == 120  # last 30% of 400 days


def test_grid_is_optimised_in_sample_and_walk_forward_runs():
    grid = param_grid(lookback=[60, 120], skip=[5])
    report = evaluate_strategy(_random_walk(1100), momentum, grid=grid, n_bootstrap=50)
    assert report.params in grid
    assert report.n_combinations == 2
    assert report.walk_forward is not None
    assert any("parameter sets were tried" in w for w in report.warnings)


def test_params_and_grid_together_are_rejected():
    with pytest.raises(ValueError):
        evaluate_strategy(_random_walk(), momentum, params={}, grid=[{}])


def test_format_report_contains_the_key_sections():
    grid = param_grid(lookback=[60, 120], skip=[5])
    text = format_report(evaluate_strategy(_random_walk(1100), momentum, grid=grid, n_bootstrap=50))
    for section in ("VERDICT", "IN-SAMPLE", "OUT-OF-SAMPLE", "WALK-FORWARD", "survivorship"):
        assert section in text


def _always_cash(prices: pd.Series) -> pd.Series:
    return pd.Series(0.0, index=prices.index)


def test_sitting_in_cash_does_not_look_skilful():
    # Cash earning 4% has a steady return and almost no volatility: if the Sharpe did
    # not subtract the cash rate, it would look like a fantastic strategy.
    report = evaluate_strategy(_random_walk(), _always_cash, cash_rate=0.04, n_bootstrap=50)
    oos_sharpe = report.out_of_sample.loc["sharpe", "strategy"]
    assert np.isnan(oos_sharpe) or abs(oos_sharpe) < 1e-6
    assert report.out_of_sample.loc["annual_return", "strategy"] == pytest.approx(0.04)


def test_cash_rate_helps_strategies_that_are_often_out_of_the_market():
    prices = _random_walk(drift=0.0003)
    params = {"lookback": 60, "skip": 5}
    without = evaluate_strategy(prices, momentum, params=params, n_bootstrap=50)
    with_cash = evaluate_strategy(prices, momentum, params=params, cash_rate=0.04, n_bootstrap=50)
    gain = lambda r: r.out_of_sample.loc["total_return", "strategy"]  # noqa: E731
    bh = lambda r: r.out_of_sample.loc["total_return", "buy_and_hold"]  # noqa: E731
    assert gain(with_cash) > gain(without)
    assert bh(with_cash) == pytest.approx(bh(without))  # always invested: no idle cash


def test_report_mentions_the_cash_rate():
    report = evaluate_strategy(_random_walk(), buy_and_hold, cash_rate=0.03, n_bootstrap=50)
    assert "idle cash earns 3.00%" in format_report(report)


def test_crypto_prices_are_annualised_with_365_days():
    # Bitcoin trades every day: one year is 365 prices, not 252.
    idx = pd.date_range("2018-01-01", periods=4 * 365)
    prices = pd.Series(100 * 1.0002 ** np.arange(len(idx)), index=idx)
    report = evaluate_strategy(prices, buy_and_hold, n_bootstrap=50)
    assert report.periods_per_year == 365
    expected = 1.0002 ** 365 - 1  # one calendar year of daily +0.02%
    assert report.out_of_sample.loc["annual_return", "buy_and_hold"] == pytest.approx(expected, rel=1e-6)


def test_stocks_keep_252_days():
    assert evaluate_strategy(_random_walk(), buy_and_hold, n_bootstrap=50).periods_per_year == 252
