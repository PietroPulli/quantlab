"""Tests for quantlab.metrics (hand-built data, no internet)."""

import numpy as np
import pandas as pd
import pytest

from quantlab.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    compare,
    drawdown,
    max_drawdown,
    sharpe_ratio,
    summary,
    total_return,
)


def test_total_return_compounds():
    assert total_return(pd.Series([0.10, -0.10])) == pytest.approx(-0.01)


def test_annualized_return_of_one_year_equals_total_return():
    r = pd.Series([0.001] * 252)
    assert annualized_return(r) == pytest.approx(total_return(r))


def test_annualized_return_of_two_years_is_the_geometric_mean():
    r = pd.Series([0.001] * 504)
    assert annualized_return(r) == pytest.approx(1.001**252 - 1)


def test_volatility_scales_with_square_root_of_time():
    r = pd.Series([0.01, -0.01] * 50)
    assert annualized_volatility(r) == pytest.approx(r.std() * np.sqrt(252))


def test_sharpe_known_value_and_sign():
    r = pd.Series([0.02, 0.0] * 50)  # mean 1%, std ~1% per day
    expected = r.mean() / r.std() * np.sqrt(252)
    assert sharpe_ratio(r) == pytest.approx(expected)
    assert sharpe_ratio(-r) == pytest.approx(-expected)


def test_sharpe_is_nan_when_returns_do_not_move():
    assert np.isnan(sharpe_ratio(pd.Series([0.001] * 10)))


def test_risk_free_rate_lowers_sharpe():
    r = pd.Series([0.02, 0.0] * 50)
    assert sharpe_ratio(r, risk_free=0.05) < sharpe_ratio(r)


def test_max_drawdown_known_path():
    # equity: 1.10 -> 0.55 -> 0.66 ; worst fall from the 1.10 peak is -50%
    assert max_drawdown(pd.Series([0.10, -0.50, 0.20])) == pytest.approx(-0.5)


def test_loss_on_first_day_counts_as_drawdown():
    assert max_drawdown(pd.Series([-0.20, 0.10])) == pytest.approx(-0.2)


def test_drawdown_is_zero_at_new_highs():
    assert (drawdown(pd.Series([0.01, 0.02, 0.03])) == 0).all()


def test_summary_and_compare_shapes():
    r = pd.Series([0.01, -0.005] * 100)
    s = summary(r)
    assert list(s.index) == [
        "total_return", "annual_return", "annual_volatility", "sharpe", "max_drawdown",
    ]
    table = compare({"strategy": r, "buy_and_hold": r * 0.5})
    assert list(table.columns) == ["strategy", "buy_and_hold"]
    assert table.loc["annual_volatility", "buy_and_hold"] == pytest.approx(
        table.loc["annual_volatility", "strategy"] / 2
    )


def test_calmar_known_value():
    # Day 1 loses 20% (the worst drawdown), then the equity climbs steadily to 1.10
    # after exactly one year: annual return 10%, max drawdown -20% -> Calmar 0.5.
    daily = (1.10 / 0.80) ** (1 / 251) - 1
    r = pd.Series([-0.20] + [daily] * 251)
    assert calmar_ratio(r) == pytest.approx(0.5)


def test_calmar_is_nan_without_drawdown():
    assert np.isnan(calmar_ratio(pd.Series([0.001] * 10)))
    assert np.isnan(calmar_ratio(pd.Series([], dtype=float)))


def test_compare_passes_the_risk_free_rate_to_sharpe():
    r = pd.Series([0.02, 0.0] * 50)
    table = compare({"a": r}, risk_free=0.05)
    assert table.loc["sharpe", "a"] == pytest.approx(sharpe_ratio(r, risk_free=0.05))
