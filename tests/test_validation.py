"""Tests for quantlab.validation (hand-built data, no internet)."""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from quantlab.strategies import momentum
from quantlab.validation import (
    block_bootstrap,
    confidence_interval,
    optimize,
    param_grid,
    sharpe_difference_bootstrap,
    split_date,
    walk_forward,
)


def _random_walk(n: int = 300, drift: float = 0.0, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))), index=idx)


def constant(prices: pd.Series, direction: float) -> pd.Series:
    """Toy strategy: always hold `direction` (1 long, -1 short)."""
    return pd.Series(direction, index=prices.index)


def test_split_date_is_chronological():
    idx = pd.bdate_range("2024-01-01", periods=10)
    assert split_date(idx, 0.7) == idx[7]
    with pytest.raises(ValueError):
        split_date(idx, 1.5)


def test_param_grid_builds_every_combination():
    assert param_grid(a=[1, 2], b=[3]) == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def test_optimize_picks_the_best_sharpe():
    uptrend = _random_walk(drift=0.002)
    best, scores = optimize(uptrend, constant, param_grid(direction=[-1.0, 1.0]))
    assert best == {"direction": 1.0}
    assert len(scores) == 2


def test_optimize_never_sees_prices_after_end():
    prices = _random_walk()
    end = prices.index[200]
    seen = []

    def spy(p: pd.Series) -> pd.Series:
        seen.append(p.index.max())
        return constant(p, 1.0)

    optimize(prices, spy, [{}], end=end)
    assert max(seen) < end


def test_walk_forward_shape_and_folds():
    prices = _random_walk(300)
    grid = param_grid(lookback=[20, 40], skip=[5])
    oos, folds = walk_forward(prices, momentum, grid, train_days=100, test_days=50)
    assert oos.index[0] == prices.index[100]
    assert len(oos) == 200
    assert len(folds) == 4
    assert folds["test_start"].iloc[1] == prices.index[150]


def test_walk_forward_does_not_use_future_prices():
    # Running on a shorter history must give identical out-of-sample returns on
    # the common days: nothing in the past may depend on prices that come later.
    prices = _random_walk(300)
    grid = param_grid(lookback=[20, 40], skip=[5])
    full, _ = walk_forward(prices, momentum, grid, train_days=100, test_days=50)
    short, _ = walk_forward(prices.iloc[:230], momentum, grid, train_days=100, test_days=50)
    pdt.assert_series_equal(short, full.iloc[: len(short)])


def test_walk_forward_rejects_too_little_data():
    with pytest.raises(ValueError):
        walk_forward(_random_walk(100), momentum, [{}], train_days=80, test_days=50)


def test_bootstrap_is_reproducible_with_a_seed():
    r = _random_walk().pct_change().dropna()
    a = block_bootstrap(r, lambda x: x.mean(), n_samples=50, seed=1)
    b = block_bootstrap(r, lambda x: x.mean(), n_samples=50, seed=1)
    c = block_bootstrap(r, lambda x: x.mean(), n_samples=50, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (50,)


def test_one_block_as_long_as_the_data_returns_the_original():
    r = pd.Series([0.01, -0.02, 0.03, 0.0])
    samples = block_bootstrap(r, lambda x: x.sum(), n_samples=5, block_size=4)
    assert np.allclose(samples, r.sum())


def test_bootstrap_keeps_columns_paired():
    a = pd.Series(np.arange(100.0))
    data = pd.DataFrame({"a": a, "b": 2 * a})
    samples = block_bootstrap(data, lambda d: (d["b"] - 2 * d["a"]).abs().max(), n_samples=20)
    assert (samples == 0).all()


def test_identical_strategies_have_zero_sharpe_difference():
    r = _random_walk().pct_change().dropna()
    samples = sharpe_difference_bootstrap(r, r, n_samples=20)
    assert np.allclose(samples, 0)


def test_confidence_interval_known_values():
    low, high = confidence_interval(np.arange(101.0), level=0.95)
    assert (low, high) == pytest.approx((2.5, 97.5))


def test_confidence_interval_of_all_nan_samples_is_nan():
    low, high = confidence_interval(np.array([np.nan, np.nan]))
    assert np.isnan(low) and np.isnan(high)
