"""Tests for quantlab.strategies (hand-built data, no internet)."""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from quantlab.strategies import (
    STRATEGIES,
    buy_and_hold,
    mean_reversion,
    momentum,
    moving_average_crossover,
)

SMALL_PARAMS = {
    "buy_and_hold": {},
    "momentum": {"lookback": 20, "skip": 5},
    "moving_average_crossover": {"fast": 5, "slow": 20},
    "mean_reversion": {"window": 10},
}


def _random_walk(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


@pytest.mark.parametrize("name", STRATEGIES)
def test_signals_do_not_depend_on_future_prices(name):
    # Core no-look-ahead check: computing on a shorter history must give the
    # same signals on the common days. If a strategy peeked ahead, they would differ.
    prices = _random_walk()
    strategy, params = STRATEGIES[name], SMALL_PARAMS[name]
    full = strategy(prices, **params)
    for cut in (50, 150, 299):
        pdt.assert_series_equal(strategy(prices.iloc[:cut], **params), full.iloc[:cut])


@pytest.mark.parametrize("name", STRATEGIES)
def test_signals_are_long_or_flat(name):
    signals = STRATEGIES[name](_random_walk(), **SMALL_PARAMS[name])
    assert set(signals.unique()) <= {0.0, 1.0}
    assert not signals.isna().any()


def test_buy_and_hold_is_always_long():
    assert (buy_and_hold(_random_walk(10)) == 1.0).all()


def test_momentum_long_in_uptrend_flat_in_downtrend():
    idx = pd.bdate_range("2020-01-01", periods=60)
    up = pd.Series(np.linspace(100, 160, 60), index=idx)
    assert momentum(up, lookback=20, skip=5).iloc[-1] == 1.0
    assert momentum(up[::-1].set_axis(idx), lookback=20, skip=5).iloc[-1] == 0.0


def test_momentum_is_flat_during_warm_up():
    signals = momentum(_random_walk(), lookback=20, skip=5)
    assert (signals.iloc[:20] == 0).all()


def test_crossover_follows_the_trend():
    idx = pd.bdate_range("2020-01-01", periods=60)
    up = pd.Series(np.linspace(100, 160, 60), index=idx)
    signals = moving_average_crossover(up, fast=5, slow=20)
    assert (signals.iloc[:19] == 0).all()  # slow average not defined yet
    assert (signals.iloc[19:] == 1).all()


def test_mean_reversion_buys_the_dip_and_sells_the_recovery():
    idx = pd.bdate_range("2020-01-01", periods=40)
    values = [100.0 + (i % 2) for i in range(40)]  # quiet market around 100.5
    values[25] = 90.0  # sudden dip
    values[26:] = [100.0 + (i % 2) for i in range(26, 40)]  # back to normal
    signals = mean_reversion(pd.Series(values, index=idx), window=10)
    assert signals.iloc[24] == 0.0
    assert signals.iloc[25] == 1.0  # z-score very negative -> buy
    assert signals.iloc[-1] == 0.0  # price back above its mean -> sold


@pytest.mark.parametrize(
    "call",
    [
        lambda p: momentum(p, lookback=10, skip=10),
        lambda p: moving_average_crossover(p, fast=20, slow=5),
        lambda p: mean_reversion(p, entry_z=0.5, exit_z=0.0),
    ],
)
def test_invalid_parameters_are_rejected(call):
    with pytest.raises(ValueError):
        call(_random_walk(50))
