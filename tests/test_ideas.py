"""Tests for quantlab.ideas (hand-built data, no internet)."""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from quantlab.ideas import FACTOR_NAMES, IDEAS


def _random_walk(n: int = 600, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=idx)


def _fake_factor(name: str, ticker: str = "") -> pd.Series:
    """Deterministic factor data covering the test period (VIX-like, curve, surprises)."""
    idx = pd.bdate_range("2019-01-01", periods=1000)
    wave = np.sin(np.arange(len(idx)) / 20)
    values = {"vix": 25 + 10 * wave, "yield_curve": wave, "earnings": 3 * wave}[name]
    return pd.Series(values, index=idx)


@pytest.mark.parametrize("name", IDEAS)
def test_every_idea_gives_long_or_flat_signals(name):
    strategy, params = IDEAS[name].build("AAPL", _fake_factor)
    signals = strategy(_random_walk(), **params)
    assert set(signals.unique()) <= {0.0, 1.0}
    assert not signals.isna().any()


@pytest.mark.parametrize("name", IDEAS)
def test_no_idea_looks_at_future_prices(name):
    prices = _random_walk()
    strategy, params = IDEAS[name].build("AAPL", _fake_factor)
    full = strategy(prices, **params)
    for cut in (250, 450):
        pdt.assert_series_equal(strategy(prices.iloc[:cut], **params), full.iloc[:cut])


def test_every_factor_used_by_ideas_has_a_label():
    asked = []
    for idea in IDEAS.values():
        idea.build("AAPL", lambda name, ticker="": asked.append(name) or _fake_factor(name))
    assert set(asked) <= set(FACTOR_NAMES)


def test_only_the_earnings_idea_needs_a_company():
    assert [i.name for i in IDEAS.values() if i.companies_only] == ["Compra dopo trimestrali sopra le attese"]
