"""Tests for quantlab.rules (hand-built data, no internet)."""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from quantlab.rules import Condition, Indicator, holds, rule_strategy
from quantlab.strategies import breakout, moving_average_crossover

PRICE = Indicator("price")


def _random_walk(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


RULES = {
    "above_average": (Condition(PRICE, ">", Indicator("sma", 20)), None),
    "dip_and_recover": (
        Condition(Indicator("zscore", 10), "<", -1.0),
        Condition(Indicator("zscore", 10), ">", 0.0),
    ),
    "positive_return": (Condition(Indicator("return", 20), ">", 0.0), None),
    "dip_in_uptrend": (
        [Condition(PRICE, ">", Indicator("sma", 50)), Condition(Indicator("zscore", 10), "<", -1.0)],
        Condition(Indicator("zscore", 10), ">", 0.0),
    ),
}


@pytest.mark.parametrize("name", RULES)
def test_rules_do_not_depend_on_future_prices(name):
    prices = _random_walk()
    entry, exit_ = RULES[name]
    full = rule_strategy(prices, entry, exit_)
    for cut in (50, 150, 299):
        pdt.assert_series_equal(rule_strategy(prices.iloc[:cut], entry, exit_), full.iloc[:cut])


@pytest.mark.parametrize("name", RULES)
def test_rule_signals_are_long_or_flat(name):
    signals = rule_strategy(_random_walk(), *RULES[name])
    assert set(signals.unique()) <= {0.0, 1.0}


def test_rule_reproduces_moving_average_crossover():
    # Long while the 5-day average is above the 20-day one = the built-in crossover.
    prices = _random_walk()
    rule = Condition(Indicator("sma", 5), ">", Indicator("sma", 20))
    pdt.assert_series_equal(rule_strategy(prices, rule), moving_average_crossover(prices, 5, 20))


def test_rule_reproduces_breakout():
    prices = _random_walk()
    entry = Condition(PRICE, ">=", Indicator("high", 10))
    exit_ = Condition(PRICE, "<=", Indicator("low", 10))
    pdt.assert_series_equal(rule_strategy(prices, entry, exit_), breakout(prices, 10))


def test_exit_rule_keeps_the_position_until_triggered():
    idx = pd.bdate_range("2020-01-01", periods=6)
    prices = pd.Series([100.0, 90, 95, 99, 105, 98], index=idx)
    entry = Condition(PRICE, "<", 92.0)  # buy below 92
    exit_ = Condition(PRICE, ">", 100.0)  # sell above 100
    signals = rule_strategy(prices, entry, exit_)
    assert signals.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_warm_up_days_are_flat():
    signals = rule_strategy(_random_walk(), Condition(PRICE, ">", Indicator("sma", 20)))
    assert (signals.iloc[:19] == 0).all()


@pytest.mark.parametrize(
    "condition",
    [
        Condition(Indicator("magic"), ">", 0.0),
        Condition(Indicator("sma", 1), ">", 0.0),
        Condition(PRICE, "==", 0.0),
    ],
)
def test_invalid_rules_are_rejected(condition):
    with pytest.raises(ValueError):
        rule_strategy(_random_walk(50), condition)


def test_a_list_of_conditions_means_all_of_them():
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.Series([90.0, 95, 100, 105, 110], index=idx)
    rule = [Condition(PRICE, ">", 92.0), Condition(PRICE, "<", 108.0)]
    assert holds(prices, rule).tolist() == [False, True, True, True, False]


def test_single_condition_and_list_of_one_are_the_same():
    prices = _random_walk()
    c = Condition(PRICE, ">", Indicator("sma", 20))
    pdt.assert_series_equal(rule_strategy(prices, c), rule_strategy(prices, [c]))


def test_empty_rule_is_rejected():
    with pytest.raises(ValueError):
        rule_strategy(_random_walk(50), [])


def _fear_index(prices: pd.Series, seed: int = 1) -> pd.Series:
    """A fake VIX-like series on the same days, between 10 and 40."""
    rng = np.random.default_rng(seed)
    return pd.Series(25 + 15 * np.sin(np.arange(len(prices)) / 15) + rng.normal(0, 1, len(prices)),
                     index=prices.index)


def test_external_source_replaces_the_traded_prices():
    prices = _random_walk()
    vix = _fear_index(prices)
    signals = rule_strategy(prices, Condition(Indicator("price", source=vix, label="VIX"), "<", 20.0))
    pdt.assert_series_equal(signals, (vix < 20).astype(float))


def test_external_source_does_not_leak_the_future():
    # Same check as for prices: changing the source AFTER a date must not change signals up to it.
    prices = _random_walk()
    vix = _fear_index(prices)
    rule = Condition(Indicator("sma", 10, source=vix), ">", 25.0)
    original = rule_strategy(prices, rule)
    future_changed = vix.copy()
    future_changed.iloc[150:] = 99.0
    changed = rule_strategy(prices, Condition(Indicator("sma", 10, source=future_changed), ">", 25.0))
    pdt.assert_series_equal(changed.iloc[:150], original.iloc[:150])


def test_external_source_on_other_days_uses_the_last_known_value():
    idx = pd.bdate_range("2024-01-01", periods=5)  # Mon..Fri
    prices = pd.Series(100.0, index=idx)
    # the source has no value on Wednesday (a holiday on its market) but has one on Saturday
    source = pd.Series([10.0, 11, 13, 14, 99], index=[idx[0], idx[1], idx[3], idx[4], idx[4] + pd.Timedelta(days=1)])
    base = Indicator("price", source=source)._base(prices)
    assert base.tolist() == [10.0, 11, 11, 13, 14]  # Wednesday carries Tuesday's value; Saturday never used


def test_indicators_with_sources_can_be_compared():
    vix = _fear_index(_random_walk())
    assert Indicator("price", source=vix, label="VIX") == Indicator("price", source=vix * 2, label="VIX")
