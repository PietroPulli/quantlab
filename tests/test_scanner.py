"""Tests for quantlab.scanner (hand-built data, no internet)."""

import numpy as np
import pandas as pd
import pytest

from quantlab.rules import Condition, Indicator, rule_strategy
from quantlab.scanner import best_idea, bonferroni_level, scan
from quantlab.strategies import buy_and_hold


def _trending_regimes(n: int = 1200, seed: int = 0) -> pd.Series:
    """Long up and down trends with little noise: a trend rule has a real edge here."""
    rng = np.random.default_rng(seed)
    drift = np.where((np.arange(n) // 120) % 2 == 0, 0.004, -0.004)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(drift + rng.normal(0, 0.004, n))), index=idx)


TREND = (rule_strategy, {"entry": Condition(Indicator("price"), ">", Indicator("sma", 20)), "exit": None})
HOLD = (buy_and_hold, {})


def test_bonferroni_level():
    assert bonferroni_level(0.95, 1) == pytest.approx(0.95)
    assert bonferroni_level(0.95, 8) == pytest.approx(1 - 0.05 / 8)
    with pytest.raises(ValueError):
        bonferroni_level(0.95, 0)


def test_a_real_edge_is_found_and_named():
    table = scan(_trending_regimes(), {"trend": TREND, "hold": HOLD}, n_bootstrap=300)
    best = best_idea(table)
    assert best is not None and best["idea"] == "trend"
    assert table.set_index("idea").loc["hold", "verdict"] == "neutral"  # buy & hold vs itself


def test_more_ideas_make_every_test_stricter():
    prices = _trending_regimes()
    alone = scan(prices, {"trend": TREND}, n_bootstrap=300).iloc[0]
    crowd = scan(prices, {"trend": TREND, **{f"hold{i}": HOLD for i in range(7)}}, n_bootstrap=300)
    strict = crowd.set_index("idea").loc["trend"]
    assert crowd.attrs["level"] > 0.99
    assert strict["ci_low"] <= alone["ci_low"] and strict["ci_high"] >= alone["ci_high"]  # wider interval


def test_no_edge_means_no_best_idea():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2018-01-01", periods=900)
    noise = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 900))), index=idx)
    table = scan(noise, {"hold": HOLD, "trend": TREND}, n_bootstrap=300)
    assert best_idea(table) is None


def test_scan_reports_what_each_idea_says_today():
    table = scan(_trending_regimes(), {"trend": TREND, "hold": HOLD}, n_bootstrap=100)
    assert set(table["signal_today"]) <= {0.0, 1.0}
    assert table.set_index("idea").loc["hold", "signal_today"] == 1.0
