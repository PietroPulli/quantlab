import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from quantlab.allocation import dual_momentum, fixed_weights, trend_allocation, volatility_target
from quantlab.ml import month_ends

IDX = pd.bdate_range("2010-01-01", periods=900)


def _walk(seed: int, drift: float = 0.0003, vol: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(drift, vol, len(IDX))))


def _assets() -> pd.DataFrame:
    return pd.DataFrame({name: _walk(seed) for seed, name in enumerate(["SPY", "EFA", "AGG", "IEF", "VNQ"])},
                        index=IDX)


def _tbill() -> pd.Series:
    return pd.Series(2.0, index=IDX)


def _future_tripled(prices):
    cut = IDX[600]
    changed = prices.copy()
    changed[changed.index > cut] *= 3
    return cut, changed


@pytest.mark.parametrize("rule", [
    lambda p: volatility_target(p["SPY"]),
    lambda p: trend_allocation(p[["SPY", "EFA", "IEF", "VNQ"]]),
    lambda p: dual_momentum(p, _tbill()),
])
def test_rules_do_not_look_ahead(rule):
    prices = _assets()
    cut, changed = _future_tripled(prices)
    before, after = rule(prices), rule(changed)
    pdt.assert_frame_equal(before[before.index <= cut], after[after.index <= cut])


def test_rules_decide_on_month_ends_with_valid_weights():
    prices = _assets()
    for weights in (volatility_target(prices["SPY"]), trend_allocation(prices), dual_momentum(prices, _tbill())):
        assert weights.index.isin(month_ends(IDX)).all()  # last trading day of each month
        assert (weights >= 0).all().all() and (weights.sum(axis=1) <= 1 + 1e-12).all()


def test_volatility_target_invests_less_when_the_market_is_wild():
    calm = np.cumprod(np.r_[1.0, np.where(np.arange(399) % 2, 1.003, 0.997)])  # ~4.8% annual volatility
    wild = calm[-1] * np.cumprod(np.where(np.arange(400) % 2, 1.03, 0.97))  # ~48%
    prices = pd.Series(np.r_[calm, wild] * 100, index=pd.bdate_range("2015-01-01", periods=800), name="SPY")
    weights = volatility_target(prices, target_vol=0.15)["SPY"]
    assert weights.iloc[3] == 1.0  # calm: capped at 100%
    assert weights.iloc[-1] == pytest.approx(0.15 / 0.476, rel=0.05)


def test_trend_allocation_drops_assets_below_their_average():
    up = np.linspace(100, 200, len(IDX))
    prices = pd.DataFrame({"UP": up, "DOWN": up[::-1]}, index=IDX)
    weights = trend_allocation(prices, months=10)
    assert (weights["UP"] == 0.5).all()
    assert (weights["DOWN"] == 0.0).all()


def test_dual_momentum_picks_the_stronger_market_or_bonds():
    t = np.arange(len(IDX))
    rising = pd.DataFrame({"SPY": 100 + 0.05 * t, "EFA": 100 + 0.10 * t, "AGG": 100.0 + 0 * t}, index=IDX)
    weights = dual_momentum(rising, _tbill())
    assert (weights["EFA"] == 1.0).all()  # both beat cash, EFA rose more
    falling = rising.assign(SPY=200 - 0.05 * t)
    weights = dual_momentum(falling, _tbill())
    assert (weights["AGG"] == 1.0).all()  # US stocks below cash -> bonds


def test_fixed_weights_rebalance_every_month():
    weights = fixed_weights(_assets(), {"SPY": 0.6, "IEF": 0.4})
    assert (weights["SPY"] == 0.6).all() and len(weights) == len(IDX.to_period("M").unique())
