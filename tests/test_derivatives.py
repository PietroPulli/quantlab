"""Tests for quantlab.derivatives: textbook values, put-call parity, Monte Carlo agreement, greeks."""

import math

import numpy as np
import pandas as pd
import pytest

from quantlab import derivatives as d

# The classic textbook case (Hull): S = K = 100, one year, r = 5%, sigma = 20%.
CASE = dict(spot=100.0, strike=100.0, years=1.0, rate=0.05, vol=0.20)


def test_textbook_prices():
    assert d.black_scholes(**CASE, kind="call") == pytest.approx(10.4506, abs=1e-4)
    assert d.black_scholes(**CASE, kind="put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
def test_put_call_parity(strike):
    """call - put = S - K e^(-rT): owning a call and selling a put is the same as a forward."""
    args = {**CASE, "strike": strike}
    call, put = d.black_scholes(**args, kind="call"), d.black_scholes(**args, kind="put")
    assert call - put == pytest.approx(args["spot"] - strike * math.exp(-args["rate"] * args["years"]), abs=1e-10)


def test_at_expiry_the_price_is_the_intrinsic_value():
    assert d.black_scholes(110, 100, 0, 0.05, 0.2, "call") == 10
    assert d.black_scholes(110, 100, 0, 0.05, 0.2, "put") == 0
    assert d.black_scholes(90, 100, 0, 0.05, 0.2, "put") == 10


def test_without_volatility_the_call_is_spot_minus_discounted_strike():
    assert d.black_scholes(100, 100, 1, 0.05, 0, "call") == pytest.approx(100 - 100 * math.exp(-0.05))


def test_more_volatility_makes_both_options_dearer():
    for kind in d.KINDS:
        assert d.black_scholes(**{**CASE, "vol": 0.3}, kind=kind) > d.black_scholes(**CASE, kind=kind)


@pytest.mark.parametrize("kind", d.KINDS)
@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
def test_monte_carlo_agrees_with_black_scholes(kind, strike):
    args = {**CASE, "strike": strike}
    price, error = d.monte_carlo_price(**args, kind=kind, paths=200_000, seed=42)
    assert abs(price - d.black_scholes(**args, kind=kind)) < 3 * error  # within 3 standard errors
    assert error < 0.05


def test_monte_carlo_is_reproducible_with_the_same_seed():
    assert d.monte_carlo_price(**CASE, seed=7) == d.monte_carlo_price(**CASE, seed=7)
    assert d.monte_carlo_price(**CASE, seed=7) != d.monte_carlo_price(**CASE, seed=8)


def test_more_paths_means_a_smaller_error():
    _, few = d.monte_carlo_price(**CASE, paths=10_000)
    _, many = d.monte_carlo_price(**CASE, paths=1_000_000)
    assert many < few / 5  # error falls like 1 / sqrt(paths): 100x paths -> 10x smaller


@pytest.mark.parametrize("kind", d.KINDS)
def test_greeks_match_small_changes_in_the_inputs(kind):
    """Each greek must equal the price change measured by nudging its input (finite differences)."""
    g = d.greeks(**CASE, kind=kind)

    def price(**change):
        return d.black_scholes(**{**CASE, **change}, kind=kind)

    h = 0.01
    assert g["delta"] == pytest.approx((price(spot=100 + h) - price(spot=100 - h)) / (2 * h), abs=1e-6)
    assert g["gamma"] == pytest.approx((price(spot=100 + h) - 2 * price() + price(spot=100 - h)) / h**2, abs=1e-4)
    assert g["vega"] == pytest.approx((price(vol=0.21) - price(vol=0.19)) / 2, abs=1e-3)  # per 1 vol point
    assert g["rho"] == pytest.approx((price(rate=0.06) - price(rate=0.04)) / 2, abs=1e-3)
    assert g["theta"] == pytest.approx(price(years=1 - 1 / 365) - price(), abs=1e-4)


def test_delta_ranges():
    assert 0 < d.greeks(**CASE, kind="call")["delta"] < 1
    assert -1 < d.greeks(**CASE, kind="put")["delta"] < 0


@pytest.mark.parametrize("kind", d.KINDS)
@pytest.mark.parametrize("vol", [0.05, 0.2, 0.8])
def test_implied_volatility_gives_back_the_volatility_used(kind, vol):
    price = d.black_scholes(**{**CASE, "vol": vol}, kind=kind)
    assert d.implied_volatility(price, 100, 100, 1, 0.05, kind) == pytest.approx(vol, abs=1e-6)


def test_implied_volatility_refuses_impossible_prices():
    with pytest.raises(ValueError):
        d.implied_volatility(0.01, 150, 100, 1, 0.05, "call")  # below the intrinsic value of ~55


def test_historical_volatility_finds_the_volatility_of_a_simulated_series():
    rng = np.random.default_rng(0)
    log_returns = rng.normal(0, 0.25 / math.sqrt(252), 5000)  # 25% a year
    prices = pd.Series(100 * np.exp(np.cumsum(log_returns)))
    assert d.historical_volatility(prices, window=5000) == pytest.approx(0.25, abs=0.01)


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        d.black_scholes(**CASE, kind="straddle")
    with pytest.raises(ValueError):
        d.black_scholes(-1, 100, 1, 0.05, 0.2)
    with pytest.raises(ValueError):
        d.greeks(100, 100, 0, 0.05, 0.2)
