"""European options: Black-Scholes price and greeks, a Monte Carlo check, implied volatility.

Conventions used everywhere in this module:
- `years` is the time to expiry in years (30 days = 30 / 365);
- `rate` and `vol` are annual and continuously compounded (0.03 = 3%, 0.20 = 20%);
- `kind` is "call" (right to buy at the strike) or "put" (right to sell at the strike).

Black-Scholes assumes the price moves like a geometric Brownian motion with constant volatility
and no jumps. Real markets break this (crashes, volatility that changes): the model is the common
language of option markets, not a description of reality. The Monte Carlo pricer simulates the
same assumption path by path, so the two must agree: that agreement is what the tests check.
"""

import math

import numpy as np
import pandas as pd

from quantlab.metrics import TRADING_DAYS

KINDS = ("call", "put")


def norm_cdf(x: float) -> float:
    """P(Z <= x) for a standard normal Z, via the error function (no SciPy needed)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x: float) -> float:
    """Density of the standard normal at x: the bell curve."""
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def _check(spot: float, strike: float, years: float, vol: float, kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be 'call' or 'put', not {kind!r}")
    if spot <= 0 or strike <= 0 or years < 0 or vol < 0:
        raise ValueError("spot and strike must be positive, years and vol not negative")


def intrinsic_value(spot: float, strike: float, kind: str) -> float:
    """What the option is worth if exercised right now: max(S - K, 0) for a call, max(K - S, 0) for a put."""
    return max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)


def _d1_d2(spot: float, strike: float, years: float, rate: float, vol: float) -> tuple[float, float]:
    """The two standardised distances of Black-Scholes. N(d2) = risk-neutral probability of ending in the money."""
    spread = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + vol**2 / 2) * years) / spread
    return d1, d1 - spread


def black_scholes(spot: float, strike: float, years: float, rate: float, vol: float, kind: str = "call") -> float:
    """Fair price of a European option under Black-Scholes.

    call = S N(d1) - K e^(-rT) N(d2);  put = K e^(-rT) N(-d2) - S N(-d1).
    At expiry (or with zero volatility) the uncertainty is gone and the price is the
    discounted intrinsic value.
    """
    _check(spot, strike, years, vol, kind)
    discount = math.exp(-rate * years)
    if years == 0 or vol == 0:
        forward = spot * math.exp(rate * years)  # where the price surely goes without randomness
        return discount * intrinsic_value(forward, strike, kind)
    d1, d2 = _d1_d2(spot, strike, years, rate, vol)
    if kind == "call":
        return spot * norm_cdf(d1) - strike * discount * norm_cdf(d2)
    return strike * discount * norm_cdf(-d2) - spot * norm_cdf(-d1)


def greeks(spot: float, strike: float, years: float, rate: float, vol: float, kind: str = "call") -> dict[str, float]:
    """How the price reacts to each input, in the units traders quote.

    delta: change in price for +1 in the spot     gamma: change in delta for +1 in the spot
    vega:  change for +1 volatility point (1%)    theta: change after one calendar day
    rho:   change for +1 rate point (1%)
    """
    _check(spot, strike, years, vol, kind)
    if years == 0 or vol == 0:
        raise ValueError("greeks need time to expiry and volatility above zero")
    d1, d2 = _d1_d2(spot, strike, years, rate, vol)
    discount = math.exp(-rate * years)
    gamma = norm_pdf(d1) / (spot * vol * math.sqrt(years))
    vega = spot * norm_pdf(d1) * math.sqrt(years)
    decay = -spot * norm_pdf(d1) * vol / (2 * math.sqrt(years))  # time value lost, same for call and put
    if kind == "call":
        delta, theta = norm_cdf(d1), decay - rate * strike * discount * norm_cdf(d2)
        rho = strike * years * discount * norm_cdf(d2)
    else:
        delta, theta = norm_cdf(d1) - 1, decay + rate * strike * discount * norm_cdf(-d2)
        rho = -strike * years * discount * norm_cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega / 100, "theta": theta / 365, "rho": rho / 100}


def monte_carlo_price(spot: float, strike: float, years: float, rate: float, vol: float, kind: str = "call",
                      paths: int = 100_000, seed: int = 0) -> tuple[float, float]:
    """Price by simulation: (price, standard error). Should match black_scholes within ~2 standard errors.

    Each path draws the price at expiry S_T = S e^((r - σ²/2)T + σ√T Z), with Z standard normal,
    pays the option's payoff there, and the price is the discounted average payoff.
    Antithetic variates: every Z is also used as -Z, which cancels part of the noise for free.
    """
    _check(spot, strike, years, vol, kind)
    z = np.random.default_rng(seed).standard_normal(paths // 2)
    z = np.concatenate([z, -z])
    terminal = spot * np.exp((rate - vol**2 / 2) * years + vol * math.sqrt(years) * z)
    payoff = np.maximum(terminal - strike, 0) if kind == "call" else np.maximum(strike - terminal, 0)
    discounted = math.exp(-rate * years) * payoff
    # Antithetic pairs are not independent: the error is measured on the pair averages.
    pairs = (discounted[: paths // 2] + discounted[paths // 2:]) / 2
    return float(discounted.mean()), float(pairs.std(ddof=1) / math.sqrt(len(pairs)))


def implied_volatility(price: float, spot: float, strike: float, years: float, rate: float,
                       kind: str = "call", low: float = 1e-4, high: float = 5.0, tol: float = 1e-8) -> float:
    """The volatility that makes Black-Scholes return `price`: what the market "expects", read backwards.

    Bisection: the option price grows with volatility, so halve [low, high] until it is tight.
    Raises ValueError if no volatility in [low, high] gives that price (e.g. below intrinsic value).
    """
    def gap(vol: float) -> float:
        return black_scholes(spot, strike, years, rate, vol, kind) - price

    if gap(low) > 0 or gap(high) < 0:
        raise ValueError("no volatility between low and high gives this price")
    while high - low > tol:
        mid = (low + high) / 2
        low, high = (mid, high) if gap(mid) < 0 else (low, mid)
    return (low + high) / 2


def historical_volatility(prices: pd.Series, window: int = TRADING_DAYS) -> float:
    """Annualised volatility of daily log returns over the last `window` days: the usual input for `vol`."""
    log_returns = np.log(prices / prices.shift(1)).dropna().iloc[-window:]
    if len(log_returns) < 2:
        raise ValueError("need at least 3 prices")
    return float(log_returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
