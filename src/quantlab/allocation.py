"""Published monthly allocation rules (see docs/RICERCA_STRATEGIE.md for sources and parameters).

Each function returns target weights: one row per month end, decided with data up to that day,
ready for portfolio.run_portfolio_backtest.
"""

import numpy as np
import pandas as pd

from quantlab.ml import month_ends

TRADING_DAYS = 252


def _month_end_rows(prices: pd.DataFrame | pd.Series) -> pd.DatetimeIndex:
    return month_ends(prices.dropna(how="all").index if isinstance(prices, pd.DataFrame) else prices.dropna().index)


def volatility_target(prices: pd.Series, target_vol: float = 0.15, window: int = 21) -> pd.DataFrame:
    """Moreira & Muir (2017): weight = min(1, target_vol / volatility of the last `window` days)."""
    daily = prices.pct_change(fill_method=None)
    realized = daily.rolling(window).std() * np.sqrt(TRADING_DAYS)  # annualized, known at today's close
    weight = (target_vol / realized).clip(upper=1.0)  # never above 100%: no leverage
    ends = _month_end_rows(prices)
    return weight.loc[ends].dropna().to_frame(prices.name)


def trend_allocation(prices: pd.DataFrame, months: int = 10) -> pd.DataFrame:
    """Faber (2007): equal weights; each asset held only if above its `months`-month average, else cash.

    Decisions start when every asset has `months` month-end prices.
    """
    ends = _month_end_rows(prices)
    monthly = prices.loc[ends]
    average = monthly.rolling(months).mean()  # NaN until an asset has `months` month ends
    ready = average.notna().all(axis=1)
    above = (monthly > average).astype(float)
    return (above[ready] / prices.shape[1])


def dual_momentum(
    prices: pd.DataFrame, tbill_yield: pd.Series, us: str = "SPY", world: str = "EFA", bonds: str = "AGG",
    months: int = 12,
) -> pd.DataFrame:
    """Antonacci (2014): if US stocks beat T-bills over `months`, hold the better of US and world stocks; else bonds.

    `tbill_yield` is the annual T-bill yield in percent (^IRX). Decisions start when all three assets
    have a full lookback.
    """
    ends = _month_end_rows(prices)
    monthly = prices.loc[ends, [us, world, bonds]]
    momentum = monthly / monthly.shift(months) - 1  # return over the last `months` month ends
    # T-bill return over the same window: compound the daily yield of the last months*21 trading days.
    daily_yield = tbill_yield.reindex(prices.index).ffill() / 100 / TRADING_DAYS
    tbill = (1 + daily_yield).rolling(months * 21).apply(np.prod, raw=True) - 1
    tbill = tbill.reindex(ends)

    ready = momentum.notna().all(axis=1) & tbill.notna()
    weights = pd.DataFrame(0.0, index=ends[ready], columns=[us, world, bonds])
    for day in weights.index:
        if momentum.loc[day, us] > tbill.loc[day]:  # absolute momentum: stocks beat cash
            best = us if momentum.loc[day, us] >= momentum.loc[day, world] else world  # relative momentum
            weights.loc[day, best] = 1.0
        else:
            weights.loc[day, bonds] = 1.0
    return weights


def fixed_weights(prices: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Reference portfolio rebalanced to the same weights at every month end."""
    ends = _month_end_rows(prices[list(weights)].dropna())
    return pd.DataFrame({name: w for name, w in weights.items()}, index=ends)
