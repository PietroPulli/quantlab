"""Backtest of a portfolio of several assets, rebalanced only on decision days.

Between decisions the weights drift with prices (an asset that rises becomes a bigger
share of the portfolio), exactly like a real account that is not touched.
"""

import numpy as np
import pandas as pd

from quantlab.backtest import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE

TRADING_DAYS = 252


def daily_cash_from_yield(yield_percent: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Daily return of cash from an annual T-bill yield in percent (e.g. ^IRX = 4.5 -> 4.5% a year)."""
    annual = yield_percent.reindex(index).ffill().fillna(0.0) / 100  # missing days: last known yield
    return (1 + annual) ** (1 / TRADING_DAYS) - 1


def run_portfolio_backtest(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    cash_return: pd.Series | float = 0.0,
) -> pd.DataFrame:
    """Backtest target weights decided on some days (rows of `targets`) and held until the next decision.

    `targets` has one row per decision day (decided with data up to that close) and one column
    per asset of `prices`; weights must be >= 0 and sum to at most 1 (the rest is cash).
    The new weights start earning from the next day, and the trade pays costs on that day.
    `cash_return` is the DAILY return of uninvested cash (a number or a daily Series).

    Returns a DataFrame indexed like `prices`: net_return, cost, invested (share of capital in assets).
    """
    if commission < 0 or slippage < 0:
        raise ValueError("commission and slippage must be >= 0")
    targets = targets.reindex(columns=prices.columns).fillna(0.0)
    if (targets < 0).any().any() or (targets.sum(axis=1) > 1 + 1e-9).any():
        raise ValueError("weights must be >= 0 and sum to at most 1 (no leverage, no shorting)")
    if not targets.index.isin(prices.index).all():
        raise ValueError("every decision day must be a day of `prices`")
    if ((targets > 0) & prices.loc[targets.index].isna()).any().any():
        raise ValueError("a weight was given to an asset with no price on that day")

    returns = prices.pct_change(fill_method=None).fillna(0.0).to_numpy()
    cash = (cash_return.reindex(prices.index).fillna(0.0).to_numpy() if isinstance(cash_return, pd.Series)
            else np.full(len(prices), float(cash_return)))
    decision_row = {prices.index.get_loc(day): row.to_numpy() for day, row in targets.iterrows()}

    weights = np.zeros(prices.shape[1])  # start fully in cash
    pending = None  # target decided yesterday, applied today
    net, costs, invested = np.zeros(len(prices)), np.zeros(len(prices)), np.zeros(len(prices))
    for t in range(len(prices)):
        cost = 0.0
        if pending is not None:  # trade at yesterday's close into the new weights
            cost = np.abs(pending - weights).sum() * (commission + slippage)
            weights = pending
            pending = None
        growth = weights * (1 + returns[t])  # each asset's share after today's move
        cash_share = 1 - weights.sum()
        day_return = growth.sum() + cash_share * (1 + cash[t]) - 1
        net[t] = day_return - cost
        costs[t] = cost
        invested[t] = weights.sum()
        weights = growth / (1 + day_return)  # drift: today's winners weigh more tomorrow
        if t in decision_row:
            pending = decision_row[t]

    return pd.DataFrame({"net_return": net, "cost": costs, "invested": invested}, index=prices.index)
