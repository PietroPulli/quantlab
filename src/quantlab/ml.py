"""Machine learning on one asset: monthly in/out decisions learned from past examples.

Each example is one month end: the features describe the market up to that day,
the target says whether the price rose over the following month.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of every month in `index` (the days we decide on)."""
    days = pd.Series(index, index=index)
    return pd.DatetimeIndex(days.groupby(index.to_period("M")).max().values)


def make_features(prices: pd.Series, vix: pd.Series | None = None) -> pd.DataFrame:
    """One row per month end, built only from data known on that day.

    Rows without a full history (e.g. the first year, needed by ret_12m) are dropped.
    """
    daily = prices.pct_change(fill_method=None)
    features = pd.DataFrame(
        {
            "ret_1m": prices.pct_change(21, fill_method=None),  # return over the last ~1 month (21 trading days)
            "ret_3m": prices.pct_change(63, fill_method=None),
            "ret_6m": prices.pct_change(126, fill_method=None),
            "ret_12m": prices.pct_change(252, fill_method=None),
            "vol_1m": daily.rolling(21).std() * np.sqrt(TRADING_DAYS),  # annualized volatility of the last month
            "dist_ma200": prices / prices.rolling(200).mean() - 1,  # +5% = price 5% above its 200-day average
        }
    )
    if vix is not None:
        # VIX may miss a few days of the SPY calendar: carry the last known value forward (never backward).
        features["vix"] = vix.reindex(prices.index).ffill()
    return features.loc[month_ends(prices.index)].dropna()


def make_target(prices: pd.Series) -> pd.Series:
    """At each month end: 1.0 if the price is higher at the next month end, else 0.0.

    The last month end has no next month yet: its target is NaN (unknown).
    """
    ends = month_ends(prices.index)
    at_ends = prices.loc[ends]
    next_return = at_ends.shift(-1) / at_ends - 1  # shift(-1): the NEXT month end's price, the future we predict
    return (next_return > 0).astype(float).where(next_return.notna())
