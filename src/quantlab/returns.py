"""Return calculations from price series."""

import numpy as np
import pandas as pd


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Simple returns r_t = P_t / P_{t-1} - 1. The first row is NaN (no previous price)."""
    return prices.pct_change(fill_method=None)


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Log returns r_t = ln(P_t / P_{t-1}). The first row is NaN (no previous price)."""
    return np.log(prices / prices.shift(1))
