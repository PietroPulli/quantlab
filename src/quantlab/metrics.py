"""Performance metrics computed from a series of periodic (daily) simple returns."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # trading days per year, used to annualise daily figures
MIN_STD = 1e-12  # below this a std is floating-point noise, not real variability


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """252 for markets closed at weekends (stocks, ETFs), 365 for 7-days-a-week ones (crypto).

    Counted from the dates themselves: more than 300 observations per calendar year means
    the asset also trades at weekends.
    """
    if len(index) < 2:
        return TRADING_DAYS
    span_days = (index[-1] - index[0]).days
    per_year = (len(index) - 1) / span_days * 365.25 if span_days else TRADING_DAYS
    return 365 if per_year > 300 else TRADING_DAYS


def total_return(returns: pd.Series) -> float:
    """Compounded return over the whole period: prod(1 + r) - 1."""
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Geometric average return per year (CAGR)."""
    n = len(returns)
    if n == 0:
        return np.nan
    growth = (1 + returns).prod()
    return float(growth ** (periods_per_year / n) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Standard deviation of returns scaled by sqrt(periods): volatility grows like sqrt(time)."""
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised Sharpe ratio. `risk_free` is an annual rate, converted to per-period."""
    rf_per_period = (1 + risk_free) ** (1 / periods_per_year) - 1
    excess = returns - rf_per_period
    std = excess.std(ddof=1)
    # A constant series gives std ~1e-19 in floating point, not exactly 0: without
    # a tolerance the ratio would explode. `not >` also catches NaN (< 2 observations).
    if not std > MIN_STD:
        return np.nan
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def drawdown(returns: pd.Series) -> pd.Series:
    """Percentage below the previous equity peak, day by day (0 = at a new high).

    The starting capital (1.0) counts as the first peak, so a loss on day one
    is already a drawdown.
    """
    equity = (1 + returns).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    return equity / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss, as a negative number (e.g. -0.34 = -34%)."""
    return float(drawdown(returns).min()) if len(returns) else np.nan


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annual return divided by the size of the worst drawdown (return per unit of pain).

    Undefined (NaN) when there was no drawdown at all: dividing by zero would give
    infinity, which would look like a perfect strategy instead of "not measurable".
    """
    mdd = max_drawdown(returns)
    if not mdd < 0:  # also catches NaN (empty series)
        return np.nan
    return float(annualized_return(returns, periods_per_year) / abs(mdd))


def summary(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS, risk_free: float = 0.0
) -> pd.Series:
    """The standard metrics in one Series, ready to be put side by side in a table.

    `risk_free` (annual) is subtracted in the Sharpe ratio: if idle cash earns interest,
    the Sharpe must measure the return *above* that, or sitting in cash would look skilful.
    """
    return pd.Series(
        {
            "total_return": total_return(returns),
            "annual_return": annualized_return(returns, periods_per_year),
            "annual_volatility": annualized_volatility(returns, periods_per_year),
            "sharpe": sharpe_ratio(returns, risk_free, periods_per_year),
            "max_drawdown": max_drawdown(returns),
        }
    )


def compare(
    results: dict[str, pd.Series], periods_per_year: int = TRADING_DAYS, risk_free: float = 0.0
) -> pd.DataFrame:
    """Metrics table with one column per strategy, e.g. {"strategy": r1, "buy_and_hold": r2}."""
    return pd.DataFrame(
        {name: summary(r, periods_per_year, risk_free) for name, r in results.items()}
    )
