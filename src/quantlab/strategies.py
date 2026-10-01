"""Classic long/flat strategies. Each one turns prices into daily signals (1 = long, 0 = cash).

Rule for every strategy here: the signal on day t may only use prices up to and
including day t. The backtest then shifts it by one day before trading.
During the warm-up period (not enough history yet) the signal is 0.
"""

import pandas as pd


def buy_and_hold(prices: pd.Series) -> pd.Series:
    """Always long. This is the benchmark every other strategy must beat."""
    return pd.Series(1.0, index=prices.index)


def momentum(prices: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Long if the price rose over the past `lookback` days, ignoring the last `skip` days.

    The classic "12-1" momentum: 12-month return excluding the most recent month,
    because very short-term returns tend to reverse instead of persist.
    """
    if not 0 <= skip < lookback:
        raise ValueError("need 0 <= skip < lookback")
    past_return = prices.shift(skip) / prices.shift(lookback) - 1
    return (past_return > 0).astype(float)  # NaN > 0 is False -> 0 during warm-up


def moving_average_crossover(prices: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Long when the fast moving average is above the slow one (trend following)."""
    if not 0 < fast < slow:
        raise ValueError("need 0 < fast < slow")
    fast_ma = prices.rolling(fast).mean()  # mean of the last `fast` prices, t included
    slow_ma = prices.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float)


def mean_reversion(
    prices: pd.Series, window: int = 20, entry_z: float = -1.0, exit_z: float = 0.0
) -> pd.Series:
    """Buy when the price is unusually low vs its recent average, sell when it is back.

    z-score = (price - rolling mean) / rolling std. Enter long when z < entry_z,
    exit when z > exit_z, and in between keep whatever position we had.
    """
    if not entry_z < exit_z:
        raise ValueError("need entry_z < exit_z")
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    z = (prices - mean) / std

    # Mark only the days where a decision happens (1 = enter, 0 = exit), leave the
    # rest empty, then forward-fill: each day carries the last decision taken.
    signal = pd.Series(float("nan"), index=prices.index)
    signal[z < entry_z] = 1.0
    signal[z > exit_z] = 0.0
    return signal.ffill().fillna(0.0)


def breakout(prices: pd.Series, window: int = 50) -> pd.Series:
    """Go long on a new `window`-day high, go to cash on a new `window`-day low.

    In between keep the last decision, with the same forward-fill trick as mean_reversion.
    """
    if window < 2:
        raise ValueError("need window >= 2")
    high = prices.rolling(window).max()  # highest price of the last `window` days, t included
    low = prices.rolling(window).min()

    is_high = prices >= high  # today is the highest of the window
    is_low = prices <= low  # today is the lowest of the window
    # In a flat window today is both the high and the low: that is not a decision,
    # so we keep the previous position instead of letting "exit" win by default.
    signal = pd.Series(float("nan"), index=prices.index)
    signal[is_high & ~is_low] = 1.0
    signal[is_low & ~is_high] = 0.0
    return signal.ffill().fillna(0.0)


STRATEGIES = {
    "buy_and_hold": buy_and_hold,
    "momentum": momentum,
    "moving_average_crossover": moving_average_crossover,
    "mean_reversion": mean_reversion,
    "breakout": breakout,
}
