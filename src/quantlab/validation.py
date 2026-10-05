"""Out-of-sample tools: chronological split, parameter search, walk-forward, block bootstrap.

A strategy here is any function `strategy(prices, **params) -> signals` that never
looks ahead (see tests/test_strategies.py). That property lets us compute signals
on the whole history up to a date and then keep only the window we care about:
indicators get their warm-up from genuinely past data.
"""

import itertools
from collections.abc import Callable

import numpy as np
import pandas as pd

from quantlab.backtest import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, run_backtest
from quantlab.metrics import TRADING_DAYS, sharpe_ratio

Strategy = Callable[..., pd.Series]


def split_date(index: pd.DatetimeIndex, in_sample_fraction: float = 0.7) -> pd.Timestamp:
    """First out-of-sample date: the first `in_sample_fraction` of days are in-sample.

    The split is chronological, never random: shuffling days would let the
    in-sample period "see" the future.
    """
    if not 0 < in_sample_fraction < 1:
        raise ValueError("in_sample_fraction must be between 0 and 1")
    return index[int(len(index) * in_sample_fraction)]


def param_grid(**options: list) -> list[dict]:
    """All combinations of parameter values: param_grid(a=[1, 2], b=[3]) -> [{a:1,b:3}, {a:2,b:3}]."""
    names = list(options)
    return [dict(zip(names, values)) for values in itertools.product(*options.values())]


def strategy_returns(
    prices: pd.Series,
    strategy: Strategy,
    params: dict,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    cash_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """Net daily returns of `strategy(prices, **params)` after costs (idle cash earns `cash_rate`)."""
    signals = strategy(prices, **params)
    return run_backtest(prices, signals, commission, slippage, cash_rate, periods_per_year)["net_return"]


def optimize(
    prices: pd.Series,
    strategy: Strategy,
    grid: list[dict],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    cash_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> tuple[dict, pd.DataFrame]:
    """Pick the params with the best Sharpe on the dates [start, end) only.

    Prices after `end` are never passed to the strategy, so the choice cannot
    depend on them. Returns (best_params, table of every combination's score).
    """
    history = prices[prices.index < end] if end is not None else prices
    rows = []
    for params in grid:
        net = strategy_returns(history, strategy, params, commission, slippage, cash_rate, periods_per_year)
        window = net[net.index >= start] if start is not None else net
        rows.append({**params, "sharpe": sharpe_ratio(window, cash_rate, periods_per_year)})
    scores = pd.DataFrame(rows)
    # NaN Sharpe (strategy never traded) must never win: rank it below everything.
    best = int(scores["sharpe"].fillna(-np.inf).to_numpy().argmax())
    return grid[best], scores


def walk_forward(
    prices: pd.Series,
    strategy: Strategy,
    grid: list[dict],
    train_days: int = 756,
    test_days: int = 252,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    cash_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> tuple[pd.Series, pd.DataFrame]:
    """Re-optimise on a rolling training window, then trade the next unseen window.

    Default: choose params on the last 3 years (756 days), use them for 1 year
    (252 days), slide forward by 1 year, repeat. Every out-of-sample day is traded
    with params chosen without seeing it. Returns (net returns over the
    out-of-sample days, one row per fold describing what was chosen).
    """
    n = len(prices)
    if train_days < 2 or test_days < 1 or train_days + test_days > n:
        raise ValueError("not enough data for one train + test window")

    signals = pd.Series(0.0, index=prices.index)  # flat until the first test window
    folds = []
    for test_start in range(train_days, n, test_days):
        test_end = min(test_start + test_days, n)
        best, scores = optimize(
            prices,
            strategy,
            grid,
            start=prices.index[test_start - train_days],
            end=prices.index[test_start],
            commission=commission,
            slippage=slippage,
            cash_rate=cash_rate,
            periods_per_year=periods_per_year,
        )
        # Signals for the test window, computed only from prices up to test_end.
        window_signals = strategy(prices.iloc[:test_end], **best).iloc[test_start:test_end]
        signals.iloc[test_start:test_end] = window_signals.to_numpy()
        folds.append(
            {
                "test_start": prices.index[test_start],
                "test_end": prices.index[test_end - 1],
                "params": best,
                "train_sharpe": scores["sharpe"].max(),
            }
        )

    net = run_backtest(prices, signals, commission, slippage, cash_rate, periods_per_year)["net_return"]
    return net.iloc[train_days:], pd.DataFrame(folds)


def block_bootstrap(
    data: pd.Series | pd.DataFrame,
    statistic: Callable[[pd.Series | pd.DataFrame], float],
    n_samples: int = 1000,
    block_size: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Distribution of `statistic` over resampled histories (moving block bootstrap).

    Each fake history is built by gluing random blocks of `block_size` consecutive
    days until it is as long as the original. Blocks (not single days) keep the
    volatility clustering of real markets. With a DataFrame, all columns are
    resampled on the same days, so paired comparisons stay paired.
    """
    n = len(data)
    if not 1 <= block_size <= n:
        raise ValueError("block_size must be between 1 and len(data)")
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block_size)  # ceiling division
    starts = rng.integers(0, n - block_size + 1, size=(n_samples, n_blocks))
    # starts[:, :, None] + arange -> every day of every block; trim to length n.
    positions = (starts[:, :, None] + np.arange(block_size)).reshape(n_samples, -1)[:, :n]
    return np.array([statistic(data.iloc[p]) for p in positions])


def sharpe_difference_bootstrap(
    strategy_net: pd.Series,
    benchmark_net: pd.Series,
    n_samples: int = 1000,
    block_size: int = 20,
    seed: int = 42,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> np.ndarray:
    """Bootstrap samples of Sharpe(strategy) - Sharpe(benchmark), resampling days in pairs."""
    paired = pd.DataFrame({"strategy": strategy_net, "benchmark": benchmark_net})
    return block_bootstrap(
        paired,
        lambda d: (sharpe_ratio(d["strategy"], risk_free, periods_per_year)
                   - sharpe_ratio(d["benchmark"], risk_free, periods_per_year)),
        n_samples,
        block_size,
        seed,
    )


def confidence_interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Central interval containing `level` of the bootstrap samples (NaNs ignored)."""
    if np.isnan(samples).all():  # e.g. a strategy that never invests has no Sharpe at all
        return np.nan, np.nan
    tail = (1 - level) / 2 * 100
    low, high = np.nanpercentile(samples, [tail, 100 - tail])
    return float(low), float(high)
