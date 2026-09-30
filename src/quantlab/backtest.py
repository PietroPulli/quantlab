"""Vectorised single-asset backtest: from signals to net returns, costs included."""

import numpy as np
import pandas as pd

DEFAULT_COMMISSION = 0.001  # 0.10% of traded value per trade (typical retail broker)
DEFAULT_SLIPPAGE = 0.0005  # 0.05% lost to bid-ask spread and price impact per trade


def signals_to_positions(signals: pd.Series) -> pd.Series:
    """Position held on day t = signal computed at the close of day t-1.

    This one-day shift is what prevents look-ahead bias: a signal that uses the
    close of day t can only be traded after that close, so it earns day t+1's return.
    Before the first signal we are flat (position 0).
    """
    return signals.shift(1).fillna(0.0)


def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
) -> pd.DataFrame:
    """Backtest one asset. Signals are target exposures in [-1, 1] (1 = fully long).

    Returns a DataFrame indexed like `prices` with columns:
    position, asset_return, gross_return, cost, net_return, equity (starts from 1).
    """
    if commission < 0 or slippage < 0:
        raise ValueError("commission and slippage must be >= 0")
    if not signals.index.equals(prices.index):
        raise ValueError("signals and prices must share the same index")
    if signals.isna().any() or (signals.abs() > 1).any():
        raise ValueError("signals must be non-missing and within [-1, 1] (no leverage)")

    position = signals_to_positions(signals)
    asset_return = prices.pct_change(fill_method=None).fillna(0.0)
    gross = position * asset_return
    # Every change of position is a trade: we pay costs on the traded fraction of capital.
    turnover = position.diff().fillna(position).abs()
    cost = turnover * (commission + slippage)
    net = gross - cost

    return pd.DataFrame(
        {
            "position": position,
            "asset_return": asset_return,
            "gross_return": gross,
            "cost": cost,
            "net_return": net,
            "equity": (1 + net).cumprod(),
        }
    )


def turnover_per_year(positions: pd.Series, periods_per_year: int = 252) -> float:
    """Average fraction of capital traded per year (1.0 = the whole portfolio once)."""
    traded = positions.diff().fillna(positions).abs().sum()
    return float(traded / len(positions) * periods_per_year) if len(positions) else np.nan
