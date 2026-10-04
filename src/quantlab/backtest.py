"""Vectorised single-asset backtest: from signals to net returns, costs included."""

import numpy as np
import pandas as pd

DEFAULT_COMMISSION = 0.001  # 0.10% of traded value per trade (typical retail broker)
DEFAULT_SLIPPAGE = 0.0005  # 0.05% lost to bid-ask spread and price impact per trade
TRADING_DAYS = 252


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
    cash_rate: float = 0.0,
) -> pd.DataFrame:
    """Backtest one asset. Signals are target exposures in [-1, 1] (1 = fully long).

    `cash_rate` is the annual interest earned by the capital that is not invested
    (e.g. 0.04 for T-bills at 4%). Without it, a strategy that is often in cash
    would be unfairly penalised against buy & hold.

    Returns a DataFrame indexed like `prices` with columns: position, asset_return,
    cash_return, gross_return, cost, net_return, equity (starts from 1).
    """
    if commission < 0 or slippage < 0:
        raise ValueError("commission and slippage must be >= 0")
    if not -0.1 < cash_rate < 0.5:
        raise ValueError("cash_rate is an annual rate, e.g. 0.04 for 4%")
    if not signals.index.equals(prices.index):
        raise ValueError("signals and prices must share the same index")
    if signals.isna().any() or (signals.abs() > 1).any():
        raise ValueError("signals must be non-missing and within [-1, 1] (no leverage)")

    position = signals_to_positions(signals)
    asset_return = prices.pct_change(fill_method=None).fillna(0.0)
    # Compound annual rate -> daily rate, paid on the idle fraction of capital.
    daily_cash = (1 + cash_rate) ** (1 / TRADING_DAYS) - 1
    cash_return = (1 - position.abs()) * daily_cash
    cash_return.iloc[:1] = 0.0  # day 0 is the starting point: no time has passed yet
    gross = position * asset_return + cash_return
    # Every change of position is a trade: we pay costs on the traded fraction of capital.
    turnover = position.diff().fillna(position).abs()
    cost = turnover * (commission + slippage)
    net = gross - cost

    return pd.DataFrame(
        {
            "position": position,
            "asset_return": asset_return,
            "cash_return": cash_return,
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


def trade_log(prices: pd.Series, positions: pd.Series) -> pd.DataFrame:
    """List of trades: date, "buy"/"sell", traded fraction of capital and execution price.

    The position held on day t was decided and traded at the close of day t-1, so a
    change of position between t and t+1 is a trade executed on day t at that close.
    """
    change = positions.diff().fillna(positions)  # change[t]: traded at the close of t-1
    traded = change.shift(-1).fillna(0.0)  # traded[t]: executed at the close of t
    executed = traded[traded != 0]
    return pd.DataFrame(
        {
            "action": np.where(executed > 0, "buy", "sell"),
            "size": executed.abs(),
            "price": prices[executed.index],
        },
        index=executed.index,
    )


def current_signal(signals: pd.Series) -> tuple[pd.Timestamp, float, pd.Timestamp]:
    """What the rule says at the last available close, and since when it has said so.

    Returns (date of the last signal, its value, first date of the current streak).
    A signal computed at the close of that date is traded from the next session on.
    """
    if signals.empty:
        raise ValueError("no signals")
    last = signals.iloc[-1]
    different = (signals != last).to_numpy().nonzero()[0]  # positions where the value differs
    since = signals.index[different[-1] + 1] if len(different) else signals.index[0]
    return signals.index[-1], float(last), since
