"""Build a long/flat strategy from simple rules, e.g. "buy when price > 100-day average".

A rule compares an indicator with another indicator or with a number. A strategy is
an entry rule plus an optional exit rule. Every indicator on day t uses only prices
up to and including day t, so the strategies built here have no look-ahead bias
(the backtest then shifts the signal by one day before trading).
"""

import operator
from dataclasses import dataclass

import pandas as pd

# name -> (description, function(prices, window) -> Series)
INDICATORS = {
    "price": ("Price", lambda p, w: p),
    "sma": ("Moving average", lambda p, w: p.rolling(w).mean()),
    "return": ("Return over the window", lambda p, w: p / p.shift(w) - 1),
    "zscore": ("z-score vs moving average", lambda p, w: (p - p.rolling(w).mean()) / p.rolling(w).std()),
    "high": ("Highest price of the window", lambda p, w: p.rolling(w).max()),
    "low": ("Lowest price of the window", lambda p, w: p.rolling(w).min()),
}

OPERATORS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}


@dataclass(frozen=True)
class Indicator:
    name: str  # a key of INDICATORS
    window: int = 1  # days of history used (ignored by "price")

    def compute(self, prices: pd.Series) -> pd.Series:
        if self.name not in INDICATORS:
            raise ValueError(f"unknown indicator {self.name!r}, choose from {list(INDICATORS)}")
        if self.name != "price" and self.window < 2:
            raise ValueError(f"{self.name} needs a window of at least 2 days")
        return INDICATORS[self.name][1](prices, self.window)


@dataclass(frozen=True)
class Condition:
    left: Indicator
    op: str  # a key of OPERATORS
    right: Indicator | float  # another indicator, or a fixed number

    def evaluate(self, prices: pd.Series) -> pd.Series:
        """True on the days the condition holds. Days without enough history are False."""
        if self.op not in OPERATORS:
            raise ValueError(f"unknown operator {self.op!r}, choose from {list(OPERATORS)}")
        left = self.left.compute(prices)
        right = self.right.compute(prices) if isinstance(self.right, Indicator) else self.right
        return OPERATORS[self.op](left, right)  # any comparison with NaN is False: warm-up -> no signal


Rule = Condition | list[Condition]  # a list means ALL its conditions must hold (AND)


def holds(prices: pd.Series, rule: Rule) -> pd.Series:
    """True on the days the rule holds; for a list, on the days every condition holds."""
    conditions = [rule] if isinstance(rule, Condition) else rule
    if not conditions:
        raise ValueError("a rule needs at least one condition")
    result = conditions[0].evaluate(prices)
    for condition in conditions[1:]:
        result = result & condition.evaluate(prices)
    return result


def rule_strategy(prices: pd.Series, entry: Rule, exit: Rule | None = None) -> pd.Series:
    """Daily signals (1 = long, 0 = cash) from an entry rule and an optional exit rule.

    Each rule is one Condition or a list of Conditions that must all hold.
    Without an exit rule we are long exactly while the entry rule holds.
    With one, we enter when `entry` becomes true and stay long until `exit` is true
    (same forward-fill trick as mean_reversion). A day where both are true is not a
    decision, so the previous position is kept.
    """
    go_long = holds(prices, entry)
    if exit is None:
        return go_long.astype(float)
    go_flat = holds(prices, exit)
    signal = pd.Series(float("nan"), index=prices.index)
    signal[go_long & ~go_flat] = 1.0
    signal[go_flat & ~go_long] = 0.0
    return signal.ffill().fillna(0.0)
