"""Try every idea on one asset and keep only an edge that survives the multiple-testing penalty.

Testing K ideas and keeping the best is a lottery with K tickets: some idea will look good
by luck. The Bonferroni correction makes each test stricter: the bootstrap interval of the
Sharpe difference is computed at level 1 - (1 - level) / K instead of `level`
(8 ideas at 95% -> 99.4%). An idea "beats buy & hold" only if even that interval is above zero.
"""

import pandas as pd

from quantlab.backtest import current_signal
from quantlab.report import evaluate_strategy
from quantlab.validation import Strategy, confidence_interval


def bonferroni_level(level: float, n_tests: int) -> float:
    """Confidence level each of `n_tests` tests needs so that the family keeps `level`."""
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1")
    return 1 - (1 - level) / n_tests


def scan(prices: pd.Series, ideas: dict[str, tuple[Strategy, dict]], cash_rate: float = 0.0,
         level: float = 0.95, n_bootstrap: int = 500, seed: int = 42,
         in_sample_fraction: float = 0.7) -> pd.DataFrame:
    """One row per idea: out-of-sample results, corrected verdict and what it says today.

    The first `in_sample_fraction` of the days is history only; the rest is the judged test period.
    """
    strict = bonferroni_level(level, len(ideas))
    rows = []
    for name, (strategy, params) in ideas.items():
        report = evaluate_strategy(prices, strategy, params=params, name=name, cash_rate=cash_rate,
                                   n_bootstrap=n_bootstrap, seed=seed, in_sample_fraction=in_sample_fraction)
        low, high = confidence_interval(report.sharpe_diff_samples, strict)
        verdict = "good" if low > 0 else ("bad" if high < 0 else "neutral")
        _, today, since = current_signal(strategy(prices, **params))
        oos = report.out_of_sample
        rows.append({
            "idea": name,
            "total_return": oos.loc["total_return", "strategy"],
            "bh_total_return": oos.loc["total_return", "buy_and_hold"],
            "sharpe_diff": report.sharpe_diff,
            "ci_low": low,
            "ci_high": high,
            "verdict": verdict,
            "signal_today": today,
            "since": since,
        })
    table = pd.DataFrame(rows)
    table.attrs["level"] = strict  # remembered for display
    table.attrs["history"] = (prices.index[0], prices.index[-1])  # data used
    table.attrs["test_start"] = report.split if rows else None  # first day of the judged period
    return table


def best_idea(table: pd.DataFrame) -> pd.Series | None:
    """The idea with the largest Sharpe gain among those that pass the corrected test, or None."""
    passed = table[table["verdict"] == "good"]
    if passed.empty:
        return None
    return passed.loc[passed["sharpe_diff"].idxmax()]
