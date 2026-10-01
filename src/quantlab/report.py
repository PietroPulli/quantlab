"""The "truth machine": one call that evaluates a strategy honestly against buy & hold."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.backtest import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, run_backtest
from quantlab.metrics import compare
from quantlab.strategies import buy_and_hold
from quantlab.validation import (
    Strategy,
    confidence_interval,
    optimize,
    sharpe_difference_bootstrap,
    split_date,
    strategy_returns,
    walk_forward,
)

OVERFIT_SHARPE_DROP = 0.5  # in-sample Sharpe this much above out-of-sample -> warn


@dataclass
class StrategyReport:
    name: str
    params: dict
    n_combinations: int  # how many parameter sets were tried (more tries = more luck)
    split: pd.Timestamp
    commission: float
    slippage: float
    returns: pd.DataFrame  # daily net returns: columns "strategy", "buy_and_hold"
    in_sample: pd.DataFrame  # metrics table, before `split`
    out_of_sample: pd.DataFrame  # metrics table, from `split` on
    sharpe_diff: float  # out-of-sample Sharpe(strategy) - Sharpe(buy & hold)
    sharpe_diff_ci: tuple[float, float]  # 95% block-bootstrap interval of sharpe_diff
    share_beating: float  # fraction of bootstrap samples where the strategy wins
    walk_forward: pd.DataFrame | None = None  # metrics table, None if no grid given
    walk_forward_folds: pd.DataFrame | None = None
    cash_rate: float = 0.0  # annual interest on idle cash, also subtracted in every Sharpe
    verdict: str = ""
    warnings: list[str] = field(default_factory=list)


def _buy_and_hold_from(
    prices: pd.Series, start: pd.Timestamp, commission: float, slippage: float, cash_rate: float
) -> pd.Series:
    """Buy & hold that only starts investing on `start`, so it pays the same entry cost."""
    signals = (prices.index >= start).astype(float)
    net = run_backtest(prices, pd.Series(signals, index=prices.index), commission, slippage, cash_rate)
    return net["net_return"][prices.index >= start]


def _verdict(ci: tuple[float, float]) -> str:
    low, high = ci
    if low > 0:
        return ("BEATS buy & hold out-of-sample on a risk-adjusted basis: the whole 95% "
                "bootstrap interval of the Sharpe difference is above zero.")
    if high < 0:
        return ("WORSE than buy & hold out-of-sample: the whole 95% bootstrap interval "
                "of the Sharpe difference is below zero.")
    return ("NO EVIDENCE it beats buy & hold: out-of-sample, the Sharpe difference is "
            "within what luck alone can produce (95% bootstrap interval contains zero).")


def evaluate_strategy(
    prices: pd.Series,
    strategy: Strategy,
    params: dict | None = None,
    grid: list[dict] | None = None,
    name: str | None = None,
    in_sample_fraction: float = 0.7,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    cash_rate: float = 0.0,
    n_bootstrap: int = 1000,
    block_size: int = 20,
    seed: int = 42,
    walk_forward_train: int = 756,
    walk_forward_test: int = 252,
) -> StrategyReport:
    """Evaluate a strategy on one asset, net of costs, always next to buy & hold.

    Give fixed `params`, or a `grid` to choose from: the choice is then made on the
    in-sample period only and judged on the out-of-sample one (plus walk-forward).
    """
    if params is not None and grid is not None:
        raise ValueError("give either params or grid, not both")
    split = split_date(prices.index, in_sample_fraction)
    if grid is not None:
        params, _ = optimize(
            prices, strategy, grid, end=split, commission=commission, slippage=slippage, cash_rate=cash_rate
        )
    params = params or {}
    n_combinations = len(grid) if grid is not None else 1

    returns = pd.DataFrame(
        {
            "strategy": strategy_returns(prices, strategy, params, commission, slippage, cash_rate),
            "buy_and_hold": strategy_returns(prices, buy_and_hold, {}, commission, slippage, cash_rate),
        }
    )
    is_mask = returns.index < split
    in_sample = compare(dict(returns[is_mask].items()), risk_free=cash_rate)
    out_of_sample = compare(dict(returns[~is_mask].items()), risk_free=cash_rate)

    oos = returns[~is_mask]
    samples = sharpe_difference_bootstrap(
        oos["strategy"], oos["buy_and_hold"], n_bootstrap, block_size, seed, cash_rate
    )
    ci = confidence_interval(samples)
    sharpe_diff = out_of_sample.loc["sharpe", "strategy"] - out_of_sample.loc["sharpe", "buy_and_hold"]

    wf_table, wf_folds = None, None
    if grid is not None and len(prices) >= walk_forward_train + walk_forward_test:
        wf_net, wf_folds = walk_forward(
            prices, strategy, grid, walk_forward_train, walk_forward_test, commission, slippage, cash_rate
        )
        bh_net = _buy_and_hold_from(prices, wf_net.index[0], commission, slippage, cash_rate)
        wf_table = compare({"strategy": wf_net, "buy_and_hold": bh_net}, risk_free=cash_rate)

    warnings = []
    is_sharpe = in_sample.loc["sharpe", "strategy"]
    oos_sharpe = out_of_sample.loc["sharpe", "strategy"]
    if is_sharpe - oos_sharpe > OVERFIT_SHARPE_DROP:
        warnings.append(
            f"Sharpe drops from {is_sharpe:.2f} in-sample to {oos_sharpe:.2f} out-of-sample: "
            "likely overfitting or a regime change."
        )
    if n_combinations > 1:
        warnings.append(
            f"{n_combinations} parameter sets were tried: in-sample results are optimistic "
            "by construction. Trust the out-of-sample and walk-forward numbers."
        )
    if np.isnan(oos_sharpe):
        warnings.append("The strategy never traded out-of-sample.")

    return StrategyReport(
        name=name or getattr(strategy, "__name__", "strategy"),
        params=params,
        n_combinations=n_combinations,
        split=split,
        commission=commission,
        slippage=slippage,
        cash_rate=cash_rate,
        returns=returns,
        in_sample=in_sample,
        out_of_sample=out_of_sample,
        sharpe_diff=float(sharpe_diff),
        sharpe_diff_ci=ci,
        share_beating=float(np.mean(samples > 0)),  # NaN samples count as "not beating"
        walk_forward=wf_table,
        walk_forward_folds=wf_folds,
        verdict=_verdict(ci),
        warnings=warnings,
    )


def _format_table(table: pd.DataFrame) -> str:
    """Metrics as text: percentages for returns/risk, two decimals for Sharpe."""
    shown = table.copy().astype(object)
    for metric in shown.index:
        fmt = "{:.2f}" if metric == "sharpe" else "{:.1%}"
        shown.loc[metric] = [fmt.format(v) if pd.notna(v) else "n/a" for v in table.loc[metric]]
    return shown.to_string()


def format_report(report: StrategyReport) -> str:
    """Plain-text report, readable in a terminal or a notebook."""
    start, end = report.returns.index[0].date(), report.returns.index[-1].date()
    low, high = report.sharpe_diff_ci
    lines = [
        f"STRATEGY REPORT: {report.name} {report.params}",
        f"Period {start} -> {end}, out-of-sample from {report.split.date()}",
        f"Costs per trade: commission {report.commission:.2%}, slippage {report.slippage:.2%}; "
        f"idle cash earns {report.cash_rate:.2%} a year (subtracted in Sharpe)",
        "",
        "VERDICT: " + report.verdict,
        f"  Out-of-sample Sharpe difference {report.sharpe_diff:+.2f} "
        f"(95% interval {low:+.2f} to {high:+.2f}; strategy ahead in "
        f"{report.share_beating:.0%} of bootstrap samples)",
        "",
        "IN-SAMPLE (used to choose parameters)",
        _format_table(report.in_sample),
        "",
        "OUT-OF-SAMPLE (never seen while choosing)",
        _format_table(report.out_of_sample),
    ]
    if report.walk_forward is not None:
        chosen = report.walk_forward_folds["params"].astype(str).value_counts()
        lines += [
            "",
            f"WALK-FORWARD ({len(report.walk_forward_folds)} yearly re-optimisations)",
            _format_table(report.walk_forward),
            "Params chosen: " + ", ".join(f"{p} x{n}" for p, n in chosen.items()),
        ]
    lines += ["", "WARNINGS"] + [f"- {w}" for w in report.warnings]
    lines += [
        "- Single asset chosen today: survivorship bias makes every result optimistic.",
        "- Past performance does not predict future returns. This is research, not advice.",
    ]
    return "\n".join(lines)
