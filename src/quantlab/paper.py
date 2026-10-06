"""Paper trading: a virtual account that follows the ideas day by day, going forward.

The capital is split in equal "slots", one per (ticker, idea). Each night, for every
new completed close, a slot follows its idea's signal: fully invested or fully in cash, trading at
that close and paying commission + slippage, exactly like the backtest. Next to it, a
buy & hold slot on the same ticker buys on the first day and never trades: the honest
benchmark. Running twice on the same data changes nothing (dates already processed are skipped).
"""

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class Slot:
    ticker: str
    idea: str
    cash: float  # strategy: cash and shares
    shares: float = 0.0
    bench_cash: float = 0.0  # buy & hold benchmark on the same ticker
    bench_shares: float = 0.0
    last_date: str | None = None  # last close already processed (ISO date)

    def value(self, price: float) -> float:
        return self.cash + self.shares * price

    def bench_value(self, price: float) -> float:
        return self.bench_cash + self.bench_shares * price


def new_slots(slots_config: list[dict], start_cash: float) -> list[Slot]:
    """Equal split of the starting cash; the benchmark of each slot starts with the same amount."""
    each = start_cash / len(slots_config)
    return [Slot(s["ticker"], s["idea"], cash=each, bench_cash=each) for s in slots_config]


def step(slot: Slot, day: pd.Timestamp, price: float, signal: float, cost: float,
         daily_cash: float) -> tuple[Slot, dict | None]:
    """Process one close: interest on idle cash, then trade if the signal asks for it."""
    first_day = slot.last_date is None
    if not first_day:  # interest accrues between closes, not on the starting day
        slot.cash *= 1 + daily_cash
    trade = None
    if signal > 0 and slot.shares == 0 and slot.cash > 0:
        slot.shares = slot.cash * (1 - cost) / price
        trade = {"action": "buy", "shares": slot.shares, "value": slot.cash, "cost": slot.cash * cost}
        slot.cash = 0.0
    elif signal == 0 and slot.shares > 0:
        gross = slot.shares * price
        trade = {"action": "sell", "shares": slot.shares, "value": gross, "cost": gross * cost}
        slot.cash, slot.shares = gross * (1 - cost), 0.0
    if first_day:  # the benchmark buys once, on the first close, and holds forever
        slot.bench_shares = slot.bench_cash * (1 - cost) / price
        slot.bench_cash = 0.0
    slot.last_date = day.date().isoformat()
    if trade:
        trade = {"date": slot.last_date, "ticker": slot.ticker, "idea": slot.idea, "price": price, **trade}
    return slot, trade


def advance(slot: Slot, prices: pd.Series, signals: pd.Series, start: str, cost: float,
            daily_cash: float) -> tuple[Slot, list[dict], list[dict]]:
    """Process every close after the slot's last processed date (and not before `start`).

    Returns the updated slot, the trades made and one value row per processed close.
    """
    after = slot.last_date or "0000-01-01"
    new_days = [d for d in prices.index if d.date().isoformat() > after and d.date().isoformat() >= start]
    trades, values = [], []
    for day in new_days:
        price = float(prices.loc[day])
        slot, trade = step(slot, day, price, float(signals.loc[day]), cost, daily_cash)
        if trade:
            trades.append(trade)
        values.append({"date": slot.last_date, "ticker": slot.ticker, "idea": slot.idea,
                       "position": 1 if slot.shares > 0 else 0,
                       "value": slot.value(price), "bench_value": slot.bench_value(price)})
    return slot, trades, values


def completed_closes(prices: pd.Series, today: pd.Timestamp) -> pd.Series:
    """Only days that are over: today's bar may be an intraday price, not a close.

    Run after midnight UTC, every market's previous day is final (Milan, Wall Street,
    gold futures, and the crypto daily bar that ends at 00:00 UTC).
    """
    return prices[prices.index.normalize() < today.normalize()]


def portfolio_history(values: pd.DataFrame) -> pd.DataFrame:
    """Total value per date of the strategy and of the benchmark, summed over slots.

    Slots trade on different calendars (crypto every day, stocks Mon-Fri): on a date where a
    slot has no close, its last known value is carried forward.
    """
    if values.empty:
        return pd.DataFrame(columns=["value", "bench_value"])
    wide = values.pivot_table(index="date", columns=["ticker", "idea"], values=["value", "bench_value"])
    wide = wide.sort_index().ffill()
    return pd.DataFrame({"value": wide["value"].sum(axis=1), "bench_value": wide["bench_value"].sum(axis=1)})


def slots_to_records(slots: list[Slot]) -> list[dict]:
    return [asdict(s) for s in slots]


def slots_from_records(records: list[dict]) -> list[Slot]:
    return [Slot(**r) for r in records]


def run_day(config: dict, state: dict | None, today: pd.Timestamp, get_prices, factor) -> tuple[dict, list, list, list]:
    """One evening run over every slot.

    `get_prices(ticker, start)` returns the daily closes up to now; `factor` is the
    quantlab.ideas factor function. Returns (new state, trades, value rows, problems).
    A slot whose data cannot be fetched is skipped and retried at the next run.
    """
    from quantlab.ideas import IDEAS
    from quantlab.metrics import infer_periods_per_year

    if state is None:  # first run: the account starts with yesterday's (first completed) close
        state = {"start": (today - pd.Timedelta(days=1)).date().isoformat(),
                 "slots": slots_to_records(new_slots(config["slots"], config["start_cash"]))}
    cost = config["commission"] + config["slippage"]
    history_from = (pd.Timestamp(state["start"]) - pd.DateOffset(years=3)).date().isoformat()  # for the indicators
    slots, trades, values, problems = [], [], [], []
    for slot in slots_from_records(state["slots"]):
        try:
            prices = get_prices(slot.ticker, history_from).dropna()
            ppy = infer_periods_per_year(prices.index)
            prices = completed_closes(prices, today)
            strategy, params = IDEAS[slot.idea].build(slot.ticker, factor)
            signals = strategy(prices, **params)
            daily_cash = (1 + config["cash_rate"]) ** (1 / ppy) - 1
            slot, new_trades, new_values = advance(slot, prices, signals, state["start"], cost, daily_cash)
            trades += new_trades
            values += new_values
        except Exception as exc:  # network, delisted ticker...: keep the slot as it was
            problems.append(f"{slot.ticker} / {slot.idea}: {exc}")
        slots.append(slot)
    return {"start": state["start"], "slots": slots_to_records(slots)}, trades, values, problems
