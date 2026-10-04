"""Macro-economic series from FRED (Federal Reserve Bank of St. Louis), made point-in-time.

The trap with macro data: the value "for August" is published weeks later and may be
revised afterwards. A backtest that uses it on the 1st of August is cheating. So every
series here is shifted by a conservative publication lag: on day t a rule only sees
the observations that had already been published on day t.
"""

import io
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


@dataclass(frozen=True)
class MacroSeries:
    fred_id: str
    lag_days: int  # days from the observation date until the value is surely public
    yoy: bool = False  # turn an index level into a year-over-year % change


MACRO = {
    # CPI not seasonally adjusted: never revised after publication. Released mid next month.
    "inflation": MacroSeries("CPIAUCNS", lag_days=45, yoy=True),
    # Unemployment rate: released the first Friday of next month (small annual revisions).
    "unemployment": MacroSeries("UNRATE", lag_days=35),
    # Effective Fed funds rate, monthly average: published at the start of next month.
    "fed_funds": MacroSeries("FEDFUNDS", lag_days=35),
    # 10-year minus 2-year Treasury yield (daily market data): known after the close.
    "yield_curve": MacroSeries("T10Y2Y", lag_days=1),
}


def download_fred(fred_id: str) -> pd.Series:
    """Full history of one FRED series (no API key needed for the public CSV)."""
    raw = urllib.request.urlopen(FRED_CSV.format(fred_id), timeout=30).read().decode()
    df = pd.read_csv(io.StringIO(raw), parse_dates=["observation_date"], index_col="observation_date")
    return pd.to_numeric(df[fred_id], errors="coerce").dropna()  # "." marks missing days


def point_in_time(series: pd.Series, lag_days: int) -> pd.Series:
    """Re-date each observation to the day it became public: observation date + lag."""
    if lag_days < 0:
        raise ValueError("lag_days must be >= 0")
    shifted = series.copy()
    shifted.index = shifted.index + pd.Timedelta(days=lag_days)
    return shifted


def yoy_percent(monthly: pd.Series) -> pd.Series:
    """Year-over-year change in %, e.g. a price index -> inflation rate."""
    return (monthly / monthly.shift(12) - 1).dropna() * 100


def load_macro(name: str, cache_dir: str | Path = "data/") -> pd.Series:
    """A MACRO series ready for rules: transformed, point-in-time, cached for the day."""
    spec = MACRO[name]
    path = Path(cache_dir) / f"fred_{spec.fred_id}_{date.today()}.parquet"
    if path.exists():
        raw = pd.read_parquet(path)[spec.fred_id]
    else:
        raw = download_fred(spec.fred_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_frame(spec.fred_id).to_parquet(path)
    values = yoy_percent(raw) if spec.yoy else raw
    return point_in_time(values, spec.lag_days)
