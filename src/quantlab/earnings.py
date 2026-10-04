"""Quarterly earnings surprises (reported EPS vs analysts' estimate), made point-in-time.

A surprise is usable from the first close after it was announced: same day if the
company reported before the 16:00 New York close, next day if after it (or if the
time is unknown). Only the surprise is available for free with a long history; full
historical balance sheets are not, so they are not used.
"""

from datetime import date
from pathlib import Path

import pandas as pd

MARKET_CLOSE_HOUR = 16  # New York time


def download_earnings(ticker: str) -> pd.DataFrame:
    """Earnings dates with 'Surprise(%)' from Yahoo Finance (index: announcement time)."""
    import yfinance as yf  # imported here so tests never need the network stack

    df = yf.Ticker(ticker).get_earnings_dates(limit=100)
    if df is None or df.empty:
        raise ValueError(f"No earnings history for {ticker} (ETFs, indices and crypto have none)")
    return df


def known_from(announced: pd.Timestamp) -> pd.Timestamp:
    """First date whose close can use an announcement made at `announced` (New York time)."""
    local = announced.tz_convert("America/New_York") if announced.tzinfo else announced
    day = pd.Timestamp(local.date())
    time_unknown = local.hour == 0 and local.minute == 0
    after_close = local.hour >= MARKET_CLOSE_HOUR
    return day + pd.Timedelta(days=1) if (after_close or time_unknown) else day


def surprise_series(earnings: pd.DataFrame) -> pd.Series:
    """Surprise in % indexed by the date it became usable (one value per date)."""
    reported = earnings["Surprise(%)"].dropna()
    surprises = pd.Series(reported.to_numpy(), index=[known_from(t) for t in reported.index], dtype=float)
    surprises = surprises.sort_index()
    return surprises[~surprises.index.duplicated(keep="last")].rename("earnings_surprise")


def load_earnings_surprises(ticker: str, cache_dir: str | Path = "data/") -> pd.Series:
    """surprise_series for `ticker`, cached for the day."""
    path = Path(cache_dir) / f"earnings_{ticker}_{date.today()}.parquet"
    if path.exists():
        return pd.read_parquet(path)["earnings_surprise"]
    series = surprise_series(download_earnings(ticker))
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame().to_parquet(path)
    return series
