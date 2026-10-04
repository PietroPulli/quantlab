"""Download, cache and validate adjusted close prices."""

import hashlib
from pathlib import Path

import pandas as pd

MAX_DAILY_JUMP = 0.25  # daily moves larger than 25% are flagged (catches 2:1 and 3:2 splits)
MAX_DATE_GAP_DAYS = 7  # calendar days between consecutive rows before flagging a gap


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance (one column per ticker)."""
    import yfinance as yf  # imported here so tests never need the network stack

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    if raw.empty:
        raise ValueError(f"No data returned for {tickers} between {start} and {end}")
    prices = raw["Close"]
    if isinstance(prices, pd.Series):  # single ticker on some yfinance versions
        prices = prices.to_frame(tickers[0])
    return prices[list(tickers)]


def _cache_path(tickers: list[str], start: str, end: str, cache_dir: str | Path) -> Path:
    """Deterministic cache file name: same request -> same file."""
    key = f"{'-'.join(sorted(tickers))}|{start}|{end}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    return Path(cache_dir) / f"prices_{digest}.parquet"


def load_prices(
    tickers: list[str], start: str, end: str, cache_dir: str | Path = "data/"
) -> pd.DataFrame:
    """Return prices from the local parquet cache, downloading and saving on a miss."""
    path = _cache_path(tickers, start, end, cache_dir)
    if path.exists():
        return pd.read_parquet(path)
    prices = download_prices(tickers, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    return prices


def validate_prices(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems found in the price table (empty = OK)."""
    problems: list[str] = []

    if df.index.has_duplicates:
        n = int(df.index.duplicated().sum())
        problems.append(f"{n} duplicate date(s)")

    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        if n_missing:
            problems.append(f"{col}: {n_missing} missing value(s)")
        n_bad = int((series <= 0).sum())
        if n_bad:
            problems.append(f"{col}: {n_bad} price(s) <= 0")
        # Only compare consecutive valid, positive prices
        valid = series[series > 0]
        jumps = valid.pct_change(fill_method=None).abs()
        n_jumps = int((jumps > MAX_DAILY_JUMP).sum())
        if n_jumps:
            problems.append(
                f"{col}: {n_jumps} daily jump(s) larger than {MAX_DAILY_JUMP:.0%}"
            )

    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        gaps = df.index.to_series().diff().dt.days
        n_gaps = int((gaps > MAX_DATE_GAP_DAYS).sum())
        if n_gaps:
            problems.append(
                f"{n_gaps} gap(s) of more than {MAX_DATE_GAP_DAYS} calendar days in dates"
            )

    return problems


def latest_news(ticker: str, count: int = 5) -> list[dict]:
    """Latest headlines about `ticker` from Yahoo Finance: title, publisher, link, published.

    Context only: free sources keep no dated archive of past news, so headlines cannot
    be backtested honestly and never enter any calculation in quantlab.
    """
    import yfinance as yf  # imported here so tests never need the network stack

    items = yf.Search(ticker, news_count=count).news or []
    return [
        {
            "title": item.get("title", ""),
            "publisher": item.get("publisher", ""),
            "link": item.get("link", ""),
            "published": pd.to_datetime(item.get("providerPublishTime"), unit="s", utc=True),
        }
        for item in items[:count]
        if item.get("title")
    ]


def relevant_news(items: list[dict], terms: list[str]) -> list[dict]:
    """Keep the headlines that mention at least one of `terms` as a whole word (any case).

    A Yahoo search for "AAPL" also returns stories about other companies: a headline
    is kept only if it names what we are looking at, e.g. "Apple".
    """
    import re

    patterns = [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in terms if t]
    return [item for item in items if any(p.search(item["title"]) for p in patterns)]
