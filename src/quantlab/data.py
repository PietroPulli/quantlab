"""Download, cache and validate adjusted close prices."""

import hashlib
import time
from pathlib import Path

import pandas as pd

MAX_DAILY_JUMP = 0.25  # daily moves larger than 25% are flagged (catches 2:1 and 3:2 splits)
MAX_DATE_GAP_DAYS = 7  # calendar days between consecutive rows before flagging a gap
RETRY_WAITS = (2, 5)  # seconds before the 2nd and 3rd attempt: Yahoo often refuses cloud servers briefly
_sleep = time.sleep  # replaced by a no-op in tests


def _yahoo_download(what: str, **kwargs) -> pd.DataFrame:
    """yfinance.download with retries: an empty answer or an error is tried again, then raised."""
    import yfinance as yf  # imported here so tests never need the network stack

    for wait in (*RETRY_WAITS, None):
        try:
            raw = yf.download(**kwargs, auto_adjust=True, progress=False)
            if not raw.empty:
                return raw
            problem = f"No data returned for {what} between {kwargs['start']} and {kwargs['end']}"
        except Exception as exc:  # network error, rate limit...
            problem = f"Download of {what} failed: {exc}"
        if wait is None:
            raise ValueError(problem)
        _sleep(wait)


def day_string(when) -> str:
    """Any date-like value -> 'YYYY-MM-DD'. Yahoo rejects dates with a time ('2026-10-05 00:00:00')."""
    return pd.Timestamp(when).date().isoformat()


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance (one column per ticker)."""
    start, end = day_string(start), day_string(end)
    raw = _yahoo_download(str(tickers), tickers=tickers, start=start, end=end)
    prices = raw["Close"]
    if isinstance(prices, pd.Series):  # single ticker on some yfinance versions
        prices = prices.to_frame(tickers[0])
    return prices[list(tickers)]


def _cache_path(tickers: list[str], start: str, end: str, cache_dir: str | Path) -> Path:
    """Deterministic cache file name: same request -> same file."""
    key = f"{'-'.join(sorted(tickers))}|{start}|{end}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    return Path(cache_dir) / f"prices_{digest}.parquet"


# ---------------------------------------------------------------- market snapshot
# A folder of CSV files (one per ticker, daily OHLCV) refreshed every night by GitHub Actions.
# Reading it avoids asking Yahoo from cloud servers, which it often refuses.
SNAPSHOT_LAG_DAYS = 5  # a snapshot ending up to 5 days before the request still counts (weekends, holidays)


def snapshot_path(snapshot_dir: str | Path, ticker: str) -> Path:
    """'^VIX' -> market/_VIX.csv, 'GC=F' -> market/GC_F.csv (safe file names)."""
    safe = ticker.replace("^", "_").replace("=", "_").replace("/", "_")
    return Path(snapshot_dir) / f"{safe}.csv"


def save_snapshot(ohlcv: pd.DataFrame, snapshot_dir: str | Path, ticker: str) -> Path:
    """Write the candles to market/<ticker>.csv (rounded to 6 decimals to keep git diffs small)."""
    path = snapshot_path(snapshot_dir, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    ohlcv.round(6).rename_axis("Date").to_csv(path)
    return path


def read_snapshot(snapshot_dir: str | Path | None, ticker: str) -> pd.DataFrame | None:
    """The saved candles of `ticker`, or None if there is no snapshot folder or no file for it."""
    if snapshot_dir is None or not snapshot_path(snapshot_dir, ticker).exists():
        return None
    return pd.read_csv(snapshot_path(snapshot_dir, ticker), index_col="Date", parse_dates=True)


def _from_snapshot(snapshot: pd.DataFrame | None, start: str, end: str, stale_ok: bool) -> pd.DataFrame | None:
    """The [start, end) slice of the snapshot if it covers the request (or anyway, if stale_ok)."""
    if snapshot is None or snapshot.empty:
        return None
    covers = (snapshot.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=SNAPSHOT_LAG_DAYS)
              and snapshot.index[-1] >= pd.Timestamp(end) - pd.Timedelta(days=SNAPSHOT_LAG_DAYS + 1))
    if not (covers or stale_ok):
        return None
    part = snapshot[(snapshot.index >= start) & (snapshot.index < end)]
    return part if len(part) else None


def load_prices(
    tickers: list[str], start: str, end: str, cache_dir: str | Path = "data/",
    snapshot_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Prices from the snapshot (if it covers the request), else the parquet cache, else Yahoo.

    If Yahoo fails and a snapshot exists, its data is used even if it ends a little early.
    """
    start, end = day_string(start), day_string(end)
    snapshot = read_snapshot(snapshot_dir, tickers[0]) if len(tickers) == 1 else None
    hit = _from_snapshot(snapshot, start, end, stale_ok=False)
    if hit is not None:
        return hit[["Close"]].rename(columns={"Close": tickers[0]})
    path = _cache_path(tickers, start, end, cache_dir)
    if path.exists():
        return pd.read_parquet(path)
    try:
        prices = download_prices(tickers, start, end)
    except ValueError:
        fallback = _from_snapshot(snapshot, start, end, stale_ok=True)
        if fallback is None:
            raise
        return fallback[["Close"]].rename(columns={"Close": tickers[0]})
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


OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Daily open, high, low, close (adjusted) and volume of one ticker, for candlestick charts."""
    start, end = day_string(start), day_string(end)
    raw = _yahoo_download(ticker, tickers=ticker, start=start, end=end)
    if isinstance(raw.columns, pd.MultiIndex):  # recent yfinance: (field, ticker) columns
        raw = raw.xs(ticker, axis=1, level=1)
    return raw[OHLCV].dropna(subset=["Close"])


def load_ohlcv(ticker: str, start: str, end: str, cache_dir: str | Path = "data/",
               snapshot_dir: str | Path | None = None) -> pd.DataFrame:
    """Candles from the snapshot, else the parquet cache, else Yahoo (snapshot as fallback)."""
    start, end = day_string(start), day_string(end)
    snapshot = read_snapshot(snapshot_dir, ticker)
    hit = _from_snapshot(snapshot, start, end, stale_ok=False)
    if hit is not None:
        return hit[OHLCV]
    path = _cache_path([f"ohlcv-{ticker}"], start, end, cache_dir)
    if path.exists():
        return pd.read_parquet(path)
    try:
        ohlcv = download_ohlcv(ticker, start, end)
    except ValueError:
        fallback = _from_snapshot(snapshot, start, end, stale_ok=True)
        if fallback is None:
            raise
        return fallback[OHLCV]
    path.parent.mkdir(parents=True, exist_ok=True)
    ohlcv.to_parquet(path)
    return ohlcv
