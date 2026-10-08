"""Tests for quantlab.data (hand-built data, no internet)."""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from quantlab import data
from quantlab.data import latest_news, load_prices, relevant_news, validate_prices



@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """Retries must not really sleep during tests."""
    import quantlab.data

    monkeypatch.setattr(quantlab.data, "_sleep", lambda seconds: None)


def _clean() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=10)
    return pd.DataFrame({"AAA": np.linspace(100, 110, 10)}, index=idx)


def test_clean_data_has_no_problems():
    assert validate_prices(_clean()) == []


def test_negative_price_is_flagged():
    df = _clean()
    df.iloc[3, 0] = -5.0
    assert any("<= 0" in p for p in validate_prices(df))


def test_missing_value_is_flagged():
    df = _clean()
    df.iloc[4, 0] = np.nan
    assert any("missing" in p for p in validate_prices(df))


def test_date_gap_is_flagged():
    df = _clean().drop(_clean().index[3:8])  # remove a whole stretch of days
    assert any("gap" in p for p in validate_prices(df))


def test_suspicious_jump_is_flagged():
    df = _clean()
    df.iloc[5:, 0] = df.iloc[5:, 0] / 2  # unadjusted 2-for-1 split look
    assert any("jump" in p for p in validate_prices(df))


def test_three_for_two_split_is_flagged():
    df = _clean()
    df.iloc[5:, 0] = df.iloc[5:, 0] * 2 / 3  # unadjusted 3-for-2 split look (-33%)
    assert any("jump" in p for p in validate_prices(df))


def test_normal_daily_move_is_not_flagged():
    df = _clean()
    df.iloc[5:, 0] = df.iloc[5:, 0] * 0.9  # a -10% day is a real move, not a data error
    assert validate_prices(df) == []


def test_duplicate_dates_are_flagged():
    df = _clean()
    df = pd.concat([df, df.iloc[[2]]])
    assert any("duplicate" in p for p in validate_prices(df))


def test_load_prices_uses_cache(tmp_path, monkeypatch):
    calls = []

    def fake_download(tickers, start, end):
        calls.append(1)
        return _clean()

    monkeypatch.setattr(data, "download_prices", fake_download)
    first = load_prices(["AAA"], "2024-01-01", "2024-02-01", cache_dir=tmp_path)
    second = load_prices(["AAA"], "2024-01-01", "2024-02-01", cache_dir=tmp_path)
    assert len(calls) == 1  # second call came from the cache
    pdt.assert_frame_equal(first, second, check_freq=False)


def test_cache_key_ignores_ticker_order(tmp_path):
    a = data._cache_path(["A", "B"], "2020-01-01", "2021-01-01", tmp_path)
    b = data._cache_path(["B", "A"], "2020-01-01", "2021-01-01", tmp_path)
    assert a == b


def test_download_raises_on_empty(monkeypatch):
    import types, sys

    fake = types.SimpleNamespace(download=lambda *a, **k: pd.DataFrame())
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    with pytest.raises(ValueError):
        data.download_prices(["AAA"], "2024-01-01", "2024-02-01")


def test_latest_news_keeps_only_the_useful_fields(monkeypatch):
    import sys
    import types

    class FakeSearch:
        def __init__(self, query, news_count):
            self.news = [
                {"title": "Apple sale", "publisher": "X", "link": "https://x", "providerPublishTime": 0, "uuid": "1"},
                {"title": "", "publisher": "Y"},  # no title: dropped
            ]

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Search=FakeSearch))
    news = latest_news("AAPL")
    assert news == [{"title": "Apple sale", "publisher": "X", "link": "https://x",
                     "published": pd.Timestamp("1970-01-01", tz="UTC")}]


def test_relevant_news_keeps_only_headlines_naming_the_asset():
    items = [{"title": "Apple's AI advantage"}, {"title": "Amazon stock falls"},
             {"title": "Pineapple prices"}, {"title": "APPLE hits record"}]
    kept = [i["title"] for i in relevant_news(items, ["Apple"])]
    assert kept == ["Apple's AI advantage", "APPLE hits record"]  # whole word, any case


def test_relevant_news_with_no_terms_keeps_nothing():
    assert relevant_news([{"title": "Anything"}], []) == []


def test_load_ohlcv_flattens_yfinance_columns_and_caches(monkeypatch, tmp_path):
    import sys
    import types

    calls = []
    idx = pd.bdate_range("2024-01-01", periods=3)
    columns = pd.MultiIndex.from_product([["Close", "High", "Low", "Open", "Volume"], ["AAPL"]])
    raw = pd.DataFrame(np.arange(15, dtype=float).reshape(3, 5), index=idx, columns=columns)

    def fake_download(tickers, start, end, auto_adjust, progress):
        calls.append(tickers)
        return raw

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    from quantlab.data import load_ohlcv

    first = load_ohlcv("AAPL", "2024-01-01", "2024-01-04", cache_dir=tmp_path)
    second = load_ohlcv("AAPL", "2024-01-01", "2024-01-04", cache_dir=tmp_path)
    assert calls == ["AAPL"]  # second call served by the cache
    assert list(first.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert first["Close"].tolist() == [0.0, 5.0, 10.0]
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_dates_with_a_time_are_sent_to_yahoo_as_plain_days(monkeypatch, tmp_path):
    import sys
    import types

    asked = []

    def fake_download(tickers, start, end, auto_adjust, progress):  # called with keywords
        asked.append((start, end))
        idx = pd.bdate_range("2026-01-05", periods=3)
        return pd.DataFrame({("Close", "QQQ"): [1.0, 2.0, 3.0]}, index=idx)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    load_prices(["QQQ"], "2023-08-25 00:00:00", pd.Timestamp("2026-10-05"), cache_dir=tmp_path)
    assert asked == [("2023-08-25", "2026-10-05")]


def test_yahoo_refusals_are_retried_then_reported(monkeypatch):
    import sys
    import types

    import quantlab.data as data

    calls = []

    def flaky(tickers, start, end, auto_adjust, progress):
        calls.append(1)
        if len(calls) < 3:
            return pd.DataFrame()  # refused twice
        return pd.DataFrame({("Close", "SPY"): [1.0]}, index=pd.to_datetime(["2026-01-05"]))

    monkeypatch.setattr(data, "_sleep", lambda s: None)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=flaky))
    assert data.download_prices(["SPY"], "2026-01-01", "2026-01-10")["SPY"].tolist() == [1.0]
    assert len(calls) == 3

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=lambda **k: pd.DataFrame()))
    with pytest.raises(ValueError, match="No data returned"):
        data.download_prices(["SPY"], "2026-01-01", "2026-01-10")


def _snapshot(tmp_path, start="2020-01-01", end="2026-10-02"):
    from quantlab.data import save_snapshot

    idx = pd.bdate_range(start, end)
    close = pd.Series(range(1, len(idx) + 1), index=idx, dtype=float)
    ohlcv = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 10.0})
    save_snapshot(ohlcv, tmp_path / "market", "^VIX")
    return close


def _no_yahoo(monkeypatch, calls):
    import sys
    import types

    def refuse(**kwargs):
        calls.append(kwargs["tickers"])
        return pd.DataFrame()

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=refuse))


def test_snapshot_round_trip_with_safe_file_names(tmp_path):
    from quantlab.data import read_snapshot, snapshot_path

    close = _snapshot(tmp_path)
    assert snapshot_path(tmp_path / "market", "^VIX").name == "_VIX.csv"
    assert snapshot_path(tmp_path / "market", "GC=F").name == "GC_F.csv"
    back = read_snapshot(tmp_path / "market", "^VIX")
    pdt.assert_series_equal(back["Close"], close, check_names=False, check_freq=False)
    assert read_snapshot(tmp_path / "market", "SPY") is None  # no file for it
    assert read_snapshot(None, "^VIX") is None  # no snapshot folder at all


def test_snapshot_serves_covered_requests_without_downloading(monkeypatch, tmp_path):
    from quantlab.data import load_ohlcv, snapshot_path

    close = _snapshot(tmp_path)
    calls = []
    _no_yahoo(monkeypatch, calls)
    assert snapshot_path(tmp_path / "market", "^VIX").name == "_VIX.csv"
    # ends on Friday 2 Oct; asked "until Monday 5 Oct": still covered (weekend)
    prices = load_prices(["^VIX"], "2021-01-01", "2026-10-05", cache_dir=tmp_path, snapshot_dir=tmp_path / "market")
    assert calls == []
    assert prices["^VIX"].iloc[-1] == close.iloc[-1] and prices.index[0] == pd.Timestamp("2021-01-01")
    candles = load_ohlcv("^VIX", "2021-01-01", "2026-10-05", cache_dir=tmp_path, snapshot_dir=tmp_path / "market")
    assert list(candles.columns) == ["Open", "High", "Low", "Close", "Volume"] and calls == []


def test_snapshot_not_covering_goes_to_yahoo_and_is_the_fallback_if_refused(monkeypatch, tmp_path):
    close = _snapshot(tmp_path, end="2026-06-30")  # three months old
    calls = []
    _no_yahoo(monkeypatch, calls)
    prices = load_prices(["^VIX"], "2021-01-01", "2026-10-05", cache_dir=tmp_path, snapshot_dir=tmp_path / "market")
    assert calls  # it tried Yahoo first, which refused
    assert prices["^VIX"].iloc[-1] == close.iloc[-1]  # then used the snapshot rather than failing


def test_without_snapshot_a_refusal_is_still_an_error(monkeypatch, tmp_path):
    calls = []
    _no_yahoo(monkeypatch, calls)
    with pytest.raises(ValueError):
        load_prices(["XYZ"], "2021-01-01", "2026-10-05", cache_dir=tmp_path, snapshot_dir=tmp_path / "market")
