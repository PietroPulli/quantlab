"""Tests for quantlab.data (hand-built data, no internet)."""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from quantlab import data
from quantlab.data import latest_news, load_prices, relevant_news, validate_prices


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
