"""Tests for quantlab.earnings (hand-built data, no internet)."""

import numpy as np
import pandas as pd
import pytest

import quantlab.earnings as earnings
from quantlab.earnings import known_from, load_earnings_surprises, surprise_series
from quantlab.rules import Condition, Indicator, rule_strategy

NY = "America/New_York"


def test_after_the_close_is_usable_from_the_next_day():
    assert known_from(pd.Timestamp("2026-07-30 16:00", tz=NY)) == pd.Timestamp("2026-07-31")


def test_before_the_close_is_usable_the_same_day():
    assert known_from(pd.Timestamp("2026-07-30 07:30", tz=NY)) == pd.Timestamp("2026-07-30")


def test_unknown_time_is_treated_as_after_the_close():
    assert known_from(pd.Timestamp("2026-07-30 00:00", tz=NY)) == pd.Timestamp("2026-07-31")


def test_other_time_zones_are_converted_to_new_york():
    # 21:30 in Rome = 15:30 in New York: still before the close
    assert known_from(pd.Timestamp("2026-07-30 21:30", tz="Europe/Rome")) == pd.Timestamp("2026-07-30")


def _earnings_table() -> pd.DataFrame:
    idx = pd.DatetimeIndex(["2026-10-29 16:00", "2026-07-30 16:00", "2026-04-30 07:00"]).tz_localize(NY)
    return pd.DataFrame({"EPS Estimate": [1.98, 1.89, 1.94], "Reported EPS": [np.nan, 2.02, 2.01],
                         "Surprise(%)": [np.nan, 6.74, -3.46]}, index=idx)


def test_surprise_series_drops_future_quarters_and_sorts():
    s = surprise_series(_earnings_table())
    assert s.index.tolist() == [pd.Timestamp("2026-04-30"), pd.Timestamp("2026-07-31")]
    assert s.tolist() == [-3.46, 6.74]  # the October quarter has no result yet


def test_a_rule_on_surprises_uses_them_only_once_public():
    days = pd.bdate_range("2026-07-28", "2026-08-04")
    prices = pd.Series(100.0, index=days)
    rule = Condition(Indicator("price", source=surprise_series(_earnings_table())), ">", 0.0)
    signals = rule_strategy(prices, rule)
    assert (signals[:"2026-07-30"] == 0).all()  # last known surprise is April's -3.46%
    assert (signals["2026-07-31":] == 1).all()  # the +6.74% announced after the close of 30 July


def test_load_earnings_surprises_caches(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(earnings, "download_earnings", lambda t: calls.append(t) or _earnings_table())
    first = load_earnings_surprises("AAPL", cache_dir=tmp_path)
    second = load_earnings_surprises("AAPL", cache_dir=tmp_path)
    assert calls == ["AAPL"]
    assert first.tolist() == second.tolist() == [-3.46, 6.74]


def test_no_history_is_an_error(monkeypatch):
    class Empty:
        def get_earnings_dates(self, limit):
            return pd.DataFrame()

    import sys
    import types
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=lambda t: Empty()))
    with pytest.raises(ValueError):
        earnings.download_earnings("SPY")
