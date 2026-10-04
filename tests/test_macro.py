"""Tests for quantlab.macro (hand-built data, no internet)."""

import pandas as pd
import pandas.testing as pdt
import pytest

import quantlab.macro as macro
from quantlab.macro import MACRO, load_macro, point_in_time, yoy_percent
from quantlab.rules import Condition, Indicator, rule_strategy


def _monthly(values: list[float], start: str = "2020-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"))


def test_point_in_time_moves_each_value_to_its_publication_day():
    cpi = _monthly([1.0, 2.0])
    known = point_in_time(cpi, lag_days=45)
    assert known.index.tolist() == [pd.Timestamp("2020-02-15"), pd.Timestamp("2020-03-17")]  # 2020 is a leap year
    assert known.tolist() == [1.0, 2.0]


def test_a_rule_cannot_see_august_data_in_august():
    # August inflation is published in mid September: a rule on 31 August must still
    # see July's value. This is the look-ahead trap of macro data.
    inflation = point_in_time(_monthly([2.0, 9.0], start="2024-07-01"), lag_days=45)  # Jul, Aug
    days = pd.bdate_range("2024-08-20", "2024-09-30")
    prices = pd.Series(100.0, index=days)
    signals = rule_strategy(prices, Condition(Indicator("price", source=inflation), ">", 5.0))
    assert (signals[: "2024-09-13"] == 0).all()  # August's 9% not public yet
    assert (signals["2024-09-16":] == 1).all()  # public from 15 September (a Sunday)


def test_yoy_percent():
    index = _monthly([100.0] * 12 + [103.0])
    assert yoy_percent(index).tolist() == pytest.approx([3.0])


def test_negative_lag_is_rejected():
    with pytest.raises(ValueError):
        point_in_time(_monthly([1.0]), lag_days=-1)


def test_load_macro_transforms_lags_and_caches(monkeypatch, tmp_path):
    calls = []

    def fake_download(fred_id):
        calls.append(fred_id)
        return _monthly([100.0] * 12 + [102.0])

    monkeypatch.setattr(macro, "download_fred", fake_download)
    first = load_macro("inflation", cache_dir=tmp_path)
    second = load_macro("inflation", cache_dir=tmp_path)  # from the cache
    assert calls == [MACRO["inflation"].fred_id]
    assert first.tolist() == pytest.approx([2.0])
    assert first.index[0] == pd.Timestamp("2021-01-01") + pd.Timedelta(days=45)
    pdt.assert_series_equal(first, second, check_freq=False, check_names=False)
