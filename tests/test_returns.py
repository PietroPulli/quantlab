"""Tests for quantlab.returns (hand-built data, no internet)."""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from quantlab.returns import log_returns, simple_returns


def _prices() -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=5)
    return pd.Series([100.0, 110.0, 99.0, 99.0, 108.9], index=idx)


def test_constant_series_has_zero_returns():
    p = pd.Series(50.0, index=pd.bdate_range("2024-01-01", periods=6))
    assert (simple_returns(p).dropna() == 0).all()
    assert (log_returns(p).dropna() == 0).all()


def test_first_return_is_nan():
    assert np.isnan(simple_returns(_prices()).iloc[0])
    assert np.isnan(log_returns(_prices()).iloc[0])


def test_simple_return_known_values():
    r = simple_returns(_prices())
    assert np.isclose(r.iloc[1], 0.10)
    assert np.isclose(r.iloc[2], -0.10)


def test_log_and_simple_are_consistent():
    p = _prices()
    pdt.assert_series_equal(np.log(1 + simple_returns(p)), log_returns(p))


def test_works_on_dataframe():
    p = pd.DataFrame({"A": _prices(), "B": _prices() * 2})
    r = simple_returns(p)
    assert list(r.columns) == ["A", "B"]
    pdt.assert_series_equal(r["A"], r["B"], check_names=False)
