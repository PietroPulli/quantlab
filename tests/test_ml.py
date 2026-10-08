import numpy as np
import pandas as pd
import pandas.testing as pdt

from quantlab.ml import make_features, make_target, month_ends


def _prices(n: int = 600, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n))), index=idx)


def test_month_ends_are_the_last_trading_day_of_each_month():
    idx = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-28"])
    assert list(month_ends(idx)) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-28")]


def test_features_do_not_look_ahead():
    """Changing the future must not change any feature of the past."""
    prices = _prices()
    vix = pd.Series(20.0, index=prices.index)
    cut = prices.index[450]
    future_changed = prices.copy()
    future_changed[future_changed.index > cut] *= 3  # a huge jump after `cut`
    before = make_features(prices, vix)
    after = make_features(future_changed, vix)
    pdt.assert_frame_equal(before[before.index <= cut], after[after.index <= cut])


def test_features_start_once_a_full_year_is_available():
    prices = _prices()
    features = make_features(prices)
    assert features.index[0] >= prices.index[252]
    assert not features.isna().any().any()
    assert list(features.columns) == ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_1m", "dist_ma200"]


def test_vix_missing_days_take_the_previous_value():
    prices = _prices()
    vix = pd.Series(np.arange(len(prices), dtype=float), index=prices.index)
    last_month_end = month_ends(prices.index)[-1]
    features = make_features(prices, vix.drop(last_month_end))  # VIX missing on that day
    assert features.loc[last_month_end, "vix"] == vix.shift(1)[last_month_end]


def test_target_says_if_the_next_month_end_is_higher():
    idx = pd.DatetimeIndex(["2024-01-31", "2024-02-15", "2024-02-29", "2024-03-29", "2024-04-30"])
    prices = pd.Series([100.0, 50.0, 110.0, 105.0, 120.0], index=idx)
    target = make_target(prices)
    # Jan->Feb end: 100 -> 110 up; Feb->Mar: 110 -> 105 down; Mar->Apr: up; April: future unknown.
    assert target.tolist()[:3] == [1.0, 0.0, 1.0]
    assert np.isnan(target.iloc[-1])
