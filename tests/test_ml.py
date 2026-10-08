import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from quantlab.ml import (
    daily_signals,
    default_models,
    make_features,
    make_target,
    month_ends,
    walk_forward_predict,
)


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


class _RecordingModel:
    """Fake classifier: remembers what it was trained on, predicts the share of ups seen."""

    trained_on: list = []

    def fit(self, x, y):
        _RecordingModel.trained_on.append((x.index.max(), y.copy()))
        self.classes_ = np.array([0.0, 1.0])
        self.p = y.mean()
        return self

    def predict_proba(self, x):
        return np.array([[1 - self.p, self.p]])


def _monthly(n: int = 40, seed: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    features = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=idx)
    target = pd.Series(rng.integers(0, 2, n).astype(float), index=idx)
    target.iloc[-1] = np.nan  # the last outcome is still unknown
    return features, target


def test_walk_forward_trains_only_on_months_before_the_decision():
    features, target = _monthly()
    _RecordingModel.trained_on = []
    probabilities = walk_forward_predict(features, target, _RecordingModel, min_train=12)
    assert list(probabilities.index) == list(features.index[12:])
    for decision_day, (last_trained, _) in zip(probabilities.index, _RecordingModel.trained_on):
        assert last_trained < decision_day


def test_walk_forward_ignores_the_outcome_it_predicts():
    """Changing the outcome of month i (and later ones) must not change the prediction made at i."""
    features, target = _monthly()
    before = walk_forward_predict(features, target, _RecordingModel, min_train=12)
    i = 25
    changed = target.copy()
    changed.iloc[i:] = 1 - changed.iloc[i:]
    after = walk_forward_predict(features, changed, _RecordingModel, min_train=12)
    pdt.assert_series_equal(before.iloc[: i - 12 + 1], after.iloc[: i - 12 + 1])


def test_walk_forward_with_a_single_class_uses_its_frequency():
    features, target = _monthly()
    target[:] = 1.0
    probabilities = walk_forward_predict(features, target, _RecordingModel, min_train=12)
    assert (probabilities == 1.0).all()


def test_walk_forward_rejects_tiny_training_sets():
    features, target = _monthly()
    with pytest.raises(ValueError):
        walk_forward_predict(features, target, _RecordingModel, min_train=1)


def test_default_models_learn_an_obvious_rule():
    """Sanity check of the real models: 'up when feature a > 0' must be found."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("1990-01-31", periods=200, freq="ME")
    features = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)}, index=idx)
    target = (features["a"] > 0).astype(float)
    for name, make_model in default_models().items():
        probabilities = walk_forward_predict(features, target, make_model, min_train=150)
        hits = ((probabilities > 0.5).astype(float) == target.reindex(probabilities.index)).mean()
        assert hits >= 0.9, name


def test_daily_signals_hold_each_decision_until_the_next_month_end():
    probabilities = pd.Series([0.7, 0.3], index=pd.DatetimeIndex(["2024-01-31", "2024-02-29"]))
    days = pd.bdate_range("2024-01-29", "2024-03-05")
    signals = daily_signals(probabilities, days)
    assert signals.index[0] == pd.Timestamp("2024-01-31")  # nothing before the first decision
    assert signals["2024-01-31":"2024-02-28"].eq(1.0).all()
    assert signals["2024-02-29":].eq(0.0).all()
