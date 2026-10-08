"""Machine learning on one asset: monthly in/out decisions learned from past examples.

Each example is one month end: the features describe the market up to that day,
the target says whether the price rose over the following month.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of every month in `index` (the days we decide on)."""
    days = pd.Series(index, index=index)
    return pd.DatetimeIndex(days.groupby(index.to_period("M")).max().values)


def make_features(prices: pd.Series, vix: pd.Series | None = None) -> pd.DataFrame:
    """One row per month end, built only from data known on that day.

    Rows without a full history (e.g. the first year, needed by ret_12m) are dropped.
    """
    daily = prices.pct_change(fill_method=None)
    features = pd.DataFrame(
        {
            "ret_1m": prices.pct_change(21, fill_method=None),  # return over the last ~1 month (21 trading days)
            "ret_3m": prices.pct_change(63, fill_method=None),
            "ret_6m": prices.pct_change(126, fill_method=None),
            "ret_12m": prices.pct_change(252, fill_method=None),
            "vol_1m": daily.rolling(21).std() * np.sqrt(TRADING_DAYS),  # annualized volatility of the last month
            "dist_ma200": prices / prices.rolling(200).mean() - 1,  # +5% = price 5% above its 200-day average
        }
    )
    if vix is not None:
        # VIX may miss a few days of the SPY calendar: carry the last known value forward (never backward).
        features["vix"] = vix.reindex(prices.index).ffill()
    return features.loc[month_ends(prices.index)].dropna()


def make_target(prices: pd.Series) -> pd.Series:
    """At each month end: 1.0 if the price is higher at the next month end, else 0.0.

    The last month end has no next month yet: its target is NaN (unknown).
    """
    ends = month_ends(prices.index)
    at_ends = prices.loc[ends]
    next_return = at_ends.shift(-1) / at_ends - 1  # shift(-1): the NEXT month end's price, the future we predict
    return (next_return > 0).astype(float).where(next_return.notna())


# ---------------------------------------------------------------- models
SEED = 42


def default_models() -> dict[str, Callable[[], object]]:
    """Three classifiers of increasing power, each built fresh (untrained) by its factory.

    Settings are fixed in advance and deliberately cautious: with ~400 monthly examples a
    flexible model can learn the past by heart (min_samples_leaf / max_depth limit that).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        # Linear: one weight per feature. StandardScaler puts features on the same scale first.
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0)),
        # Many decision trees, each on a random part of the data; their votes are averaged.
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, max_features="sqrt", random_state=SEED, n_jobs=-1
        ),
        # Small trees added one after another, each correcting the errors of the previous ones.
        "gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED
        ),
    }


def walk_forward_predict(
    features: pd.DataFrame,
    target: pd.Series,
    make_model: Callable[[], object],
    min_train: int = 120,
) -> pd.Series:
    """Probability that the next month is up, decided at every month end from `min_train` on.

    At month end i the model is trained again from scratch on rows 0..i-1 only: their
    outcome ended at a month end <= i, so it was already known. Row i's own outcome is
    the future we predict. Training grows by one month at every step (expanding window).
    """
    if min_train < 2:
        raise ValueError("min_train must be at least 2")
    target = target.reindex(features.index)
    probabilities = {}
    for i in range(min_train, len(features)):
        x_train, y_train = features.iloc[:i], target.iloc[:i]
        known = y_train.notna()
        x_train, y_train = x_train[known], y_train[known]
        if y_train.nunique() < 2:  # only ups (or only downs) so far: nothing to learn, use the frequency
            probabilities[features.index[i]] = float(y_train.mean())
            continue
        model = make_model()
        model.fit(x_train, y_train)
        up_column = list(model.classes_).index(1.0)
        probabilities[features.index[i]] = float(model.predict_proba(features.iloc[[i]])[0, up_column])
    return pd.Series(probabilities, name="p_up")


def daily_signals(probabilities: pd.Series, index: pd.DatetimeIndex, threshold: float = 0.5) -> pd.Series:
    """Month-end decisions (1 = invested if p_up > threshold, else 0) spread over the daily `index`.

    Each decision is the signal from its month end until the next one. Days before the first
    decision are dropped. run_backtest then shifts it: the position starts the day after.
    """
    decisions = (probabilities > threshold).astype(float)
    signals = decisions.reindex(index).ffill()  # ffill: keep the last decision, never use a later one
    return signals[signals.notna()]
