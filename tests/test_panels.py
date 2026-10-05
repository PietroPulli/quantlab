"""Tests for the app's table builders (app/panels.py): rows must never shift."""

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from panels import drawdown_table, trips_table  # noqa: E402

from quantlab.backtest import round_trips, run_backtest  # noqa: E402


def test_trips_table_keeps_each_trade_on_one_row():
    idx = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.Series([10.0, 11, 12, 11, 10, 10, 12, 13, 12, 14], index=idx)
    signals = pd.Series([1.0, 1, 0, 0, 1, 1, 0, 0, 1, 1], index=idx)
    trips = round_trips(prices, run_backtest(prices, signals)["position"])
    trips = trips[trips["entry_date"] > idx[0]]  # filtering leaves gaps in the index, as in the app
    table = trips_table(trips)
    assert list(table.index) == [1, 2]
    assert table.loc[1, "Entrata"] == "05/01/2024" and table.loc[1, "Uscita"] == "09/01/2024"
    assert table.loc[2, "Entrata"] == "11/01/2024" and table.loc[2, "Uscita"] == "aperta"
    assert not table.isna().any().any()


def test_drawdown_table_rows_match_the_periods():
    idx = pd.bdate_range("2024-01-01", periods=7)
    r = pd.Series([0.10, -0.10, -0.10, 0.20, 0.10, -0.10, 0.05], index=idx)
    table = drawdown_table(r)
    assert list(table.index) == [1, 2]
    assert table.loc[1, "Profondità"] == "-19,0%" and table.loc[1, "Recupero"] == "05/01/2024"
    assert table.loc[2, "Recupero"] == "aperta"
