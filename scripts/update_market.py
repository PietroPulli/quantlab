"""Refresh the market snapshot: python scripts/update_market.py

Downloads the daily candles of every ticker in market/tickers.json and rewrites market/<ticker>.csv.
The app reads these files first, so it rarely has to ask Yahoo from a cloud server.
A ticker that cannot be downloaded keeps its previous file.
"""

import json
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantlab.data import download_ohlcv, save_snapshot  # noqa: E402

MARKET = ROOT / "market"


def main() -> int:
    spec = json.loads((MARKET / "tickers.json").read_text(encoding="utf-8"))
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    tomorrow = (today + timedelta(days=1)).date().isoformat()
    failed = []
    for ticker in spec["tickers"]:
        try:
            ohlcv = download_ohlcv(ticker, spec["since"], tomorrow)
            ohlcv = ohlcv[ohlcv.index < today]  # only finished days: today's bar may be intraday
            save_snapshot(ohlcv, MARKET, ticker)
            print(f"{ticker:10} {len(ohlcv):6d} days, last {ohlcv.index[-1].date()}")
        except Exception as exc:  # keep yesterday's file, try again tomorrow
            failed.append(ticker)
            print(f"{ticker:10} WARNING not updated: {exc}")
    return 0 if len(failed) < len(spec["tickers"]) else 1  # fail the job only if nothing worked


if __name__ == "__main__":
    raise SystemExit(main())
