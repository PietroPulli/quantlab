"""Evening run of the paper account: python scripts/paper_trade.py

Reads paper/config.json and paper/state.json, processes the new closes, appends to
paper/trades.csv and paper/values.csv, saves the new state. Run every evening by
.github/workflows/paper.yml; running it twice on the same day changes nothing.
"""

import json
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantlab.data import download_prices  # noqa: E402
from quantlab.ideas import download_factor  # noqa: E402
from quantlab.paper import run_day  # noqa: E402

PAPER = ROOT / "paper"


def fresh_prices(ticker: str, start: str) -> pd.Series:
    """Always download (no cache): tonight's close must be included."""
    tomorrow = (pd.Timestamp.now(tz="UTC") + timedelta(days=1)).date().isoformat()
    return download_prices([ticker], start, tomorrow)[ticker]


def append_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> int:
    config = json.loads((PAPER / "config.json").read_text(encoding="utf-8"))
    state_path = PAPER / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    with tempfile.TemporaryDirectory() as cache:
        state, trades, values, problems = run_day(config, state, today, fresh_prices, download_factor(cache))
    append_csv(PAPER / "trades.csv", trades)
    append_csv(PAPER / "values.csv", values)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"{today.date()}: {len(values)} new closes, {len(trades)} trades")
    for t in trades:
        print(f"  {t['date']} {t['action']:4} {t['ticker']:10} {t['idea']} @ {t['price']:.2f}")
    for p in problems:
        print(f"  WARNING {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
