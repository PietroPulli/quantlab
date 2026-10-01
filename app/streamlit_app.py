"""Quantlab web app: pick a strategy and an asset, get an honest verdict against buy & hold.

Run from the project root with:  streamlit run app/streamlit_app.py

This file is only the user interface. Every number comes from the quantlab library
(the same tested functions used in the notebooks), never from code written here.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from quantlab import data
from quantlab.metrics import calmar_ratio, drawdown
from quantlab.report import evaluate_strategy, format_report
from quantlab.strategies import breakout, mean_reversion, momentum, moving_average_crossover
from quantlab.validation import param_grid

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# For each strategy: the function, a human description, its sliders and the grid
# tried when the user lets the in-sample period choose the parameters.
# Slider format: name -> (label, min, max, default, step)
STRATEGY_UI = {
    "Breakout": {
        "func": breakout,
        "rule": "Compra quando il prezzo fa il massimo degli ultimi N giorni, vende quando fa il minimo.",
        "params": {"window": ("Finestra N (giorni)", 5, 250, 50, 5)},
        "grid": param_grid(window=[20, 50, 100, 200]),
    },
    "Momentum 12-1": {
        "func": momentum,
        "rule": "Compra se il prezzo è salito nell'ultimo periodo (ignorando l'ultimo mese), altrimenti contanti.",
        "params": {
            "lookback": ("Periodo di osservazione (giorni)", 21, 504, 252, 21),
            "skip": ("Giorni recenti da ignorare", 0, 63, 21, 1),
        },
        "grid": param_grid(lookback=[126, 252], skip=[0, 21]),
    },
    "Incrocio medie mobili": {
        "func": moving_average_crossover,
        "rule": "Compra quando la media veloce è sopra quella lenta (il trend sale), altrimenti contanti.",
        "params": {
            "fast": ("Media veloce (giorni)", 5, 100, 50, 5),
            "slow": ("Media lenta (giorni)", 20, 300, 200, 10),
        },
        "grid": param_grid(fast=[20, 50], slow=[100, 200]),
    },
    "Mean reversion": {
        "func": mean_reversion,
        "rule": "Compra quando il prezzo è insolitamente basso rispetto alla sua media, vende quando ci torna.",
        "params": {
            "window": ("Finestra della media (giorni)", 5, 100, 20, 5),
            "entry_z": ("Entra sotto z =", -3.0, -0.5, -1.0, 0.25),
        },
        "grid": param_grid(window=[10, 20], entry_z=[-1.5, -1.0]),
    },
}


@st.cache_data(show_spinner="Scarico i prezzi...")
def get_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Prices for one ticker, from the local cache or Yahoo Finance (cached per session too)."""
    return data.load_prices([ticker], start, end, cache_dir=DATA_DIR)[ticker].dropna()


st.set_page_config(page_title="Quantlab", page_icon="📈", layout="wide")
st.title("Quantlab: la strategia regge davvero?")
st.caption(
    "Scegli una regola di trading e un titolo. Quantlab la simula sul passato, costi inclusi, "
    "e la confronta con il semplice *compra e tieni*. Non è un consiglio di investimento: "
    "dice solo se una regola **avrebbe** funzionato."
)

# ---- Inputs (sidebar) ----
with st.sidebar:
    st.header("1. Cosa provare")
    ticker = st.text_input("Titolo (ticker Yahoo)", value="SPY").strip().upper()
    col_a, col_b = st.columns(2)
    start = col_a.date_input("Dal", value=date(2015, 1, 1), min_value=date(1995, 1, 1))
    end = col_b.date_input("Al", value=date(2025, 12, 31), max_value=date.today())

    name = st.selectbox("Strategia", list(STRATEGY_UI))
    ui = STRATEGY_UI[name]
    st.info(ui["rule"])

    st.header("2. Parametri")
    optimize_params = st.toggle(
        "Fai scegliere i parametri al passato",
        help="Prova alcune combinazioni sul primo 70% dei dati e tiene la migliore. "
        "Il giudizio vero si fa poi sul restante 30%, mai visto durante la scelta.",
    )
    params = {}
    if optimize_params:
        st.caption(f"Combinazioni provate: {ui['grid']}")
    else:
        for key, (label, lo, hi, default, step) in ui["params"].items():
            params[key] = st.slider(label, lo, hi, default, step, key=f"{name}-{key}")

    st.header("3. Costi per operazione")
    commission = st.number_input("Commissione (%)", 0.0, 1.0, 0.10, 0.01) / 100
    slippage = st.number_input("Slippage (%)", 0.0, 1.0, 0.05, 0.01) / 100

    run = st.button("Metti alla prova", type="primary", use_container_width=True)

# A button is True only on the run right after the click: remember it, so the
# results stay on screen (and update) when the user then moves a slider.
if run:
    st.session_state.started = True
if not st.session_state.get("started"):
    st.info("Scegli strategia e titolo a sinistra, poi premi **Metti alla prova**.")
    st.stop()

# ---- Data ----
try:
    prices = get_prices(ticker, str(start), str(end))
except Exception as exc:  # network errors, unknown tickers, empty downloads
    st.error(f"Non riesco a ottenere i prezzi di {ticker}: {exc}")
    st.stop()

problems = data.validate_prices(prices.to_frame(ticker))
if problems:
    st.warning("Possibili problemi nei dati: " + "; ".join(problems))
if len(prices) < 504:
    st.error("Servono almeno 2 anni di dati: allarga il periodo.")
    st.stop()

# ---- The truth machine ----
try:
    with st.spinner("Simulo la strategia e faccio il bootstrap..."):
        report = evaluate_strategy(
            prices,
            ui["func"],
            params=None if optimize_params else params,
            grid=ui["grid"] if optimize_params else None,
            name=f"{name} su {ticker}",
            commission=commission,
            slippage=slippage,
        )
except ValueError as exc:  # e.g. fast average longer than the slow one
    st.error(f"Parametri non validi: {exc}")
    st.stop()

# ---- Verdict ----
low, high = report.sharpe_diff_ci
if low > 0:
    st.success("### ✅ Batte il compra e tieni\nAnche considerando la fortuna, il vantaggio regge.")
elif high < 0:
    st.error("### ❌ Peggio del compra e tieni\nAnche considerando la fortuna, lo svantaggio è netto.")
else:
    st.warning(
        "### ⚖️ Nessuna prova che batta il compra e tieni\n"
        "La differenza è compatibile con la fortuna."
    )
st.write(
    f"Parametri usati: `{report.params}` · periodo di prova (out-of-sample) dal "
    f"**{report.split.date()}**. Differenza di Sharpe {report.sharpe_diff:+.2f}, "
    f"intervallo al 95% da {low:+.2f} a {high:+.2f}. "
    f"La strategia è avanti nel {report.share_beating:.0%} dei campioni bootstrap."
)
for warning in report.warnings:
    st.warning(warning)

# ---- Numbers ----
oos_returns = report.returns[report.returns.index >= report.split]
oos = report.out_of_sample.copy()
oos.loc["calmar"] = [calmar_ratio(oos_returns[c]) for c in oos.columns]
labels = {"strategy": name, "buy_and_hold": "Compra e tieni"}
row_names = {
    "total_return": "Guadagno totale",
    "annual_return": "Guadagno annuo",
    "annual_volatility": "Volatilità annua",
    "sharpe": "Sharpe",
    "max_drawdown": "Peggior batosta",
    "calmar": "Calmar",
}


def pretty(table: pd.DataFrame) -> pd.DataFrame:
    """Metrics table as strings: percentages for returns and risk, two decimals for ratios."""
    shown = table.rename(index=row_names, columns=labels).astype(object)
    for metric in shown.index:
        fmt = "{:.2f}" if metric in ("Sharpe", "Calmar") else "{:.1%}"
        shown.loc[metric] = [fmt.format(v) if pd.notna(v) else "n/d" for v in shown.loc[metric]]
    return shown


left, right = st.columns([3, 2])
with left:
    st.subheader("Quanto diventa 1 € (periodo di prova, costi inclusi)")
    st.line_chart((1 + oos_returns).cumprod().rename(columns=labels))
    st.subheader("Perdita dal massimo precedente")
    st.area_chart(oos_returns.apply(drawdown).rename(columns=labels))
with right:
    st.subheader("Periodo di prova")
    st.table(pretty(oos))
    st.subheader("Periodo di scelta (in-sample)")
    st.caption("Qui la strategia è avvantaggiata: i parametri sono stati scelti guardando questi dati.")
    st.table(pretty(report.in_sample))
    if report.walk_forward is not None:
        st.subheader("Walk-forward")
        st.caption("Riscelta dei parametri ogni anno, usando solo i 3 anni precedenti.")
        st.table(pretty(report.walk_forward))

with st.expander("Report completo in testo"):
    st.code(format_report(report), language=None)
