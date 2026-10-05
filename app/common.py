"""Pieces shared by the simple and the advanced view: style, strategies, prices, charts."""

from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from quantlab import data, earnings, macro
from quantlab.macro import MACRO
from quantlab.report import StrategyReport
from quantlab.rules import Condition, Indicator
from quantlab.strategies import breakout, mean_reversion, momentum, moving_average_crossover
from quantlab.validation import param_grid

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Small additions on top of .streamlit/config.toml: header, verdict, KPI strip, tables.
INK, MUTED, LINE, SURFACE = "#14181f", "#5d6673", "#dfe3e8", "#ffffff"
POS, NEG = "#127a4b", "#b42318"  # gains and losses only
CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
.ql-brand {{ display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }}
.ql-brand b {{ font-size:1.25rem; letter-spacing:-0.01em; }}
.ql-brand span {{ color:{MUTED}; font-size:0.85rem; }}
.ql-context {{ color:{MUTED}; font-size:0.82rem; margin:2px 0 10px; }}
.ql-context b {{ color:{INK}; font-weight:600; }}
.ql-verdict {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:4px; padding:14px 18px;
               display:flex; gap:18px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }}
.ql-pill {{ font:600 0.7rem 'Geist Mono', monospace; letter-spacing:0.06em; text-transform:uppercase;
            padding:4px 8px; border-radius:3px; white-space:nowrap; }}
.ql-good .ql-pill {{ background:#e3f3ea; color:{POS}; }}
.ql-bad .ql-pill {{ background:#fbe7e5; color:{NEG}; }}
.ql-neutral .ql-pill {{ background:#f3eedf; color:#7a5d12; }}
.ql-verdict .ql-head {{ font-weight:600; font-size:1rem; }}
.ql-verdict .ql-sub {{ color:{MUTED}; font-size:0.85rem; }}
.ql-verdict .ql-stats {{ margin-left:auto; display:flex; gap:22px; flex-wrap:wrap; }}
.ql-kpis {{ display:grid; grid-template-columns:repeat(var(--cols, 4), minmax(0, 1fr)); gap:1px;
            background:{LINE}; border:1px solid {LINE}; border-radius:4px; overflow:hidden; margin-bottom:14px; }}
.ql-kpi {{ padding:12px 16px; background:{SURFACE}; min-width:0; }}
@media (max-width: 760px) {{ .ql-kpis {{ grid-template-columns:repeat(2, minmax(0, 1fr)); }} }}
.ql-label {{ font-size:0.72rem; color:{MUTED}; text-transform:uppercase; letter-spacing:0.05em; }}
.ql-value {{ font:500 1.3rem 'Geist Mono', monospace; color:{INK}; margin-top:2px; white-space:nowrap; }}
.ql-bench {{ font:400 0.75rem 'Geist Mono', monospace; color:{MUTED}; margin-top:2px; }}
.ql-pos {{ color:{POS}; }} .ql-neg {{ color:{NEG}; }}
.ql-signal {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:4px; padding:14px 18px; }}
.ql-signal .ql-state {{ font:600 1.5rem 'Geist Mono', monospace; }}
[data-testid="stTable"] td, [data-testid="stTable"] th {{ font-size:0.82rem; }}
[data-testid="stTable"] td {{ font-family:'Geist Mono', monospace; font-variant-numeric:tabular-nums; }}
[data-testid="stTabs"] button p {{ font-size:0.85rem; font-weight:500; }}
</style>
"""

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

INDICATOR_LABELS = {
    "price": "Prezzo",
    "sma": "Media mobile",
    "return": "Rendimento degli ultimi N giorni",
    "zscore": "z-score (distanza dalla media)",
    "high": "Massimo degli ultimi N giorni",
    "low": "Minimo degli ultimi N giorni",
}
# Factors usable inside rules: name -> (kind, key, short label).
#   yahoo:    daily market data, key = Yahoo ticker
#   macro:    FRED series made point-in-time, key = name in quantlab.macro.MACRO
#   earnings: the chosen company's own quarterly surprises
FACTORS = {
    "VIX (paura del mercato)": ("yahoo", "^VIX", "VIX"),
    "Tasso USA a 10 anni (%)": ("yahoo", "^TNX", "Tasso 10 anni"),
    "Tasso USA a 3 mesi (%)": ("yahoo", "^IRX", "Tasso 3 mesi"),
    "Dollaro (indice DXY)": ("yahoo", "DX-Y.NYB", "Dollaro"),
    "Petrolio WTI": ("yahoo", "CL=F", "Petrolio"),
    "Oro": ("yahoo", "GC=F", "Oro"),
    "Inflazione USA (% annuo)": ("macro", "inflation", "Inflazione"),
    "Disoccupazione USA (%)": ("macro", "unemployment", "Disoccupazione"),
    "Tassi della Fed (%)": ("macro", "fed_funds", "Tassi Fed"),
    "Curva dei tassi 10 anni - 2 anni (%)": ("macro", "yield_curve", "Curva tassi"),
    "Sorpresa dell'ultima trimestrale (%)": ("earnings", None, "Sorpresa utili"),
}


def factor_note(name: str) -> str:
    """One line on how a factor is kept honest (when its values become usable)."""
    kind, key, _ = FACTORS[name]
    if kind == "macro":
        return (f"Dato pubblicato in ritardo: nel test ogni valore diventa visibile solo "
                f"{MACRO[key].lag_days} giorni dopo la data a cui si riferisce, come nella realtà.")
    if kind == "earnings":
        return ("Utile annunciato contro le attese degli analisti. Usato solo dalla prima chiusura dopo "
                "l'annuncio. Esiste solo per singole aziende, non per ETF, indici o crypto.")
    return "Dato di mercato giornaliero: usato dal giorno stesso della chiusura."
OPERATOR_LABELS = {">": "sopra (>)", "<": "sotto (<)", ">=": "sopra o uguale (≥)", "<=": "sotto o uguale (≤)"}

COLORS = {"strategy": "#1f4aa8", "buy_and_hold": "#9aa1ab"}  # strategy in cobalt, benchmark in grey
# Vega-Lite expression: the year on January ticks, an Italian month abbreviation otherwise.
MONTHS_IT = ("['gen','feb','mar','apr','mag','giu','lug','ago','set','ott','nov','dic'][month(datum.value)]"
             " + ' ' + timeFormat(datum.value, '%y')")  # e.g. "ott 22"
TIME_AXIS = alt.X("date:T", title=None, axis=alt.Axis(labelExpr=MONTHS_IT))
# Italian number labels: swap the separators, 1,800.50 -> 1.800,50
NUMBERS_IT = ("replace(replace(replace(datum.label, regexp(',', 'g'), '_'), "
              "regexp('[.]', 'g'), ','), regexp('_', 'g'), '.')")


@st.cache_data(show_spinner="Scarico i prezzi...")
def get_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Prices for one ticker, from the local cache or Yahoo Finance (cached per session too)."""
    return data.load_prices([ticker], start, end, cache_dir=DATA_DIR)[ticker].dropna()


@st.cache_data(show_spinner="Scarico i dati...")
def load_factor(name: str, ticker: str = "") -> pd.Series:
    """Full history of a factor up to today (rules only ever read its past values).

    `ticker` is needed only by the earnings factor, which belongs to one company.
    """
    kind, key, _ = FACTORS[name]
    if kind == "macro":
        return macro.load_macro(key, DATA_DIR)
    if kind == "earnings":
        return earnings.load_earnings_surprises(ticker, DATA_DIR)
    return get_prices(key, "2000-01-01", str(date.today()))


@st.cache_data(ttl=1800, show_spinner=False)
def _news(query: str) -> list[dict]:
    return data.latest_news(query, count=10)


def show_news(query: str, terms: list[str] | None = None) -> None:
    """Latest headlines as context, clearly kept apart from the calculations.

    With `terms`, only headlines that name the asset are shown (a search also returns
    stories about other companies). Without, the raw search results are shown, saying so.
    """
    st.subheader("Ultime notizie")
    try:
        found = _news(query)
    except Exception:  # news are optional: never break the page for them
        found = []
    news = data.relevant_news(found, terms)[:5] if terms else found[:5]
    if not news:
        st.caption(f"Nessuna notizia recente che parli direttamente di {terms[0] if terms else query}.")
    elif not terms:
        st.caption(f"Risultati della ricerca «{query}» su Yahoo Finance: non tutti riguardano direttamente il titolo.")
    for item in news:
        when = item["published"].tz_convert("Europe/Rome").strftime("%d/%m %H:%M")
        st.markdown(f"[{item['title']}]({item['link']})<br><span style='color:#6b736c;font-size:0.8rem'>"
                    f"{item['publisher']} · {when}</span>", unsafe_allow_html=True)
    st.caption("Solo contesto: le notizie non entrano nei calcoli. Non esiste un archivio gratuito di notizie "
               "datate con cui verificare onestamente se una regola basata sulle notizie avrebbe funzionato.")


def verdict_tone(report: StrategyReport) -> str:
    """'good', 'bad' or 'neutral' from the bootstrap interval of the Sharpe difference."""
    low, high = report.sharpe_diff_ci
    if low > 0:
        return "good"
    if high < 0:
        return "bad"
    return "neutral"


def describe_indicator(ind: Indicator | float) -> str:
    """Short Italian label, e.g. 'Media mobile 100g' or '0.05'."""
    if not isinstance(ind, Indicator):
        return f"{ind:g}"
    label = INDICATOR_LABELS[ind.name].replace(" degli ultimi N giorni", "")
    if ind.label:  # computed on a factor, e.g. "VIX" or "Media mobile 20g di VIX"
        return ind.label if ind.name == "price" else f"{label} {ind.window}g di {ind.label}"
    return label if ind.name == "price" else f"{label} {ind.window}g"


def describe(rule: Condition | list[Condition]) -> str:
    """'Prezzo > Media mobile 200g', or several conditions joined by 'E'."""
    conditions = [rule] if isinstance(rule, Condition) else rule
    return " E ".join(
        f"{describe_indicator(c.left)} {c.op} {describe_indicator(c.right)}" for c in conditions
    )


def comparison_chart(table: pd.DataFrame, labels: dict, y_title: str, y_format: str) -> alt.Chart:
    """Strategy vs buy & hold as two lines with fixed, distinct colours."""
    long = table.rename_axis("date").reset_index().melt("date", var_name="series", value_name="value")
    long["series"] = long["series"].map(labels)
    return (
        alt.Chart(long)
        .mark_line(strokeWidth=1.6)
        .encode(
            x=TIME_AXIS,
            y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format=y_format, labelExpr=NUMBERS_IT), scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top", labelLimit=0),
                            scale=alt.Scale(domain=[labels[k] for k in COLORS], range=list(COLORS.values()))),
            tooltip=[alt.Tooltip("date:T", title="Data"), alt.Tooltip("series:N", title="Serie"),
                     alt.Tooltip("value:Q", title=y_title, format=y_format)],
        )
    )
