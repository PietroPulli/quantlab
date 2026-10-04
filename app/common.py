"""Pieces shared by the simple and the advanced view: style, strategies, prices, charts."""

from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from quantlab import data
from quantlab.report import StrategyReport
from quantlab.rules import Condition, Indicator
from quantlab.strategies import breakout, mean_reversion, momentum, moving_average_crossover
from quantlab.validation import param_grid

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Small additions on top of .streamlit/config.toml.
CSS = """
<style>
.ql-verdict { background:#ffffff; border:1px solid #d5d9d3; border-radius:4px; padding:18px 20px;
              display:grid; gap:6px; margin-bottom:8px; }
.ql-tag { font:500 0.72rem 'IBM Plex Mono', monospace; letter-spacing:0.08em; text-transform:uppercase;
          color:#6b736c; }
.ql-good .ql-tag { color:#2e6b4a; } .ql-bad .ql-tag { color:#9b2f2f; } .ql-neutral .ql-tag { color:#8a6a1c; }
.ql-title { font:600 1.45rem 'IBM Plex Serif', serif; line-height:1.2; color:#1b232c; }
.ql-big { font:600 2rem 'IBM Plex Serif', serif; line-height:1.15; color:#1b232c; }
.ql-text { color:#4a535c; }
.ql-facts { display:flex; flex-wrap:wrap; gap:8px 32px; margin:10px 0 0; padding-top:12px;
            border-top:1px solid #e3e6e1; }
.ql-facts div { min-width:0; }
.ql-facts dt { font-size:0.75rem; color:#6b736c; }
.ql-facts dd { margin:0; font:500 0.95rem 'IBM Plex Mono', monospace; overflow-wrap:anywhere; color:#1b232c; }
[data-testid="stTable"] td { font-family:'IBM Plex Mono', monospace; font-variant-numeric:tabular-nums; }
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
# Market-wide factors with a daily history, usable inside rules: name -> (Yahoo ticker, short label)
FACTORS = {
    "VIX (paura del mercato)": ("^VIX", "VIX"),
    "Tasso USA a 10 anni (%)": ("^TNX", "Tasso 10 anni"),
    "Tasso USA a 3 mesi (%)": ("^IRX", "Tasso 3 mesi"),
    "Dollaro (indice DXY)": ("DX-Y.NYB", "Dollaro"),
    "Petrolio WTI": ("CL=F", "Petrolio"),
    "Oro": ("GC=F", "Oro"),
}
OPERATOR_LABELS = {">": "sopra (>)", "<": "sotto (<)", ">=": "sopra o uguale (≥)", "<=": "sotto o uguale (≤)"}

COLORS = {"strategy": "#1f4e79", "buy_and_hold": "#b5762a"}  # ink blue vs ochre
# Vega-Lite expression: the year on January ticks, an Italian month abbreviation otherwise.
MONTHS_IT = ("month(datum.value) == 0 ? year(datum.value) + '' : "
             "['gen','feb','mar','apr','mag','giu','lug','ago','set','ott','nov','dic'][month(datum.value)]")
TIME_AXIS = alt.X("date:T", title=None, axis=alt.Axis(labelExpr=MONTHS_IT))
# Italian number labels: swap the separators, 1,800.50 -> 1.800,50
NUMBERS_IT = ("replace(replace(replace(datum.label, regexp(',', 'g'), '_'), "
              "regexp('[.]', 'g'), ','), regexp('_', 'g'), '.')")


@st.cache_data(show_spinner="Scarico i prezzi...")
def get_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Prices for one ticker, from the local cache or Yahoo Finance (cached per session too)."""
    return data.load_prices([ticker], start, end, cache_dir=DATA_DIR)[ticker].dropna()


def load_factor(name: str) -> pd.Series:
    """Full daily history of a factor up to today (rules only ever read past values of it)."""
    return get_prices(FACTORS[name][0], "2000-01-01", str(date.today()))


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
        .mark_line(strokeWidth=1.8)
        .encode(
            x=TIME_AXIS,
            y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format=y_format, labelExpr=NUMBERS_IT), scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top", labelLimit=0),
                            scale=alt.Scale(domain=[labels[k] for k in COLORS], range=list(COLORS.values()))),
            tooltip=[alt.Tooltip("date:T", title="Data"), alt.Tooltip("series:N", title="Serie"),
                     alt.Tooltip("value:Q", title=y_title, format=y_format)],
        )
    )
