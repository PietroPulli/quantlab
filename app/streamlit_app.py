"""Quantlab web app: pick a strategy and an asset, get an honest verdict against buy & hold.

Run from the project root with:  streamlit run app/streamlit_app.py

This file is only the user interface. Every number comes from the quantlab library
(the same tested functions used in the notebooks), never from code written here.
"""

import html
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from quantlab import data
from quantlab.backtest import current_signal, run_backtest, trade_log
from quantlab.metrics import calmar_ratio, drawdown
from quantlab.report import evaluate_strategy, format_report
from quantlab.rules import Condition, Indicator, rule_strategy
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


INDICATOR_LABELS = {
    "price": "Prezzo",
    "sma": "Media mobile",
    "return": "Rendimento degli ultimi N giorni",
    "zscore": "z-score (distanza dalla media)",
    "high": "Massimo degli ultimi N giorni",
    "low": "Minimo degli ultimi N giorni",
}
OPERATOR_LABELS = {">": "sopra (>)", "<": "sotto (<)", ">=": "sopra o uguale (≥)", "<=": "sotto o uguale (≤)"}


def describe_indicator(ind: Indicator | float) -> str:
    """Short Italian label, e.g. 'Media mobile 100g' or '0.05'."""
    if not isinstance(ind, Indicator):
        return f"{ind:g}"
    label = INDICATOR_LABELS[ind.name].replace(" degli ultimi N giorni", "")
    return label if ind.name == "price" else f"{label} {ind.window}g"


def describe(rule: Condition | list[Condition]) -> str:
    """'Prezzo > Media mobile 200g', or several conditions joined by 'E'."""
    conditions = [rule] if isinstance(rule, Condition) else rule
    return " E ".join(
        f"{describe_indicator(c.left)} {c.op} {describe_indicator(c.right)}" for c in conditions
    )


def indicator_input(label: str, key: str, default: Indicator) -> Indicator:
    """Two widgets (which indicator, how many days) -> an Indicator."""
    names = list(INDICATOR_LABELS)
    name = st.selectbox(label, names, index=names.index(default.name),
                        format_func=INDICATOR_LABELS.get, key=f"{key}-name")
    if name == "price":
        return Indicator("price")
    start_window = default.window if default.window >= 2 else 20  # "price" has no window
    window = st.number_input("N giorni", 2, 504, start_window, key=f"{key}-window")
    return Indicator(name, int(window))


def condition_input(key: str, left: Indicator, op: str, right: Indicator | float) -> Condition:
    """Widgets for one condition: <indicator> <operator> <indicator or number>.

    `left`, `op` and `right` are only the starting values shown to the user.
    """
    right_is_number = not isinstance(right, Indicator)
    left_ind = indicator_input("Se", f"{key}-left", left)
    op = st.selectbox("è", list(OPERATOR_LABELS), index=list(OPERATOR_LABELS).index(op),
                      format_func=OPERATOR_LABELS.get, key=f"{key}-op")
    kinds = ["un indicatore", "un numero"]
    kind = st.radio("di", kinds, index=int(right_is_number), horizontal=True, key=f"{key}-kind")
    if kind == "un numero":
        right_value: Indicator | float = st.number_input(
            "Numero", value=float(right) if right_is_number else 0.0, step=0.01, format="%.2f", key=f"{key}-number",
            help="Rendimenti in decimali (0.05 = +5%). z-score in deviazioni standard (-1 = 1 sotto la media).",
        )
    else:
        default = Indicator("sma", 50) if right_is_number else right
        right_value = indicator_input("Indicatore", f"{key}-right", default)
    return Condition(left_ind, op, right_value)


@st.cache_data(show_spinner="Scarico i prezzi...")
def get_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Prices for one ticker, from the local cache or Yahoo Finance (cached per session too)."""
    return data.load_prices([ticker], start, end, cache_dir=DATA_DIR)[ticker].dropna()


st.set_page_config(page_title="Quantlab", layout="wide")

# Small additions on top of .streamlit/config.toml: the verdict block and table numbers.
st.markdown(
    """
    <style>
    .ql-verdict { background:#ffffff; border:1px solid #d5d9d3; border-radius:4px; padding:18px 20px;
                  display:grid; gap:6px; margin-bottom:8px; }
    .ql-tag { font:500 0.72rem 'IBM Plex Mono', monospace; letter-spacing:0.08em; text-transform:uppercase; }
    .ql-good .ql-tag { color:#2e6b4a; } .ql-bad .ql-tag { color:#9b2f2f; } .ql-neutral .ql-tag { color:#8a6a1c; }
    .ql-title { font:600 1.45rem 'IBM Plex Serif', serif; line-height:1.2; color:#1b232c; }
    .ql-text { color:#4a535c; }
    .ql-facts { display:flex; flex-wrap:wrap; gap:8px 32px; margin:10px 0 0; padding-top:12px;
                border-top:1px solid #e3e6e1; }
    .ql-facts div { min-width:0; }
    .ql-facts dt { font-size:0.75rem; color:#6b736c; }
    .ql-facts dd { margin:0; font:500 0.95rem 'IBM Plex Mono', monospace; overflow-wrap:anywhere; color:#1b232c; }
    [data-testid="stTable"] td { font-family:'IBM Plex Mono', monospace; font-variant-numeric:tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Quantlab")
st.caption(
    "Scegli una regola di trading e un titolo. Quantlab la simula sul passato con costi reali "
    "e la confronta con il semplice compra e tieni. Non è un consiglio di investimento: "
    "dice solo se una regola avrebbe funzionato."
)

# ---- Inputs (sidebar) ----
with st.sidebar:
    st.header("1. Cosa provare")
    ticker = st.text_input("Titolo (ticker Yahoo)", value="SPY").strip().upper()
    col_a, col_b = st.columns(2)
    start = col_a.date_input("Dal", value=date(2015, 1, 1), min_value=date(1995, 1, 1))
    end = col_b.date_input("Al", value=date(2025, 12, 31), max_value=date.today())

    mode = st.radio("Tipo di strategia", ["Strategia pronta", "Crea la tua regola"], key="mode", horizontal=True)

    if mode == "Strategia pronta":
        name = st.selectbox("Strategia", list(STRATEGY_UI))
        ui = STRATEGY_UI[name]
        st.caption(ui["rule"])
        strategy_func = ui["func"]

        st.header("2. Parametri")
        optimize_params = st.toggle(
            "Fai scegliere i parametri al passato",
            help="Prova alcune combinazioni sul primo 70% dei dati e tiene la migliore. "
            "Il giudizio vero si fa poi sul restante 30%, mai visto durante la scelta.",
        )
        params, grid = {}, None
        if optimize_params:
            grid = ui["grid"]
            st.caption(f"Combinazioni provate: {grid}")
        else:
            for key, (label, lo, hi, default, step) in ui["params"].items():
                params[key] = st.slider(label, lo, hi, default, step, key=f"{name}-{key}")
    else:
        st.header("2. La tua regola")
        st.markdown("**Compra quando...**")
        entry = condition_input("entry", Indicator("price"), ">", Indicator("sma", 100))
        if st.toggle("Aggiungi una condizione (E)", key="entry_and",
                     help="Compra solo nei giorni in cui sono vere entrambe le condizioni."):
            st.markdown("**...e anche quando...**")
            entry = [entry, condition_input("entry2", Indicator("zscore", 10), "<", -1.0)]
        st.caption(f"Entrata: {describe(entry)}")
        exit_rule = None
        if st.toggle("Aggiungi una regola di uscita", key="has_exit",
                     help="Senza uscita sei investito solo nei giorni in cui la regola di entrata è vera. "
                     "Con un'uscita, una volta entrato resti dentro finché non scatta l'uscita."):
            st.markdown("**Vendi quando...**")
            exit_rule = condition_input("exit", Indicator("price"), "<", Indicator("sma", 50))
            st.caption(f"Uscita: {describe(exit_rule)}")
        strategy_func, grid = rule_strategy, None
        params = {"entry": entry, "exit": exit_rule}
        name = "La tua regola"

    st.header("3. Costi per operazione")
    commission = st.number_input("Commissione (%)", 0.0, 1.0, 0.10, 0.01) / 100
    slippage = st.number_input("Slippage (%)", 0.0, 1.0, 0.05, 0.01) / 100
    cash_rate = st.number_input(
        "Interesse sui contanti (% annuo)", 0.0, 10.0, 2.0, 0.25, key="cash_rate",
        help="Quando la strategia è fuori dal mercato, i soldi rendono questo tasso (come BOT o conto deposito). "
        "2% ≈ media dei T-bill USA 2015-2025: quasi 0 fino al 2021, 4-5% dal 2023. "
        "Lo stesso tasso viene sottratto nello Sharpe, così stare in contanti non sembra bravura.",
    ) / 100

    run = st.button("Metti alla prova", type="primary", width="stretch")

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
            strategy_func,
            params=None if grid else params,
            grid=grid,
            name=f"{name} su {ticker}",
            commission=commission,
            slippage=slippage,
            cash_rate=cash_rate,
        )
except ValueError as exc:  # e.g. fast average longer than the slow one
    st.error(f"Parametri non validi: {exc}")
    st.stop()

# ---- Verdict ----
low, high = report.sharpe_diff_ci
if low > 0:
    tone, tag, title, text = ("good", "Batte il benchmark", "Batte il compra e tieni",
                              "Anche tenendo conto della fortuna, il vantaggio regge.")
elif high < 0:
    tone, tag, title, text = ("bad", "Sotto il benchmark", "Peggio del compra e tieni",
                              "Anche tenendo conto della fortuna, lo svantaggio è netto.")
else:
    tone, tag, title, text = ("neutral", "Nessuna prova", "Nessuna prova che batta il compra e tieni",
                              "La differenza è compatibile con la fortuna.")
if strategy_func is rule_strategy:
    used = describe(entry) + (f"; uscita: {describe(exit_rule)}" if exit_rule else "")
else:
    # e.g. {"window": 50} -> "Finestra N (giorni) 50", using the same labels as the sliders
    used = ", ".join(f"{ui['params'][k][0]} {v}" for k, v in report.params.items()) or "nessun parametro"
facts = {
    "Regola": str(used),
    "Periodo di prova dal": str(report.split.date()),
    "Differenza di Sharpe": f"{report.sharpe_diff:+.2f}",
    "Intervallo al 95%": f"{low:+.2f} … {high:+.2f}",
    "Strategia avanti in": f"{report.share_beating:.0%} dei campioni",
}
facts_html = "".join(f"<div><dt>{k}</dt><dd>{html.escape(v)}</dd></div>" for k, v in facts.items())
st.markdown(
    f'<div class="ql-verdict ql-{tone}"><div class="ql-tag">Verdetto · {tag}</div>'
    f'<div class="ql-title">{title}</div><div class="ql-text">{text}</div>'
    f'<dl class="ql-facts">{facts_html}</dl></div>',
    unsafe_allow_html=True,
)

# Data snooping: every new idea tested on the same data is another lottery ticket.
# Remember what was tried on each ticker in this session (a grid counts as all its combinations).
tried = st.session_state.setdefault("tried", {}).setdefault(ticker, {})
tried[f"{name}: {grid if grid else used}"] = len(grid) if grid else 1
if len(tried) > 1:
    st.warning(
        f"Su {ticker} in questa sessione hai provato {len(tried)} strategie diverse "
        f"({sum(tried.values())} combinazioni di parametri in tutto). Più idee provi sugli stessi dati, "
        "più è probabile che la migliore sembri buona solo per fortuna. "
        "Il verdetto qui sopra non ne tiene conto: vale per un solo tentativo."
    )
for warning in report.warnings:
    st.warning(warning)

# ---- What the rule says today ----
# The test above stops at the chosen end date; here we apply the same rule, with the same
# parameters, to prices up to the latest close available.
st.subheader("Cosa dice la regola oggi")
try:
    live_prices = get_prices(ticker, str(start), str(date.today()))
    day, value, since = current_signal(strategy_func(live_prices, **report.params))
except Exception as exc:  # network problems, or a rule that cannot be applied
    st.info(f"Non riesco a calcolare il segnale di oggi: {exc}")
else:
    state = "DENTRO" if value > 0 else "FUORI"
    meaning = ("essere investiti in" if value > 0 else "stare in contanti, fuori da")
    with st.container(border=True):
        cols = st.columns([1, 2])
        cols[0].markdown(f'<div class="ql-tag">Segnale alla chiusura del {day.date()}</div>'
                         f'<div class="ql-title">{state}</div>', unsafe_allow_html=True)
        cols[1].markdown(
            f"Con i prezzi fino al **{day.date()}** (ultima chiusura {live_prices.iloc[-1]:.2f}), "
            f"la regola dice di {meaning} **{ticker}**, ininterrottamente dal **{since.date()}**. "
            "Il segnale vale dalla seduta successiva."
        )
    st.caption(
        "Questo è il risultato della tua regola applicata ai prezzi di oggi, non un consiglio di investimento. "
        "Quanto fidarsi della regola lo dice il verdetto qui sopra"
        + (": e qui non ha battuto il compra e tieni." if tone != "good" else ".")
    )

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


COLORS = {"strategy": "#1f4e79", "buy_and_hold": "#b5762a"}  # ink blue vs ochre
# Vega-Lite expression: the year on January ticks, an Italian month abbreviation otherwise.
MONTHS_IT = ("month(datum.value) == 0 ? year(datum.value) + '' : "
             "['gen','feb','mar','apr','mag','giu','lug','ago','set','ott','nov','dic'][month(datum.value)]")
TIME_AXIS = alt.X("date:T", title=None, axis=alt.Axis(labelExpr=MONTHS_IT))


def comparison_chart(table: pd.DataFrame, y_title: str, y_format: str) -> alt.Chart:
    """Strategy vs buy & hold as two lines with fixed, distinct colours."""
    long = table.rename_axis("date").reset_index().melt("date", var_name="series", value_name="value")
    long["series"] = long["series"].map(labels)
    return (
        alt.Chart(long)
        .mark_line(strokeWidth=1.8)
        .encode(
            x=TIME_AXIS,
            y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format=y_format), scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=[labels[k] for k in COLORS], range=list(COLORS.values()))),
            tooltip=[alt.Tooltip("date:T", title="Data"), alt.Tooltip("series:N", title="Serie"),
                     alt.Tooltip("value:Q", title=y_title, format=y_format)],
        )
    )


left, right = st.columns([3, 2])
with left:
    st.subheader("Quanto diventa 1 € (periodo di prova, costi inclusi)")
    st.altair_chart(comparison_chart((1 + oos_returns).cumprod(), "Valore di 1 €", ".2f"), width="stretch")
    st.subheader("Perdita dal massimo precedente")
    st.altair_chart(comparison_chart(oos_returns.apply(drawdown), "Perdita", ".0%"), width="stretch")
with right:
    st.subheader("Periodo di prova")
    st.table(pretty(oos))
    st.subheader("Periodo di scelta (in-sample)")
    if grid:
        st.caption("Qui la strategia è avvantaggiata: i parametri sono stati scelti guardando questi dati.")
    else:
        st.caption("Parametri fissi: nessuna ottimizzazione automatica. Ma se li hai scelti tu "
                   "dopo aver guardato i grafici, li hai scelti sul passato anche tu.")
    st.table(pretty(report.in_sample))
    if report.walk_forward is not None:
        st.subheader("Walk-forward")
        st.caption("Riscelta dei parametri ogni anno, usando solo i 3 anni precedenti.")
        st.table(pretty(report.walk_forward))

# ---- When does it trade? ----
positions = run_backtest(prices, strategy_func(prices, **report.params), commission, slippage, cash_rate)["position"]
trades = trade_log(prices, positions)
trades = trades[trades.index >= report.split]
oos_prices = prices[prices.index >= report.split]

st.subheader("Quando compra e vende (periodo di prova)")
years = len(oos_prices) / 252
st.caption(
    f"{len(trades)} operazioni in {years:.1f} anni ({len(trades) / years:.1f} all'anno). "
    "Ogni operazione paga commissione e slippage: più operazioni, più costi."
)
price_line = (
    alt.Chart(oos_prices.rename("price").rename_axis("date").reset_index())
    .mark_line(color="#7d847c", strokeWidth=1.3)
    .encode(x=TIME_AXIS, y=alt.Y("price:Q", title="Prezzo", scale=alt.Scale(zero=False)))
)
trade_points = (
    alt.Chart(trades.rename_axis("date").reset_index())
    .mark_point(filled=True, size=160, opacity=1)
    .encode(
        x="date:T",
        y="price:Q",
        color=alt.Color("action:N", title=None,
                        scale=alt.Scale(domain=["buy", "sell"], range=["#2e6b4a", "#9b2f2f"]),
                        legend=alt.Legend(labelExpr="datum.label == 'buy' ? 'Compra' : 'Vende'")),
        shape=alt.Shape("action:N", scale=alt.Scale(domain=["buy", "sell"], range=["triangle-up", "triangle-down"]),
                        legend=None),
        tooltip=[alt.Tooltip("date:T", title="Data"), alt.Tooltip("action:N", title="Operazione"),
                 alt.Tooltip("price:Q", title="Prezzo", format=".2f")],
    )
)
st.altair_chart(price_line + trade_points, width="stretch")

with st.expander("Report completo in testo"):
    st.code(format_report(report), language=None)
