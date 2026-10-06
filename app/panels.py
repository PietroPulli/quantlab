"""Result panels shared by both views: verdict bar, KPI strip, charts and tables.

Only presentation lives here: every number is computed by the quantlab library.
"""

import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import COLORS, INK, LINE, MUTED, NEG, NUMBERS_IT, POS, TIME_AXIS
from quantlab.backtest import round_trips, run_backtest, trade_log, trade_stats
from quantlab.metrics import (
    calendar_returns,
    calmar_ratio,
    drawdown,
    drawdown_periods,
    monthly_returns,
    rolling_sharpe,
    rolling_volatility,
    summary,
)
from quantlab.report import StrategyReport

MONTHS = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"]


# ---------------------------------------------------------------- formatting
def pct(x: float, signed: bool = False, decimals: int = 1) -> str:
    """0.1234 -> '12,3%' (Italian decimal comma); NaN -> 'n/d'."""
    if x is None or pd.isna(x):
        return "n/d"
    return f"{x * 100:{'+' if signed else ''}.{decimals}f}%".replace(".", ",")


def num(x: float, decimals: int = 2, signed: bool = False) -> str:
    """1234.5 -> '1.234,50': Italian separators (thousands '.', decimals ',')."""
    if x is None or pd.isna(x):
        return "n/d"
    text = f"{x:{'+' if signed else ''},.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def euro(x: float) -> str:
    return f"{x:,.0f} €".replace(",", ".")


def day(ts) -> str:
    return "aperta" if pd.isna(ts) else pd.Timestamp(ts).strftime("%d/%m/%Y")


# ---------------------------------------------------------------- analytics
def analyse(report: StrategyReport, prices: pd.Series, strategy, commission: float, slippage: float,
            cash_rate: float, period: str = "test") -> dict:
    """Everything the panels show, for the test period (out-of-sample) or the full history."""
    positions = run_backtest(prices, strategy(prices, **report.params), commission, slippage, cash_rate,
                             report.periods_per_year)["position"]
    start = report.split if period == "test" else report.returns.index[0]
    keep = report.returns.index >= start
    returns = report.returns[keep]
    trips = round_trips(prices, positions, commission + slippage)
    trips = trips[trips["entry_date"] >= start] if period == "test" else trips
    trades = trade_log(prices, positions)
    years = len(returns) / report.periods_per_year
    return {
        "start": start,
        "returns": returns,
        "metrics": pd.DataFrame({c: summary(returns[c], report.periods_per_year, cash_rate) for c in returns}),
        "calmar": {c: calmar_ratio(returns[c], report.periods_per_year) for c in returns},
        "exposure": float(positions[keep].mean()),
        "trades": trades[trades.index >= start],
        "trips": trips,
        "stats": trade_stats(trips[~trips["open"]]),  # an open trade has no result yet
        "years": years,
        "prices": prices[prices.index >= start],
    }


# ---------------------------------------------------------------- blocks
def context_line(parts: list[str]) -> None:
    st.markdown('<div class="ql-context">' + " &nbsp;·&nbsp; ".join(parts) + "</div>", unsafe_allow_html=True)


VERDICTS = {  # tone -> (pill, headline, explanation)
    "good": ("Supera il benchmark", "Batte il compra e tieni",
             "Anche tenendo conto della fortuna (bootstrap), il vantaggio regge."),
    "bad": ("Sotto il benchmark", "Peggio del compra e tieni",
            "Anche tenendo conto della fortuna (bootstrap), lo svantaggio è netto."),
    "neutral": ("Nessuna evidenza", "Nessuna prova che batta il compra e tieni",
                "La differenza rispetto al benchmark è compatibile con la fortuna."),
}


def verdict_bar(report: StrategyReport, tone: str, plain: str | None = None) -> None:
    """One line: verdict pill, headline, and the three numbers it is based on."""
    pill, head, sub = VERDICTS[tone]
    low, high = report.sharpe_diff_ci
    stats = "" if plain else (
        f'<div class="ql-stats">'
        f'<div><div class="ql-label">Diff. Sharpe</div><div class="ql-bench">{num(report.sharpe_diff, signed=True)}</div></div>'
        f'<div><div class="ql-label">Intervallo 95%</div><div class="ql-bench">{num(low, signed=True)} … {num(high, signed=True)}</div></div>'
        f'<div><div class="ql-label">Avanti nei campioni</div><div class="ql-bench">{pct(report.share_beating, decimals=0)}</div></div>'
        f'</div>')
    st.markdown(
        f'<div class="ql-verdict ql-{tone}"><span class="ql-pill">{pill}</span>'
        f'<div><div class="ql-head">{head}</div><div class="ql-sub">{html.escape(plain or sub)}</div></div>'
        f'{stats}</div>', unsafe_allow_html=True)


def kpi_strip(items: list[tuple]) -> None:
    """items: (label, value, benchmark text or None, +1 if better / -1 if worse / 0)."""
    cells = []
    for label, value, bench, better in items:
        cls = {1: "ql-pos", -1: "ql-neg"}.get(better, "")
        bench_html = f'<div class="ql-bench">{bench}</div>' if bench else ""
        cells.append(f'<div class="ql-kpi"><div class="ql-label">{label}</div>'
                     f'<div class="ql-value {cls}">{value}</div>{bench_html}</div>')
    cols = len(items) if len(items) <= 6 else (len(items) + 1) // 2  # 8 items -> 2 rows of 4
    st.markdown(f'<div class="ql-kpis" style="--cols:{cols}">' + "".join(cells) + "</div>", unsafe_allow_html=True)


def _better(a: float, b: float, higher_is_better: bool = True) -> int:
    if pd.isna(a) or pd.isna(b) or np.isclose(a, b):
        return 0
    return 1 if (a > b) == higher_is_better else -1


def standard_kpis(a: dict) -> list[tuple]:
    m, c = a["metrics"], a["calmar"]
    s, b = m["strategy"], m["buy_and_hold"]
    return [
        ("Rendimento totale", pct(s["total_return"], True), f"B&H {pct(b['total_return'], True)}",
         _better(s["total_return"], b["total_return"])),
        ("Rendimento annuo", pct(s["annual_return"], True), f"B&H {pct(b['annual_return'], True)}",
         _better(s["annual_return"], b["annual_return"])),
        ("Volatilità annua", pct(s["annual_volatility"]), f"B&H {pct(b['annual_volatility'])}",
         _better(s["annual_volatility"], b["annual_volatility"], higher_is_better=False)),
        ("Sharpe", num(s["sharpe"]), f"B&H {num(b['sharpe'])}", _better(s["sharpe"], b["sharpe"])),
        ("Max drawdown", pct(s["max_drawdown"]), f"B&H {pct(b['max_drawdown'])}",
         _better(s["max_drawdown"], b["max_drawdown"])),
        ("Calmar", num(c["strategy"]), f"B&H {num(c['buy_and_hold'])}", _better(c["strategy"], c["buy_and_hold"])),
        ("Esposizione", pct(a["exposure"], decimals=0), "B&H 100%", 0),
        ("Operazioni / anno", num(len(a["trades"]) / a["years"], 1), f"{len(a['trades'])} in totale", 0),
    ]


# ---------------------------------------------------------------- charts
def _series_color(labels: dict) -> alt.Color:
    return alt.Color("series:N", title=None, legend=alt.Legend(orient="top", labelLimit=0, title=None),
                     scale=alt.Scale(domain=[labels[k] for k in COLORS], range=list(COLORS.values())))


def _long(table: pd.DataFrame, labels: dict) -> pd.DataFrame:
    out = table.rename_axis("date").reset_index().melt("date", var_name="series", value_name="value")
    out["series"] = out["series"].map(labels)
    return out


def crosshair(table: pd.DataFrame, fmt) -> alt.Chart:
    """Broker-style hover: a vertical line follows the pointer and shows every value of that day.

    `table`: one column per series (column name = label shown), indexed by date. Values are
    pre-formatted with `fmt` so the tooltip uses Italian separators.
    """
    shown = table.copy()
    names = [f"s{i}" for i in range(len(shown.columns))]  # safe field names; labels go in the titles
    titles = list(shown.columns)
    shown.columns = names
    for c in names:
        shown[c] = shown[c].map(fmt)
    shown = shown.rename_axis("date").reset_index()
    near = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False, clear="pointerout")
    return (alt.Chart(shown).mark_rule(color="#9aa3ae", strokeWidth=1, strokeDash=[3, 3])
            .encode(x="date:T", opacity=alt.condition(near, alt.value(0.8), alt.value(0)),
                    tooltip=[alt.Tooltip("date:T", title="Data", format="%d/%m/%Y"),
                             *[alt.Tooltip(f"{n}:N", title=t) for n, t in zip(names, titles)]])
            .add_params(near))


def equity_chart(returns: pd.DataFrame, labels: dict, start_value: float = 1.0, log: bool = False,
                 y_title: str = "Valore di 1 €", height: int = 320) -> alt.Chart:
    values = start_value * (1 + returns).cumprod()
    data = _long(values, labels)
    y = alt.Y("value:Q", title=y_title, scale=alt.Scale(type="log" if log else "linear", zero=False),
              axis=alt.Axis(labelExpr=NUMBERS_IT, format=",.2f" if start_value == 1 else ",.0f"))
    lines = alt.Chart(data).mark_line(strokeWidth=1.6).encode(x=TIME_AXIS, y=y, color=_series_color(labels))
    hover = crosshair(values.rename(columns=labels), (lambda v: num(v)) if start_value == 1 else euro)
    return alt.layer(lines, hover).properties(height=height)


def drawdown_chart(returns: pd.DataFrame, labels: dict) -> alt.Chart:
    falls = returns.apply(drawdown)
    data = _long(falls, labels)
    y = alt.Y("value:Q", title="Distanza dal massimo", axis=alt.Axis(format=".0%", labelExpr=NUMBERS_IT))
    area = (alt.Chart(data[data["series"] == labels["strategy"]], height=180)
            .mark_area(color=COLORS["strategy"], opacity=0.18, line={"color": COLORS["strategy"], "strokeWidth": 1.2})
            .encode(x=TIME_AXIS, y=y))
    bench = (alt.Chart(data[data["series"] == labels["buy_and_hold"]])
             .mark_line(color=COLORS["buy_and_hold"], strokeWidth=1.1).encode(x=TIME_AXIS, y=y))
    return area + bench + crosshair(falls.rename(columns=labels), pct)


def annual_chart(returns: pd.DataFrame, labels: dict, height: int = 260) -> alt.Chart:
    table = pd.DataFrame({c: calendar_returns(returns[c]) for c in returns})
    data = table.rename_axis("year").reset_index().melt("year", var_name="series", value_name="value")
    data["series"] = data["series"].map(labels)
    return (alt.Chart(data, height=height).mark_bar()
            .encode(x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0)), xOffset="series:N",
                    y=alt.Y("value:Q", title="Rendimento", axis=alt.Axis(format=".0%", labelExpr=NUMBERS_IT)),
                    color=_series_color(labels),
                    tooltip=[alt.Tooltip("year:O", title="Anno"), alt.Tooltip("series:N", title="Serie"),
                             alt.Tooltip("value:Q", title="Rendimento", format="+.1%")]))


def monthly_heatmap(returns: pd.Series) -> alt.Chart:
    table = monthly_returns(returns)
    data = table.stack().dropna().rename("value").reset_index()  # months without data stay empty
    data["mese"] = data["month"].map(lambda m: MONTHS[m - 1])
    data["label"] = data["value"].map(lambda v: pct(v, signed=True))
    limit = max(0.01, float(data["value"].abs().quantile(0.95))) if len(data) else 0.05
    base = alt.Chart(data, height=max(120, 26 * table.shape[0])).encode(
        x=alt.X("mese:O", sort=MONTHS, title=None, axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("year:O", title=None))
    cells = base.mark_rect(stroke="#0c0f14", strokeWidth=1).encode(
        color=alt.Color("value:Q", legend=None,
                        scale=alt.Scale(domain=[-limit, 0, limit], range=[NEG, "#1b2129", POS], clamp=True,
                                        interpolate="rgb")),
        tooltip=[alt.Tooltip("year:O", title="Anno"), alt.Tooltip("mese:N", title="Mese"),
                 alt.Tooltip("label:N", title="Rendimento")])
    text = base.mark_text(font="Geist Mono", fontSize=10).encode(
        text="label:N",
        color=alt.value("#ffffff"))
    return cells + text


def rolling_chart(returns: pd.DataFrame, labels: dict, kind: str, window: int, ppy: int, rf: float) -> alt.Chart:
    if kind == "sharpe":
        table = returns.apply(lambda r: rolling_sharpe(r, window, rf, ppy))
        y = alt.Y("value:Q", title=f"Sharpe su {window} giorni", axis=alt.Axis(labelExpr=NUMBERS_IT))
    else:
        table = returns.apply(lambda r: rolling_volatility(r, window, ppy))
        y = alt.Y("value:Q", title=f"Volatilità su {window} giorni", axis=alt.Axis(format=".0%", labelExpr=NUMBERS_IT))
    table = table.dropna()
    lines = (alt.Chart(_long(table, labels), height=200).mark_line(strokeWidth=1.3)
             .encode(x=TIME_AXIS, y=y, color=_series_color(labels)))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=LINE).encode(y="y:Q")
    return zero + lines + crosshair(table.rename(columns=labels), num if kind == "sharpe" else pct)


def trades_chart(prices: pd.Series, trades: pd.DataFrame) -> alt.Chart:
    line = (alt.Chart(prices.rename("price").rename_axis("date").reset_index(), height=300)
            .mark_line(color="#aeb5bf", strokeWidth=1.1)
            .encode(x=TIME_AXIS, y=alt.Y("price:Q", title="Prezzo", scale=alt.Scale(zero=False),
                                         axis=alt.Axis(labelExpr=NUMBERS_IT))))
    points = (alt.Chart(trades.rename_axis("date").reset_index())
              .mark_point(filled=True, size=90, opacity=1)
              .encode(x="date:T", y="price:Q",
                      color=alt.Color("action:N", title=None, scale=alt.Scale(domain=["buy", "sell"], range=[POS, NEG]),
                                      legend=alt.Legend(orient="top",
                                                        labelExpr="datum.label == 'buy' ? 'Acquisto' : 'Vendita'")),
                      shape=alt.Shape("action:N", legend=None,
                                      scale=alt.Scale(domain=["buy", "sell"], range=["triangle-up", "triangle-down"])),
                      tooltip=[alt.Tooltip("date:T", title="Data"), alt.Tooltip("action:N", title="Operazione"),
                               alt.Tooltip("price:Q", title="Prezzo", format=",.2f")]))
    return line + points + crosshair(prices.rename("Prezzo").to_frame(), num)


def bootstrap_chart(samples: np.ndarray, ci: tuple[float, float]) -> alt.Chart:
    data = pd.DataFrame({"d": samples[~np.isnan(samples)]})
    bars = (alt.Chart(data, height=220).mark_bar(color="#3a4554")
            .encode(x=alt.X("d:Q", bin=alt.Bin(maxbins=45), title="Sharpe strategia − Sharpe compra e tieni"),
                    y=alt.Y("count():Q", title="Campioni")))
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#ffffff", strokeWidth=1.5).encode(x="x:Q")
    bounds = (alt.Chart(pd.DataFrame({"x": list(ci)}))
              .mark_rule(color=COLORS["strategy"], strokeDash=[4, 3], strokeWidth=1.5).encode(x="x:Q"))
    return bars + zero + bounds


# ---------------------------------------------------------------- tables
ROW_NAMES = {"total_return": "Rendimento totale", "annual_return": "Rendimento annuo",
             "annual_volatility": "Volatilità annua", "sharpe": "Sharpe", "max_drawdown": "Max drawdown"}


def metrics_table(table: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """quantlab.metrics.compare output -> formatted strings, one column per series."""
    rows = {}
    for key, name in ROW_NAMES.items():
        signed = key in ("total_return", "annual_return")
        rows[name] = [num(v) if key == "sharpe" else pct(v, signed=signed) for v in table.loc[key]]
    return pd.DataFrame(rows, index=[labels[c] for c in table.columns]).T


def annual_table(returns: pd.DataFrame, labels: dict) -> pd.DataFrame:
    table = pd.DataFrame({labels[c]: calendar_returns(returns[c]) for c in returns})
    table["Differenza"] = table.iloc[:, 0] - table.iloc[:, 1]
    return table.map(lambda v: pct(v, signed=True)).rename_axis("Anno")


def _numbered(columns: dict) -> pd.DataFrame:
    """Table with rows numbered from 1. Columns are passed as plain lists on purpose: a pandas
    Series would be aligned on its own index (0, 1, ...) and every row would shift by one."""
    columns = {name: list(values) for name, values in columns.items()}
    n = len(next(iter(columns.values()))) if columns else 0
    return pd.DataFrame(columns, index=pd.RangeIndex(1, n + 1, name="#"))


def drawdown_table(returns: pd.Series) -> pd.DataFrame:
    periods = drawdown_periods(returns)
    return _numbered({
        "Inizio": periods["start"].map(day), "Minimo": periods["trough"].map(day),
        "Recupero": periods["recovery"].map(day), "Profondità": periods["depth"].map(pct),
        "Giorni al minimo": periods["days_to_trough"].astype(int).astype(str),
        "Giorni al recupero": periods["days_to_recover"].map(lambda d: "—" if pd.isna(d) else str(int(d))),
    })


def trips_table(trips: pd.DataFrame) -> pd.DataFrame:
    return _numbered({
        "Entrata": trips["entry_date"].map(day), "Prezzo entrata": trips["entry_price"].map(num),
        "Uscita": [("aperta" if o else day(d)) for d, o in zip(trips["exit_date"], trips["open"])],
        "Prezzo uscita": trips["exit_price"].map(num),
        "Risultato netto": trips["net_return"].map(lambda v: pct(v, signed=True)),
        "Giorni": trips["days"].astype(str),
    })


def trade_kpis(stats: dict) -> list[tuple]:
    return [
        ("Operazioni chiuse", str(stats["trades"]), "acquisto + vendita", 0),
        ("Vincenti", pct(stats["win_rate"], decimals=0), None, 0),
        ("Guadagno medio", pct(stats["avg_win"], True), None, 0),
        ("Perdita media", pct(stats["avg_loss"], True), None, 0),
        ("Profit factor", num(stats["profit_factor"]), "guadagni / perdite", 0),
        ("Durata media", f"{num(stats['avg_days'], 0)} gg", None, 0),
    ]


def signal_panel(ticker: str, day_, value: float, since, last_price: float, tone: str,
                 who: str = "La regola") -> None:
    """What the rule says at the latest close, with the reminder that it is not advice."""
    state = "DENTRO" if value > 0 else "FUORI"
    meaning = "essere investiti in" if value > 0 else "stare in contanti, fuori da"
    st.markdown(
        f'<div class="ql-signal"><div class="ql-label">Segnale alla chiusura del {day_:%d/%m/%Y}</div>'
        f'<div class="ql-state">{state}</div>'
        f'<div class="ql-sub">{who} dice di {meaning} <b>{html.escape(ticker)}</b>, ininterrottamente dal '
        f'{since:%d/%m/%Y} (ultima chiusura {num(last_price)}). Il segnale vale dalla seduta successiva.</div></div>',
        unsafe_allow_html=True)
    st.caption("Regola applicata ai prezzi di oggi: non un consiglio di investimento."
               + ("" if tone == "good" else " Sul passato non ha battuto il compra e tieni."))


def chart(c: alt.Chart) -> None:
    """Render an Altair chart with the app's chart settings."""
    st.altair_chart(c.properties(background="transparent").configure_view(strokeWidth=0)
                    .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#1b2129", domainColor=LINE,
                                    tickColor=LINE, labelFont="Geist", titleFont="Geist", titleFontWeight=500)
                    .configure_legend(labelColor=INK, labelFont="Geist", padding=0, offset=6, labelFontSize=11),
                    width="stretch")



# ---------------------------------------------------------------- broker-style instrument page
RANGES = {"1M": 21, "3M": 63, "6M": 126, "1A": 252, "5A": 1260, "MAX": None}  # trading days shown
MA_COLORS = ["#d6aa3c", "#a77bff"]


def quote_header(name: str, ticker: str, ohlcv: pd.DataFrame) -> None:
    """Instrument header: last price, day change, day and 52-week range, volume."""
    last, prev = ohlcv.iloc[-1], ohlcv.iloc[-2]
    change = last["Close"] / prev["Close"] - 1
    year = ohlcv.tail(252)
    cls = "ql-pos" if change >= 0 else "ql-neg"
    volume = f"{last['Volume']:,.0f}".replace(",", ".") if last["Volume"] else "n/d"
    meta = [("Apertura", num(last["Open"])), ("Min / Max giorno", f"{num(last['Low'])} – {num(last['High'])}"),
            ("Min / Max 52 sett.", f"{num(year['Low'].min())} – {num(year['High'].max())}"),
            ("Volume", volume), ("Chiusura del", ohlcv.index[-1].strftime("%d/%m/%Y"))]
    meta_html = "".join(f"<div><span>{k}</span>{v}</div>" for k, v in meta)
    st.markdown(
        f'<div class="ql-quote"><div><div class="ql-name">{html.escape(name)}'
        + (f'<span class="ql-sym">{html.escape(ticker)}</span>' if ticker != name else "") + '</div>'
        f'<div><span class="ql-px">{num(last["Close"])}</span>'
        f'<span class="ql-chg {cls}">{num(last["Close"] - prev["Close"], signed=True)} '
        f'({pct(change, True, 2)})</span></div></div>'
        f'<div class="ql-meta">{meta_html}</div></div>', unsafe_allow_html=True)


def candle_chart(ohlcv: pd.DataFrame, days: int | None, trades: pd.DataFrame | None = None,
                 averages: tuple[int, ...] = (50, 200)) -> alt.VConcatChart:
    """Candlesticks with moving averages and optional trade markers, volume bars below."""
    data = ohlcv.copy()
    for n in averages:
        data[f"MA{n}"] = data["Close"].rolling(n).mean()  # on the full history, then cut to the range
    data = data.tail(days) if days else data
    data = data.rename_axis("date").reset_index()
    data["up"] = data["Close"] >= data["Open"]
    color = alt.condition("datum.up", alt.value(POS), alt.value(NEG))
    width = max(1, min(8, 700 // max(1, len(data))))
    x = TIME_AXIS
    base = alt.Chart(data)
    y = alt.Y("Low:Q", title=None, scale=alt.Scale(zero=False), axis=alt.Axis(labelExpr=NUMBERS_IT, orient="right"))
    wick = base.mark_rule(strokeWidth=1).encode(x=x, y=y, y2="High:Q", color=color)
    body = base.mark_bar(size=width).encode(
        x=x, y="Open:Q", y2="Close:Q", color=color,
        tooltip=[alt.Tooltip("date:T", title="Data"), alt.Tooltip("Open:Q", title="Apertura", format=",.2f"),
                 alt.Tooltip("High:Q", title="Massimo", format=",.2f"),
                 alt.Tooltip("Low:Q", title="Minimo", format=",.2f"),
                 alt.Tooltip("Close:Q", title="Chiusura", format=",.2f")])
    layers = [wick, body]
    for n, c in zip(averages, MA_COLORS):
        layers.append(base.mark_line(color=c, strokeWidth=1.2, opacity=0.9).encode(x=x, y=f"MA{n}:Q"))
    if trades is not None and not trades.empty:
        shown = trades[trades.index >= data["date"].iloc[0]].rename_axis("date").reset_index()
        layers.append(
            alt.Chart(shown).mark_point(filled=True, size=110, opacity=1, stroke="#0c0f14", strokeWidth=1)
            .encode(x="date:T", y="price:Q",
                    color=alt.Color("action:N", legend=None,
                                    scale=alt.Scale(domain=["buy", "sell"], range=[POS, NEG])),
                    shape=alt.Shape("action:N", legend=None,
                                    scale=alt.Scale(domain=["buy", "sell"], range=["triangle-up", "triangle-down"])),
                    tooltip=[alt.Tooltip("date:T", title="Data"), alt.Tooltip("action:N", title="Operazione"),
                             alt.Tooltip("price:Q", title="Prezzo", format=",.2f")]))
    day = data.set_index("date")[["Open", "High", "Low", "Close"]].rename(
        columns={"Open": "Apertura", "High": "Massimo", "Low": "Minimo", "Close": "Chiusura"})
    day["Volume"] = data.set_index("date")["Volume"]
    layers.append(crosshair(day, lambda v: num(v) if abs(v) < 1e7 else f"{v:,.0f}".replace(",", ".")))
    price = alt.layer(*layers).properties(height=340)
    volume = base.mark_bar(opacity=0.45, size=width).encode(
        x=alt.X("date:T", title=None, axis=None), y=alt.Y("Volume:Q", title=None, axis=None), color=color,
    ).properties(height=60)
    return alt.vconcat(price, volume, spacing=2)


def ma_legend(averages: tuple[int, ...] = (50, 200), with_trades: bool = False) -> None:
    items = [f'<span style="color:{c}">━</span> Media {n} giorni' for n, c in zip(averages, MA_COLORS)]
    if with_trades:
        items += [f'<span style="color:{POS}">▲</span> acquisto della strategia',
                  f'<span style="color:{NEG}">▼</span> vendita']
    st.markdown('<div class="ql-context">' + " &nbsp; ".join(items) + "</div>", unsafe_allow_html=True)


def watchlist(rows: list[dict], active: str = "", title: str = "Watchlist") -> None:
    """Broker-style list: name, ticker, last price and day change. rows: name, ticker, last, change."""
    body = [f'<div class="ql-watch ql-mini"><div class="ql-watch-head">{html.escape(title)}</div>']
    for r in rows:
        cls = "ql-pos" if r["change"] >= 0 else "ql-neg"
        act = " ql-active" if r["name"] == active else ""
        body.append(f'<div class="ql-watch-row{act}"><div><b>{html.escape(r["name"])}</b>'
                    f'<small>{html.escape(r["ticker"])}</small></div>'
                    f'<div class="ql-num">{num(r["last"])}</div>'
                    f'<div class="ql-num {cls}">{pct(r["change"], True, 2)}</div></div>')
    body.append("</div>")
    st.markdown("".join(body), unsafe_allow_html=True)
