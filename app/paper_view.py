"""Demo account view: how the ideas are doing going forward, with virtual money."""

import json
import os
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from common import COLORS, NUMBERS_IT, TIME_AXIS, app_factor, get_prices
from panels import chart, context_line, crosshair, euro, kpi_strip, num, pct
from quantlab.backtest import run_backtest
from quantlab.ideas import IDEAS
from quantlab.paper import portfolio_history, replay

PAPER_DIR = Path(os.environ.get("QUANTLAB_PAPER_DIR", Path(__file__).resolve().parents[1] / "paper"))


def _read(name: str) -> pd.DataFrame:
    path = PAPER_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def summary() -> dict | None:
    """Current value of the demo account and of its buy & hold twin, or None before the first run."""
    values = _read("values.csv")
    if values.empty:
        return None
    history = portfolio_history(values)
    start = json.loads((PAPER_DIR / "state.json").read_text(encoding="utf-8"))["start"]
    return {"value": history["value"].iloc[-1], "bench": history["bench_value"].iloc[-1],
            "date": pd.Timestamp(history.index[-1]), "start": pd.Timestamp(start)}


def render() -> None:
    st.subheader("Conto demo: le idee alla prova con soldi finti")
    config = json.loads((PAPER_DIR / "config.json").read_text(encoding="utf-8"))
    live, fast = st.tabs(["Dal vivo", "Replay accelerato"])
    with live:
        _live(config)
    with fast:
        _replay(config)


def _account_chart(history: pd.DataFrame) -> None:
    """Strategy account vs its buy & hold twin, with the crosshair."""
    named = history.rename(columns={"value": "Conto demo", "bench_value": "Compra e tieni"})
    long = named.rename_axis("date").reset_index().melt("date", var_name="series", value_name="value")
    lines = alt.Chart(long).mark_line(strokeWidth=1.6, point=len(history) < 30).encode(
        x=TIME_AXIS,
        y=alt.Y("value:Q", title="Valore (€)", scale=alt.Scale(zero=False),
                axis=alt.Axis(format=",.0f", labelExpr=NUMBERS_IT)),
        color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["Conto demo", "Compra e tieni"], range=list(COLORS.values()))))
    chart(alt.layer(lines, crosshair(named, euro)).properties(height=300))


def _live(config: dict) -> None:
    state_path = PAPER_DIR / "state.json"
    values, trades = _read("values.csv"), _read("trades.csv")
    slots = pd.DataFrame(config["slots"])
    st.caption(
        f"Capitale iniziale {euro(config['start_cash'])}, diviso in parti uguali tra {len(slots)} scomparti. "
        "Ogni notte un programma automatico su GitHub legge i prezzi di chiusura del giorno prima, "
        "applica le idee e registra acquisti e vendite con costi reali "
        f"({pct(config['commission'] + config['slippage'], decimals=2)} per operazione). Accanto, lo stesso "
        "capitale in compra e tieni sugli stessi titoli, per un confronto onesto."
    )

    if values.empty or not state_path.exists():
        st.info("Il conto non ha ancora registrato una chiusura: parte con la prima esecuzione notturna "
                "(01:00 UTC, le 3:00 in Italia), sui prezzi di chiusura del giorno prima. Ecco come è configurato:")
        st.table(slots.rename(columns={"ticker": "Titolo", "idea": "Idea"}).set_index("Titolo"))
        return

    start = json.loads(state_path.read_text(encoding="utf-8"))["start"]
    history = portfolio_history(values)
    history.index = pd.to_datetime(history.index)
    now, bench = history["value"].iloc[-1], history["bench_value"].iloc[-1]
    start_cash = config["start_cash"]
    context_line([f"attivo dal <b>{pd.Timestamp(start):%d/%m/%Y}</b>",
                  f"ultimo aggiornamento <b>{history.index[-1]:%d/%m/%Y}</b>",
                  f"{len(history)} chiusure registrate"])
    kpi_strip([
        ("Valore del conto", euro(now), f"compra e tieni {euro(bench)}", 1 if now > bench else -1 if now < bench else 0),
        ("Rendimento dall'inizio", pct(now / start_cash - 1, True), f"compra e tieni {pct(bench / start_cash - 1, True)}", 0),
        ("Differenza", euro(now - bench), "conto − compra e tieni", 0),
        ("Operazioni", str(len(trades)), "acquisti e vendite", 0),
    ])

    _account_chart(history)

    st.markdown("##### Gli scomparti oggi")
    last = values.sort_values("date").groupby(["ticker", "idea"], sort=False).tail(1)
    each = start_cash / len(slots)
    st.table(pd.DataFrame({
        "Titolo": last["ticker"], "Idea": last["idea"],
        "Posizione": last["position"].map({1: "DENTRO", 0: "FUORI"}),
        "Valore": last["value"].map(euro),
        "Rendimento": (last["value"] / each - 1).map(lambda v: pct(v, True)),
        "Compra e tieni": (last["bench_value"] / each - 1).map(lambda v: pct(v, True)),
        "Ultima chiusura": pd.to_datetime(last["date"]).dt.strftime("%d/%m/%Y"),
    }).set_index("Titolo"))

    st.markdown("##### Operazioni")
    if trades.empty:
        st.caption("Ancora nessuna operazione.")
    else:
        shown = trades.sort_values("date", ascending=False)
        st.dataframe(pd.DataFrame({
            "Data": pd.to_datetime(shown["date"]).dt.strftime("%d/%m/%Y").to_numpy(),
            "Operazione": shown["action"].map({"buy": "Acquisto", "sell": "Vendita"}).to_numpy(),
            "Titolo": shown["ticker"].to_numpy(), "Idea": shown["idea"].to_numpy(),
            "Prezzo": shown["price"].map(num).to_numpy(), "Importo": shown["value"].map(euro).to_numpy(),
            "Costi": shown["cost"].map(lambda v: num(v)).to_numpy(),
        }), hide_index=True, width="stretch")

    st.caption(
        "Come leggerlo: poche settimane o pochi mesi dicono pochissimo, perché le operazioni sono poche e il caso "
        "pesa molto; servono anni. Il conto mostra soprattutto come si comportano le regole dal vivo: quando entrano, "
        "quando escono, quanto costano. I valori sono nella valuta di ciascun titolo, senza effetto cambio. "
        "Gli scomparti si cambiano modificando paper/config.json nel repository."
    )



@st.cache_data(show_spinner=False, ttl=6 * 3600)
def _run_replay(config_json: str, start: str, end: str) -> tuple[list, list, list, dict]:
    """Replay the nightly program over [start, end]; also the backtest of each slot as a check."""
    config = json.loads(config_json)
    # dates as "YYYY-MM-DD": the data source does not accept a time of day
    first = (pd.Timestamp(start) - pd.DateOffset(years=4)).date().isoformat()  # history for the indicators
    after_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
    history = {}
    for slot in config["slots"]:
        if slot["ticker"] not in history:
            history[slot["ticker"]] = get_prices(slot["ticker"], first, after_end)
    _, trades, values, problems = replay(config, history, app_factor, start, end)
    # The same ideas in the backtest engine, over the same days: the two must agree.
    each, cost = config["start_cash"] / len(config["slots"]), (config["commission"], config["slippage"])
    checks = {}
    for slot in config["slots"]:
        try:
            prices = history[slot["ticker"]]
            strategy, params = IDEAS[slot["idea"]].build(slot["ticker"], app_factor)
            period = prices[(prices.index >= start) & (prices.index <= end)]
            signals = strategy(prices, **params)[period.index]
            net = run_backtest(period, signals, *cost, config["cash_rate"])["net_return"]
            checks[(slot["ticker"], slot["idea"])] = float(each * (1 + net).prod())
        except Exception:  # the slot failed in the replay too: reported there
            continue
    return trades, values, problems, checks


def _replay(config: dict) -> None:
    st.caption(
        "Lo stesso programma del conto dal vivo, fatto girare notte per notte su un periodo passato. Ogni notte "
        "vede solo i prezzi fino a quella sera, come se il futuro non esistesse ancora: due anni in pochi secondi."
    )
    today = date.today()
    c1, c2, c3 = st.columns([1, 1, 1])
    start = c1.date_input("Dal", value=today - timedelta(days=2 * 365), min_value=date(2005, 1, 1),
                          max_value=today - timedelta(days=60), key="replay_start", format="DD/MM/YYYY")
    end = c2.date_input("Al", value=today - timedelta(days=1), min_value=start + timedelta(days=30),
                        max_value=today - timedelta(days=1), key="replay_end", format="DD/MM/YYYY")
    c3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    if c3.button("Avvia replay", type="primary", key="replay_run", width="stretch"):
        st.session_state.replay_started = True
    if not st.session_state.get("replay_started"):
        return

    with st.spinner("Rivivo il periodo notte per notte..."):
        trades, values, problems, checks = _run_replay(json.dumps(config, sort_keys=True), str(start), str(end))
    for problem in problems:
        st.warning(f"Scomparto saltato: {problem}")
    if not values:
        st.info("Nessuna chiusura nel periodo scelto.")
        return
    values = pd.DataFrame(values)
    history = portfolio_history(values)
    history.index = pd.to_datetime(history.index)
    start_cash = config["start_cash"]
    now, bench = history["value"].iloc[-1], history["bench_value"].iloc[-1]
    context_line([f"replay <b>{history.index[0]:%d/%m/%Y} – {history.index[-1]:%d/%m/%Y}</b>",
                  f"{len(history)} chiusure", f"{len(config['slots'])} scomparti"])
    kpi_strip([
        ("Valore finale", euro(now), f"compra e tieni {euro(bench)}", 1 if now > bench else -1 if now < bench else 0),
        ("Rendimento", pct(now / start_cash - 1, True), f"compra e tieni {pct(bench / start_cash - 1, True)}", 0),
        ("Differenza", euro(now - bench), "conto − compra e tieni", 0),
        ("Operazioni", str(len(trades)), "acquisti e vendite", 0),
    ])
    _account_chart(history)

    st.markdown("##### Scomparti a fine periodo")
    last = values.sort_values("date").groupby(["ticker", "idea"], sort=False).tail(1)
    each = start_cash / len(config["slots"])
    backtest = [checks.get((t, i)) for t, i in zip(last["ticker"], last["idea"])]
    st.table(pd.DataFrame({
        "Titolo": last["ticker"].to_numpy(), "Idea": last["idea"].to_numpy(),
        "Replay": [euro(v) for v in last["value"]],
        "Compra e tieni": [pct(v / each - 1, True) for v in last["bench_value"]],
        "Rendimento replay": [pct(v / each - 1, True) for v in last["value"]],
        "Controllo backtest": [euro(b) if b else "n/d" for b in backtest],
        "Scarto": [pct(v / b - 1, True, 2) if b else "n/d" for v, b in zip(last["value"], backtest)],
    }).set_index("Titolo"))
    st.caption("Controllo backtest: lo stesso scomparto calcolato dal motore di backtest, un codice diverso. "
               "Lo scarto viene solo dal modo di applicare i costi (moltiplicando o sommando) e deve restare "
               "minimo: se fosse grande, uno dei due conti sarebbe sbagliato.")

    if trades:
        with st.expander(f"Tutte le {len(trades)} operazioni"):
            shown = pd.DataFrame(trades).sort_values("date", ascending=False)
            st.dataframe(pd.DataFrame({
                "Data": pd.to_datetime(shown["date"]).dt.strftime("%d/%m/%Y").to_numpy(),
                "Operazione": shown["action"].map({"buy": "Acquisto", "sell": "Vendita"}).to_numpy(),
                "Titolo": shown["ticker"].to_numpy(), "Idea": shown["idea"].to_numpy(),
                "Prezzo": shown["price"].map(num).to_numpy(), "Importo": shown["value"].map(euro).to_numpy(),
            }), hide_index=True, width="stretch")
