"""Demo account view: how the ideas are doing going forward, with virtual money."""

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from common import COLORS, TIME_AXIS
from panels import chart, context_line, euro, kpi_strip, num, pct
from quantlab.paper import portfolio_history

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
    st.subheader("Conto demo: le idee alla prova dal vivo, con soldi finti")
    config = json.loads((PAPER_DIR / "config.json").read_text(encoding="utf-8"))
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

    long = history.rename(columns={"value": "Conto demo", "bench_value": "Compra e tieni"})
    long = long.rename_axis("date").reset_index().melt("date", var_name="series", value_name="value")
    chart(alt.Chart(long, height=300).mark_line(strokeWidth=1.6, point=len(history) < 30).encode(
        x=TIME_AXIS,
        y=alt.Y("value:Q", title="Valore (€)", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["Conto demo", "Compra e tieni"], range=list(COLORS.values()))),
        tooltip=[alt.Tooltip("date:T", title="Data"), alt.Tooltip("series:N", title="Serie"),
                 alt.Tooltip("value:Q", title="Valore", format=",.2f")]))

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
