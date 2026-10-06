"""Scanner view: every idea on several assets, with the multiple-testing correction."""

from datetime import date

import pandas as pd
import streamlit as st

from common import app_factor, get_prices
from panels import context_line, num, pct
from quantlab.ideas import IDEAS
from quantlab.scanner import best_idea, scan
from simple import ASSETS, CASH_RATE, YEARS_OF_HISTORY

VERDICT_TEXT = {"good": "Batte il B&H", "bad": "Peggio del B&H", "neutral": "Nessuna evidenza"}


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def _scan_asset(ticker: str, day: str) -> tuple[pd.DataFrame, list[str]]:
    """Scan one asset (cached for the day). Returns the table and the ideas that could not apply."""
    today = date.fromisoformat(day)
    prices = get_prices(ticker, str(date(today.year - YEARS_OF_HISTORY, 1, 1)), day)
    built, skipped = {}, []
    for name, idea in IDEAS.items():
        try:
            built[name] = idea.build(ticker, app_factor)
        except Exception:  # e.g. earnings asked for an index or a crypto
            skipped.append(name)
    return scan(prices, built, cash_rate=CASH_RATE, n_bootstrap=300), skipped


def render() -> None:
    st.subheader("Scanner: c'è un'idea che batte davvero il compra e tieni?")
    st.caption(
        "Per ogni titolo prova tutte le idee pronte sugli ultimi anni (periodo di prova) e tiene solo quelle che "
        "battono il compra e tieni anche dopo la correzione per i tentativi multipli. Provare molte idee e "
        "tenere la migliore è come comprare molti biglietti della lotteria: per questo il test è più severo."
    )
    names = [n for n, t in ASSETS.items() if t]
    c1, c2 = st.columns([3, 2])
    chosen = c1.multiselect("Titoli", names, default=names[:5], key="scan_assets")
    extra = c2.text_input("Altri ticker Yahoo (separati da virgola)", key="scan_extra", placeholder="MSFT, ENEL.MI")
    tickers = [(n, ASSETS[n]) for n in chosen]
    tickers += [(t.strip().upper(), t.strip().upper()) for t in extra.split(",") if t.strip()]

    if st.button("Analizza", type="primary", key="scan_run"):
        st.session_state.scan_started = True
    if not st.session_state.get("scan_started") or not tickers:
        st.caption(f"{len(IDEAS)} idee per titolo: ci vuole circa mezzo minuto per titolo la prima volta.")
        return

    rows, details = [], {}
    progress = st.progress(0.0, text="Analizzo...")
    for i, (label, ticker) in enumerate(tickers):
        progress.progress(i / len(tickers), text=f"Analizzo {label}...")
        try:
            table, skipped = _scan_asset(ticker, str(date.today()))
        except Exception as exc:  # unknown ticker, too little history, network
            rows.append({"Titolo": label, "Idea con vantaggio dimostrato": f"errore: {exc}"[:80],
                         "Cosa dice oggi": "", "Idee provate": 0})
            continue
        details[label] = (table, skipped)
        best = best_idea(table)
        rows.append({
            "Titolo": label,
            "Idea con vantaggio dimostrato": best["idea"] if best is not None else "nessuna",
            "Cosa dice oggi": ("DENTRO" if best["signal_today"] > 0 else "FUORI") if best is not None
            else "nessuna indicazione",
            "Idee provate": len(table),
        })
    progress.empty()

    level = next(iter(details.values()))[0].attrs["level"] if details else 0.95
    context_line([f"<b>{len(tickers)}</b> titoli", f"<b>{len(IDEAS)}</b> idee per titolo",
                  f"intervallo richiesto <b>{pct(level)}</b> invece di 95%", "periodo di prova: ultimi anni",
                  "costi reali inclusi"])
    summary = pd.DataFrame(rows).set_index("Titolo")
    st.table(summary)
    found = (summary["Idea con vantaggio dimostrato"] != "nessuna").sum()
    st.caption(
        f"{found} titoli su {len(summary)} hanno un'idea con un vantaggio che regge alla correzione. "
        "Dove la risposta è «nessuna», sul passato recente nessuna regola ha fatto meglio del semplice comprare "
        "e tenere in modo distinguibile dalla fortuna. Non è un consiglio di investimento: è il risultato di regole "
        "fisse applicate ai prezzi di oggi."
    )

    st.markdown("##### Dettaglio per titolo")
    for label, (table, skipped) in details.items():
        with st.expander(label):
            shown = pd.DataFrame({
                "Idea": table["idea"],
                "Rendimento": table["total_return"].map(lambda v: pct(v, signed=True)),
                "Compra e tieni": table["bh_total_return"].map(lambda v: pct(v, signed=True)),
                "Diff. Sharpe": table["sharpe_diff"].map(lambda v: num(v, signed=True)),
                "Intervallo corretto": [f"{num(lo, signed=True)} … {num(hi, signed=True)}"
                                        for lo, hi in zip(table["ci_low"], table["ci_high"])],
                "Esito": table["verdict"].map(VERDICT_TEXT),
                "Oggi": table["signal_today"].map(lambda v: "DENTRO" if v > 0 else "FUORI"),
            }).set_index("Idea")
            st.table(shown)
            if skipped:
                st.caption("Non applicabili a questo titolo: " + ", ".join(skipped) + ".")
