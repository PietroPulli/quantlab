"""Scanner view: every idea on several assets, with the multiple-testing correction."""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from common import app_factor, get_prices
from panels import context_line, num, pct
from quantlab.ideas import IDEAS
from quantlab.scanner import best_idea, scan
from simple import ASSETS, CASH_RATE, YEARS_OF_HISTORY

MIN_HISTORY_DAYS = 365  # calendar days of history needed before the test (200-day average, 12-month momentum)
VERDICT_TEXT = {"good": "Batte il B&H", "bad": "Peggio del B&H", "neutral": "Nessuna evidenza"}


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def _scan_asset(ticker: str, start: str, end: str, test_share: float) -> tuple[pd.DataFrame, list[str]]:
    """Scan one asset on [start, end], judging the last `test_share` of the days (cached).

    Returns the table and the ideas that could not apply to this asset.
    """
    prices = get_prices(ticker, start, end)
    if len(prices) < 504:
        raise ValueError("meno di 2 anni di prezzi nel periodo scelto")
    built, skipped = {}, []
    for name, idea in IDEAS.items():
        try:
            built[name] = idea.build(ticker, app_factor)
        except Exception:  # e.g. earnings asked for an index or a crypto
            skipped.append(name)
    table = scan(prices, built, cash_rate=CASH_RATE, n_bootstrap=300, in_sample_fraction=1 - test_share)
    return table, skipped


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

    # ---- period: data range and how much of it is judged ----
    today = date.today()
    p1, p2, p3 = st.columns([1, 1, 2])
    start = p1.date_input("Dati dal", value=date(today.year - YEARS_OF_HISTORY, 1, 1), min_value=date(1995, 1, 1),
                          max_value=today - timedelta(days=3 * 365), key="scan_start", format="DD/MM/YYYY")
    end = p2.date_input("al", value=today, min_value=start + timedelta(days=3 * 365), max_value=today,
                        key="scan_end", format="DD/MM/YYYY")
    share = p3.slider("Periodo di prova (giudicato): ultimo", 20, 90, 30, 5, format="%d%%", key="scan_share",
                      help="I giorni prima servono solo come storia per gli indicatori (media a 200 giorni, "
                           "momentum a 12 mesi) e non entrano nel giudizio: ne serve almeno 1 anno. "
                           "Più lungo il periodo di prova, più affidabile il verdetto.") / 100
    test_from = start + (end - start) * (1 - share)  # calendar approximation; exact dates come with the results
    st.caption(f"Giudizio sul periodo **{test_from:%d/%m/%Y} – {end:%d/%m/%Y}** "
               f"(circa {num((end - test_from).days / 365.25, 1)} anni); "
               f"storia per gli indicatori dal {start:%d/%m/%Y}.")
    if (test_from - start).days < MIN_HISTORY_DAYS:
        st.warning("Prima del periodo di prova resta meno di 1 anno di storia: le idee che guardano 200 giorni o "
                   "12 mesi indietro non avrebbero i dati per partire. Anticipa la data di inizio o abbassa la "
                   "percentuale.")
        return

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
            table, skipped = _scan_asset(ticker, str(start), str(end), share)
        except Exception as exc:  # unknown ticker, too little history, network
            rows.append({"Titolo": label, "Periodo di prova": "", "Idea con vantaggio dimostrato": f"errore: {exc}"[:80],
                         "Cosa dice oggi": "", "Idee provate": 0})
            continue
        details[label] = (table, skipped)
        best = best_idea(table)
        test_start, end = table.attrs["test_start"], table.attrs["history"][1]
        rows.append({
            "Titolo": label,
            "Periodo di prova": f"{test_start:%d/%m/%Y} – {end:%d/%m/%Y}",
            "Idea con vantaggio dimostrato": best["idea"] if best is not None else "nessuna",
            "Cosa dice oggi": ("DENTRO" if best["signal_today"] > 0 else "FUORI") if best is not None
            else "nessuna indicazione",
            "Idee provate": len(table),
        })
    progress.empty()

    first = next(iter(details.values()))[0].attrs if details else {"level": 0.95}
    period = []
    if "history" in first:
        start, end = first["history"]
        period = [f"storia {start:%d/%m/%Y} – {end:%d/%m/%Y}",
                  f"giudizio sul periodo di prova dal <b>{first['test_start']:%d/%m/%Y}</b> (ultimo 30%)"]
    context_line([f"<b>{len(tickers)}</b> titoli", f"<b>{len(IDEAS)}</b> idee per titolo", *period,
                  f"intervallo richiesto <b>{pct(first['level'])}</b> invece di 95%", "costi inclusi"])
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
        test_start, end = table.attrs["test_start"], table.attrs["history"][1]
        years = (end - test_start).days / 365.25
        yearly = lambda v: pct((1 + v) ** (1 / years) - 1, signed=True)  # noqa: E731  total -> per year
        with st.expander(f"{label} · periodo di prova {test_start:%d/%m/%Y} – {end:%d/%m/%Y} ({num(years, 1)} anni)"):
            shown = pd.DataFrame({
                "Idea": table["idea"],
                f"Rendimento dal {test_start:%d/%m/%Y}": table["total_return"].map(lambda v: pct(v, signed=True)),
                "Annuo": table["total_return"].map(yearly),
                "Compra e tieni (stesso periodo)": table["bh_total_return"].map(lambda v: pct(v, signed=True)),
                "B&H annuo": table["bh_total_return"].map(yearly),
                "Diff. Sharpe": table["sharpe_diff"].map(lambda v: num(v, signed=True)),
                "Intervallo corretto": [f"{num(lo, signed=True)} … {num(hi, signed=True)}"
                                        for lo, hi in zip(table["ci_low"], table["ci_high"])],
                "Esito": table["verdict"].map(VERDICT_TEXT),
                "Oggi": table["signal_today"].map(lambda v: "DENTRO" if v > 0 else "FUORI"),
            }).set_index("Idea")
            st.table(shown)
            if skipped:
                st.caption("Non applicabili a questo titolo: " + ", ".join(skipped) + ".")
