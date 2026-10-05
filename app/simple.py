"""Simple view: three plain choices, one plain answer. Same truth machine underneath."""

from datetime import date

import html

import streamlit as st

from common import FACTORS, get_prices, load_factor, show_news, verdict_tone
from panels import (
    analyse,
    annual_chart,
    chart,
    context_line,
    equity_chart,
    euro,
    kpi_strip,
    num,
    pct,
    signal_panel,
    verdict_bar,
)
from quantlab.backtest import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, current_signal
from quantlab.report import evaluate_strategy
from quantlab.rules import Condition, Indicator, rule_strategy
from quantlab.strategies import breakout, mean_reversion, momentum

CASH_RATE = 0.02  # same default as the advanced view
YEARS_OF_HISTORY = 10

ASSETS = {  # what people recognise -> Yahoo ticker
    "S&P 500 (le 500 maggiori aziende USA)": "SPY",
    "Nasdaq 100 (tecnologia USA)": "QQQ",
    "Borsa italiana (FTSE MIB)": "FTSEMIB.MI",
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Ferrari": "RACE.MI",
    "ENI": "ENI.MI",
    "Intesa Sanpaolo": "ISP.MI",
    "Bitcoin": "BTC-USD",
    "Oro": "GC=F",
    "Altro titolo...": None,
}

def _on_factor(factor: str, op: str, level: float, ticker: str = "") -> Condition:
    """Condition on the value of a factor, e.g. VIX < 25 or yield curve > 0."""
    series = load_factor(factor, ticker)
    return Condition(Indicator("price", source=series, label=FACTORS[factor][2]), op, level)


VIX = "VIX (paura del mercato)"

# name -> (strategy, parameters or a function of the ticker that builds them, explanation).
# Factor ideas build their parameters only when chosen, so their data is downloaded only then.
# Words a headline must contain to be about the asset (Yahoo headlines are in English).
NEWS_TERMS = {
    "SPY": ["S&P 500", "S&P", "Wall Street"],
    "QQQ": ["Nasdaq"],
    "FTSEMIB.MI": ["Italy", "Italian", "FTSE MIB", "Milan"],
    "AAPL": ["Apple"],
    "TSLA": ["Tesla"],
    "RACE.MI": ["Ferrari"],
    "ENI.MI": ["Eni"],
    "ISP.MI": ["Intesa"],
    "BTC-USD": ["Bitcoin"],
    "GC=F": ["gold"],
}

IDEAS = {
    "Segui la tendenza": (
        rule_strategy,
        {"entry": Condition(Indicator("price"), ">", Indicator("sma", 200)), "exit": None},
        "Resto investito quando il prezzo è sopra la sua media degli ultimi 200 giorni (circa 10 mesi). "
        "Quando scende sotto, tengo i soldi da parte.",
    ),
    "Compra quando sfonda verso l'alto": (
        breakout,
        {"window": 50},
        "Compro quando il prezzo tocca il massimo degli ultimi 50 giorni, vendo quando tocca il minimo.",
    ),
    "Compra dopo un forte calo": (
        mean_reversion,
        {},
        "Compro quando il prezzo è sceso molto sotto la sua media dell'ultimo mese, vendo quando ci torna.",
    ),
    "Compra ciò che è salito nell'ultimo anno": (
        momentum,
        {},
        "Resto investito se nell'ultimo anno il prezzo è salito (senza contare l'ultimo mese), altrimenti no.",
    ),
    "Esci quando il mercato ha paura": (
        rule_strategy,
        lambda ticker: {"entry": _on_factor(VIX, "<", 25.0), "exit": None},
        "Resto investito finché il VIX, l'indice della paura del mercato americano, è sotto 25. "
        "Quando sale sopra, cioè quando c'è agitazione, tengo i soldi da parte.",
    ),
    "Compra quando il mercato ha paura": (
        rule_strategy,
        lambda ticker: {"entry": _on_factor(VIX, ">", 30.0), "exit": _on_factor(VIX, "<", 20.0)},
        "Compro quando il VIX supera 30 (panico: i prezzi sono spesso scesi molto) e rivendo quando "
        "torna sotto 20, cioè quando la calma è tornata.",
    ),
    "Esci quando la curva dei tassi si inverte": (
        rule_strategy,
        lambda ticker: {"entry": _on_factor("Curva dei tassi 10 anni - 2 anni (%)", ">", 0.0), "exit": None},
        "Quando i tassi a 2 anni superano quelli a 10 anni (curva «invertita»), storicamente è spesso "
        "arrivata una recessione. Resto investito solo quando la curva è normale.",
    ),
    "Compra dopo trimestrali sopra le attese": (
        rule_strategy,
        lambda ticker: {"entry": _on_factor("Sorpresa dell'ultima trimestrale (%)", ">", 0.0, ticker),
                        "exit": None},
        "Resto investito se l'ultima trimestrale dell'azienda ha battuto le attese degli analisti, "
        "fuori se le ha mancate. Solo per singole aziende (Apple, Ferrari, ENI...), non per indici o crypto.",
    ),
}

SENTENCES = {  # tone -> the verdict in plain words
    "good": "Questa idea ha battuto il semplice comprare e tenere, e il vantaggio non sembra dovuto alla fortuna.",
    "bad": "Questa idea ha fatto chiaramente peggio del semplice comprare e tenere.",
    "neutral": "Non c'è prova che questa idea sia meglio del semplice comprare e tenere: "
               "la differenza potrebbe essere solo fortuna.",
}


def _open_advanced(ticker: str) -> None:
    """Button callback: show the same asset in the advanced view."""
    st.session_state.view = "Approfondita"
    st.session_state.adv_ticker = ticker
    st.session_state.started = True


def render() -> None:
    st.subheader("Un'idea di investimento avrebbe funzionato?")
    c1, c2, c3 = st.columns([3, 3, 2])
    asset = c1.selectbox("Su cosa?", list(ASSETS), key="simple_asset")
    ticker = ASSETS[asset]
    if ticker is None:
        ticker = c1.text_input("Ticker di Yahoo Finance", value="MSFT", key="simple_ticker",
                               help="Lo trovi cercando il nome su finance.yahoo.com, es. ENI.MI o MSFT.")
        ticker, asset = ticker.strip().upper(), ticker.strip().upper()
    idea = c2.selectbox("Quale idea?", list(IDEAS), key="simple_idea")
    strategy, params, explanation = IDEAS[idea]
    c2.caption(explanation)
    amount = c3.number_input("Con quanti euro?", 100, 1_000_000, 1000, 100, key="simple_amount")

    if st.button("Verifica l'idea", type="primary", key="simple_run"):
        st.session_state.simple_started = True
    if not st.session_state.get("simple_started"):
        st.caption("Scegli e premi Verifica. La risposta arriva in pochi secondi.")
        return

    try:
        params = params(ticker) if callable(params) else params
    except Exception as exc:  # no network, or no earnings for an index/ETF
        st.error(f"Questa idea non si può applicare a {asset}: {exc}")
        return

    today = date.today()
    try:
        prices = get_prices(ticker, str(date(today.year - YEARS_OF_HISTORY, 1, 1)), str(today))
    except Exception as exc:  # unknown ticker, network problems
        st.error(f"Non trovo i prezzi di {ticker}. Controlla il nome su finance.yahoo.com. ({exc})")
        return
    if len(prices) < 504:
        st.error(f"Di {asset} ci sono meno di 2 anni di prezzi: troppo pochi per un giudizio.")
        return

    with st.spinner("Faccio i conti..."):
        report = evaluate_strategy(prices, strategy, params=params, name=idea, cash_rate=CASH_RATE)
        # Only the test period is judged: the earlier years are used as history, never scored.
        a = analyse(report, prices, strategy, DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, CASH_RATE, "test")

    growth = amount * (1 + a["returns"]).cumprod()
    final, final_bh = growth["strategy"].iloc[-1], growth["buy_and_hold"].iloc[-1]
    m = a["metrics"]
    tone = verdict_tone(report)
    years_text = num(a["years"], 1)

    context_line([f"<b>{html.escape(asset)}</b>", f"<b>{html.escape(idea)}</b>",
                  f"ultimi {years_text} anni (dal {report.split:%d/%m/%Y})", "costi reali inclusi"])
    verdict_bar(report, tone, plain=f"{euro(amount)} sarebbero diventati {euro(final)}; comprando e tenendo "
                                    f"{euro(final_bh)}. {SENTENCES[tone]}")
    kpi_strip([
        ("Capitale finale", euro(final), f"comprare e tenere {euro(final_bh)}", 1 if final > final_bh else -1),
        ("Rendimento annuo", pct(m.loc["annual_return", "strategy"], True),
         f"comprare e tenere {pct(m.loc['annual_return', 'buy_and_hold'], True)}", 0),
        ("Calo massimo", pct(m.loc["max_drawdown", "strategy"]),
         f"comprare e tenere {pct(m.loc['max_drawdown', 'buy_and_hold'])}", 0),
        ("Tempo investito", pct(a["exposure"], decimals=0), "dei giorni di borsa", 0),
        ("Operazioni", str(len(a["trades"])), f"in {years_text} anni", 0),
    ])

    labels = {"strategy": idea, "buy_and_hold": "Comprare e tenere"}
    st.markdown(f"##### Come sarebbero cambiati {euro(amount)}")
    chart(equity_chart(a["returns"], labels, start_value=amount, y_title="Valore (€)"))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Anno per anno")
        chart(annual_chart(a["returns"], labels))
    with right:
        st.markdown("##### Oggi")
        day, value, since = current_signal(strategy(prices, **params))
        signal_panel(asset, day, value, since, prices.iloc[-1], tone, who="L'idea")
        st.button("Apri l'analisi completa", key="simple_open_advanced", on_click=_open_advanced,
                  args=(ticker,), help="Stesso titolo nella vista Approfondita: rischio, operazioni, robustezza.")

    st.caption(
        f"Come è fatto il conto: {YEARS_OF_HISTORY} anni di prezzi giornalieri; il giudizio riguarda solo "
        "l'ultimo 30%, mai usato per scegliere niente. Costi reali inclusi: 0,10% di commissione e 0,05% "
        "di slippage a ogni operazione; quando l'idea è fuori dal mercato i soldi rendono il 2% l'anno."
    )

    terms = NEWS_TERMS.get(ticker)
    show_news(terms[0] if terms else ticker, terms)
