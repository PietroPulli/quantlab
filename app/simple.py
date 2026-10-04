"""Simple view: three plain choices, one plain answer. Same truth machine underneath."""

from datetime import date

import pandas as pd
import streamlit as st

from common import comparison_chart, get_prices, verdict_tone
from quantlab.backtest import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, current_signal, run_backtest, trade_log
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

IDEAS = {  # name -> (strategy, parameters, explanation in plain words)
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
}

VERDICTS = {  # tone -> (tag, sentence)
    "good": ("Ha funzionato", "Questa idea ha battuto il semplice comprare e tenere, "
                              "e il vantaggio non sembra dovuto alla fortuna."),
    "bad": ("Ha fatto peggio", "Questa idea ha fatto chiaramente peggio del semplice comprare e tenere."),
    "neutral": ("Nessun vantaggio dimostrato", "Non c'è prova che questa idea sia meglio del semplice "
                                               "comprare e tenere: la differenza potrebbe essere solo fortuna."),
}


def euro(x: float) -> str:
    """1234.5 -> '1.235 €' (Italian thousands separator)."""
    return f"{x:,.0f} €".replace(",", ".")


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
        positions = run_backtest(prices, strategy(prices, **params), DEFAULT_COMMISSION, DEFAULT_SLIPPAGE,
                                 CASH_RATE)["position"]

    # Only the test period counts: the earlier years are used as history, never judged.
    in_test = report.returns.index >= report.split
    growth = amount * (1 + report.returns[in_test]).cumprod()
    years = in_test.sum() / 252
    trades = trade_log(prices, positions)
    n_trades = int((trades.index >= report.split).sum())
    worst = report.out_of_sample.loc["max_drawdown"]
    tone = verdict_tone(report)
    tag, sentence = VERDICTS[tone]
    years_text = f"{years:.1f}".replace(".", ",")

    st.markdown(
        f'<div class="ql-verdict ql-{tone}">'
        f'<div class="ql-tag">{tag}</div>'
        f'<div class="ql-big">{euro(amount)} sarebbero diventati {euro(growth["strategy"].iloc[-1])}</div>'
        f'<div class="ql-text">Con l\'idea «{idea}» su {asset}, negli ultimi {years_text} anni. '
        f'Comprando e tenendo sarebbero diventati <b>{euro(growth["buy_and_hold"].iloc[-1])}</b>.</div>'
        f'<div class="ql-text"><b>{sentence}</b></div>'
        f'<dl class="ql-facts">'
        f'<div><dt>Momento peggiore</dt><dd>{worst["strategy"]:.0%} (comprando e tenendo {worst["buy_and_hold"]:.0%})</dd></div>'
        f'<div><dt>Operazioni</dt><dd>{n_trades} in {years_text} anni</dd></div>'
        f'<div><dt>Tempo investito</dt><dd>{positions[in_test].mean():.0%} dei giorni</dd></div>'
        f'</dl></div>',
        unsafe_allow_html=True,
    )

    labels = {"strategy": idea, "buy_and_hold": "Comprare e tenere"}
    st.altair_chart(comparison_chart(growth, labels, "Valore (€)", ",.0f"), width="stretch")

    day, value, since = current_signal(strategy(prices, **params))
    with st.container(border=True):
        state = "DENTRO" if value > 0 else "FUORI"
        st.markdown(f'<div class="ql-tag">Oggi, con i prezzi del {day:%d/%m/%Y}</div>'
                    f'<div class="ql-title">L\'idea dice: {state}</div>', unsafe_allow_html=True)
        st.caption(
            ("Cioè: essere investiti" if value > 0 else "Cioè: stare in contanti")
            + f" su {asset}, così dal {since:%d/%m/%Y}. È il risultato dell'idea applicata ai prezzi di oggi, "
            "non un consiglio di investimento."
        )

    st.caption(
        f"Come è fatto il conto: {YEARS_OF_HISTORY} anni di prezzi giornalieri; il giudizio riguarda solo "
        "l'ultimo 30%, mai usato per scegliere niente. Costi reali inclusi: 0,10% di commissione e 0,05% "
        "di slippage a ogni operazione; quando l'idea è fuori dal mercato i soldi rendono il 2% l'anno. "
        "Per cambiare i parametri, scrivere regole tue e vedere tutte le statistiche, passa alla vista Approfondita."
    )
