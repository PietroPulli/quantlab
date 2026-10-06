"""Simple view: a broker-style instrument page, plus one plain test of an idea on it."""

import html
from datetime import date

import streamlit as st

import paper_view
from common import app_factor, get_ohlcv, quotes, show_news, verdict_tone
from panels import (
    RANGES,
    analyse,
    annual_chart,
    annual_table,
    candle_chart,
    chart,
    context_line,
    equity_chart,
    euro,
    kpi_strip,
    ma_legend,
    num,
    pct,
    quote_header,
    signal_panel,
    verdict_bar,
    watchlist,
)
from quantlab.backtest import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, current_signal
from quantlab.ideas import IDEAS
from quantlab.report import evaluate_strategy

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

SENTENCES = {  # tone -> the verdict in plain words
    "good": "Vantaggio sul compra e tieni che regge al test della fortuna.",
    "bad": "Peggio del compra e tieni, e non per sfortuna.",
    "neutral": "Differenza compatibile con la fortuna.",
}
MARKETS = (("VIX", "^VIX"), ("Tasso USA 10 anni", "^TNX"), ("Dollaro (DXY)", "DX-Y.NYB"),
           ("Petrolio WTI", "CL=F"), ("EUR / USD", "EURUSD=X"))


def _open_demo() -> None:
    st.session_state.view = "Conto demo"


def _open_advanced(ticker: str) -> None:
    """Button callback: show the same asset in the advanced view."""
    st.session_state.view = "Approfondita"
    st.session_state.adv_ticker = ticker
    st.session_state.started = True


def short(name: str) -> str:
    """'S&P 500 (le 500 maggiori aziende USA)' -> 'S&P 500'."""
    return name.split(" (")[0]


def render() -> None:
    side, main = st.columns([1, 3.2], gap="medium")

    # ---- left: instrument picker and watchlist ----
    with side:
        asset = st.selectbox("Titolo", list(ASSETS), key="simple_asset", format_func=short)
        ticker = ASSETS[asset]
        if ticker is None:
            ticker = st.text_input("Ticker di Yahoo Finance", value="MSFT", key="simple_ticker",
                                   help="Lo trovi cercando il nome su finance.yahoo.com, es. ENI.MI o MSFT.")
            ticker = asset = ticker.strip().upper()
        listed = tuple((short(n), t) for n, t in ASSETS.items() if t)
        watchlist(quotes(listed), active=short(asset), title="Watchlist")
        watchlist(quotes(MARKETS), title="Mercati")
        _demo_box()
        terms = NEWS_TERMS.get(ticker)
        show_news(terms[0] if terms else ticker, terms)

    with main:
        try:
            ohlcv = get_ohlcv(ticker, YEARS_OF_HISTORY)
        except Exception as exc:  # unknown ticker, network problems
            st.error(f"Non trovo i prezzi di {ticker}. Controlla il nome su finance.yahoo.com. ({exc})")
            return
        quote_header(short(asset), ticker, ohlcv)
        days = RANGES[st.radio("Periodo", list(RANGES), index=3, key="chart_range", horizontal=True,
                               label_visibility="collapsed")]
        chart_box = st.container()  # drawn after the test below, so it can show the strategy's trades

        # ---- the idea test ----
        st.markdown("##### Metti alla prova un'idea su questo titolo")
        c1, c2, c3 = st.columns([3, 1.3, 1])
        idea = c1.selectbox("Quale idea?", list(IDEAS), key="simple_idea", label_visibility="collapsed")
        amount = c2.number_input("Con quanti euro?", 100, 1_000_000, 1000, 100, key="simple_amount",
                                 label_visibility="collapsed")
        if c3.button("Verifica l'idea", type="primary", key="simple_run", width="stretch"):
            st.session_state.simple_started = True
        st.caption(IDEAS[idea].explanation)
        result = _test_idea(asset, ticker, idea, amount, ohlcv["Close"]) if st.session_state.get("simple_started") \
            else None
        with chart_box:
            chart(candle_chart(ohlcv, days, trades=result["trades"] if result else None))
            ma_legend(with_trades=result is not None)
        if result is not None:
            _show_result(asset, ticker, idea, amount, **result)


def _test_idea(asset: str, ticker: str, idea: str, amount: float, prices) -> dict | None:
    try:
        strategy, params = IDEAS[idea].build(ticker, app_factor)
    except Exception as exc:  # no network, or no earnings for an index/ETF
        st.error(f"Questa idea non si può applicare a {asset}: {exc}")
        return None
    if len(prices) < 504:
        st.error(f"Di {asset} ci sono meno di 2 anni di prezzi: troppo pochi per un giudizio.")
        return None
    with st.spinner("Faccio i conti..."):
        report = evaluate_strategy(prices, strategy, params=params, name=idea, cash_rate=CASH_RATE)
        # Only the test period is judged: the earlier years are used as history, never scored.
        a = analyse(report, prices, strategy, DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, CASH_RATE, "test")
    signal = current_signal(strategy(prices, **params))
    return {"report": report, "a": a, "trades": a["trades"], "signal": signal, "last_price": prices.iloc[-1]}


def _demo_box() -> None:
    """Side panel: the demo account at a glance, with a link to its page."""
    info = paper_view.summary()
    if info is None:
        return
    diff = info["value"] - info["bench"]
    cls = "ql-pos" if diff >= 0 else "ql-neg"
    st.markdown(
        f'<div class="ql-watch ql-mini"><div class="ql-watch-head">Conto demo · dal {info["start"]:%d/%m}</div>'
        f'<div class="ql-watch-row"><div><b>Strategie</b><small>al {info["date"]:%d/%m/%Y}</small></div>'
        f'<div class="ql-num">{euro(info["value"])}</div><div class="ql-num {cls}">{euro(diff)}</div></div>'
        f'<div class="ql-watch-row"><div><b>Compra e tieni</b><small>stessi titoli</small></div>'
        f'<div class="ql-num">{euro(info["bench"])}</div><div></div></div></div>', unsafe_allow_html=True)
    st.button("Apri il conto demo", key="simple_open_demo", on_click=_open_demo, width="stretch")


def _show_result(asset, ticker, idea, amount, report, a, trades, signal, last_price) -> None:
    growth = amount * (1 + a["returns"]).cumprod()
    final, final_bh = growth["strategy"].iloc[-1], growth["buy_and_hold"].iloc[-1]
    m = a["metrics"]
    tone = verdict_tone(report)
    years_text = num(a["years"], 1)

    context_line([f"<b>{html.escape(short(asset))}</b>", f"<b>{html.escape(idea)}</b>",
                  f"periodo di prova {report.split:%d/%m/%Y} – oggi ({years_text} anni)", "costi inclusi"])
    verdict_bar(report, tone, plain=f"{euro(amount)} sarebbero diventati {euro(final)}, "
                                    f"con il compra e tieni {euro(final_bh)}. {SENTENCES[tone]}")
    kpi_strip([
        ("Capitale finale", euro(final), f"B&H {euro(final_bh)}", 1 if final > final_bh else -1),
        ("Rendimento annuo", pct(m.loc["annual_return", "strategy"], True),
         f"B&H {pct(m.loc['annual_return', 'buy_and_hold'], True)}", 0),
        ("Calo massimo", pct(m.loc["max_drawdown", "strategy"]), f"B&H {pct(m.loc['max_drawdown', 'buy_and_hold'])}", 0),
        ("Tempo investito", pct(a["exposure"], decimals=0), "dei giorni di borsa", 0),
        ("Operazioni", str(len(a["trades"])), f"in {years_text} anni", 0),
    ])

    labels = {"strategy": idea, "buy_and_hold": "Compra e tieni"}
    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.markdown(f"##### Valore di {euro(amount)}")
        chart(equity_chart(a["returns"], labels, start_value=amount, y_title="", height=420))
    with right:
        st.markdown("##### Segnale di oggi")
        day, value, since = signal
        signal_panel(short(asset), day, value, since, last_price, tone, who="L'idea")
        stats, last = a["stats"], a["trades"]
        last_trade = (f'{"Acquisto" if last["action"].iloc[-1] == "buy" else "Vendita"} '
                      f'{last.index[-1]:%d/%m/%Y}') if len(last) else "nessuna"
        rows = [("Operazioni chiuse", str(stats["trades"])), ("Vincenti", pct(stats["win_rate"], decimals=0)),
                ("Guadagno medio", pct(stats["avg_win"], True)), ("Perdita media", pct(stats["avg_loss"], True)),
                ("Durata media", f'{num(stats["avg_days"], 0)} giorni'), ("Ultima operazione", last_trade)]
        st.markdown('<div class="ql-watch ql-mini"><div class="ql-watch-head">Operazioni nel periodo di prova</div>'
                    + "".join(f'<div class="ql-watch-row"><div>{k}</div><div></div><div class="ql-num">{v}</div></div>'
                              for k, v in rows) + "</div>", unsafe_allow_html=True)
        st.button("Analisi completa", key="simple_open_advanced", on_click=_open_advanced, args=(ticker,),
                  width="stretch", help="Stesso titolo nella vista Approfondita: rischio, operazioni, robustezza.")
    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.markdown("##### Rendimento per anno")
        chart(annual_chart(a["returns"], labels, height=150))
    with right:
        st.markdown("##### Tabella annuale")
        st.table(annual_table(a["returns"], labels))
    with st.expander("Come è calcolato"):
        st.markdown(
            f"- {YEARS_OF_HISTORY} anni di prezzi giornalieri; il giudizio usa solo l'ultimo 30%, mai usato per "
            "scegliere niente.\n"
            "- Ogni operazione paga 0,10% di commissione e 0,05% di slippage; fuori dal mercato i soldi "
            "rendono il 2% annuo.\n"
            "- «Differenza compatibile con la fortuna»: rimescolando 1.000 volte la storia (bootstrap), "
            "il vantaggio non è stabilmente sopra zero.")
