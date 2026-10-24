"""Options view: price a European option with Black-Scholes, check it by Monte Carlo, read the greeks."""

import altair as alt
import pandas as pd
import streamlit as st

from common import COLORS, get_ohlcv
from panels import chart, kpi_strip, num, pct
from quantlab import derivatives as d

UNDERLYINGS = {"S&P 500 (SPY)": "SPY", "Nasdaq 100 (QQQ)": "QQQ", "Apple": "AAPL", "Tesla": "TSLA",
               "Ferrari": "RACE.MI", "Oro": "GC=F", "Bitcoin": "BTC-USD"}
RATE = 0.04  # default risk-free rate, roughly the 3-month T-bill: change it in the box


def _market_inputs(ticker: str) -> tuple[float, float]:
    """Last close and one-year historical volatility of the underlying, as starting values."""
    close = get_ohlcv(ticker, years=2)["Close"]
    return float(close.iloc[-1]), d.historical_volatility(close)


def _value_curve(strike: float, years: float, rate: float, vol: float, kind: str, spot: float) -> alt.Chart:
    """Option value today and at expiry across underlying prices from -40% to +40%."""
    spots = [spot * (0.6 + 0.8 * i / 160) for i in range(161)]
    rows = [{"Prezzo del sottostante": s, "Valore": d.black_scholes(s, strike, years, rate, vol, kind), "Curva": "Oggi"}
            for s in spots]
    rows += [{"Prezzo del sottostante": s, "Valore": d.intrinsic_value(s, strike, kind), "Curva": "Alla scadenza"}
             for s in spots]
    return (alt.Chart(pd.DataFrame(rows)).mark_line()
            .encode(x=alt.X("Prezzo del sottostante:Q", scale=alt.Scale(zero=False)), y="Valore:Q",
                    color=alt.Color("Curva:N", legend=alt.Legend(orient="top"),
                                    scale=alt.Scale(domain=["Oggi", "Alla scadenza"],
                                                    range=[COLORS["strategy"], COLORS["buy_and_hold"]])))
            .properties(height=320))


def render() -> None:
    st.markdown("##### Opzioni · prezzo teorico con Black-Scholes")
    st.caption("Un'opzione call dà il diritto di comprare il sottostante al prezzo di esercizio (strike) "
               "entro la scadenza; una put dà il diritto di venderlo. Qui calcoli quanto vale in teoria: "
               "è uno strumento per studiare, non una quotazione e non un consiglio.")

    c1, c2, c3 = st.columns(3)
    name = c1.selectbox("Sottostante", list(UNDERLYINGS), key="opt_underlying")
    kind = c2.radio("Tipo", ["call", "put"], key="opt_kind", horizontal=True)
    try:
        spot, hist_vol = _market_inputs(UNDERLYINGS[name])
    except Exception as exc:  # no network, too little history
        st.error(f"Non trovo i prezzi di {name}: {exc}")
        return
    c3.metric("Ultimo prezzo", num(spot), help="Chiusura più recente da Yahoo Finance.")

    c1, c2, c3, c4 = st.columns(4)
    strike = c1.number_input("Strike", min_value=0.01, value=float(round(spot)), key="opt_strike")
    days = c2.number_input("Giorni alla scadenza", min_value=1, max_value=3650, value=90, key="opt_days")
    vol = c3.number_input("Volatilità annua %", min_value=1.0, max_value=300.0, value=round(hist_vol * 100, 1),
                          key="opt_vol", help="Di partenza: la volatilità storica dell'ultimo anno.") / 100
    rate = c4.number_input("Tasso privo di rischio %", min_value=-2.0, max_value=20.0, value=RATE * 100,
                           key="opt_rate") / 100
    years = days / 365

    price = d.black_scholes(spot, strike, years, rate, vol, kind)
    mc, error = d.monte_carlo_price(spot, strike, years, rate, vol, kind, paths=200_000, seed=0)
    g = d.greeks(spot, strike, years, rate, vol, kind)
    kpi_strip([
        ("Prezzo Black-Scholes", num(price), f"{pct(price / spot)} del sottostante", 0),
        ("Monte Carlo (200.000 scenari)", num(mc), f"± {num(2 * error)} (95%)", 0),
        ("Valore intrinseco", num(d.intrinsic_value(spot, strike, kind)), "se esercitata oggi", 0),
        ("Valore temporale", num(price - d.intrinsic_value(spot, strike, kind)), "il prezzo dell'incertezza", 0),
    ])

    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.markdown("##### Valore al variare del prezzo")
        chart(_value_curve(strike, years, rate, vol, kind, spot))
    with right:
        st.markdown("##### Greche")
        st.table(pd.DataFrame({
            "Valore": [num(g["delta"], 3), num(g["gamma"], 4), num(g["vega"], 3), num(g["theta"], 3), num(g["rho"], 3)],
            "Cosa dice": ["variazione del prezzo se il sottostante sale di 1",
                          "variazione del delta se il sottostante sale di 1",
                          "variazione se la volatilità sale di 1 punto",
                          "quanto vale in meno domani, a parità di tutto",
                          "variazione se il tasso sale di 1 punto"],
        }, index=["Delta", "Gamma", "Vega", "Theta", "Rho"]))
    with st.expander("Come è calcolato e cosa non dice"):
        st.markdown(
            "- **Black-Scholes** suppone che il prezzo si muova a caso con volatilità costante e senza salti, "
            "e che l'opzione si eserciti solo alla scadenza (europea).\n"
            "- **Monte Carlo** simula 200.000 prezzi possibili alla scadenza con le stesse ipotesi e fa la media dei "
            "guadagni: se le due cifre coincidono (entro il margine), i conti sono giusti.\n"
            "- La **volatilità storica** misura il passato; il mercato prezza le opzioni con la volatilità *attesa* "
            "(implicita), di solito diversa. Crolli e salti improvvisi rendono le opzioni lontane dallo strike "
            "più care di quanto dica il modello.")
