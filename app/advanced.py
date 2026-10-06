"""Advanced view: every parameter, custom rules and the full analysis, organised in tabs."""

import html
from datetime import date

import streamlit as st

from common import (
    FACTORS,
    INDICATOR_LABELS,
    OPERATOR_LABELS,
    STRATEGY_UI,
    describe,
    factor_note,
    get_ohlcv,
    get_prices,
    load_factor,
    show_news,
    verdict_tone,
)
from panels import (
    RANGES,
    analyse,
    annual_chart,
    annual_table,
    bootstrap_chart,
    candle_chart,
    chart,
    context_line,
    drawdown_chart,
    drawdown_table,
    equity_chart,
    kpi_strip,
    ma_legend,
    metrics_table,
    monthly_heatmap,
    num,
    pct,
    quote_header,
    rolling_chart,
    signal_panel,
    standard_kpis,
    trade_kpis,
    trades_chart,
    trips_table,
    verdict_bar,
)
from quantlab import data
from quantlab.backtest import current_signal
from quantlab.metrics import drawdown_periods
from quantlab.report import evaluate_strategy, format_report
from quantlab.rules import Condition, Indicator, rule_strategy


ASSET = "il titolo scelto"


def indicator_input(label: str, key: str, default: Indicator, ticker: str) -> Indicator:
    """Widgets (on what, which indicator, how many days) -> an Indicator."""
    on = st.selectbox(f"{label}: calcolato su", [ASSET, *FACTORS], key=f"{key}-source",
                      help="Il titolo stesso, oppure un fattore che muove il mercato.")
    source, short = None, ""
    if on != ASSET:
        st.caption(factor_note(on))
        try:
            source, short = load_factor(on, ticker), FACTORS[on][2]
        except Exception as exc:  # e.g. earnings asked for an ETF, or no network
            st.error(f"Non riesco a usare «{on}» per {ticker}: {exc}")
            st.stop()
    names = list(INDICATOR_LABELS)
    name = st.selectbox("Indicatore", names, index=names.index(default.name), key=f"{key}-name",
                        format_func=lambda n: "Valore" if (n == "price" and source is not None) else INDICATOR_LABELS[n])
    if name == "price":
        return Indicator("price", source=source, label=short)
    start_window = default.window if default.window >= 2 else 20  # "price" has no window
    window = st.number_input("N giorni", 2, 504, start_window, key=f"{key}-window")
    return Indicator(name, int(window), source=source, label=short)


def condition_input(key: str, left: Indicator, op: str, right: Indicator | float, ticker: str) -> Condition:
    """Widgets for one condition: <indicator> <operator> <indicator or number>.

    `left`, `op` and `right` are only the starting values shown to the user.
    """
    right_is_number = not isinstance(right, Indicator)
    left_ind = indicator_input("Se", f"{key}-left", left, ticker)
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
        right_value = indicator_input("Confronta con", f"{key}-right", default, ticker)
    return Condition(left_ind, op, right_value)


def render() -> None:
    # ---- Inputs (sidebar) ----
    with st.sidebar:
        st.header("1. Cosa provare")
        ticker = st.text_input("Titolo (ticker Yahoo)", value="SPY", key="adv_ticker").strip().upper()
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
            entry = condition_input("entry", Indicator("price"), ">", Indicator("sma", 100), ticker)
            if st.toggle("Aggiungi una condizione (E)", key="entry_and",
                         help="Compra solo nei giorni in cui sono vere entrambe le condizioni."):
                st.markdown("**...e anche quando...**")
                entry = [entry, condition_input("entry2", Indicator("zscore", 10), "<", -1.0, ticker)]
            st.caption(f"Entrata: {describe(entry)}")
            exit_rule = None
            if st.toggle("Aggiungi una regola di uscita", key="has_exit",
                         help="Senza uscita sei investito solo nei giorni in cui la regola di entrata è vera. "
                         "Con un'uscita, una volta entrato resti dentro finché non scatta l'uscita."):
                st.markdown("**Vendi quando...**")
                exit_rule = condition_input("exit", Indicator("price"), "<", Indicator("sma", 50), ticker)
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

    # ---- Header: context, verdict, key numbers ----
    tone = verdict_tone(report)
    if strategy_func is rule_strategy:
        used = describe(entry) + (f"; uscita: {describe(exit_rule)}" if exit_rule else "")
    else:
        # e.g. {"window": 50} -> "Finestra N (giorni) 50", using the same labels as the sliders
        used = ", ".join(f"{ui['params'][k][0]} {v}" for k, v in report.params.items()) or "nessun parametro"
    first, last = prices.index[0], prices.index[-1]
    try:
        ohlcv = get_ohlcv(ticker)
        quote_header(ticker, ticker, ohlcv)
    except Exception:  # the header is a nice-to-have: the analysis works without it
        ohlcv = None
    context_line([
        f"<b>{html.escape(ticker)}</b>", f"<b>{html.escape(name)}</b> ({html.escape(used)})",
        f"dati {first:%d/%m/%Y} – {last:%d/%m/%Y}", f"prova dal <b>{report.split:%d/%m/%Y}</b>",
        f"costi {pct(commission + slippage, decimals=2)} per operazione", f"contanti {pct(cash_rate)} annuo",
    ])
    verdict_bar(report, tone)

    period = st.radio("Periodo analizzato", ["Periodo di prova", "Storia completa"], key="adv_period",
                      horizontal=True, label_visibility="collapsed",
                      help="Il verdetto si basa sempre sul periodo di prova (dati mai usati per scegliere). "
                           "La storia completa include anche gli anni usati per la scelta.")
    a = analyse(report, prices, strategy_func, commission, slippage, cash_rate,
                "test" if period == "Periodo di prova" else "full")
    kpi_strip(standard_kpis(a))

    # Data snooping: every new idea tested on the same data is another lottery ticket.
    tried = st.session_state.setdefault("tried", {}).setdefault(ticker, {})
    tried[f"{name}: {grid if grid else used}"] = len(grid) if grid else 1
    if len(tried) > 1:
        st.warning(
            f"Su {ticker} in questa sessione hai provato {len(tried)} strategie diverse "
            f"({sum(tried.values())} combinazioni di parametri in tutto). Più idee provi sugli stessi dati, "
            "più è probabile che la migliore sembri buona solo per fortuna. "
            "Il verdetto non ne tiene conto: vale per un solo tentativo."
        )
    for warning in report.warnings:
        st.warning(warning)

    labels = {"strategy": name, "buy_and_hold": "Compra e tieni"}
    returns = a["returns"]
    tabs = st.tabs(["Sintesi", "Rendimenti", "Rischio", "Operazioni", "Robustezza", "Segnale e notizie", "Dati"])

    with tabs[0]:  # Summary
        if ohlcv is not None:
            days = RANGES[st.radio("Periodo grafico", list(RANGES), index=3, key="adv_chart_range",
                                   horizontal=True, label_visibility="collapsed")]
            chart(candle_chart(ohlcv, days, trades=a["trades"]))
            ma_legend(with_trades=True)
        log = st.toggle("Scala logaritmica", key="adv_log",
                        help="Su periodi lunghi rende confrontabili le variazioni percentuali di inizio e fine.")
        st.markdown("##### Valore di 1 € investito, costi inclusi")
        chart(equity_chart(returns, labels, log=log))
        st.markdown("##### Distanza dal massimo precedente")
        chart(drawdown_chart(returns, labels))

    with tabs[1]:  # Returns
        left, right = st.columns([3, 2])
        with left:
            st.markdown("##### Rendimento per anno solare")
            chart(annual_chart(returns, labels))
        with right:
            st.markdown("##### Tabella annuale")
            st.table(annual_table(returns, labels))
            st.caption("Il primo e l'ultimo anno possono essere parziali: contano solo i giorni del periodo analizzato.")
        st.markdown(f"##### Rendimenti mensili: {html.escape(name)}")
        chart(monthly_heatmap(returns["strategy"]))
        st.markdown("##### Rendimenti mensili: compra e tieni")
        chart(monthly_heatmap(returns["buy_and_hold"]))

    with tabs[2]:  # Risk
        st.markdown("##### I cali più profondi della strategia")
        if drawdown_periods(returns["strategy"]).empty:
            st.caption("Nessun calo nel periodo.")
        else:
            st.table(drawdown_table(returns["strategy"]))
        window = min(report.periods_per_year, max(20, len(returns) // 4))
        left, right = st.columns(2)
        with left:
            st.markdown("##### Volatilità mobile")
            chart(rolling_chart(returns, labels, "vol", window, report.periods_per_year, cash_rate))
        with right:
            st.markdown("##### Sharpe mobile")
            chart(rolling_chart(returns, labels, "sharpe", window, report.periods_per_year, cash_rate))
        st.caption(f"Finestra mobile di {window} giorni di borsa.")

    with tabs[3]:  # Trades
        st.caption(
            f"{len(a['trades'])} operazioni in {num(a['years'], 1)} anni "
            f"({num(len(a['trades']) / a['years'], 1)} all'anno). "
            "Ogni operazione paga commissione e slippage: più operazioni, più costi."
        )
        chart(trades_chart(a["prices"], a["trades"]))
        kpi_strip(trade_kpis(a["stats"]))  # every strategy in the app is long/flat: round trips exist
        if not a["trips"].empty:
            st.markdown("##### Elenco operazioni")
            st.dataframe(trips_table(a["trips"]), width="stretch", height=min(420, 38 + 35 * len(a["trips"])))

    with tabs[4]:  # Robustness
        left, right = st.columns(2)
        with left:
            st.markdown("##### Periodo di scelta (in-sample)")
            st.caption("Qui la strategia è avvantaggiata: i parametri sono stati scelti guardando questi dati."
                       if grid else "Parametri fissi: nessuna ottimizzazione automatica. Ma se li hai scelti tu "
                                    "dopo aver guardato i grafici, li hai scelti sul passato anche tu.")
            st.table(metrics_table(report.in_sample, labels))
        with right:
            st.markdown("##### Periodo di prova (out-of-sample)")
            st.caption("Dati mai usati per scegliere niente: è su questo periodo che si basa il verdetto.")
            st.table(metrics_table(report.out_of_sample, labels))
        if report.walk_forward is not None:
            st.markdown("##### Walk-forward")
            st.caption("Ogni anno i parametri vengono riscelti usando solo i 3 anni precedenti.")
            st.table(metrics_table(report.walk_forward, labels))
        st.markdown("##### Bootstrap: quanto conta la fortuna")
        chart(bootstrap_chart(report.sharpe_diff_samples, report.sharpe_diff_ci))
        st.caption(
            f"La storia del periodo di prova è stata rimescolata {len(report.sharpe_diff_samples)} volte "
            "a blocchi di 20 giorni. Ogni barra conta quante volte la differenza di Sharpe è caduta in quella "
            "zona. Linee tratteggiate: intervallo al 95%. Se contiene lo zero (linea nera), il vantaggio "
            "potrebbe essere fortuna."
        )

    with tabs[5]:  # Signal and news
        try:
            live_prices = get_prices(ticker, str(start), str(date.today()))
            day_, value, since = current_signal(strategy_func(live_prices, **report.params))
        except Exception as exc:  # network problems, or a rule that cannot be applied
            st.info(f"Non riesco a calcolare il segnale di oggi: {exc}")
        else:
            signal_panel(ticker, day_, value, since, live_prices.iloc[-1], tone)
        show_news(ticker)

    with tabs[6]:  # Data
        export = report.returns.rename(columns={"strategy": "strategia", "buy_and_hold": "compra_e_tieni"})
        st.download_button("Scarica i rendimenti giornalieri (CSV)", export.to_csv().encode("utf-8"),
                           file_name=f"quantlab_{ticker}.csv", mime="text/csv")
        st.markdown("##### Report completo")
        st.code(format_report(report), language=None)
