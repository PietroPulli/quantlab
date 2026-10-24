"""The Assistente as an agent: the model decides which analyses to run, Quantlab computes them.

The model never computes a number. It asks for one through a tool ("compare SPY and QQQ since
2005"), this module runs the real library code on real prices and hands back the results as
ready-written Italian strings ("+9,8%"), and the model reasons on them. It can ask again if a
result raises a doubt, up to MAX_ROUNDS times, then it must answer.
"""

import json
import re
from collections.abc import Callable, Iterator
from datetime import date, timedelta

import pandas as pd

from quantlab import metrics
from quantlab.ideas import IDEAS, Factor, Idea
from quantlab.scanner import scan

MAX_ROUNDS = 6  # rounds of tool calls before the model must answer with what it has
MAX_TICKERS = 6  # per comparison: each one is a download
MIN_DAYS = 252  # one year of prices, or the yearly numbers mean nothing
CASH_RATE = 0.02  # same as the app: idle cash earns this, and the Sharpe is measured above it

DISCLAIMER = ("⚠️ *Analisi automatica di Quantlab sui prezzi passati, non una consulenza finanziaria. "
              "Il passato non garantisce il futuro: la decisione e la responsabilità sono tue.*")

RECHECK = ("Nella risposta ci sono numeri ma non hai chiamato nessuno strumento. Se parlano di mercati "
           "(rendimenti, perdite, anni, confronti), chiama ora gli strumenti e rispondi con i loro risultati. "
           "Se erano solo esempi di teoria, riscrivi la stessa risposta.")
VERDICTS = {"good": "batte il compra e tieni (regge al test della fortuna)",
            "bad": "peggio del compra e tieni, e non per sfortuna",
            "neutral": "differenza compatibile con la fortuna"}

# The tools in the format Ollama expects. Names and descriptions are in Italian: the model reads them.
_DATE = {"type": "string", "description": "Data AAAA-MM-GG. Se manca: 20 anni fa per 'dal', oggi per 'al'."}
TOOLS = [
    {"type": "function", "function": {
        "name": "confronta_titoli",
        "description": "Confronta da 1 a 6 titoli/ETF sullo STESSO periodo (quello in cui ci sono dati per "
                       "tutti): rendimento annuo, volatilità, Sharpe, perdita massima, anno peggiore e migliore. "
                       "Usalo per scegliere tra investimenti o per guardare un periodo preciso (es. 2000-2010).",
        "parameters": {"type": "object", "required": ["tickers"], "properties": {
            "tickers": {"type": "array", "items": {"type": "string"},
                        "description": "Ticker Yahoo Finance, es. [\"SPY\", \"QQQ\", \"GC=F\"]"},
            "dal": _DATE, "al": _DATE}}}},
    {"type": "function", "function": {
        "name": "rendimenti_per_anno",
        "description": "Il rendimento di ogni anno solare di un titolo: per vedere crolli, recuperi, anni fermi.",
        "parameters": {"type": "object", "required": ["ticker"], "properties": {
            "ticker": {"type": "string", "description": "Ticker Yahoo Finance, es. \"SPY\""},
            "dal": _DATE}}}},
    {"type": "function", "function": {
        "name": "prova_le_idee",
        "description": "Prova tutte le idee di Quantlab (trend, momentum, VIX, curva dei tassi...) su un titolo "
                       "contro il compra e tieni, con costi e test della fortuna, sull'ultimo 30% degli ultimi "
                       "10 anni. Dice se entrare e uscire dal mercato avrebbe aggiunto valore. Lento: ~30 s.",
        "parameters": {"type": "object", "required": ["ticker"], "properties": {
            "ticker": {"type": "string", "description": "Ticker Yahoo Finance, es. \"SPY\""}}}}},
]

# price loader: (ticker, start "YYYY-MM-DD", end "YYYY-MM-DD") -> daily closes
Loader = Callable[[str, str, str], pd.Series]
# one call to the model: (messages, tools) -> the reply message {"role", "content", "tool_calls"?}
Ask = Callable[[list[dict], list[dict] | None], dict]


def pct(x: float) -> str:
    """0.0981 -> '+9,8%' (Italian decimals, sign always shown)."""
    return f"{x * 100:+.1f}%".replace(".", ",")


def _num(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def _day(value: str | None, default: date) -> str:
    """A date the model wrote, checked: falls back to `default` if missing or not a date."""
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d") if value else default.isoformat()
    except ValueError:
        return default.isoformat()


def describe(name: str, args: dict) -> str:
    """What a tool call is doing, in a line for the user: 'Confronto SPY, QQQ dal 2005-01-01'."""
    if name == "confronta_titoli":
        since = f" dal {args['dal']}" if args.get("dal") else " (ultimi 20 anni)"
        until = f" al {args['al']}" if args.get("al") else ""
        return f"Confronto {', '.join(args.get('tickers', []))}{since}{until}"
    if name == "rendimenti_per_anno":
        return f"Rendimenti anno per anno di {args.get('ticker', '?')}"
    if name == "prova_le_idee":
        return f"Test della fortuna sulle idee di Quantlab per {args.get('ticker', '?')}"
    return f"Strumento sconosciuto: {name}"


class Toolbox:
    """Runs the tools on real prices. `load` and `factor` come from the caller (app cache, or fakes in tests).

    Results are remembered: the model often asks the same thing twice in one conversation.
    """

    def __init__(self, load: Loader, factor: Factor, ideas: dict[str, Idea] = IDEAS,
                 n_bootstrap: int = 300, today: date | None = None):
        self.load, self.factor, self.ideas, self.n_bootstrap = load, factor, ideas, n_bootstrap
        self.today = today or date.today()
        self._memo: dict[str, dict] = {}

    def run(self, name: str, args: dict) -> dict:
        """The result of one tool call, or {"errore": ...}: a mistake is reported to the model, never raised."""
        key = json.dumps([name, args], sort_keys=True, default=str)
        if key not in self._memo:
            tool = {"confronta_titoli": self.compare, "rendimenti_per_anno": self.yearly,
                    "prova_le_idee": self.ideas_test}.get(name)
            try:
                self._memo[key] = tool(**args) if tool else {"errore": f"lo strumento {name} non esiste"}
            except Exception as exc:  # unknown ticker, no data, wrong arguments
                self._memo[key] = {"errore": str(exc) or type(exc).__name__}
        return self._memo[key]

    def _prices(self, ticker: str, start: str, end: str) -> pd.Series:
        prices = self.load(ticker.strip().upper(), start, end).dropna()
        if len(prices) < MIN_DAYS:
            raise ValueError(f"meno di un anno di prezzi per {ticker} in questo periodo")
        return prices

    def compare(self, tickers: list[str], dal: str | None = None, al: str | None = None) -> dict:
        start = _day(dal, self.today - timedelta(days=20 * 365))
        end = _day(al, self.today)
        loaded, errors = {}, {}
        for ticker in list(dict.fromkeys(tickers))[:MAX_TICKERS]:
            try:
                loaded[ticker.strip().upper()] = self._prices(ticker, start, end)
            except Exception as exc:
                errors[ticker] = str(exc) or "dati non trovati"
        if not loaded:
            return {"errore": "nessun titolo con dati", "titoli_senza_dati": errors}
        # same period for all, or an older asset would look better just for having more history
        common = pd.DataFrame(loaded).dropna()
        if len(common) < MIN_DAYS:
            raise ValueError("meno di un anno in cui ci sono prezzi per tutti i titoli insieme")
        results = {}
        for ticker in common:
            returns = common[ticker].pct_change().dropna()
            stats = metrics.summary(returns, risk_free=CASH_RATE)
            years = metrics.calendar_returns(returns)
            results[ticker] = {
                "rendimento_annuo": pct(stats["annual_return"]),
                "rendimento_totale": pct(stats["total_return"]),
                "volatilita_annua": pct(stats["annual_volatility"]).lstrip("+"),
                "sharpe": _num(stats["sharpe"]),
                "perdita_massima": pct(stats["max_drawdown"]),
                "anno_peggiore": f"{years.idxmin()}: {pct(years.min())}",
                "anno_migliore": f"{years.idxmax()}: {pct(years.max())}",
            }
        result = {"periodo_comune": f"{common.index[0]:%Y-%m-%d} – {common.index[-1]:%Y-%m-%d}",
                  "titoli": results}
        if errors:
            result["titoli_senza_dati"] = errors
        return result

    def yearly(self, ticker: str, dal: str | None = None) -> dict:
        start = _day(dal, self.today - timedelta(days=20 * 365))
        returns = self._prices(ticker, start, self.today.isoformat()).pct_change().dropna()
        years = metrics.calendar_returns(returns)
        last = years.index[-1]
        out = {str(year): pct(value) for year, value in years.items()}
        if returns.index[-1].year == last and returns.index[-1].month < 12:
            out[f"{last} (da inizio anno)"] = out.pop(str(last))  # not a full year yet
        return {"ticker": ticker.upper(), "rendimenti": out}

    def ideas_test(self, ticker: str) -> dict:
        ticker = ticker.strip().upper()
        prices = self._prices(ticker, (self.today - timedelta(days=10 * 365)).isoformat(), self.today.isoformat())
        built = {}
        for name, idea in self.ideas.items():
            try:
                built[name] = idea.build(ticker, self.factor)
            except Exception:  # e.g. earnings asked for an index or a crypto
                pass
        table = scan(prices, built, cash_rate=CASH_RATE, n_bootstrap=self.n_bootstrap)
        start, end = table.attrs["test_start"], prices.index[-1]
        return {
            "ticker": ticker,
            "periodo_giudicato": f"{start:%Y-%m-%d} – {end:%Y-%m-%d}",
            "compra_e_tieni_rendimento_totale": pct(table["bh_total_return"].iloc[0]),
            "idee": {row["idea"]: {"rendimento_totale": pct(row["total_return"]),
                                   "verdetto": VERDICTS[row["verdict"]]}
                     for _, row in table.iterrows()},
            "idee_che_battono_il_compra_e_tieni": int((table["verdict"] == "good").sum()),
        }


def _arguments(call: dict) -> dict:
    """The arguments of a tool call: Ollama gives a dict, some models a JSON string."""
    args = call.get("function", {}).get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return {}
    return args if isinstance(args, dict) else {}


def run(messages: list[dict], toolbox: Toolbox, ask: Ask, max_rounds: int = MAX_ROUNDS) -> Iterator[dict]:
    """Let the model work: yields {"type": "tool", "name", "args", "result"} for each analysis,
    then one {"type": "answer", "content"}.

    Small models often answer from memory even when told to use the tools: an answer with numbers
    but no analysis behind it is sent back once, asking to run the tools (or confirm it was theory).
    """
    messages, used_tools, checked = list(messages), False, False
    for _ in range(max_rounds):
        reply = ask(messages, TOOLS)
        calls = reply.get("tool_calls") or []
        content = reply.get("content", "")
        if not calls and not used_tools and not checked and re.search(r"\d", content):
            checked = True
            messages += [{"role": "assistant", "content": content}, {"role": "user", "content": RECHECK}]
            continue
        if not calls:
            yield {"type": "answer", "content": content}
            return
        used_tools = True
        messages.append({"role": "assistant", "content": reply.get("content", ""), "tool_calls": calls})
        for call in calls:
            name, args = call.get("function", {}).get("name", ""), _arguments(call)
            result = toolbox.run(name, args)
            yield {"type": "tool", "name": name, "args": args, "result": result}
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result, ensure_ascii=False)})
    # out of rounds: no more tools, answer with what is there
    messages.append({"role": "user", "content": "Basta analisi: rispondi ora usando i numeri che hai già."})
    yield {"type": "answer", "content": ask(messages, None).get("content", "")}
