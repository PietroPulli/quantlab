"""The ready-made investment ideas, shared by the app, the scanner and the paper account.

Each idea knows how to build its strategy for a ticker. Ideas based on external data
(VIX, yield curve, earnings) get that data from a `factor` function passed in by the
caller, so the app can use its cache, scripts can download directly and tests can fake it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from quantlab.rules import Condition, Indicator, rule_strategy
from quantlab.strategies import breakout, mean_reversion, momentum
from quantlab.validation import Strategy

# factor(name, ticker) -> daily series; name is one of FACTOR_NAMES
Factor = Callable[[str, str], pd.Series]
FACTOR_NAMES = {"vix": "VIX", "yield_curve": "Curva tassi", "earnings": "Sorpresa utili"}


@dataclass(frozen=True)
class Idea:
    name: str
    explanation: str  # in plain Italian, shown to users
    build: Callable[[str, Factor], tuple[Strategy, dict]]  # (ticker, factor) -> (strategy, params)
    companies_only: bool = False  # needs data that exists only for single companies


def _on(factor: Factor, name: str, op: str, level: float, ticker: str = "") -> Condition:
    """Condition on the value of an external factor, e.g. VIX < 25."""
    return Condition(Indicator("price", source=factor(name, ticker), label=FACTOR_NAMES[name]), op, level)


IDEAS = {idea.name: idea for idea in [
    Idea("Segui la tendenza",
         "Resto investito quando il prezzo è sopra la sua media degli ultimi 200 giorni (circa 10 mesi). "
         "Quando scende sotto, tengo i soldi da parte.",
         lambda t, f: (rule_strategy, {"entry": Condition(Indicator("price"), ">", Indicator("sma", 200)),
                                       "exit": None})),
    Idea("Compra quando sfonda verso l'alto",
         "Compro quando il prezzo tocca il massimo degli ultimi 50 giorni, vendo quando tocca il minimo.",
         lambda t, f: (breakout, {"window": 50})),
    Idea("Compra dopo un forte calo",
         "Compro quando il prezzo è sceso molto sotto la sua media dell'ultimo mese, vendo quando ci torna.",
         lambda t, f: (mean_reversion, {})),
    Idea("Compra ciò che è salito nell'ultimo anno",
         "Resto investito se nell'ultimo anno il prezzo è salito (senza contare l'ultimo mese), altrimenti no.",
         lambda t, f: (momentum, {})),
    Idea("Esci quando il mercato ha paura",
         "Resto investito finché il VIX, l'indice della paura del mercato americano, è sotto 25. "
         "Quando sale sopra, cioè quando c'è agitazione, tengo i soldi da parte.",
         lambda t, f: (rule_strategy, {"entry": _on(f, "vix", "<", 25.0), "exit": None})),
    Idea("Compra quando il mercato ha paura",
         "Compro quando il VIX supera 30 (panico: i prezzi sono spesso scesi molto) e rivendo quando "
         "torna sotto 20, cioè quando la calma è tornata.",
         lambda t, f: (rule_strategy, {"entry": _on(f, "vix", ">", 30.0), "exit": _on(f, "vix", "<", 20.0)})),
    Idea("Esci quando la curva dei tassi si inverte",
         "Quando i tassi a 2 anni superano quelli a 10 anni (curva «invertita»), storicamente è spesso "
         "arrivata una recessione. Resto investito solo quando la curva è normale.",
         lambda t, f: (rule_strategy, {"entry": _on(f, "yield_curve", ">", 0.0), "exit": None})),
    Idea("Compra dopo trimestrali sopra le attese",
         "Resto investito se l'ultima trimestrale dell'azienda ha battuto le attese degli analisti, "
         "fuori se le ha mancate. Solo per singole aziende (Apple, Ferrari, ENI...), non per indici o crypto.",
         lambda t, f: (rule_strategy, {"entry": _on(f, "earnings", ">", 0.0, t), "exit": None}),
         companies_only=True),
]}


def download_factor(cache_dir: str | Path = "data/") -> Factor:
    """A `factor` function that downloads (and caches) the real data, for scripts."""
    from quantlab.data import load_prices
    from quantlab.earnings import load_earnings_surprises
    from quantlab.macro import load_macro

    def factor(name: str, ticker: str = "") -> pd.Series:
        if name == "vix":
            return load_prices(["^VIX"], "2000-01-01", str(date.today()), cache_dir)["^VIX"].dropna()
        if name == "yield_curve":
            return load_macro("yield_curve", cache_dir)
        if name == "earnings":
            return load_earnings_surprises(ticker, cache_dir)
        raise ValueError(f"unknown factor {name!r}")

    return factor
