# Roadmap — MVP in ~8 settimane (6–8 ore/settimana)

Obiettivo dell'MVP: **io posso prendere un'idea di strategia e in 10 minuti ottenere un report
onesto** che dice se regge o no.

| Settimana | Fase | Risultato concreto |
|---|---|---|
| 1 | **0 — Setup** | Python, Git, GitHub, Claude Code funzionanti. Primo test che passa. |
| 2–3 | **1 — Dati** | Scarico e salvo in cache prezzi storici di qualsiasi ticker, puliti e verificati. |
| 4–5 | **2 — Motore di backtest** | Da segnali a rendimenti netti, con costi e test anti look-ahead. Metriche base. |
| 6 | **3 — Strategie classiche** | Buy & hold, momentum, incrocio medie mobili, mean reversion. |
| 7–8 | **4 — Macchina della verità** | Split in/out-of-sample, walk-forward, bootstrap, report finale leggibile. |

**Checkpoint dopo la fase 2:** mostro i primi risultati ai ragazzi di finanza e raccolgo
cosa vorrebbero verificare loro.

## Dopo l'MVP (da decidere insieme agli altri)
- [x] Interfaccia semplice (Streamlit) per chi non programma
- [x] Portafogli multi-asset e ribilanciamento
- [x] Modulo derivati: Black-Scholes, greche, Monte Carlo (`derivatives.py`, scheda Opzioni)
- [x] Report pubblici: primo report "Tre strategie famose battono il compra e tieni?" (24/10/2026)
- [x] Assistente AI locale (Ollama, senza chiavi), anche dal telefono via Tailscale

## Regola del tempo
Se una fase sfora di più di una settimana, si taglia lo scope, non si allunga.
Meglio un MVP semplice e usato che uno completo e mai finito.
