# QuantLab

[![tests](https://github.com/PietroPulli/quantlab/actions/workflows/tests.yml/badge.svg)](https://github.com/PietroPulli/quantlab/actions/workflows/tests.yml)

**La macchina della verità per le strategie di investimento.**
Prende un'idea di trading e mostra onestamente se avrebbe battuto il semplice comprare e tenere:
dopo i costi, su dati mai visti, e al netto della fortuna.

> Non fornisce consigli di investimento: dice solo se una regola *avrebbe* funzionato sul passato.

## Cosa fa
- **Dati**: prezzi giornalieri di qualsiasi titolo Yahoo Finance (azioni, ETF, indici, crypto,
  materie prime, valute), salvati in cache locale e controllati (buchi, salti, prezzi ≤ 0).
- **Backtest**: il segnale di oggi diventa posizione solo da domani (niente look-ahead),
  commissioni e slippage sempre inclusi, interesse sui contanti quando si è fuori dal mercato.
- **Strategie**: 5 pronte (buy & hold, breakout, momentum 12-1, incrocio di medie, mean reversion)
  e un costruttore di regole ("compra quando il prezzo è sopra la media a 200 giorni E il VIX è sotto 25").
- **Fattori nelle regole**: VIX, tassi USA, dollaro, petrolio, oro (dati di mercato), inflazione,
  disoccupazione, tassi Fed, curva dei tassi (FRED) e sorprese sugli utili trimestrali, ciascuno
  usato solo da quando era davvero pubblico.
- **Macchina della verità**: split in-sample / out-of-sample, walk-forward, bootstrap a blocchi
  della differenza di Sharpe, verdetto (batte / nessuna prova / peggio) e avvisi su overfitting
  e su troppi tentativi.
- **App** con due viste: *Semplice* (tre scelte, una risposta in euro) e *Approfondita*
  (tutti i parametri e le statistiche). Le notizie compaiono solo come contesto: non entrano nei calcoli.

## Avvio rapido
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"
pytest                          # 166 test, tutti offline
streamlit run app/streamlit_app.py
```
L'app si apre su http://localhost:8501.

## Struttura
| Percorso | Contenuto |
|---|---|
| `src/quantlab/data.py` | download, cache e controllo dei prezzi; notizie |
| `src/quantlab/backtest.py` | da segnali a rendimenti netti; elenco operazioni; segnale di oggi |
| `src/quantlab/metrics.py` | rendimento, volatilità, Sharpe, drawdown, Calmar |
| `src/quantlab/strategies.py` | le strategie pronte |
| `src/quantlab/rules.py` | costruttore di regole su prezzo o fattori esterni |
| `src/quantlab/macro.py` | serie FRED rese "point-in-time" |
| `src/quantlab/earnings.py` | sorprese sugli utili, datate al primo giorno utilizzabile |
| `src/quantlab/validation.py` | split, ottimizzazione, walk-forward, bootstrap |
| `src/quantlab/report.py` | `evaluate_strategy`: tutto il giudizio in una chiamata |
| `app/` | l'app Streamlit (`simple.py`, `advanced.py`, `common.py`) |
| `notebooks/` | esplorazione dei dati e prime prove |
| `tests/` | un file di test per ogni modulo |

## Limiti noti
- **Bias di sopravvivenza**: Yahoo ha solo le aziende che esistono ancora.
- **Tasso dei contanti** fisso (default 2% annuo), non la serie storica reale.
- **Molti tentativi**: l'app avvisa quando provi tante idee sullo stesso titolo, ma il verdetto
  non viene corretto statisticamente.
- **Sentiment dai social**: escluso, perché non esiste uno storico gratuito che permetta un test onesto.

## Documentazione
- [Visione](docs/VISIONE.md) · [Roadmap](docs/ROADMAP.md)
- Lezioni: [Fase 0-1](docs/FASE_0_1.md) · [Fase 1](docs/LEZIONE_FASE1.md) ·
  [Fasi 2-4](docs/LEZIONE_FASI_2_4.md) · [App, regole e fattori](docs/LEZIONE_APP_E_FATTORI.md)
