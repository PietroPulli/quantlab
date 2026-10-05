# Lezione — App, regole, fattori esterni

Questa lezione copre tutto ciò che è stato aggiunto dopo le fasi 2–4: nuove metriche e
strategie, l'app, il costruttore di regole, i fattori esterni (VIX, macro, trimestrali).
Il filo conduttore è sempre lo stesso: **a ogni data, il software può usare solo ciò che
quel giorno era davvero noto.** Quasi ogni scelta qui sotto esiste per rispettare questa frase.

Leggi i file in quest'ordine, uno per seduta: `rules.py` → `macro.py` → `earnings.py` →
`backtest.py` (solo `trade_log`, `current_signal`, `cash_rate`) → `app/simple.py`.

---

## 1. Calmar e breakout (gli esercizi 1–2 della lezione precedente)
- `calmar_ratio` = rendimento annuo / |max drawdown|. Senza drawdown restituisce `NaN`:
  dividere per zero darebbe infinito, cioè una strategia "perfetta" che in realtà non è misurabile.
- `breakout` compra sul massimo degli ultimi N giorni e vende sul minimo. **Bug trovato dai test:**
  con un prezzo piatto, oggi è *sia* il massimo *sia* il minimo. Senza la correzione
  `is_high & ~is_low` la strategia vendeva senza motivo.

## 2. L'interesse sui contanti (`cash_rate`)
Quando una strategia è fuori dal mercato i soldi non restano fermi: stanno in BOT o su un
conto deposito. Senza questo, le strategie che stanno spesso fuori sono penalizzate ingiustamente.

```python
daily_cash = (1 + cash_rate) ** (1 / periods_per_year) - 1   # tasso annuo -> giornaliero composto
cash_return = (1 - position.abs()) * daily_cash             # solo sulla parte non investita
cash_return.iloc[:1] = 0.0                                   # il giorno 0 è il punto di partenza
```

**La trappola collegata:** se i contanti rendono il 4% e lo Sharpe non lo sottrae, una strategia
*sempre in contanti* guadagna sempre un po' e non oscilla mai, quindi avrebbe uno Sharpe enorme.
Per questo ogni Sharpe (tabelle, ottimizzazione, walk-forward, bootstrap) usa `risk_free=cash_rate`.
Il test `test_sitting_in_cash_does_not_look_skilful` lo verifica.

## 3. 252 o 365 giorni (`infer_periods_per_year`)
Le azioni hanno circa 252 chiusure l'anno, Bitcoin 365. Usare 252 per Bitcoin gonfia rendimento e
volatilità annui e paga troppi interessi sui contanti. La funzione lo deduce dalle date:
più di 300 osservazioni per anno di calendario → 365.

## 4. Il costruttore di regole (`rules.py`)
Tre mattoni:
- `Indicator("sma", 200)` → una serie calcolata dai prezzi (media mobile a 200 giorni);
- `Condition(sinistra, ">", destra)` → vero/falso giorno per giorno;
- `rule_strategy(prices, entry, exit)` → segnali 1/0.

Una lista di condizioni significa **E** (devono valere tutte). Con una regola di uscita si usa lo
stesso trucco di `ffill` della mean reversion: si segnano solo i giorni di decisione e poi si
"trascina" l'ultima decisione.

**Il test più elegante del progetto:** scrivendo a mano le regole del breakout e dell'incrocio di
medie, `rule_strategy` dà *esattamente* gli stessi segnali delle strategie già testate. Due codici
diversi che arrivano allo stesso risultato si controllano a vicenda.

## 5. Fattori esterni senza guardare il futuro
Un indicatore può essere calcolato su un'altra serie (`source=vix`). Il problema: VIX e titolo non
hanno sempre gli stessi giorni di borsa. La soluzione:

```python
source.reindex(prices.index, method="ffill")
```

`reindex` mette la serie sulle date del titolo. `method="ffill"` riempie un giorno mancante con
**l'ultimo valore già noto**, mai con uno successivo. Il test
`test_external_source_does_not_leak_the_future` cambia i valori *futuri* del VIX e controlla che
i segnali passati restino identici.

## 6. Dati macro "point-in-time" (`macro.py`)
L'inflazione di agosto viene pubblicata a metà settembre. Un backtest che la usa il 1° agosto
**bara**, anche se il codice sembra innocuo. Due difese:
1. **ritardo di pubblicazione**: `point_in_time(serie, lag_days)` sposta ogni valore alla data in
   cui era sicuramente pubblico (inflazione +45 giorni, disoccupazione e tassi Fed +35, curva +1);
2. **revisioni**: per l'inflazione si usa l'indice *non destagionalizzato* (`CPIAUCNS`), che non
   viene mai corretto dopo l'uscita.

## 7. Trimestrali (`earnings.py`)
`known_from(orario_annuncio)` decide da quando una sorpresa è utilizzabile:
annuncio prima delle 16:00 di New York → lo stesso giorno; dopo la chiusura o a orario
sconosciuto → il giorno dopo. Nel dubbio si sceglie sempre la data **più tardi**.

## 8. Operazioni e segnale di oggi (`backtest.py`)
- `trade_log`: la posizione del giorno t è stata decisa alla chiusura di t−1, quindi
  un'operazione va datata a t−1, al prezzo di quella chiusura. Datarla a t mostrerebbe un prezzo
  mai pagato.
- `current_signal`: l'ultimo segnale e da quando dura. È ciò che la regola dice *oggi*: non è un
  consiglio, è il risultato della regola dell'utente.

## 9. L'app
- `streamlit_app.py` disegna solo la cornice e sceglie la vista; `simple.py` e `advanced.py`
  disegnano; **tutti i numeri vengono dalla libreria**, mai da codice scritto nell'app.
- Un `st.button` vale `True` solo nel giro subito dopo il click: per questo lo stato
  "ho già premuto" è salvato in `st.session_state`.
- Le notizie sono mostrate solo come contesto e filtrate con `relevant_news`: non esiste un
  archivio gratuito di notizie datate, quindi non si possono testare onestamente.
- I test dell'app (`streamlit.testing.v1.AppTest`) usano prezzi, notizie, macro e trimestrali
  finti: girano senza internet, anche su GitHub ad ogni push.

---

## Domande di verifica
1. Perché `calmar_ratio` restituisce `NaN` e non infinito quando non c'è drawdown?
2. Una strategia è sempre in contanti al 4%. Quanto vale il suo Sharpe se *non* sottraiamo il
   tasso? E se lo sottraiamo? Perché la seconda risposta è quella giusta?
3. Perché il primo giorno del backtest non matura interessi?
4. Cosa succede ai numeri annui di Bitcoin se usiamo 252 invece di 365?
5. Scrivi con `Condition` e `Indicator` la regola "compra quando la media a 50 giorni è sopra
   quella a 200 E il VIX è sotto 25".
6. Cosa fa `reindex(..., method="ffill")`? Cosa succederebbe con `method="bfill"`?
7. Perché l'inflazione si sposta di 45 giorni e la curva dei tassi di 1 solo giorno?
8. Apple annuncia gli utili alle 16:30. Da che giorno la regola può usare la sorpresa? E se
   annunciasse alle 7:00?
9. Perché `trade_log` data un acquisto al giorno *prima* che la posizione diventi 1?
10. Su Apple, "compra dopo trimestrali sopra le attese" dà esattamente il compra e tieni. Perché?
11. Perché le notizie non entrano nei calcoli, mentre il VIX sì?

## Esercizi da fare da solo
1. **Un fattore nuovo.** Aggiungi in `app/common.py` → `FACTORS` il rame (`HG=F`), spesso citato
   come termometro dell'economia. Che riga devi scrivere? Serve un nuovo test?
2. **Rompi il point-in-time.** In una copia di `point_in_time` togli lo spostamento delle date e
   lancia `pytest tests/test_macro.py`. Quale test fallisce? Prevedilo prima.
3. **Una regola a parole tue.** Nella vista Approfondita costruisci una regola che usa un fattore
   macro e scrivi in tre righe perché *pensavi* che funzionasse e cosa dice il verdetto.
4. **Il costo dei tentativi.** Prova 10 regole diverse su SPY e annota la migliore. Rileggi
   l'avviso giallo dell'app: quanto ti fidi della migliore? Perché?
