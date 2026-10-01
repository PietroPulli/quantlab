# Lezione Fasi 2–4 — Backtest, strategie, macchina della verità

Prerequisito: `LEZIONE_FASE1.md`. Per vedere tutto in azione apri `notebooks/02_macchina_verita.ipynb`.

Il flusso completo in una riga:

```
prezzi → strategia → segnali → shift(1) → posizioni → rendimenti lordi − costi → rendimenti netti
       → metriche, sempre accanto a buy & hold → in-sample / out-of-sample / walk-forward → bootstrap → verdetto
```

---

## Fase 2 — Il motore di backtest (`backtest.py`, `metrics.py`)

### Segnale, posizione, rendimento: chi viene prima
- **Segnale del giorno t**: calcolato con la chiusura di t ("domani voglio essere long").
- **Posizione del giorno t**: quella che *ho davvero* durante il giorno t = segnale di t−1.
- **Rendimento della strategia il giorno t** = posizione_t × rendimento dell'asset_t.

```python
def signals_to_positions(signals):
    return signals.shift(1).fillna(0.0)
```
`shift(1)` sposta tutti i valori avanti di una riga: il valore di ieri finisce sulla riga di oggi.
Il primo giorno non ha un "ieri" → `NaN` → `fillna(0.0)` = sono in liquidità.

**Perché è la regola più importante del progetto:** senza shift, la strategia userebbe la chiusura
di oggi per guadagnare il rendimento di oggi, cioè quello che finisce *a quella stessa chiusura*.
Sarebbe come scommettere su una partita già finita. Il backtest sembrerebbe fantastico ma è
impossibile da replicare.

> Semplificazione dichiarata: assumiamo di eseguire l'ordine alla chiusura del giorno del segnale.
> Nella realtà si esegue un po' dopo, a un prezzo leggermente diverso: è una delle cose che lo
> **slippage** deve coprire.

### I costi
```python
turnover = position.diff().fillna(position).abs()
cost = turnover * (commission + slippage)
```
- `position.diff()` = variazione di posizione rispetto a ieri. Passare da 0 a 1 = compro tutto il
  capitale (turnover 1). Da 1 a −1 = vendo e vado short (turnover 2, paghi due volte).
- `.fillna(position)`: il primo giorno `diff` è `NaN`; la variazione vera è da 0 alla posizione iniziale.
- **Commissione** (0,10%): quello che prende il broker. **Slippage** (0,05%): spread denaro/lettera
  e prezzo che si muove mentre compri. Default **mai a zero**, e `ValueError` se negativi.
- Il costo si sottrae al rendimento il giorno in cui la posizione cambia.

### I controlli in ingresso
`run_backtest` rifiuta: costi negativi, segnali con indice diverso dai prezzi (si
allineerebbero male in silenzio), segnali `NaN`, segnali fuori da [−1, 1] (leva: non nell'MVP).
Meglio un errore subito che un risultato sbagliato che sembra giusto.

### Le metriche
| Metrica | Formula | Cosa dice |
|---|---|---|
| Rendimento totale | `prod(1 + r) − 1` | Quanto è cresciuto 1 € |
| Rendimento annuo (CAGR) | `prod(1 + r) ^ (252 / n) − 1` | Media *geometrica* per anno |
| Volatilità annua | `std(r) × √252` | Quanto oscilla. Cresce con √tempo, non col tempo |
| Sharpe | `media(r − rf) / std(r − rf) × √252` | Rendimento per unità di rischio |
| Max drawdown | `min(equity / picco − 1)` | La peggior perdita da un massimo |

Dettagli che i test hanno fatto emergere:
- **Sharpe di una serie costante**: la deviazione standard in floating point vale ~1e-19, non 0.
  Senza tolleranza (`MIN_STD = 1e-12`) lo Sharpe usciva 7×10¹⁶. Ora restituisce `NaN`.
- **Drawdown dal primo giorno**: il capitale iniziale (1) conta come primo picco (`clip(lower=1.0)`),
  altrimenti una perdita il giorno 1 non verrebbe mai contata.

---

## Fase 3 — Le strategie (`strategies.py`)

Tutte sono **long o liquidità** (1 o 0) e restituiscono 0 durante il **warm-up** (quando non c'è
ancora abbastanza storia per calcolare l'indicatore).

| Strategia | Regola | Idea economica |
|---|---|---|
| `buy_and_hold` | sempre 1 | Il benchmark. Tutto il resto deve batterlo |
| `momentum(252, 21)` | long se il prezzo di 21 giorni fa > prezzo di 252 giorni fa | Chi è salito nell'ultimo anno tende a continuare; l'ultimo mese si esclude perché nel brevissimo i rendimenti tendono a invertirsi |
| `moving_average_crossover(50, 200)` | long se media 50 gg > media 200 gg | Seguire il trend |
| `mean_reversion(20, −1, 0)` | entra se z-score < −1, esce se > 0 | I prezzi "troppo bassi" tornano alla media |

Funzioni pandas usate:
- `prices.shift(k)`: il prezzo di k giorni fa sulla riga di oggi.
- `prices.rolling(n).mean()`: media degli ultimi n valori, **oggi incluso**. I primi n−1 sono `NaN`.
- `(x > 0).astype(float)`: True/False → 1.0/0.0. Attenzione: `NaN > 0` è `False`, ecco perché il
  warm-up diventa 0 da solo.

### Il trucco della mean reversion: `ffill`
La mean reversion ha **memoria**: tra soglia di entrata e di uscita si tiene la posizione che si
aveva. Invece di un ciclo `for`:
1. creo una serie vuota (`NaN`);
2. scrivo 1 solo nei giorni in cui si entra, 0 nei giorni in cui si esce;
3. `ffill()` (forward fill) copia in avanti l'ultimo valore non vuoto → ogni giorno "ricorda"
   l'ultima decisione.

### Il test anti look-ahead delle strategie
```python
full = strategy(prices)
assert strategy(prices.iloc[:cut]).equals(full.iloc[:cut])
```
Se tolgo il futuro, il passato non deve cambiare. Se una strategia usasse prezzi futuri (es.
`shift(-1)`, oppure normalizzare con la media di *tutta* la serie), i segnali sulla storia corta
sarebbero diversi e il test fallirebbe. Lo stesso test protegge backtest e walk-forward.

---

## Fase 4 — La macchina della verità (`validation.py`, `report.py`)

### In-sample / out-of-sample
`split_date` taglia **in ordine cronologico**: il primo 70% dei giorni è in-sample, il resto
out-of-sample. Mai a caso: mescolando i giorni, l'in-sample conterrebbe pezzi di futuro.

`optimize` prova ogni combinazione di parametri e sceglie quella con lo Sharpe migliore **solo
in-sample**. In più alla strategia vengono passati solo i prezzi *prima* della data di taglio, così
la scelta non può dipendere dal futuro nemmeno per errore (c'è un test con una strategia "spia").

Perché l'out-of-sample: se provo 100 combinazioni, la migliore in-sample è in parte *fortunata*.
L'out-of-sample è un esame su domande mai viste.

### Walk-forward
Un solo out-of-sample è un solo periodo, magari fortunato. Il walk-forward ripete l'esame:
```
[ train 3 anni ][ test 1 anno ]
      [ train 3 anni ][ test 1 anno ]
            [ train 3 anni ][ test 1 anno ] ...
```
Ogni anno si ri-sceglie il parametro sugli ultimi 3 anni e lo si usa per l'anno dopo. Mettendo in
fila tutti gli anni di test ottieni una storia di rendimenti in cui **ogni giorno** è stato
tradato con parametri scelti senza vederlo. È la simulazione più vicina a "cosa avrei fatto davvero".

Scelte fatte:
- Gli indicatori si calcolano su **tutta** la storia fino a quel giorno (non solo sulla finestra
  di train), altrimenti ogni finestra perderebbe il warm-up. È lecito solo perché le strategie non
  guardano il futuro: i test della Fase 3 lo garantiscono.
- Prima del primo anno di test la strategia è in liquidità. Il benchmark del walk-forward compra il
  primo giorno di test e paga lo stesso costo di entrata: confronto alla pari.

### Bootstrap a blocchi: fortuna o abilità?
Abbiamo **una sola** storia. La domanda è: se la storia fosse andata un po' diversamente, la
strategia batterebbe ancora buy & hold?

1. Prendo i rendimenti out-of-sample di strategia e benchmark **appaiati**: stesso giorno, stessa riga.
2. Costruisco una storia finta incollando **blocchi di 20 giorni consecutivi** presi a caso.
3. Calcolo Sharpe(strategia) − Sharpe(buy & hold) su quella storia.
4. Ripeto 1000 volte (seed 42: stesso risultato ogni volta) → distribuzione della differenza.

- **Perché blocchi e non giorni singoli?** I mercati hanno *volatility clustering* (lo vedi nel
  notebook 01): giorni agitati vicino a giorni agitati. Mescolando giorni singoli distruggeresti
  questa struttura.
- **Perché appaiati?** Strategia e benchmark vivono lo stesso mercato: confrontarli su giorni
  diversi aggiungerebbe rumore finto.
- Trucco NumPy in `block_bootstrap`: `starts[:, :, None] + np.arange(block_size)` usa il
  *broadcasting* per trasformare ogni inizio di blocco in tutti i suoi 20 giorni, senza cicli.

### Il verdetto
Si guarda l'intervallo al 95% della differenza di Sharpe:
- tutto sopra 0 → **BEATS**;
- tutto sotto 0 → **WORSE**;
- contiene lo 0 → **NO EVIDENCE**: la differenza è compatibile con la fortuna.

Il report aggiunge sempre degli avvisi: quante combinazioni sono state provate, un crollo dello
Sharpe tra in-sample e out-of-sample (sospetto overfitting), e il survivorship bias.

### Cosa abbiamo scoperto (notebook 02)
Su SPY e AAPL, dal 2015 al 2025, **nessuna** delle tre strategie batte buy & hold in modo
distinguibile dalla fortuna. In walk-forward perdono tutte, e la mean reversion su SPY è
nettamente peggiore. Non è un fallimento dello strumento: è esattamente il tipo di risposta onesta
per cui esiste.

---

## Domande di verifica
1. Un segnale calcolato con la chiusura di martedì: da che giorno genera rendimento? Perché?
2. Scrivi a mano `signals_to_positions([1, 1, 0, 1])`.
3. Passare da long (1) a short (−1) costa il doppio che entrare da 0 a 1. Perché?
4. Perché la volatilità si annualizza con √252 e il rendimento no?
5. Lo Sharpe del momentum su SPY out-of-sample è più alto di buy & hold, ma la strategia ha
   guadagnato meno. Come è possibile?
6. Perché `NaN > 0` è utile nel warm-up delle strategie?
7. Cosa fa `ffill()` nella mean reversion, e cosa succederebbe senza?
8. Perché lo split in-sample/out-of-sample non può essere casuale?
9. Cosa aggiunge il walk-forward rispetto a un singolo out-of-sample?
10. Perché il bootstrap usa blocchi di 20 giorni e non giorni singoli?
11. L'intervallo bootstrap è −0,47 .. +0,56. Cosa puoi concludere? Cosa **non** puoi concludere?
12. Se provi 1000 combinazioni di parametri invece di 4, cosa succede allo Sharpe in-sample della
    migliore? E all'out-of-sample?

## Esercizi da fare da solo
1. **Nuova metrica `calmar_ratio(returns)`** in `metrics.py` = rendimento annuo / |max drawdown|.
   Test: con rendimento annuo 10% e drawdown −20% deve dare 0,5. Caso limite: drawdown 0?
2. **Nuova strategia `breakout(prices, window=50)`**: long se il prezzo di oggi è il massimo
   degli ultimi `window` giorni, esce se è il minimo. Usa `rolling(...).max()` e il trucco di
   `ffill`. Aggiungila a `STRATEGIES` con i suoi `SMALL_PARAMS`: il test anti look-ahead
   parametrizzato la controllerà da solo.
3. **Rompi apposta il look-ahead**: in una copia di `signals_to_positions` togli lo `shift(1)` e
   lancia `pytest`. Quali test falliscono? Prevedilo **prima** di lanciarlo.
4. **Sensibilità ai costi**: nel notebook 02 rilancia `evaluate_strategy` con costi 0 e con costi
   triplicati. Quale strategia ne soffre di più? Perché? (Indizio: `turnover_per_year`.)
