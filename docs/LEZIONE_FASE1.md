# Lezione Fase 1 — Dati e rendimenti

> Aggiornamento: ora `pytest` gira e tutti i test passano. L'unico test che falliva
> (`test_suspicious_jump_is_flagged`) ha portato ad abbassare la soglia dei salti al 25%.
> La prova con dati veri di SPY e AAPL e il notebook `01_esplora_dati.ipynb` sono fatti.

## 1. Concetti

### Prezzi aggiustati
Il prezzo di chiusura "grezzo" cambia per motivi che non sono guadagni o perdite. Se un'azione
fa uno **split 2 per 1**, il prezzo si dimezza da un giorno all'altro, ma chi la possiede ha il
doppio delle azioni: non ha perso nulla. Anche i **dividendi** fanno scendere il prezzo il giorno
dello stacco, ma il denaro finisce a chi possiede il titolo. I prezzi *aggiustati* correggono
tutto il passato per tenerne conto. Senza aggiustamento, un backtest vede un "crollo del 50%"
inesistente e la strategia reagisce a un fantasma. Per questo usiamo `auto_adjust=True`.

### Rendimenti semplici vs logaritmici
- Semplice: `r = P_t / P_{t-1} - 1`. Si **aggrega tra titoli** (il rendimento di un portafoglio è
  la media pesata dei rendimenti semplici).
- Logaritmico: `r = ln(P_t / P_{t-1})`. Si **somma nel tempo** (il log-rendimento di un mese è la
  somma di quelli giornalieri) ed è più comodo per la statistica.
- Legame: `ln(1 + r_semplice) = r_log`. Per rendimenti piccoli sono quasi uguali.
- Il primo giorno è `NaN`: non esiste un prezzo precedente.

### Survivorship bias
Se scarico oggi i titoli dell'indice e li testo sugli ultimi 20 anni, ho solo le aziende
**sopravvissute**. Quelle fallite o uscite dall'indice non ci sono, e i risultati vengono
gonfiati. Yahoo Finance non ce lo risolve: è un limite da dichiarare in ogni report.

### Cache
Salviamo su disco (parquet) quello che scarichiamo: **velocità**, **riproducibilità** (stessi
dati oggi e fra un mese, anche se Yahoo cambia qualcosa) e **indipendenza da internet**.

## 2. Il codice, riga per riga

### `returns.py`

```python
def simple_returns(prices):
    return prices.pct_change(fill_method=None)
```
`pct_change` calcola `P_t / P_{t-1} - 1` su ogni colonna. `fill_method=None` vieta a pandas di
riempire in silenzio i buchi (NaN) con il valore precedente: un buco resta visibile.

```python
def log_returns(prices):
    return np.log(prices / prices.shift(1))
```
`shift(1)` sposta la serie di un giorno in giù, così ogni riga vede il prezzo di ieri.
Dividiamo, poi `np.log` applica il logaritmo naturale a ogni elemento.

### `data.py`

Costanti: `MAX_DAILY_JUMP = 0.25` (25%, abbastanza bassa da catturare anche uno split 3:2, che vale −33%) e `MAX_DATE_GAP_DAYS = 7` sono le soglie di sospetto.
Sono scelte mie: si possono cambiare.

**`download_prices`**
1. `import yfinance` è *dentro* la funzione: i test non caricano lo strato di rete.
2. `yf.download(..., auto_adjust=True, progress=False)` scarica prezzi già aggiustati.
3. Se il risultato è vuoto solleviamo `ValueError`: meglio fallire subito che lavorare su niente.
4. `raw["Close"]` prende solo la chiusura (già aggiustata).
5. Se è una `Series` (un solo ticker in certe versioni) la trasformiamo in DataFrame.
6. `prices[list(tickers)]` rimette le colonne nell'ordine richiesto.

**`_cache_path`**
1. Costruisce una chiave testuale: tickers **ordinati**, start, end. L'ordinamento fa sì che
   `["A","B"]` e `["B","A"]` usino lo stesso file.
2. `hashlib.sha1(...)[:10]` la trasforma in un nome breve e sicuro per il file system.
3. Restituisce `prices_<hash>.parquet` dentro `cache_dir`.

**`load_prices`**
1. Calcola il percorso di cache. 2. Se il file esiste, `read_parquet` e ritorna.
3. Altrimenti scarica, `mkdir(parents=True, exist_ok=True)` crea la cartella, `to_parquet` salva.

**`validate_prices`** — non corregge nulla, *segnala* i problemi come lista di stringhe (lista
vuota = tutto ok).
1. `df.index.has_duplicates`: date duplicate.
2. Per ogni colonna: `isna().sum()` conta i NaN; `(series <= 0).sum()` conta i prezzi non validi.
3. Salti: prendiamo solo i prezzi positivi, `pct_change().abs()` e contiamo quelli oltre il 25%.
4. Buchi nelle date: `index.diff().dt.days` è la distanza in giorni tra righe consecutive;
   oltre 7 giorni di calendario è un buco (un weekend lungo o una festa non lo fanno scattare).

## 3. Scelte che ho fatto (dovevo decidere da solo)
- **"Buco nei dati"** = sia valore mancante (NaN) sia salto di date > 7 giorni. Il testo della
  specifica non lo distingueva.
- **Nome del file di cache** con hash di (tickers ordinati, start, end): semplice, ma cambiare
  anche solo la data di fine crea un nuovo file.
- **Nessuna pulizia automatica**: `validate_prices` segnala soltanto. Decidere cosa fare dei dati
  sporchi è una scelta di ricerca, non da nascondere in una funzione.
- **`fill_method=None`** nei rendimenti, per non nascondere i buchi.
- **Test con `monkeypatch`** per simulare il download e per verificare che la seconda chiamata
  usi la cache, senza internet.
- Non ho creato il notebook `01_esplora_dati.ipynb`: è richiesto dalla specifica ma non da te
  stanotte, e richiede di guardare grafici. Resta da fare insieme.

## 4. Domande di verifica
1. Perché uno split 2 per 1 sembra un crollo del 50% nei prezzi non aggiustati?
2. Cosa fa esattamente `auto_adjust=True`?
3. Quale dei due rendimenti si somma nel tempo e quale si aggrega tra titoli? Perché?
4. Perché il primo valore di `simple_returns` è `NaN`?
5. Cosa restituirebbe `log_returns` per una serie costante? Perché?
6. Cos'è il survivorship bias e in che direzione distorce i risultati?
7. Perché `["A","B"]` e `["B","A"]` devono dare lo stesso file di cache?
8. Perché `import yfinance` è dentro `download_prices`?
9. Cosa cambia se togliamo `fill_method=None` e nei dati c'è un NaN?
10. Perché `validate_prices` restituisce una lista invece di sollevare subito un errore?

## 5. Esercizi da fare da solo
1. **`cumulative_returns(returns)`** in `returns.py`: da rendimenti semplici alla curva
   `(1 + r).cumprod() - 1`. Scrivi il test: rendimenti `[0.1, 0.1]` danno `[0.1, 0.21]`.
2. **`annualized_volatility(returns, periods=252)`**: deviazione standard × `sqrt(periods)`.
   Test: una serie costante ha volatilità 0. (Ora esiste già in `metrics.py`: scrivila da solo
   in un file a parte e poi confronta con la versione esistente.)
3. **Un nuovo controllo in `validate_prices`**: segnala i giorni con volume o prezzo *identico*
   per più di 5 giorni di fila (dato "congelato"). Scrivi prima il test che deve fallire.
