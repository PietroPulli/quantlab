# Fase 0 e Fase 1 — Setup e Dati

## Fase 0 — Setup (settimana 1)

### Cosa devo avere alla fine
- [ ] Python 3.11+ installato, ambiente virtuale creato (`python3 -m venv .venv`)
- [ ] Dipendenze installate: `pip install -e ".[dev]"`
- [ ] `pytest` eseguito con successo (il test di prova passa)
- [ ] Repository su GitHub (privato all'inizio, pubblico quando l'MVP è pronto)
- [ ] Ho capito: cos'è un ambiente virtuale, cos'è un commit, cos'è un test

### Concetti da capire (chiedi a Claude di spiegarli)
- Differenza tra `src/` e `tests/`, perché si separa il codice dai test
- Cos'è `pyproject.toml`
- Comandi Git base: `status`, `add`, `commit`, `push`

---

## Fase 1 — Dati (settimane 2–3)

### Obiettivo
Una funzione `load_prices(tickers, start, end)` che restituisce un DataFrame di prezzi
di chiusura **aggiustati** (per split e dividendi), puliti e salvati in cache locale.

### Moduli
`src/quantlab/data.py`
- `download_prices(tickers, start, end) -> pd.DataFrame` — scarica da Yahoo Finance (`yfinance`)
- `load_prices(tickers, start, end, cache_dir="data/") -> pd.DataFrame` — usa la cache se presente,
  altrimenti scarica e salva (formato parquet)
- `validate_prices(df) -> list[str]` — restituisce i problemi trovati:
  valori mancanti, prezzi ≤ 0, salti giornalieri sospetti (es. > 50%), date duplicate

`src/quantlab/returns.py`
- `simple_returns(prices)` e `log_returns(prices)`

### Test richiesti (`tests/test_data.py`, `tests/test_returns.py`)
- I rendimenti di una serie costante sono tutti zero
- Rendimenti semplici e logaritmici sono coerenti: `log(1 + r_simple) == r_log`
- `validate_prices` segnala un prezzo negativo inserito apposta
- `validate_prices` segnala un buco nei dati
- I test **non** devono scaricare dati da internet: usare DataFrame costruiti a mano

### Concetti da capire
- Perché si usano i prezzi aggiustati e cosa succede se non lo fai (uno split sembra un crollo del 50%)
- Rendimenti semplici vs logaritmici: quando usare quali
- **Survivorship bias:** se testo solo le aziende che esistono oggi, ignoro quelle fallite → risultati gonfiati
- Perché la cache: velocità, riproducibilità, non dipendere da internet

### Nota sui dati
`yfinance` non è un'API ufficiale: va bene per uso personale e studio. Se il progetto diventa
un prodotto, servirà un fornitore di dati con licenza.

### Fase completata quando
- [ ] `load_prices(["SPY", "AAPL"], "2015-01-01", "2025-12-31")` funziona e usa la cache al secondo avvio
- [ ] Tutti i test passano
- [ ] Un notebook `notebooks/01_esplora_dati.ipynb` con un grafico dei prezzi e dei rendimenti
- [ ] So spiegare a voce cos'è il survivorship bias
