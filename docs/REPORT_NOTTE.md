# Report della notte

## Cosa ho fatto
- Creato il branch `notte-fase1` (main non toccato, nessun push).
- `src/quantlab/returns.py` + `tests/test_returns.py` → commit `cb0b8e3`.
- `src/quantlab/data.py` + `tests/test_data.py` → commit `d10bf9c`. I test non usano internet
  (DataFrame costruiti a mano, `monkeypatch` per il download).
- `docs/LEZIONE_FASE1.md` (spiegazioni, 10 domande, 3 esercizi) e questo report.

## Cosa non ha funzionato
- **Non ho potuto verificare niente con pytest.** Al primo test importato, pandas fallisce con:
  `ImportError: DLL load failed while importing json: Un criterio di controllo dell'applicazione
  ha bloccato il file.` È una policy di Windows (controllo applicazioni, tipo WDAC/AppLocker/
  Smart App Control) che blocca `pandas/_libs/json` nel `.venv`. Il test di prova
  `test_smoke.py` invece passa (non importa pandas).
- Non ho tentato di aggirarla (reinstallare pandas, sbloccare file, usare altri Python): è un
  blocco di sicurezza della tua macchina e non spettava a me decidere.
- Quindi **i commit sono marcati "NOT VERIFIED"**: il codice è scritto con cura ma mai eseguito.
  Può contenere errori (es. differenze di versione di yfinance sulla forma del DataFrame).
- Non ho testato `load_prices(["SPY","AAPL"], ...)` dal vivo, né creato il notebook
  `01_esplora_dati.ipynb`. Nessun software installato.
- `notte.log` (file non tracciato già presente) non è stato toccato né committato.

## Cosa resta da decidere / fare
1. Capire il blocco: guarda in Sicurezza di Windows → Controllo app e browser, o se il
   `.venv` è in una cartella "protetta"; prova a ricreare il `.venv`. Poi `pytest` e correggi.
2. Se pytest fallisce su qualche test, correggerlo *insieme* (è materiale didattico).
3. Confermare le mie scelte (vedi sezione 3 della lezione): definizione di "buco", soglie 50% e
   7 giorni, naming della cache.
4. Fare il notebook `01_esplora_dati.ipynb` e la prova dal vivo con SPY e AAPL.
5. Rivedere `data.py` con me prima di fare merge su main: è tua responsabilità saperlo difendere.
