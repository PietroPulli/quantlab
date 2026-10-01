# QuantLab

**La macchina della verità per le strategie di investimento.**
Prende una strategia e mostra onestamente se regge: dopo i costi, su dati mai visti,
e al netto della fortuna.

> Progetto in sviluppo. Non fornisce consigli di investimento.

## Avvio rapido
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## L'app
Scegli strategia, titolo, parametri e costi; premi un bottone e ottieni il verdetto contro buy & hold.
```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```
Si apre nel browser su http://localhost:8501.

## Documentazione
- [Visione](docs/VISIONE.md)
- [Roadmap](docs/ROADMAP.md)
- [Fase 0 e 1 — Setup e Dati](docs/FASE_0_1.md)
