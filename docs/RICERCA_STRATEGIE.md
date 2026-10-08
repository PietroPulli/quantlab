# Ricerca: un metodo migliore del buy & hold? (pre-registro)

Scritto e committato **prima** di vedere qualsiasi risultato, il 9 ottobre 2026.
Regole e parametri sono quelli pubblicati dagli autori: non si cambiano dopo aver visto i numeri.
Ogni variazione provata dopo va dichiarata come tale e non può diventare "la strategia".

## Cosa vuol dire "meglio del buy & hold"
Il benchmark è il buy & hold di SPY (azioni USA, dividendi inclusi).
Una candidata **vince** solo se:
1. il suo **Sharpe** è più alto e l'intervallo bootstrap della differenza sta tutto sopra zero
   (livello corretto con Bonferroni per 3 candidate: 98,3%);
2. la sua **perdita massima** è minore.

Il rendimento annuo viene sempre riportato, anche quando è più basso.

## Regole comuni
- Decisioni a **fine mese** con i soli dati di quel giorno; si opera dal giorno dopo.
- Costi: 0,10% di commissione + 0,05% di slippage sul valore scambiato.
- La liquidità rende il tasso vero dei T-bill a 3 mesi (`^IRX`), giorno per giorno.
- Nessuna leva: pesi tra 0 e 1, somma al massimo 1.
- Tra un ribilanciamento e l'altro i pesi si muovono con i prezzi (nessun ribilanciamento gratis).
- Periodo comune: da gennaio 2007, il primo mese in cui tutte e tre le candidate possono decidere,
  a settembre 2026. A parte si mostra anche il periodo **dopo la pubblicazione** di ciascuna,
  che è un vero out-of-sample.

## Le tre candidate

### A. Volatility targeting (Moreira & Muir, *Journal of Finance*, 2017)
Idea: quando il mercato è agitato, le perdite grandi sono più probabili e il rendimento atteso
non sale abbastanza da compensarle. Quindi si investe meno quando la volatilità è alta.
- Peso in SPY = min(1, 15% / volatilità annualizzata degli ultimi 21 giorni). Il resto è liquidità.
- 15% è un obiettivo tondo, sotto la volatilità storica di SPY (circa 19%), fissato senza guardare i risultati.
- Dopo la pubblicazione: 2017–2026.

### B. Tactical asset allocation (Faber, *Journal of Wealth Management*, 2007)
Idea: diversificare su 5 classi di asset e uscire da ciascuna quando è in tendenza negativa.
- 5 asset al 20%: azioni USA (SPY), azioni estere (EFA), titoli di Stato USA a 10 anni (IEF),
  immobiliare (VNQ), materie prime (DBC).
- Ogni asset si tiene se il prezzo di fine mese è sopra la media degli ultimi 10 fine mese;
  altrimenti il suo 20% va in liquidità.
- Dopo la pubblicazione: 2007–2026, cioè quasi tutto il periodo.

### C. Global equities momentum, "dual momentum" (Antonacci, 2014)
Idea: stare sull'azionario che sale di più, ma passare alle obbligazioni quando le azioni rendono meno della liquidità.
- Se il rendimento di SPY negli ultimi 12 mesi supera quello dei T-bill: 100% nel migliore
  tra SPY ed EFA (rendimento a 12 mesi).
- Altrimenti: 100% in obbligazioni (AGG).
- Dopo la pubblicazione: 2014–2026.

## Solo riferimenti (non candidate, non contano come tentativi)
- 60/40: 60% SPY e 40% IEF, ribilanciato ogni mese.
- I 5 asset di Faber al 20% senza regola di uscita: serve a capire se il merito è della regola o solo della diversificazione.

## Controlli di robustezza (dopo, dichiarati)
Costi raddoppiati; parametri vicini (volatilità obiettivo 10% e 20%; media a 8 e 12 mesi; momentum a 6 e 9 mesi).
Servono solo a vedere se il risultato è fragile, **non** a scegliere i parametri migliori.
