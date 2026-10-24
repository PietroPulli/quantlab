"""A language model running on this computer (Ollama): a finance chat, and a plain-Italian explainer.

No API key and no cost: the model runs locally, and nothing leaves the machine. Two uses:
- the "Assistente" tab: free questions on finance, markets and quant methods (build_chat);
- the "Spiegamelo" panel: it rewords one result, using only the numbers the library already
  produced, passed as a list of facts (build_messages). The model never computes them.
If Ollama is not running (e.g. on Streamlit Cloud), the app simply says so instead of failing.
"""

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterator

OLLAMA_URL = os.environ.get("QUANTLAB_OLLAMA_URL", "http://localhost:11434")
# Mixture of experts: 25B parameters of knowledge, but only ~4B work on each word, so it stays
# usable on a laptop CPU (18 GB on disk, ~9 GB of RAM while it answers). Smaller PC: QUANTLAB_MODEL=gemma3:4b
MODEL = os.environ.get("QUANTLAB_MODEL", "gemma4:26b")

SYSTEM_PROMPT = """Sei l'assistente di Quantlab, uno strumento che verifica onestamente le strategie \
di investimento contro il "compra e tieni" (buy & hold).

Regole, senza eccezioni:
- Rispondi in italiano semplice, a chi non ha studiato finanza. Al massimo 120 parole.
- Usa SOLO i numeri dei FATTI qui sotto, scritti esattamente così. Non fare calcoli e
  non ricavare numeri nuovi. Non inventare cifre, date o eventi di mercato.
- I FATTI riguardano tutto il periodo di prova insieme: il rendimento annuo è una media,
  non il risultato di un anno preciso. Se chiedono un anno, un mese o un dato che non è
  nei FATTI, rispondi che quel dato qui non c'è, senza stimarlo.
- Non dare mai consigli di investimento: non dire di comprare, vendere o usare la strategia.
  Se chiedono se comprare o vendere, non rispondere né sì né no: di' che Quantlab non dà
  consigli e che decidere spetta a loro.
  Spiega solo cosa sarebbe successo nel passato e quanto il risultato è affidabile.
- Ricorda che il passato non garantisce il futuro quando parli di guadagni.
- Quanto fidarsi lo dice il Verdetto nei FATTI: riportalo così com'è, senza cambiarlo.
  Solo se il Verdetto dice "compatibile con la fortuna", spiega che rimescolando la storia
  il vantaggio non resta stabilmente sopra zero, quindi non prova che l'idea funzioni.

FATTI:
{facts}"""

EXPLAIN = "Spiegami questo risultato: cosa è successo, cosa vuol dire e quanto posso fidarmi."


def facts_text(facts: dict[str, str]) -> str:
    """{"Rendimento annuo": "+7,2%"} -> '- Rendimento annuo: +7,2%' (one fact per line)."""
    return "\n".join(f"- {name}: {value}" for name, value in facts.items())


def build_messages(facts: dict[str, str], history: list[dict], question: str) -> list[dict]:
    """The chat sent to the model: rules + facts first, then the conversation so far, then the question."""
    system = {"role": "system", "content": SYSTEM_PROMPT.format(facts=facts_text(facts))}
    return [system, *history, {"role": "user", "content": question}]


CHAT_PROMPT = """Sei l'assistente di Quantlab, uno strumento per studiare finanza e ricerca quantitativa e per verificare onestamente le strategie di investimento contro il "compra e tieni" (buy & hold).

Di cosa parli: finanza, mercati, strumenti (azioni, ETF, obbligazioni, opzioni...), economia,
statistica e metodi quantitativi (backtest, rischio, rendimento, overfitting, machine learning
applicato ai mercati), Python per l'analisi dei dati finanziari, e come funziona Quantlab.
Se la domanda è su altro, dillo in una frase e riporta gentilmente il discorso sulla finanza.

Regole, senza eccezioni:
- Rispondi in italiano, chiaro e preciso, come un bravo professore a uno studente del primo anno.
  Vai dritto al punto; usa esempi numerici semplici quando aiutano, e dichiarali come esempi.
- Non dare mai consigli di investimento personali: non dire a nessuno cosa comprare, vendere o
  quanto investire, nemmeno con un "no". Puoi spiegare come funzionano le cose e i rischi.
- Non hai internet né prezzi aggiornati: non inventare prezzi, notizie o dati di oggi. Se una
  cosa non la sai o non sei sicuro, dillo invece di indovinare.
- Quando parli di guadagni passati, ricorda che il passato non garantisce il futuro.

Come funziona Quantlab:
- Ogni idea viene provata sui prezzi giornalieri, con commissioni (0,10%) e slippage (0,05%)
  sempre inclusi; fuori dal mercato i soldi rendono il 2% annuo.
- Niente look-ahead: un segnale calcolato con i dati di oggi agisce solo da domani.
- Il giudizio usa solo l'ultimo 30% della storia (out-of-sample), mai usato per scegliere niente.
- "Test della fortuna": la storia viene rimescolata 1.000 volte (bootstrap). Se l'idea batte il
  compra e tieni in quasi tutti i rimescolamenti il vantaggio è solido; se perde in quasi tutti
  è peggio, e non per sfortuna; se la differenza oscilla attorno a zero è "compatibile con la fortuna".
- Le idee pronte nell'app:
{ideas}{result}"""

RESULT_BLOCK = """

L'ultimo risultato che l'utente ha sullo schermo (usa questi numeri così come sono, senza ricalcolarli):
{facts}"""


def build_chat(history: list[dict], question: str, ideas: dict[str, str],
               facts: dict[str, str] | None = None) -> list[dict]:
    """The free chat: rules + what Quantlab is (+ the result on screen, if any), the conversation, the question."""
    result = RESULT_BLOCK.format(facts=facts_text(facts)) if facts else ""
    system = CHAT_PROMPT.format(ideas=facts_text(ideas), result=result)
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": question}]


def is_available(url: str = OLLAMA_URL, model: str = MODEL, timeout: float = 0.5) -> bool:
    """True if Ollama answers at `url` and has `model` downloaded."""
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as response:
            names = [m["name"] for m in json.load(response).get("models", [])]
    except (OSError, ValueError):  # not installed / not running / not JSON
        return False
    return any(name == model or name.split(":")[0] == model for name in names)


def stream_answer(messages: list[dict], url: str = OLLAMA_URL, model: str = MODEL,
                  timeout: float = 300) -> Iterator[str]:
    """Yield the answer piece by piece, as the model writes it (so the page fills while you read)."""
    body = json.dumps({"model": model, "messages": messages, "stream": True,
                       # no silent reasoning before answering: on a laptop CPU it more than doubles the wait
                       "think": False,
                       # stay in memory 30 min after the last question, so a study session loads it once
                       "keep_alive": "30m",
                       "options": {"temperature": 0.3}}).encode()  # low temperature: precise, not creative
    request = urllib.request.Request(f"{url}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:  # Ollama sends one JSON object per line
            if not line.strip():
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            yield chunk.get("message", {}).get("content", "")
            if chunk.get("done"):
                return
