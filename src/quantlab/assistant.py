"""A language model running on this computer (Ollama): a finance chat, and a plain-Italian explainer.

No API key and no cost: the model runs locally, and nothing leaves the machine. Two uses:
- the "Assistente" tab: free questions on finance, markets and quant methods (build_chat);
- the "Spiegamelo" panel: it rewords one result, using only the numbers the library already
  produced, passed as a list of facts (build_messages). The model never computes them.
If Ollama is not running (e.g. on Streamlit Cloud), the app simply says so instead of failing.
"""

import json
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

OLLAMA_URL = os.environ.get("QUANTLAB_OLLAMA_URL", "http://localhost:11434")
# Mixture of experts: 25B parameters of knowledge, but only ~4B work on each word, so it stays
# usable on a laptop CPU (18 GB on disk, ~9 GB of RAM while it answers). Smaller PC: QUANTLAB_MODEL=gemma3:4b
MODEL = os.environ.get("QUANTLAB_MODEL", "gemma4:26b")
CONTEXT = 8192  # tokens the model reads at once: the rules (~1,400) + the recent conversation
MAX_HISTORY = 10  # earlier messages are left out, so the rules always fit and reading stays quick

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
- Il test della fortuna rimescola la storia 1.000 volte: se l'idea vince in quasi tutti i
  rimescolamenti il vantaggio è solido; se vince quasi mai è davvero peggio del compra e tieni
  (un risultato chiaro, non incerto); se vince più o meno metà delle volte è fortuna.

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
- Di norma resta sotto le 150 parole: chi vuole approfondire te lo chiederà, e allora puoi
  scrivere di più.
- Non dare mai consigli di investimento personali: non dire a nessuno cosa comprare, vendere o
  quanto investire, nemmeno con un "no". Puoi spiegare come funzionano le cose e i rischi.
- Non hai internet: i soli prezzi che conosci sono quelli in "Mercati" qui sotto, se ci sono.
  Non inventare altri prezzi, notizie o dati di oggi. Se una cosa non la sai o non sei sicuro,
  dillo invece di indovinare.
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
{ideas}{market}{result}"""

MARKET_BLOCK = """

Mercati: ultime chiusure scaricate dall'app (dati Yahoo Finance). Riportale così come sono:
{market}"""

RESULT_BLOCK = """

L'ultimo risultato che l'utente ha sullo schermo (usa questi numeri così come sono, senza ricalcolarli):
{facts}"""


def build_chat(history: list[dict], question: str, ideas: dict[str, str],
               facts: dict[str, str] | None = None, market: dict[str, str] | None = None) -> list[dict]:
    """The free chat: rules + what Quantlab is (+ market prices and the result on screen, if any),
    then the conversation and the question."""
    system = CHAT_PROMPT.format(ideas=facts_text(ideas),
                                market=MARKET_BLOCK.format(market=facts_text(market)) if market else "",
                                result=RESULT_BLOCK.format(facts=facts_text(facts)) if facts else "")
    return [{"role": "system", "content": system}, *history[-MAX_HISTORY:], {"role": "user", "content": question}]


def save_chat(chat: list[dict], path: Path) -> None:
    """Keep the conversation on disk, so it is still there next time the app opens."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chat, ensure_ascii=False, indent=1), encoding="utf-8")


def load_chat(path: Path) -> list[dict]:
    """The saved conversation, or an empty one if there is none (or the file is broken)."""
    try:
        chat = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [m for m in chat if isinstance(m, dict) and {"role", "content"} <= m.keys()]

def is_available(url: str = OLLAMA_URL, model: str = MODEL, timeout: float = 0.5) -> bool:
    """True if Ollama answers at `url` and has `model` downloaded."""
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as response:
            names = [m["name"] for m in json.load(response).get("models", [])]
    except (OSError, ValueError):  # not installed / not running / not JSON
        return False
    return any(name == model or name.split(":")[0] == model for name in names)


def _chat_request(messages: list[dict], url: str, model: str, stream: bool, **options) -> urllib.request.Request:
    body = json.dumps({"model": model, "messages": messages, "stream": stream,
                       # no silent reasoning before answering: on a laptop CPU it more than doubles the wait
                       "think": False,
                       # stay in memory 30 min after the last question, so a study session loads it once
                       "keep_alive": "30m",
                       # low temperature: precise, not creative
                       "options": {"temperature": 0.3, "num_ctx": CONTEXT, **options}}).encode()
    return urllib.request.Request(f"{url}/api/chat", data=body, headers={"Content-Type": "application/json"})


_warming = threading.Lock()  # one warm-up at a time: Ollama answers one request at a time


def warm_up(messages: list[dict], url: str = OLLAMA_URL, model: str = MODEL) -> threading.Thread | None:
    """Load the model and read the rules in the background, while the user is still typing.

    Most of the wait before the first word is loading ~18 GB and reading the long system message.
    Ollama remembers the start of the last conversation it read, so a later question with the
    same system message starts almost immediately. If a warm-up is already running nothing new
    starts (returns None): extra ones would only queue up in front of the real question.
    Errors are ignored: it is only a head start.
    """
    if not _warming.acquire(blocking=False):
        return None

    def run() -> None:
        try:
            urllib.request.urlopen(_chat_request(messages, url, model, stream=False, num_predict=1),
                                   timeout=600).read()
        except OSError:
            pass
        finally:
            _warming.release()

    thread = threading.Thread(target=run, daemon=True)  # daemon: never keeps the app from closing
    thread.start()
    return thread

def stream_answer(messages: list[dict], url: str = OLLAMA_URL, model: str = MODEL,
                  timeout: float = 300) -> Iterator[str]:
    """Yield the answer piece by piece, as the model writes it (so the page fills while you read)."""
    request = _chat_request(messages, url, model, stream=True)
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
