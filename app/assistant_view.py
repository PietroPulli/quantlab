"""The local AI (Ollama) in the app: the "Assistente" tab (free finance chat) and the "Spiegamelo" panel."""

import os
from pathlib import Path

import streamlit as st

from common import quotes
from panels import num, pct
from quantlab import assistant
from quantlab.ideas import IDEAS

EXAMPLES = ["Cos'è lo Sharpe ratio e perché conta più del rendimento?",
            "Che cos'è l'overfitting in un backtest?",
            "Che differenza c'è tra un ETF e un'azione?",
            "Perché il compra e tieni è così difficile da battere?"]


def _chat_file() -> Path:
    """Where the Assistente chat is kept between sessions (data/ stays on this PC, never on GitHub)."""
    default = Path(__file__).resolve().parents[1] / "data" / "assistant_chat.json"
    return Path(os.environ.get("QUANTLAB_CHAT_FILE", default))


def _warm_up(system: list[dict]) -> None:
    """Start loading the model and reading these rules now, once per set of rules."""
    if st.session_state.get("warmed") != system[0]["content"]:
        st.session_state.warmed = system[0]["content"]
        assistant.warm_up(system)


def _market() -> dict[str, str]:
    """The prices the app already shows (watchlist and markets), as facts the AI may quote."""
    import simple  # here, not at the top: simple imports this module

    listed = tuple((simple.short(n), t) for n, t in simple.ASSETS.items() if t) + simple.MARKETS
    return {f"{q['name']} ({q['ticker']})": f"{num(q['last'], 2)}, {pct(q['change'], True)} nell'ultimo giorno"
            for q in quotes(listed)}


@st.cache_data(ttl=30, show_spinner=False)  # ask Ollama at most every 30 s, not on every click
def _available() -> bool:
    return assistant.is_available()


def explain_panel(facts: dict[str, str], key: str) -> None:
    """Chat about one result. `key` identifies the result: a new result starts a new chat."""
    st.markdown("##### Spiegamelo · AI locale")
    if not _available():
        st.caption(f"L'assistente gira solo sul tuo computer, gratis e senza chiavi: installa Ollama "
                   f"(ollama.com), poi `ollama pull {assistant.MODEL}` e avvia l'app in locale.")
        return

    st.session_state.last_facts = facts  # the Assistente tab can talk about it too
    if st.session_state.get("chat_key") != key:
        st.session_state.chat_key, st.session_state.chat = key, []
    chat = st.session_state.chat
    for message in chat:
        st.chat_message(message["role"]).markdown(message["content"])

    question = None
    if not chat and st.button("Spiegami questo risultato", key="chat_explain"):
        question = assistant.EXPLAIN
    asked = st.chat_input("Chiedi qualcosa su questo risultato", key="chat_input")
    question = asked or question
    if not question:
        st.caption(f"Modello {assistant.MODEL} sul tuo PC: usa solo i numeri qui sopra, non dà consigli.")
        _warm_up(assistant.build_messages(facts, [], "")[:1])
        return

    st.chat_message("user").markdown(question)
    try:
        answer = st.chat_message("assistant").write_stream(
            assistant.stream_answer(assistant.build_messages(facts, chat, question)))
    except Exception as exc:  # Ollama stopped, model removed, timeout
        st.error(f"L'assistente non risponde: {exc}")
        return
    chat += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def _not_available() -> None:
    st.info(f"L'assistente gira solo sul tuo computer, gratis e senza chiavi. Per usarlo: installa Ollama "
            f"(ollama.com), scarica il modello con `ollama pull {assistant.MODEL}` e avvia l'app in locale "
            "con `streamlit run app/streamlit_app.py`.")


def render() -> None:
    """The Assistente tab: a chat on finance, markets and quant methods, like a tutor."""
    st.markdown("##### Assistente · AI locale")
    if not _available():
        _not_available()
        return

    if "assistant_chat" not in st.session_state:
        st.session_state.assistant_chat = assistant.load_chat(_chat_file())  # pick up where you left off
    chat = st.session_state.assistant_chat
    facts = st.session_state.get("last_facts")
    left, mid, right = st.columns([2, 2, 1])
    use_result = left.toggle("Parla dell'ultimo risultato verificato", value=True, key="assistant_use_result",
                             disabled=facts is None, help="Fai prima «Verifica l'idea» nella vista Semplice.")
    use_market = mid.toggle("Dagli i prezzi di oggi della watchlist", value=True, key="assistant_use_market",
                            help="Le ultime chiusure che l'app ha già scaricato da Yahoo Finance.")
    if right.button("Nuova chat", key="assistant_clear", disabled=not chat, width="stretch"):
        chat.clear()
        assistant.save_chat(chat, _chat_file())

    ideas = {name: idea.explanation for name, idea in IDEAS.items()}
    market = _market() if use_market else None
    facts = facts if use_result else None
    for message in chat:
        st.chat_message(message["role"]).markdown(message["content"])

    question = st.chat_input("Chiedi qualcosa su finanza, mercati, metodi quant o Quantlab",
                             key="assistant_input")
    if not chat and not question:
        st.caption("Qualche idea per cominciare:")
        for i, (column, example) in enumerate(zip(st.columns(2) * 2, EXAMPLES)):
            if column.button(example, key=f"assistant_example_{i}", width="stretch"):
                question = example
        st.caption(f"Modello {assistant.MODEL} sul tuo PC. Non ha internet: conosce solo i prezzi della "
                   "watchlist. Non dà consigli di investimento e può sbagliare: verifica le cose importanti.")
    if not question:
        # The model starts loading now, while you read and type: the first answer comes much sooner.
        _warm_up(assistant.build_chat([], "", ideas, facts, market)[:1])
        return

    st.chat_message("user").markdown(question)
    messages = assistant.build_chat(chat, question, ideas, facts, market)
    try:
        with st.spinner("Sto pensando..."):  # the first answer may still wait for the model to load
            answer = st.chat_message("assistant").write_stream(assistant.stream_answer(messages))
    except Exception as exc:  # Ollama stopped, model removed, timeout
        st.error(f"L'assistente non risponde: {exc}")
        return
    chat += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    assistant.save_chat(chat, _chat_file())
