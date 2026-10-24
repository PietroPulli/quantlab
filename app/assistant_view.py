"""The local AI (Ollama) in the app: the "Assistente" tab (free finance chat) and the "Spiegamelo" panel."""

import streamlit as st

from quantlab import assistant
from quantlab.ideas import IDEAS

EXAMPLES = ["Cos'è lo Sharpe ratio e perché conta più del rendimento?",
            "Che cos'è l'overfitting in un backtest?",
            "Che differenza c'è tra un ETF e un'azione?",
            "Perché il compra e tieni è così difficile da battere?"]


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

    chat = st.session_state.setdefault("assistant_chat", [])
    facts = st.session_state.get("last_facts")
    left, right = st.columns([3, 1])
    use_result = left.toggle("Parla anche dell'ultimo risultato che ho verificato", value=True,
                             key="assistant_use_result", disabled=facts is None,
                             help="Fai prima «Verifica l'idea» nella vista Semplice.")
    if right.button("Nuova chat", key="assistant_clear", disabled=not chat, width="stretch"):
        chat.clear()

    for message in chat:
        st.chat_message(message["role"]).markdown(message["content"])

    question = st.chat_input("Chiedi qualcosa su finanza, mercati, metodi quant o Quantlab",
                             key="assistant_input")
    if not chat and not question:
        st.caption("Qualche idea per cominciare:")
        for i, (column, example) in enumerate(zip(st.columns(2) * 2, EXAMPLES)):
            if column.button(example, key=f"assistant_example_{i}", width="stretch"):
                question = example
        st.caption(f"Modello {assistant.MODEL} sul tuo PC. Non ha internet: niente prezzi o notizie di oggi. "
                   "Non dà consigli di investimento e può sbagliare: verifica le cose importanti.")
    if not question:
        return

    st.chat_message("user").markdown(question)
    ideas = {name: idea.explanation for name, idea in IDEAS.items()}
    messages = assistant.build_chat(chat, question, ideas, facts if use_result else None)
    try:
        with st.spinner("Sto pensando..."):  # the first answer also loads the model: can take a minute
            answer = st.chat_message("assistant").write_stream(assistant.stream_answer(messages))
    except Exception as exc:  # Ollama stopped, model removed, timeout
        st.error(f"L'assistente non risponde: {exc}")
        return
    chat += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
