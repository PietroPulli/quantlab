"""Tests for quantlab.assistant (no model needed: Ollama is replaced by a fake answer)."""

import io
import json

from quantlab import assistant

FACTS = {"Titolo": "SPY", "Rendimento annuo strategia": "+5,1%", "Rendimento annuo compra e tieni": "+9,8%"}


def test_facts_reach_the_model_in_the_system_message():
    messages = assistant.build_messages(FACTS, [], "Perché ha perso?")
    assert messages[0]["role"] == "system"
    assert "- Rendimento annuo strategia: +5,1%" in messages[0]["content"]
    assert "Non dare mai consigli di investimento" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "Perché ha perso?"}


def test_history_sits_between_the_rules_and_the_new_question():
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    messages = assistant.build_messages(FACTS, history, "c")
    assert [m["content"] for m in messages[1:]] == ["a", "b", "c"]


def test_not_available_when_nothing_listens():
    assert not assistant.is_available("http://127.0.0.1:9", timeout=0.2)  # port 9: nobody there


def _fake_urlopen(payload: bytes):
    def urlopen(*args, **kwargs):
        return io.BytesIO(payload)  # supports `with` and iterating line by line, like a response
    return urlopen


def test_available_only_if_the_model_is_downloaded(monkeypatch):
    tags = json.dumps({"models": [{"name": "gemma3:4b"}]}).encode()
    monkeypatch.setattr(assistant.urllib.request, "urlopen", _fake_urlopen(tags))
    assert assistant.is_available(model="gemma3:4b")
    assert assistant.is_available(model="gemma3")
    assert not assistant.is_available(model="llama3.2:3b")


def test_stream_joins_the_pieces_and_stops_when_done(monkeypatch):
    lines = [{"message": {"content": "Ciao"}}, {"message": {"content": " Pietro"}},
             {"message": {"content": ""}, "done": True}, {"message": {"content": "mai letto"}}]
    payload = b"\n".join(json.dumps(x).encode() for x in lines)
    monkeypatch.setattr(assistant.urllib.request, "urlopen", _fake_urlopen(payload))
    assert "".join(assistant.stream_answer([])) == "Ciao Pietro"


IDEAS = {"Segui la tendenza": "Investito sopra la media a 200 giorni."}


def test_chat_knows_the_rules_and_the_ideas():
    messages = assistant.build_chat([], "Cos'è lo Sharpe ratio?", IDEAS)
    system = messages[0]["content"]
    assert "- Segui la tendenza: Investito sopra la media a 200 giorni." in system
    assert "Non dare mai consigli di investimento personali" in system
    assert "ultimo risultato" not in system  # no result on screen, no result block
    assert messages[-1] == {"role": "user", "content": "Cos'è lo Sharpe ratio?"}


def test_chat_sees_the_result_on_screen_when_there_is_one():
    system = assistant.build_chat([], "E questo?", IDEAS, facts=FACTS)[0]["content"]
    assert "ultimo risultato" in system
    assert "- Rendimento annuo strategia: +5,1%" in system


def test_chat_keeps_the_conversation_in_order():
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    messages = assistant.build_chat(history, "c", IDEAS)
    assert [m["content"] for m in messages[1:]] == ["a", "b", "c"]
