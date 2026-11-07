"""The page reports through the catalog, retaining the open chat and terminal output."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from desman import global_cli, ui
from desman.chat import Chat
from desman.config import Network, open_config
from desman.core import db
from desman.core.sessions import Role, Sessions
from desman.net import protocol, sign
from helpers import make_config


@pytest.mark.parametrize("mode", ["off", "decline", "send", "secret", "empty"])
def test_report_action(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    config = make_config(api_keys={"GROQ_API_KEY": "private-key-marker"})
    if mode not in ("off", "empty"):
        config.network = Network(enabled=True, registry="https://registry.test")
        sign.node_key(sign.default_node_key_path())
    open_config().save(config)
    conn = db.open_db(":memory:")
    sessions = Sessions(conn)
    chat = Chat(sessions)
    session = sessions.add(None, Role.USER, "open chat").session_id
    chat.session_id = session
    description = (
        "private-key-marker" if mode == "secret" else "" if mode == "empty" else "It crashes"
    )
    monkeypatch.setattr(ui, "text", lambda question: description)
    previews: list[str] = []
    sent: list[Any] = []

    def confirmed(self: Chat, proposal: Any, *, in_prompt: bool = False) -> bool:
        assert in_prompt
        previews.append("\n".join(proposal.lines))
        return mode == "send"

    def send(*args: Any, **kwargs: Any) -> protocol.ReportReceived:
        sent.append(args[2])
        return protocol.ReportReceived(id="receipt", kind="bug")

    monkeypatch.setattr(Chat, "_confirmed", confirmed)
    monkeypatch.setattr(global_cli, "send_report", send)
    chat._action(ui.ACTIONS.validate_python({"type": "report_problem"}))
    assert chat.session_id == session and not chat.in_terminal
    assert len(sessions.messages(session)) == 1
    output = "\n".join(m.text for m in sessions.messages(sessions.terminal()))
    assert "private-key-marker" not in output
    if mode in ("send", "decline"):
        assert len(previews) == 1
        payload = json.loads(previews[0].split("\n", 1)[1])
        assert payload["signed"]["report"]["text"] == description
    else:
        assert not previews
    assert len(sent) == int(mode == "send")
    if mode == "off":
        assert "desman-global" in output
    if mode == "send":
        assert "receipt" in output
    conn.close()


def test_report_action_is_strict() -> None:
    with pytest.raises(ValidationError):
        ui.ACTIONS.validate_python({"type": "report_problem", "text": "untrusted"})


def test_report_uses_page_text_and_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from desman.gui.front import WebFront

    config = make_config()
    config.network = Network(enabled=True, registry="https://registry.test")
    open_config().save(config)
    sign.node_key(sign.default_node_key_path())
    conn = db.open_db(":memory:")
    chat = Chat(Sessions(conn))
    front = WebFront()
    prompts: list[dict[str, Any]] = []
    sent: list[Any] = []
    monkeypatch.setattr(global_cli, "send_report", lambda *a, **kw: sent.append(a))

    def answer() -> None:
        previous = None
        for value in ("Page problem", False):
            with front.condition:
                if not front.condition.wait_for(
                    lambda: (
                        front.prompt is not None
                        and (not prompts or front.prompt["id"] != prompts[-1]["id"])
                    ),
                    timeout=3,
                ):
                    return
                assert front.prompt is not None
                prompt = dict(front.prompt)
            prompts.append(prompt)
            previous = prompt["id"]
            front.answer({"id": previous, "value": value})

    thread = threading.Thread(target=answer, daemon=True)
    thread.start()
    with ui.use(front):
        chat._action(ui.ACTIONS.validate_python({"type": "report_problem"}))
    thread.join(3)
    assert not thread.is_alive()
    assert [p["kind"] for p in prompts] == ["text", "confirm"]
    assert "Page problem" in str(prompts[1])
    assert "registry.test" in str(prompts[1])
    assert not sent
    assert "Nothing changed" in chat.sessions.messages(chat.sessions.terminal())[-1].text
    conn.close()


def test_report_action_only_answers_an_open_line() -> None:
    import threading

    from desman.gui.front import WebFront
    from desman.gui.server import Server
    from test_gui import headers

    front = WebFront()
    app = Server(front, 12345, poll_timeout=0)
    body = b'{"type":"report_problem"}'
    assert app.handle("POST", "/action", headers(app), body).status == 409
    received: list[Any] = []
    thread = threading.Thread(target=lambda: received.append(front.line()))
    thread.start()
    with front.condition:
        assert front.condition.wait_for(lambda: front.prompt is not None, timeout=2)
    assert app.handle("POST", "/action", headers(app), body).status == 200
    thread.join(2)
    assert received == [ui.ACTIONS.validate_python({"type": "report_problem"})]
    assert app.handle("POST", "/action", headers(app), body).status == 409
