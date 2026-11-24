"""Operator page permissions, typed catalog actions, and plain-text registry data."""

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from desman import operator_cli, ui
from desman.chat import Chat
from desman.config import Config, open_config
from desman.core import db
from desman.core.sessions import Sessions
from desman.gui.front import WebFront
from desman.gui.server import Server
from desman.gui.settings import command
from desman.net import operator, sign
from desman.net import protocol as p
from test_gui import headers

REF = "1234567890ab"
OFFER = {"acceptance": "Answer the question.", "inputs": ["NAME=some text"], "days": 3}


def get(app: Server, path: str) -> Any:
    response = app.handle("GET", path, headers(app))
    assert response.status == 200
    return json.loads(response.body)


def action(operation: str, args: dict[str, Any]) -> ui.SettingsAction:
    parsed = ui.ACTIONS.validate_python(
        {
            "type": "settings",
            "id": "operator-test",
            "change": {"section": "operator", "operation": operation, "args": args},
        }
    )
    assert isinstance(parsed, ui.SettingsAction)
    return parsed


def test_operator_data_with_and_without_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = Server(WebFront(), 12345, poll_timeout=0, database=tmp_path / "page.db")
    path = operator.operator_key_path()
    assert get(app, "/sessions")["operator_key"] is False
    assert get(app, "/settings")["operator_key"] is False
    assert get(app, "/operator") == {"has_key": False}
    assert app.handle("GET", "/operator", headers(app, Cookie="")).status == 403

    key = sign.NodeKey.generate()
    operator.install_key(key)
    open_config().save(Config.model_validate({"network": {"registry": "https://registry.test"}}))
    until = datetime(2026, 11, 26, tzinfo=UTC)
    monkeypatch.setattr(
        operator.OperatorClient,
        "whoami",
        lambda self: p.OperatorIdentity(
            fingerprint=key.fingerprint, scopes=[sign.Scope.TASKS], until=until
        ),
    )
    assert get(app, "/sessions")["operator_key"] is True
    assert get(app, "/settings")["operator_key"] is True
    assert get(app, "/operator") == {
        "has_key": True,
        "fingerprint": key.fingerprint,
        "public_key": sign.encode_public(key.public),
        "registry": "https://registry.test",
        "scopes": ["tasks"],
        "until": until.isoformat(),
    }
    assert path.read_text() not in str(get(app, "/operator"))
    assert not app.database.exists()


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        ("key import", {"file": "/some key.pem", "registry": None, "replace": True}),
        ("key create", {"registry": "https://registry.test"}),
        ("key remove", {}),
        ("job add", OFFER | {"description": "Write code.", "kind": "code", "runs_code": True}),
        *[("job settle", {"id": REF, "decision": word}) for word in ("accept", "again", "drop")],
        ("cluster add", OFFER | {"goal": "Solve it"}),
        ("cluster settle", {"id": REF, "decision": "again", "note": "Keep the result."}),
        *[("cluster settle", {"id": REF, "decision": word}) for word in ("accept", "drop")],
        ("complaint dismiss", {"id": REF}),
        ("suspend", {"node": REF, "reason": "Abuse", "days": 30, "complaints": [REF]}),
        ("lift", {"node": REF, "reason": "Resolved"}),
        ("owner", {"node": REF, "name": "Owner"}),
        ("owner", {"node": REF, "clear": True}),
    ],
)
def test_operator_action_routes_to_catalog(operation: str, args: dict[str, Any]) -> None:
    entry, parsed = command(action(operation, args).change, Config())
    assert entry.name == f"operator {operation}"
    assert parsed.model_dump(mode="json", include=set(args)) == args


@pytest.mark.parametrize("confirmed", [False, True])
def test_key_create_uses_window_and_reports_public_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confirmed: bool
) -> None:
    path = operator.operator_key_path()
    questions: list[str] = []

    def confirm(question: str, *, default: bool) -> bool:
        questions.append(question)
        assert not default
        assert not path.exists()
        return confirmed

    monkeypatch.setattr(ui, "confirm", confirm)
    conn = db.open_db(tmp_path / "page.db")
    front = WebFront()
    try:
        with ui.use(front):
            Chat(Sessions(conn))._settings_action(action("key create", {}))
        assert len(questions) == 1
        assert path.exists() is confirmed
        assert front.settings_outcome is not None
        assert front.settings_outcome["ok"] is confirmed
        if confirmed:
            key = operator_cli.current_key()
            assert key is not None
            public = sign.encode_public(key.public)
            assert f"desman-registry assign {public}" in front.settings_outcome["message"]
            assert "PRIVATE KEY" not in front.settings_outcome["message"]
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("view", "reason", "message"),
    [
        ("status", "not_operator", "This key has no current operator assignment."),
        ("complaints", "scope", "The operator assignment needs the moderation scope."),
        ("inbox", "already_settled", "This item has already been settled."),
        ("inbox", "ConnectError", "The registry could not be reached."),
    ],
)
def test_operator_read_errors_use_cli_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, view: str, reason: str, message: str
) -> None:
    operator.create_key()
    open_config().save(Config.model_validate({"network": {"registry": "https://registry.test"}}))

    def refused(*args: object, **kwargs: object) -> Any:
        raise operator.OperatorError(reason)

    method = "whoami" if view == "status" else view
    monkeypatch.setattr(operator.OperatorClient, method, refused)
    app = Server(WebFront(), 12345, poll_timeout=0, database=tmp_path / "page.db")
    assert get(app, f"/operator?view={view}")["error"].startswith(message)


def test_operator_ui() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node unavailable")
    subprocess.run(
        [node, "tests/js/gui_operator.cjs", "src/desman/gui/static/app.js"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
