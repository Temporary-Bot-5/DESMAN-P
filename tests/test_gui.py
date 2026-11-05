"""The browser transports the existing UI; it never decides whether to apply a change."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.client import HTTPConnection
from typing import Any

import click
import pytest
from click.testing import CliRunner
from rich.text import Text

from desman import i18n, runs, ui
from desman.chat import Chat
from desman.cli import cli
from desman.config import open_config
from desman.core import db
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.sessions import Sessions
from desman.core.store import Store
from desman.gui.front import WebFront
from desman.gui.server import MAX_BODY, SECURITY, Server, http_server, strings
from desman.interlocutor import CHAT
from desman.live import Progress
from desman.llm.fake import FakeLLM
from helpers import make_config


@pytest.fixture
def app() -> Server:
    return Server(WebFront(), 12345, poll_timeout=0)


def headers(app: Server, **extra: str) -> dict[str, str]:
    return {
        "Host": f"127.0.0.1:{app.port}",
        "Cookie": f"desman={app.cookie}",
        "Origin": f"http://127.0.0.1:{app.port}",
        "Content-Type": "application/json",
        **extra,
    }


def test_token_becomes_private_cookie_and_redirect(app: Server) -> None:
    result = app.handle("GET", "/?token=" + app.token, {"Host": "localhost:12345"})
    assert result.status == 303 and result.headers["Location"] == "/"
    assert result.headers["Set-Cookie"] == f"desman={app.cookie}; HttpOnly; SameSite=Strict; Path=/"
    assert app.token not in result.headers["Set-Cookie"]
    assert app.handle("GET", "/?token=wrong", headers(app)).status == 403
    assert app.handle("GET", "/?token=日本", headers(app)).status == 403


@pytest.mark.parametrize(
    "host", ["evil.com", "127.0.0.1.evil.com", "127.0.0.1", "localhost:9999", "[::1]:12345"]
)
def test_host_rejected_even_for_token(app: Server, host: str) -> None:
    for path in ("/", "/?token=" + app.token):
        assert app.handle("GET", path, headers(app, Host=host)).status == 403


@pytest.mark.parametrize("cookie", ["", "desman=wrong", "other=value"])
def test_every_route_requires_cookie(app: Server, cookie: str) -> None:
    for path in (
        "/",
        "/app.js",
        "/app.css",
        "/strings.json",
        "/events?after=0",
        "/answer",
        "/stop",
        "/missing",
    ):
        assert app.handle("GET", path, headers(app, Cookie=cookie)).status == 403


@pytest.mark.parametrize(
    "origin", ["", "http://evil.com", "http://localhost:12345", "https://127.0.0.1:12345"]
)
def test_post_origin_is_exact(app: Server, origin: str) -> None:
    assert app.handle("POST", "/stop", headers(app, Origin=origin), b"{}").status == 403


def test_post_validation_and_unknown_routes(app: Server) -> None:
    assert (
        app.handle("POST", "/stop", headers(app, **{"Content-Type": "text/plain"}), b"{}").status
        == 415
    )
    assert app.handle("POST", "/stop", headers(app), b" " * (MAX_BODY + 1)).status == 413
    for body in (b"oops", b"[]", b"null", b"\xff"):
        assert app.handle("POST", "/answer", headers(app), body).status == 400
    assert app.handle("POST", "/answer", headers(app), b'{"id":1,"value":true}').status == 400
    assert app.handle("POST", "/stop", headers(app), b"{}").status == 200
    assert app.handle("POST", "/missing", headers(app), b"{}").status == 404
    assert app.handle("GET", "/missing", headers(app)).status == 404
    for after in ("-1", "bad"):
        assert app.handle("GET", "/events?after=" + after, headers(app)).status == 400


def test_security_headers_on_every_response_and_static_files(app: Server) -> None:
    for method, path, hdr, body in [
        ("GET", "/?token=" + app.token, headers(app), b""),
        ("GET", "/", {}, b""),
        ("POST", "/stop", headers(app), b"no"),
        ("POST", "/stop", headers(app), b" " * (MAX_BODY + 1)),
        *[
            ("GET", path, headers(app), b"")
            for path in ("/", "/app.js", "/app.css", "/strings.json", "/events?after=0", "/missing")
        ],
    ]:
        result = app.handle(method, path, hdr, body)
        assert all(result.headers[k] == v for k, v in SECURITY.items())
        if path in {"/app.js", "/app.css", "/strings.json"}:
            assert result.status == 200 and result.body
    assert SECURITY["Content-Security-Policy"] == (
        "default-src 'self'; style-src 'self'; style-src-attr 'unsafe-inline'; frame-ancestors 'none'"
    )
    page = app.handle("GET", "/", headers(app)).body.decode()
    assert '<script src="/app.js" defer>' in page and "<style" not in page


def next_prompt(front: WebFront, after: int = 0) -> dict[str, Any]:
    with front.condition:
        assert front.condition.wait_for(
            lambda: front.prompt is not None and front.prompt["id"] > after, timeout=5
        )
        assert front.prompt is not None
        return dict(front.prompt)


def roundtrip(
    front: WebFront,
    ask: Callable[[], Any],
    value: Any = None,
    *,
    cancel: bool = False,
    invalid: tuple[Any, ...] = (),
) -> tuple[Any, dict[str, Any]]:
    result: list[Any] = []

    def run() -> None:
        try:
            result.append(ask())
        except ui.Cancelled as exc:
            result.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    prompt = next_prompt(front)
    assert not front.answer({"id": -1, "value": value})
    for wrong in invalid:
        assert not front.answer({"id": prompt["id"], "value": wrong})
        assert front.prompt is not None
    assert front.answer({"id": prompt["id"], **({"cancel": True} if cancel else {"value": value})})
    worker.join(5)
    assert not worker.is_alive() and len(result) == 1
    return result[0], prompt


def test_prompt_roundtrips_and_validation() -> None:
    f = WebFront()
    choices = [ui.Choice("Alpha", "a", "hint"), ui.Choice("Beta", "b")]
    result, prompt = roundtrip(
        f,
        lambda: f.select("Pick", [ui.Heading("Group"), *choices], "b"),
        1,
        invalid=(True, -1, 2, "1", None),
    )
    assert result == "b" and prompt["default"] == 1
    assert prompt["items"][0] == {"heading": "Group"}
    assert roundtrip(f, lambda: f.confirm("Sure?"), True, invalid=(1, "yes", None))[0] is True
    assert roundtrip(
        f,
        lambda: f.checkbox("Pick", choices, {"a"}),
        [0, 1],
        invalid=([0, 0], [True], [2], [{}], "0"),
    )[0] == {"a", "b"}
    assert (
        roundtrip(f, lambda: f.text("Name", "default"), " Jane ", invalid=(5, "a" * 20001))[0]
        == "Jane"
    )
    result, prompt = roundtrip(f, lambda: f.line({"/help": "Help"}), "hello\nthere")
    assert result == "hello\nthere" and prompt["commands"] == {"/help": "Help"}
    assert roundtrip(f, lambda: f.confirm("Sure?"), cancel=True)[0].__class__ is ui.Cancelled
    assert roundtrip(f, f.line, cancel=True)[0] is None


def test_secret_is_never_in_events_strings_or_logs(caplog: pytest.LogCaptureFixture) -> None:
    front = WebFront()
    secret = "private-key-Abc12345"
    assert roundtrip(front, lambda: front.secret("Key"), secret)[0] == secret
    events = front.poll(0, 0)
    assert secret not in json.dumps(events) + json.dumps(strings()) + caplog.text
    assert events["events"][-1]["echo"] == "••••"


def test_output_is_escaped_bounded_and_capture_restores_streams(
    capsys: pytest.CaptureFixture[str],
) -> None:
    front = WebFront(limit=4)
    with ui.use(front), front.capture():
        ui.header("Title", "Subtitle")
        ui.section("Section")
        ui.ok("Done")
        ui.warn("Warning", stderr=True)
        ui.error("Error")
        click.echo("<script>bad</script>")
    assert not capsys.readouterr().out
    assert len(front.events) == 4
    assert "&lt;script&gt;bad&lt;/script&gt;" in json.dumps(list(front.events))
    assert "<script>bad" not in json.dumps(list(front.events))
    click.echo("terminal")
    assert capsys.readouterr().out == "terminal\n"


def test_progress_uses_web_live_and_stop_only_interrupts_when_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interrupted: list[bool] = []
    monkeypatch.setattr("desman.gui.front._thread.interrupt_main", lambda: interrupted.append(True))
    front = WebFront()
    conn = db.open_db(":memory:")
    store = Store(conn)
    task = store.create_task(
        TaskSpec(
            goal="Do it", subtasks=[SubtaskSpec(description="One", acceptance_criteria="done")]
        )
    )
    try:
        front.stop()
        assert not interrupted
        with ui.use(front), Progress(store, task, make_config()):
            assert front.busy
            assert "One" in front.poll(0, 0)["live"]
            front.stop()
            assert interrupted == [True]
        assert not front.busy
        front.stop()
        assert interrupted == [True]
        assert [e["type"] for e in front.events] == ["busy"]
        assert front.poll(0, 0)["live"] is None
    finally:
        conn.close()
    front.end()
    assert front.poll(front.seq, 0)["ended"]


def browser_chat(
    monkeypatch: pytest.MonkeyPatch, turn: str, answers: list[Any]
) -> tuple[WebFront, str]:
    llm = FakeLLM().on(CHAT, turn, '{"type":"reply","text":"OK"}')
    monkeypatch.setattr(runs, "models", lambda config, conn, **_: runs.Models(llm, None))
    open_config().save(make_config(language="en", configured=True))
    front = WebFront()
    errors: list[BaseException] = []
    dumped: list[str] = []

    def run() -> None:
        conn = db.open_db()
        try:
            with ui.use(front), front.capture():
                Chat(Sessions(conn)).run()
            dumped.append("\n".join(conn.iterdump()))
        except BaseException as exc:
            errors.append(exc)
        finally:
            conn.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    last = 0
    for value in ["please do it", *answers, "/exit"]:
        prompt = next_prompt(front, last)
        last = prompt["id"]
        assert front.answer({"id": last, "value": value})
    thread.join(5)
    assert not thread.is_alive() and not errors, errors
    return front, dumped[0]


@pytest.mark.parametrize("accept", [False, True])
def test_real_chat_confirmation_is_the_only_authority(
    monkeypatch: pytest.MonkeyPatch, accept: bool
) -> None:
    front, _ = browser_chat(
        monkeypatch,
        json.dumps({"type": "command", "name": "personalize", "args": {"on": True}}),
        [accept],
    )
    assert any(e["type"] == "prompt" and e["kind"] == "confirm" for e in front.events)
    assert open_config().load().personalize is accept


def test_real_chat_key_is_saved_but_absent_from_history_and_journal(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    key = "gsk_" + "Abc123" * 8
    front, dumped = browser_chat(
        monkeypatch,
        json.dumps(
            {"type": "command", "name": "keys add", "args": {"provider": "groq", "label": "work"}}
        ),
        [key],
    )
    assert open_config().load().keys("GROQ_API_KEY")[-1].value == key
    assert key not in dumped + json.dumps(list(front.events)) + caplog.text
    assert any(e.get("echo") == "••••" for e in front.events)


def test_poll_returns_open_prompt_after_log_is_truncated(app: Server) -> None:
    front = app.front
    result: list[Any] = []
    worker = threading.Thread(target=lambda: result.append(front.line()), daemon=True)
    worker.start()
    prompt = next_prompt(front)
    front.out(Text("output"))
    data = json.loads(app.handle("GET", f"/events?after={front.seq}", headers(app)).body)
    assert not data["events"] and data["prompt"]["id"] == prompt["id"]
    answer = json.dumps({"id": prompt["id"], "cancel": True}).encode()
    assert app.handle("POST", "/answer", headers(app), answer).status == 200
    worker.join(5)
    assert result == [None]


def test_gui_cli_flags_and_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr("desman.gui.app.run", lambda *args: calls.append(args))
    result = CliRunner().invoke(cli, ["gui", "--port", "1234", "--no-browser", "-t"])
    assert result.exit_code == 0 and calls == [(False, True, 1234, False)]
    assert CliRunner().invoke(cli, ["gui", "-c", "-t"]).exit_code == 2


def test_strings_follow_language() -> None:
    i18n.set_language("ru")
    try:
        assert strings()["send"] == "Отправить"
    finally:
        i18n.set_language("en")


def test_real_loopback_request() -> None:
    app = Server(WebFront(), 0, poll_timeout=0)
    try:
        server = http_server(app)
    except PermissionError:
        pytest.skip("sandbox does not permit loopback binding")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", app.port, timeout=5)
    try:
        connection.request("GET", "/?token=" + app.token)
        response = connection.getresponse()
        assert response.status == 303
        cookie = response.getheader("Set-Cookie")
        response.read()
        assert cookie is not None
        connection.request("GET", "/", headers={"Cookie": cookie.split(";", 1)[0]})
        response = connection.getresponse()
        assert response.status == 200 and b"/app.js" in response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.mark.parametrize("interrupted", [False, True])
def test_app_runs_wizard_then_chat_on_main_thread_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    interrupted: bool,
) -> None:
    from desman.gui import app as gui_app

    stopped = threading.Event()
    seen: list[str] = []
    apps: list[Server] = []
    browser: list[str] = []
    pauses: list[float] = []

    class HTTP:
        def serve_forever(self) -> None:
            stopped.wait(5)

        def shutdown(self) -> None:
            seen.append("shutdown")
            stopped.set()

        def server_close(self) -> None:
            seen.append("closed")

    def http(app: Server) -> HTTP:
        app.port = 4321
        apps.append(app)
        return HTTP()

    def wizard(manager: object) -> None:
        assert threading.current_thread() is threading.main_thread()
        seen.append("wizard")
        ui.note("wizard output")

    class FakeChat:
        def __init__(self, sessions: Sessions, *, resume: bool, temporary: bool) -> None:
            assert resume and not temporary
            seen.append("chat")

        def run(self) -> None:
            assert threading.current_thread() is threading.main_thread()
            if interrupted:
                raise KeyboardInterrupt
            click.echo("command output")

    monkeypatch.setattr(gui_app, "http_server", http)
    monkeypatch.setattr(gui_app, "run_wizard", wizard)
    monkeypatch.setattr(gui_app, "print_project_info", lambda manager: seen.append("banner"))
    monkeypatch.setattr(gui_app, "Chat", FakeChat)
    monkeypatch.setattr("desman.gui.app.webbrowser.open", browser.append)
    gui_app.run(True, False, 0, True, sleep=pauses.append)
    [app] = apps
    assert seen == ["wizard", "banner", "chat", "shutdown", "closed"]
    assert browser == [app.url] and app.url in capsys.readouterr().out
    assert pauses == [0.5] and app.front.ended
    assert list(app.front.events)[-1]["type"] == "end"
    captured = json.dumps(list(app.front.events))
    assert "wizard output" in captured
    assert ("command output" in captured) is not interrupted
    click.echo("restored")
    assert capsys.readouterr().out == "restored\n"


@pytest.mark.parametrize("kind", ["select", "checkbox", "confirm", "text", "secret", "line"])
def test_every_window_can_be_cancelled(kind: str) -> None:
    front = WebFront()
    choices = [ui.Choice("One", 1)]
    asks: dict[str, Callable[[], Any]] = {
        "select": lambda: front.select("Pick", choices),
        "checkbox": lambda: front.checkbox("Pick", choices, set()),
        "confirm": lambda: front.confirm("Sure?"),
        "text": lambda: front.text("Name"),
        "secret": lambda: front.secret("Key"),
        "line": front.line,
    }
    result, prompt = roundtrip(front, asks[kind], cancel=True)
    assert result is None if kind == "line" else isinstance(result, ui.Cancelled)
    assert front.prompt is None
    assert not front.answer({"id": prompt["id"], "cancel": True})
    assert list(front.events)[-1]["type"] == "prompt_closed"


def test_prose_is_not_hard_wrapped_but_tables_keep_their_layout() -> None:
    from rich.table import Table

    prose = "A long sentence with words. " * 20
    assert WebFront.render(Text(prose)).rstrip("\n") == prose
    table = Table("Column", width=100)
    table.add_row("Value")
    assert "\n" in WebFront.render(table).rstrip("\n")


def test_live_updates_preserve_transcript_and_wake_waiting_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    front = WebFront(limit=2)
    front.out(Text("Keep this chat"))
    front.live_start()
    cursor = front.seq
    revision = front.state
    waiting = threading.Event()
    original = front.condition.wait_for

    def wait_for(predicate: Callable[[], bool], timeout: float | None = None) -> bool:
        waiting.set()
        return original(predicate, timeout)

    monkeypatch.setattr(front.condition, "wait_for", wait_for)
    results: list[dict[str, Any]] = []
    worker = threading.Thread(target=lambda: results.append(front.poll(cursor, 5, revision)))
    worker.start()
    assert waiting.wait(5)
    front.live_update(Text("Working"))
    worker.join(5)
    assert not worker.is_alive()
    assert results[0]["live"] == "Working\n"
    assert results[0]["busy"] and not results[0]["events"]
    assert results[0]["state"] > revision
    for i in range(20):
        front.live_update(Text(str(i)))
    assert front.seq == cursor
    assert "Keep this chat" in front.poll(0, 0)["events"][0]["html"]
    assert front.poll(cursor, 0)["live"] == "19\n"
    front.live_stop()
    assert front.poll(cursor, 0)["live"] is None
    assert not front.busy


def test_bold_theme_styles_keep_their_colour() -> None:
    """`"bold accent"` is no rich style string; the theme names it, else headings lose colour."""
    from rich.console import Console

    from desman import ui

    console = Console(theme=ui.THEME, force_terminal=True, color_system="truecolor")
    for name, colour in (("bold accent", ui.ACCENT), ("bold warn", ui.WARN), ("bold bad", ui.BAD)):
        style = console.get_style(name)
        assert style.bold and style.color is not None
        assert style.color.triplet is not None and style.color.triplet.hex == colour.lower()
    assert ui.ACCENT.lower() in WebFront.render(Text("A", style="bold accent"))


def test_a_key_typed_into_a_line_is_not_echoed() -> None:
    """The event log is replayed on every reload: a pasted key must not stay in it."""
    front = WebFront()
    key = "gsk_" + "A1b2C3d4E5f6G7h8J9k0L1m2N3p4Q5r6S7t8"
    said = f"my key is {key} ok"
    assert roundtrip(front, front.line, said)[0] == said  # the chat itself cuts it, and warns
    echoed = json.dumps(front.poll(0, 0))
    assert key not in echoed and "my key is" in echoed
    custom = WebFront(redact=lambda text: text.replace("secret-word", "[cut]"))
    roundtrip(custom, lambda: custom.text("Name"), "a secret-word here")
    assert "secret-word" not in json.dumps(custom.poll(0, 0))


def test_a_startup_error_is_shown_on_the_page(monkeypatch: pytest.MonkeyPatch) -> None:
    from desman.gui import app as gui_app

    apps: list[Server] = []

    class HTTP:
        def serve_forever(self) -> None:
            pass

        def shutdown(self) -> None:
            pass

        def server_close(self) -> None:
            pass

    def http(app: Server) -> HTTP:
        apps.append(app)
        return HTTP()

    def broken() -> None:
        raise click.ClickException("config.json does not load: fix it or run desman config reset")

    monkeypatch.setattr(gui_app, "http_server", http)
    monkeypatch.setattr(gui_app, "load_config", broken)
    with pytest.raises(click.ClickException):  # the CLI still exits with an error
        gui_app.run(False, False, 0, False, sleep=lambda _: None)
    [app] = apps
    events = list(app.front.events)
    assert [e["type"] for e in events][-2:] == ["out", "end"]
    assert "config.json does not load" in events[-2]["html"]
