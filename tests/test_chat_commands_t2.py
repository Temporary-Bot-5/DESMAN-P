"""Explicit command lines, human inbox decisions, and kept/temporary session controls."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman import actions, commands, ui
from desman.chat import Chat
from desman.cli import cli
from desman.config import Config, open_config
from desman.core import db, operator
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.report import inbox
from desman.core.retry import FailureKind
from desman.core.sessions import Kind, Role, Sessions
from desman.core.states import Actor, SubtaskStatus
from desman.core.store import Store


@pytest.fixture
def sessions(clock: FakeClock) -> Iterator[Sessions]:
    conn = db.open_db(":memory:")
    open_config().save(Config(language="en", configured=True))
    yield Sessions(conn, clock=clock)
    conn.close()


def waiting(store: Store) -> str:
    store.create_task(
        TaskSpec(goal="goal", subtasks=[SubtaskSpec(description="fetch", acceptance_criteria="ok")])
    )
    sub = store.claim("worker")
    assert sub is not None
    store.record_failure(
        sub.id,
        SubtaskStatus.RUNNING,
        Actor.EXECUTOR,
        FailureKind.PERMANENT,
        lease=sub.lease_token,
        reason="unavailable",
    )
    return sub.id


def picker(monkeypatch: pytest.MonkeyPatch, *answers: Any) -> None:
    sequence = iter(answers)

    def select(question: str, choices: Any, default: Any = None) -> Any:
        answer = next(sequence)
        if isinstance(answer, int):
            return choices[answer].value
        return answer

    monkeypatch.setattr(ui, "select", select)
    monkeypatch.setattr(ui, "interactive", lambda: True)


def test_each_catalog_field_is_reachable_from_click() -> None:
    for entry in commands.CATALOG:
        command = commands.cli_command(cli, entry.name)
        assert command is not None
        mapping = dict(entry.cli_names)
        reachable = {mapping.get(p.name or "", p.name) for p in command.params}
        chat_only = dict(entry.chat_only)
        assert all(chat_only.values())
        assert set(entry.args.model_fields) <= reachable | chat_only.keys(), entry.name
        assert commands.find(entry.name) is entry


@pytest.mark.parametrize(
    ("line", "name", "data"),
    [
        ("desman keys list", "keys list", {}),
        (
            "list --subtasks --status done",
            "list",
            {"all": False, "subtasks": True, "status": "done", "limit": 20},
        ),
        ("language ru", "language", {"code": "ru"}),
        ("personalize off", "personalize", {"on": False}),
        ("global update --revert", "global update", {"action": "revert", "release": None}),
        ("global update --show 2", "global update", {"action": "show", "release": 2}),
        ("tuning --model planner=auto", "tuning", None),
        ('run "two words"', "run", {"goal": "two words", "title": None, "runs_code": False}),
    ],
)
def test_line_parser(line: str, name: str, data: dict[str, Any] | None) -> None:
    parsed = commands.parse_line(line, None)
    assert parsed is not None and parsed[0].name == name
    if data is not None:
        assert parsed[1].model_dump() == data


@pytest.mark.parametrize(
    "line",
    [
        'run "unterminated',
        "list --unknown",
        "list --json",
        "list --db x",
        "keys not-a-command",
        "config unknown",
    ],
)
def test_bad_lines_never_execute(sessions: Sessions, line: str) -> None:
    Chat(sessions)._typed(line)
    assert sessions.recent() == []
    assert operator.entries(sessions.conn) == []


def test_help_needs_no_required_arguments_and_runs_nothing(
    sessions: Sessions, capsys: pytest.CaptureFixture[str]
) -> None:
    Chat(sessions)._typed("fail --help")
    assert "--note" in capsys.readouterr().out
    assert operator.entries(sessions.conn) == []


def test_typed_change_has_no_confirmation_and_only_an_event(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = sessions.add(None, Role.USER, "hello")
    chat = Chat(sessions)
    chat.session_id = first.session_id
    monkeypatch.setattr(
        ui, "confirm", lambda *a, **kw: pytest.fail("typed change must not confirm")
    )
    chat._typed("personalize on")
    assert open_config().load().personalize
    messages = sessions.messages(first.session_id)
    assert len(messages) == 2 and messages[1].kind is Kind.EVENT
    assert "The person ran personalize" in messages[1].text
    entries = operator.entries(sessions.conn)
    assert any(e.action == "operator_command" and e.details["typed"] for e in entries)
    assert all(e.details.get("typed") is True for e in entries)


def test_typed_input_still_uses_hidden_field(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    def secret(question: str) -> str:
        asked.append(question)
        return ""

    monkeypatch.setattr(ui, "secret", secret)
    Chat(sessions)._typed("keys add groq")
    assert asked and sessions.recent() == []


def test_model_command_without_configuration_is_plain(
    sessions: Sessions, capsys: pytest.CaptureFixture[str]
) -> None:
    Chat(sessions)._typed('run "a goal"')
    assert "DESMAN needs configuring" in capsys.readouterr().err
    assert sessions.recent() == []


def test_completion_crosses_spaces() -> None:
    names = commands.completions()
    assert ui.slash_matches("!keys a", names) == []
    assert ui.slash_matches("/keys a", names) == [("/keys add", names["/keys add"])]
    assert all(hint for hint in names.values())


@pytest.mark.parametrize(
    ("action", "target", "answers"),
    [
        ("retry", SubtaskStatus.READY, (0, "retry")),
        ("accept", SubtaskStatus.DONE, (0, "accept")),
        ("fail", SubtaskStatus.FAILED, (0, "fail", None)),
    ],
)
def test_inbox_decisions(
    sessions: Sessions,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    target: SubtaskStatus,
    answers: tuple[Any, ...],
) -> None:
    store = Store(sessions.conn, clock=sessions.clock)
    sub_id = waiting(store)
    picker(monkeypatch, *answers)
    monkeypatch.setattr(ui, "text", lambda *a, **kw: "try a mirror")
    result = actions.choose_inbox(store)
    assert store.get_subtask(sub_id).status is target
    assert result["decisions"][0]["to"] == target
    assert any(e.details.get("command") == action for e in operator.entries(sessions.conn))
    if action == "retry":
        feedback = store.last_feedback(sub_id)
        assert feedback is not None and feedback.text == "try a mirror"


def test_inbox_stale_version_refreshes(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = Store(sessions.conn, clock=sessions.clock)
    waiting(store)
    old = inbox(store)[0]
    pending = actions.prepare_decision(store, old.ref, SubtaskStatus.READY, old.version)
    actions.apply_decision(store, pending, "retry", "operator_retry")
    sub = store.claim("worker")
    assert sub is not None
    store.record_failure(
        sub.id,
        SubtaskStatus.RUNNING,
        Actor.EXECUTOR,
        FailureKind.PERMANENT,
        lease=sub.lease_token,
        reason="again",
    )
    picker(monkeypatch, old, "accept", None)
    result = actions.choose_inbox(store)
    assert not result["decisions"] and inbox(store)
    assert "changed since version" in capsys.readouterr().err


def test_inbox_back_and_leave(sessions: Sessions, monkeypatch: pytest.MonkeyPatch) -> None:
    store = Store(sessions.conn, clock=sessions.clock)
    waiting(store)
    picker(monkeypatch, 0, None, None)
    assert actions.choose_inbox(store)["decisions"] == []
    assert len(inbox(store)) == 1


def test_inbox_catalog_input_and_read(sessions: Sessions, monkeypatch: pytest.MonkeyPatch) -> None:
    store = Store(sessions.conn, clock=sessions.clock)
    waiting(store)
    entry = commands.find("inbox")
    assert entry is not None
    picker(monkeypatch, 0, "retry")
    monkeypatch.setattr(ui, "text", lambda *a, **kw: "guidance")
    assert entry.kind_of(commands.InboxArgs()) is commands.Kind.INPUT
    assert entry.kind_of(commands.InboxArgs(list=True)) is commands.Kind.READ
    result = entry.handler(
        commands.Context(store, open_config(), lambda *a: None, lambda *a: None),
        commands.InboxArgs(),
    )
    assert isinstance(result, commands.Outcome) and result.data["decisions"]


def test_cli_inbox_interactive_and_list_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "tasks.db"
    store = Store(db.open_db(path))
    waiting(store)
    picker(monkeypatch, 0, "accept")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    # CliRunner replaces streams, so force their TTY methods during invocation.
    from unittest.mock import patch

    runner = CliRunner()
    with (
        runner.isolation(),
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
    ):
        cli.main(["inbox", "--db", str(path)], standalone_mode=False)
    assert not inbox(store)
    waiting(store)
    result = runner.invoke(cli, ["inbox", "--list", "--db", str(path)])
    assert result.exit_code == 0 and "needs_human" in result.output
    result = runner.invoke(cli, ["inbox", "--json", "--db", str(path)])
    assert json.loads(result.output)[0]["status"] == "needs_human"
    store.conn.close()


def test_list_subtasks_text_and_json(tmp_path: Path) -> None:
    path = tmp_path / "tasks.db"
    conn = db.open_db(path)
    Store(conn).create_task(
        TaskSpec(
            goal="goal",
            subtasks=[SubtaskSpec(description="unique description", acceptance_criteria="ok")],
        )
    )
    conn.close()
    runner = CliRunner()
    plain = runner.invoke(cli, ["list", "--db", str(path)])
    detailed = runner.invoke(cli, ["list", "--subtasks", "--db", str(path)])
    assert "unique description" not in plain.output
    assert "unique description" in detailed.output and "[0]" in detailed.output
    for extra, include in [([], False), (["--subtasks"], True)]:
        result = runner.invoke(cli, ["list", "--json", "--db", str(path), *extra])
        assert ("subtasks" in json.loads(result.output)[0]) is include


def test_pins_delete_all_and_latest(sessions: Sessions, clock: FakeClock) -> None:
    pinned = sessions.add(None, Role.USER, "pinned").session_id
    clock.advance(1)
    newest = sessions.add(None, Role.USER, "newest").session_id
    temporary = sessions.add(None, Role.USER, "secret", temporary=True).session_id
    sessions.pin(pinned, True)
    assert [s.id for s in sessions.recent()] == [pinned, newest]
    latest = sessions.latest()
    assert latest is not None and latest.id == newest
    with pytest.raises(sqlite3.IntegrityError):
        sessions.pin(temporary, True)
    assert sessions.delete_all() == 1
    assert sessions.get(newest) is None and sessions.messages(newest)
    assert sessions.messages(temporary) and sessions.get(pinned)
    sessions.pin(pinned, False)
    assert sessions.delete_all() == 1


def test_upgrade_33_retains_sessions() -> None:
    conn = db.connect(":memory:")
    db.migrate(conn, db.MIGRATIONS[:33])
    session = Sessions(conn).add(None, Role.USER, "kept").session_id
    db.migrate(conn)
    db.verify_schema(conn)
    current = Sessions(conn).get(session)
    assert current is not None and not current.pinned
    Sessions(conn).pin(session, True)
    assert Sessions(conn).delete_all() == 0
    conn.close()


def test_temporary_switch_and_delete_all(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = Chat(sessions)
    normal = sessions.add(None, Role.USER, "keep").session_id
    chat.session_id = normal
    chat._slash("/pin")
    chat._slash("/temporary")
    assert chat.temporary and chat.session_id is None and sessions.get(normal)
    secret = sessions.add(None, Role.USER, "secret", temporary=True).session_id
    chat.session_id = secret
    chat._slash("/temporary")
    assert not sessions.messages(secret) and chat.temporary
    chat._slash("/new")
    assert not chat.temporary and chat.session_id is None
    chat.session_id = sessions.add(None, Role.USER, "delete").session_id
    monkeypatch.setattr(ui, "confirm", lambda *a, **kw: True)
    chat._slash("/delete-all")
    assert chat.session_id is None and sessions.get(normal)
    chat.session_id = normal
    chat._slash("/pin")
    assert [e.action for e in operator.entries(sessions.conn)] == [
        "operator_session_pinned",
        "operator_sessions_deleted",
        "operator_session_unpinned",
    ]


def test_inbox_cancel_keeps_partial_results(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(sessions.conn, clock=sessions.clock)
    sub = waiting(store)
    picker(monkeypatch, 0, "fail")
    monkeypatch.setattr(ui, "text", lambda *a, **kw: "cannot do it")
    select = ui.select
    calls = 0

    def cancel_after_decision(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ui.Cancelled
        return select(*args, **kwargs)

    monkeypatch.setattr(ui, "select", cancel_after_decision)
    result = actions.choose_inbox(store)
    assert result["decisions"][0]["subtask_id"] == sub
    assert store.get_subtask(sub).status is SubtaskStatus.FAILED


def test_list_catalog_omits_subtasks_unless_requested(sessions: Sessions) -> None:
    store = Store(sessions.conn, clock=sessions.clock)
    waiting(store)
    entry = commands.find("list")
    assert entry is not None
    ctx = commands.Context(store, open_config(), lambda *a: None, lambda *a: None)
    for include in (False, True):
        outcome = entry.handler(ctx, commands.ListArgs(subtasks=include))
        assert isinstance(outcome, commands.Outcome)
        assert ("subtasks" in outcome.data[0]) is include
        if include:
            assert outcome.data[0]["subtasks"][0]["description"] == "fetch"


def test_typed_lines_skip_model_and_user_history(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines = iter(["!keys list", "!list --help", "!desman personalize on", "/exit"])
    chat = Chat(sessions, read=lambda: next(lines, None))
    monkeypatch.setattr(
        chat, "_message", lambda text: pytest.fail("explicit line became a message")
    )
    monkeypatch.setattr(
        chat, "_models", lambda *a, **kw: pytest.fail("read/change command requested models")
    )
    chat.run()
    assert sessions.recent() == []
    assert open_config().load().personalize


def test_pinned_groups_sort_by_last_message(sessions: Sessions, clock: FakeClock) -> None:
    ids = []
    for i in range(4):
        ids.append(sessions.add(None, Role.USER, str(i)).session_id)
        clock.advance(1)
    sessions.pin(ids[0], True)
    sessions.pin(ids[2], True)
    assert [s.id for s in sessions.recent()] == [ids[2], ids[0], ids[3], ids[1]]
    sessions.add(ids[0], Role.USER, "latest")
    assert [s.id for s in sessions.recent()] == [ids[0], ids[2], ids[3], ids[1]]


def test_switch_rebuilds_model_cache_policy(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman import runs
    from desman.llm.fake import FakeLLM

    caching: list[bool] = []

    def models(config: Config, conn: sqlite3.Connection, *, cache: bool) -> runs.Models:
        caching.append(cache)
        return runs.Models(FakeLLM(), None)

    monkeypatch.setattr(runs, "models", models)
    chat = Chat(sessions)
    config = open_config().load()
    assert chat._models(config) is not None
    chat._new(temporary=True)
    assert chat._models(config) is not None
    chat._new()
    assert chat._models(config) is not None
    assert caching == [True, False, True]


def test_bad_command_keys_are_not_echoed_or_kept(
    sessions: Sessions, capsys: pytest.CaptureFixture[str]
) -> None:
    key = "gsk_" + "Abc123" * 8
    chat = Chat(sessions)
    chat.session_id = sessions.add(None, Role.USER, "hello").session_id
    chat._typed("language " + key)
    chat._typed("language --unknown " + key)
    assert key not in capsys.readouterr().err
    assert key not in "\n".join(sessions.conn.iterdump())


def test_terminal_recall_redacts_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    import prompt_toolkit

    captured: dict[str, Any] = {}

    def prompt(*args: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(prompt_toolkit, "PromptSession", prompt)
    monkeypatch.setattr(ui, "_prompt", None)
    ui._chat_prompt()
    key = "gsk_" + "Abc123" * 8
    captured["history"].append_string("!language " + key)
    assert key not in "\n".join(captured["history"].get_strings())


def test_path_argument_is_converted_before_catalog_validation(tmp_path: Path) -> None:
    path = tmp_path / "key file.txt"
    path.write_text("", encoding="utf-8")
    parsed = commands.parse_line(f'keys import "{path.as_posix()}"', None)
    assert parsed is not None and parsed[1].model_dump() == {"file": str(path)}


def test_typed_run_before_first_message_does_not_create_session(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from desman import runs
    from desman.interlocutor import RETELL
    from desman.llm.fake import FakeLLM

    llm = FakeLLM()
    llm.on("Plan task", '{"subtasks": [{"description": "find", "acceptance_criteria": "found"}]}')
    llm.on("Execute subtask", '{"ok": true}')
    llm.on("Review subtask", '{"verdict": "accept"}')
    llm.on(RETELL, '{"text": "Done now."}')
    monkeypatch.setattr(runs, "models", lambda *a, **kw: runs.Models(llm, None))
    chat = Chat(sessions)
    chat._typed('run "find it"')
    assert chat.session_id is None and sessions.recent() == []
    assert "Done now." in capsys.readouterr().out
    assert sessions.conn.execute("SELECT session_id FROM tasks").fetchone()[0] is None


def test_completion_and_help_follow_language_switch(
    sessions: Sessions, capsys: pytest.CaptureFixture[str]
) -> None:
    from desman import i18n

    english = commands.completions()["/list"]
    Chat(sessions)._typed("language ru")
    assert commands.completions()["/list"] == i18n.t("List tasks, newest first.")
    assert commands.completions()["/list"] != english
    capsys.readouterr()
    commands.parse_line("list --help", None)
    assert i18n.t("List tasks, newest first.") in capsys.readouterr().out


def test_chat_inbox_returns_decisions_as_session_event(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = Chat(sessions)
    sub = waiting(chat.store)
    chat.session_id = sessions.add(None, Role.USER, "hello").session_id
    picker(monkeypatch, 0, "retry")
    monkeypatch.setattr(ui, "text", lambda *a, **kw: "use a mirror")
    chat._typed("inbox")
    feedback = chat.store.last_feedback(sub)
    assert feedback is not None and feedback.text == "use a mirror"
    messages = sessions.messages(chat.session_id)
    assert messages[-1].kind is Kind.EVENT
    assert '"waiting"' in messages[-1].text and '"decisions"' in messages[-1].text
    assert any(
        e.details.get("command") == "inbox" and e.details.get("typed")
        for e in operator.entries(sessions.conn)
    )


@pytest.mark.parametrize("pinned", [False, True])
def test_delete_all_without_unpinned_sessions_skips_confirmation(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch, pinned: bool
) -> None:
    chat = Chat(sessions)
    if pinned:
        chat.session_id = sessions.add(None, Role.USER, "keep").session_id
        sessions.pin(chat.session_id, True)
    notes: list[str] = []
    monkeypatch.setattr(ui, "note", notes.append)

    def unexpected_confirmation(*args: Any, **kwargs: Any) -> bool:
        pytest.fail("Nothing can be deleted: confirmation must not open")

    monkeypatch.setattr(ui, "confirm", unexpected_confirmation)
    chat._slash("/delete-all")
    assert notes == ["No unpinned sessions to delete."]
    assert operator.entries(sessions.conn) == []
    if pinned:
        assert chat.session_id is not None and sessions.get(chat.session_id)


def test_cli_confirmations_are_explicit_in_catalog() -> None:
    # CLI reset asks; uploads preview and require --confirm SHA256 to send.
    confirming = {"config reset", "global report", "global share", "global submit-proposal"}
    assert {entry.name for entry in commands.CATALOG if entry.typed_confirm} == confirming
    for entry in commands.CATALOG:
        command = commands.cli_command(cli, entry.name)
        assert command is not None
        if any("--confirm" in param.opts for param in command.params):
            assert entry.name in confirming, entry.name
    for name in confirming:
        command = commands.cli_command(cli, name)
        assert command is not None
        assert any(set(param.opts) & {"--yes", "--confirm"} for param in command.params)


@pytest.mark.parametrize("yes", [False, True])
def test_typed_reset_keeps_cli_confirmation(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch, yes: bool
) -> None:
    seen: list[commands.Proposal] = []

    def decline(self: Chat, proposal: commands.Proposal) -> bool:
        seen.append(proposal)
        return False

    monkeypatch.setattr(Chat, "_confirmed", decline)
    Chat(sessions)._typed("config reset" + (" --yes" if yes else ""))
    assert bool(seen) is not yes
    assert open_config().config_path.exists() is not yes


@pytest.mark.parametrize("name", ["global report", "global share", "global submit-proposal"])
def test_typed_upload_shows_proposal_before_applying(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    from dataclasses import replace

    entry = commands.find(name)
    assert entry is not None
    applied: list[bool] = []

    def apply() -> commands.Outcome:
        applied.append(True)
        return commands.Outcome({})

    proposal = commands.Proposal(['{"exact": "payload"}'], apply)
    entry = replace(entry, args=commands.NoArgs, handler=lambda ctx, args: proposal)
    monkeypatch.setattr(commands, "parse_line", lambda *a: (entry, commands.NoArgs()))
    seen: list[commands.Proposal] = []

    def decline(self: Chat, made: commands.Proposal) -> bool:
        seen.append(made)
        return False

    monkeypatch.setattr(Chat, "_confirmed", decline)
    Chat(sessions)._typed(name)
    assert seen == [proposal]
    assert applied == []


@pytest.mark.parametrize(
    ("windows", "argument", "expected"),
    [
        (True, r"C:\keys.txt", r"C:\keys.txt"),
        (True, r'"C:\my keys.txt"', r"C:\my keys.txt"),
        (False, r"C:\keys.txt", "C:keys.txt"),
        (False, r'"C:\my keys.txt"', r"C:\my keys.txt"),
        (False, "'/tmp/my keys.txt'", "/tmp/my keys.txt"),
        (True, "'C:\\keys.txt'", "'C:\\keys.txt'"),
    ],
)
def test_path_tokenization(
    windows: bool, argument: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import click

    command = commands.cli_command(cli, "keys import")
    assert command is not None
    monkeypatch.setattr(command.params[0], "type", click.STRING)
    parsed = commands.parse_line("keys import " + argument, None, windows=windows)
    assert parsed is not None
    assert str(parsed[1].model_dump()["file"]) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (r"keys import C:\keys.txt", ["keys", "import", r"C:\keys.txt"]),
        (r'keys import "C:\My Files\k.txt"', ["keys", "import", r"C:\My Files\k.txt"]),
        ('list --status="active"', ["list", "--status=active"]),
        ('protected approve --note="two words"', ["protected", "approve", "--note=two words"]),
        ("protected approve --note=can't", ["protected", "approve", "--note=can't"]),
        (r"keys import C:\Users\O'Brien\key.txt", ["keys", "import", r"C:\Users\O'Brien\key.txt"]),
        ('protected approve --note "it\'s here"', ["protected", "approve", "--note", "it's here"]),
        ('protected approve --note ""', ["protected", "approve", "--note", ""]),
    ],
)
def test_windows_tokenization(
    line: str, expected: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = commands._windows_words
    seen: list[list[str]] = []

    def capture(text: str) -> list[str]:
        words = tokenizer(text)
        seen.append(words.copy())
        return words

    import click

    command = commands.cli_command(cli, "keys import")
    assert command is not None
    monkeypatch.setattr(command.params[0], "type", click.STRING)
    monkeypatch.setattr(commands, "_windows_words", capture)
    if expected[-1] == "":
        # The tokenizer keeps the empty value; the note contract rejects it.
        with pytest.raises(commands.BadArgs, match="at least 1 character"):
            commands.parse_line(line, None, windows=True)
    else:
        parsed = commands.parse_line(line, None, windows=True)
        assert parsed is not None
        assert expected[-1].split("=", 1)[-1] in parsed[1].model_dump().values()
    assert seen == [expected]


def test_windows_unclosed_quote() -> None:
    with pytest.raises(commands.BadArgs, match="Invalid quoting"):
        commands.parse_line('protected approve --note="unfinished', None, windows=True)
