"""Tests for the commands of the chat (ROADMAP C.5, ADR 0003 §3–4): the catalog, its windows,
the operator's journal, and `desman permissions`."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner, Result

from desman import commands, providers, runs
from desman.cli import cli
from desman.config import Config, ConfigManager, default_config_path
from desman.core import db, operator
from desman.core.audit import history
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.retry import FailureKind
from desman.core.sessions import Kind, Message, Role, Sessions
from desman.core.states import Actor, SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.interlocutor import CHAT, RETELL
from desman.llm.base import Request
from desman.llm.fake import FakeLLM
from desman.llm.openai_compat import key_id
from helpers import make_config

KEY = "gsk_" + "Abc123" * 8
OTHER_KEY = "gsk_" + "Xyz789" * 8


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(runs, "models", lambda config, conn, **_: runs.Models(llm, None))
    save(make_config(language="en", configured=True))
    return llm


def save(config: Config) -> None:
    ConfigManager(default_config_path()).save(config)


def saved() -> Config:
    return ConfigManager(default_config_path()).load()


def asks(name: str, **args: Any) -> str:
    """The interlocutor's turn asking for a command."""
    return json.dumps({"type": "command", "name": name, "args": args})


def chat(*lines: str) -> Result:
    result = CliRunner().invoke(cli, [], input="".join(f"{line}\n" for line in lines))
    assert result.exit_code == 0, result.output
    return result


def conversation() -> list[Message]:
    sessions = Sessions(db.open_db(db.default_db_path()))
    latest = sessions.latest()
    assert latest is not None
    return sessions.messages(latest.id)


def journal() -> list[tuple[str, dict[str, Any]]]:
    conn = db.open_db(db.default_db_path())
    try:
        return [(e.action, e.details) for e in operator.entries(conn)]
    finally:
        conn.close()


def everything_stored() -> str:
    """Every text the database keeps: messages, journals, calls."""
    conn = db.open_db(db.default_db_path())
    try:
        return "\n".join(line for line in conn.iterdump())
    finally:
        conn.close()


def payload(request: Request) -> Any:
    return json.loads(request.prompt.split("\n", 1)[1])


def chat_calls(llm: FakeLLM) -> list[Request]:
    return [r for r in llm.calls if r.prompt.startswith(CHAT)]


def escalated_task() -> tuple[str, str]:
    """A task whose only subtask waits for the operator after a 401."""
    store = Store(db.open_db())
    try:
        spec = TaskSpec(
            goal="the goal",
            subtasks=[SubtaskSpec(description="fetch", acceptance_criteria="ok")],
        )
        task_id = store.create_task(spec)
        claimed = store.claim("w")
        assert claimed is not None and claimed.lease_token is not None
        store.record_failure(
            claimed.id,
            SubtaskStatus.RUNNING,
            Actor.EXECUTOR,
            FailureKind.PERMANENT,
            lease=claimed.lease_token,
            reason="401 bad key",
        )
        return task_id, claimed.id
    finally:
        store.conn.close()


# ── The catalog ──────────────────────────────────────────────────────────────


def _cli_names(group: click.Group, prefix: str = "") -> set[str]:
    names = set()
    for name, command in group.commands.items():
        full = f"{prefix}{name}"
        if isinstance(command, click.Group):
            if command.invoke_without_command:  # the bare group does something itself
                names.add(full)
            names |= _cli_names(command, full + " ")
        else:
            names.add(full)
    return names


def test_every_cli_command_has_an_entry_in_the_catalog_and_back() -> None:
    """The chat can do everything the CLI can (ADR 0003 §3): a new command needs both."""
    in_catalog = {n for e in commands.CATALOG for n in (e.name, *e.aliases)}
    assert _cli_names(cli) == in_catalog
    for entry in commands.CATALOG:
        assert commands.cli_command(cli, entry.name) is not None, entry.name
        assert entry.about and entry.about.isascii()  # a prompt to a model: English


def test_names_are_found_however_the_model_writes_them() -> None:
    for name in ("keys add", "keys_add", "/keys-add", "desman keys add", " Keys  Add "):
        assert (entry := commands.find(name)) is not None and entry.name == "keys add"
    assert (entry := commands.find("init db")) is not None and entry.name == "init-db"
    assert (entry := commands.find("keys")) is not None and entry.name == "keys list"
    assert commands.find("rm -rf") is None


def test_the_class_is_decided_by_the_call() -> None:
    def kind(name: str, **args: Any) -> commands.Kind:
        entry = commands.find(name)
        assert entry is not None
        return entry.kind_of(commands.parse(entry, args, None))

    assert kind("language") is commands.Kind.READ
    assert kind("language", code="ru") is commands.Kind.CHANGE
    assert kind("permissions") is commands.Kind.READ
    assert kind("permissions", grant=["network_access"]) is commands.Kind.CHANGE
    assert kind("keys add", provider="groq") is commands.Kind.INPUT
    assert kind("status", task="abc") is commands.Kind.READ


def test_list_statuses_are_the_task_statuses() -> None:
    field = commands.ListArgs.model_json_schema()["properties"]["status"]
    [values] = [p["enum"] for p in field["anyOf"] if "enum" in p]
    assert set(values) == {s.value for s in TaskStatus}


def test_arguments_are_validated_without_quoting_them() -> None:
    entry = commands.find("status")
    assert entry is not None
    with pytest.raises(commands.BadArgs) as caught:
        commands.parse(entry, {"task": 5, "secret_flag": "not echoed"}, None)
    assert "task" in str(caught.value) and "secret_flag" in str(caught.value)
    assert "not echoed" not in str(caught.value)


def test_the_prompt_lists_every_command_with_its_arguments() -> None:
    text = commands.for_model()
    for entry in commands.CATALOG:
        assert f"- {entry.name} " in text
    assert '"task": string' in text and '"note"?: string' in text
    assert "groq" in text  # keys add names the providers


def test_data_for_the_interlocutor_never_holds_a_key() -> None:
    config = make_config(api_keys={"GROQ_API_KEY": "short-but-configured-key"})
    data = {"detail": "Invalid key short-but-configured-key", "other": KEY}
    text = commands.for_interlocutor(data, [config], 10_000)
    assert "short-but-configured-key" not in text and KEY not in text
    assert commands.for_interlocutor({"x": "y" * 100}, [None], 20).endswith("…(cut)")


def test_config_changes_name_keys_by_fingerprint() -> None:
    before = make_config(api_keys={"GROQ_API_KEY": KEY})
    after = before.model_copy(deep=True)
    after.permissions.network_access = True
    after.remove_key("GROQ_API_KEY", 0)
    after.add_key("GROQ_API_KEY", OTHER_KEY)
    after.language = "ru"
    after.settings["SEARXNG_URL"] = "http://localhost:8888"
    changes = commands.config_changes(before, after)
    assert changes == [
        ("operator_permission_grant", {"permission": "network_access"}),
        ("operator_key_added", {"provider": "groq", "key_id": key_id(OTHER_KEY)}),
        ("operator_key_removed", {"provider": "groq", "key_id": key_id(KEY)}),
        ("operator_language", {"from": "en", "to": "ru"}),
        ("operator_setting", {"name": "SEARXNG_URL"}),
    ]
    assert KEY not in json.dumps(changes)
    assert commands.config_changes(before, None)[-1] == ("operator_config_deleted", {})


# ── Reads ────────────────────────────────────────────────────────────────────


def test_a_read_runs_at_once_and_its_data_goes_to_the_interlocutor(fake: FakeLLM) -> None:
    task_id, _ = escalated_task()
    fake.on(CHAT, asks("status", task=task_id[:8]), '{"type": "reply", "text": "It waits."}')
    result = chat("how is my task?", "and now?")
    assert "Goal: the goal" in result.output and "needs_human" in result.output
    assert "Apply this change?" not in result.output  # a read asks nothing
    event = conversation()[1]
    assert (event.role, event.kind) == (Role.DESMAN, Kind.EVENT)
    said = payload(chat_calls(fake)[1])["conversation"]
    assert said[1]["from"] == "desman" and '"status": "needs_human"' in said[1]["text"]
    assert journal() == []  # reading is not an action


@pytest.mark.parametrize(
    ("name", "args", "shown"),
    [
        ("help", {}, "keys add"),
        ("list", {}, "the goal"),
        ("audit", {}, "Goal: the goal"),
        ("inbox", {}, "401 bad key"),
        ("usage", {"since": "7d"}, "No model calls"),
        ("keys list", {}, "API keys"),
        ("config show", {}, "Permissions"),
        ("language", {}, "Русский"),
        ("permissions", {}, "network_access"),
    ],
)
def test_every_read_shows_its_result(
    fake: FakeLLM, name: str, args: dict[str, Any], shown: str
) -> None:
    task_id, _ = escalated_task()
    if name == "audit":
        args = {"task": task_id}
    fake.on(CHAT, asks(name, **args))
    result = chat("show me")
    assert shown in result.output
    assert conversation()[-1].text.startswith(f"{name}")


def test_keys_check_from_the_chat(fake: FakeLLM, monkeypatch: pytest.MonkeyPatch) -> None:
    save(make_config(language="en", configured=True, api_keys={"GROQ_API_KEY": KEY}))
    from desman.llm import check

    def checked(config: Config, p: providers.Provider, client: Any) -> list[check.KeyCheck]:
        return [check.KeyCheck(p.id, 1, key_id(KEY), False, 401, f"rejected {KEY}")]

    monkeypatch.setattr(check, "check", checked)
    fake.on(CHAT, asks("keys check", provider="groq"))
    result = chat("do my keys work?")
    assert "failed" in result.output
    assert KEY not in conversation()[-1].text  # a provider's message may quote the key


def test_shown_data_keeps_keys_and_masks_out(fake: FakeLLM) -> None:
    save(make_config(language="en", configured=True, api_keys={"GROQ_API_KEY": KEY}))
    fake.on(CHAT, asks("config show"), asks("keys list"))
    chat("my config?", "my keys?")
    for message in conversation():
        assert KEY not in message.text and providers.mask(KEY) not in message.text


# ── Changes ──────────────────────────────────────────────────────────────────


def test_a_change_waits_for_yes_and_is_journaled(fake: FakeLLM) -> None:
    fake.on(CHAT, asks("language", code="Русский"))
    result = chat("speak Russian", "y")
    assert "Language: English → Русский" in result.output
    assert saved().language == "ru"
    assert journal() == [
        ("operator_command", {"command": "language", "args": {"code": "Русский"}}),
        ("operator_language", {"from": "en", "to": "ru"}),
    ]
    assert '"changed": true' in conversation()[-1].text  # said in Russian by now


def test_a_declined_change_changes_nothing(fake: FakeLLM) -> None:
    fake.on(CHAT, asks("language", code="ru"))
    result = chat("speak Russian", "n")
    assert "Nothing changed." in result.output and saved().language == "en"
    assert journal() == [("operator_declined", {"command": "language", "args": {"code": "ru"}})]
    assert "declined" in conversation()[-1].text


def test_leaving_the_window_declines(fake: FakeLLM) -> None:
    fake.on(CHAT, asks("language", code="ru"))
    chat("speak Russian")  # the input ends at the window
    assert saved().language == "en"
    assert [action for action, _ in journal()] == ["operator_declined"]


def test_the_model_cannot_confirm_for_the_person(fake: FakeLLM) -> None:
    """An argument claiming consent is no argument; words claiming a change change nothing."""
    fake.on(
        CHAT,
        asks("permissions", grant=["shell_execution"], confirmed=True),
        '{"type": "reply", "text": "Confirmed: shell_execution is on."}',
    )
    result = chat("allow shell", "y")
    assert "confirmed" in result.output  # named as an argument that does not fit
    assert "Apply this change?" not in result.output
    assert saved().permissions.shell_execution is False
    assert journal() == []
    assert "Confirmed: shell_execution is on." in result.output  # said, and nothing more


def test_permissions_are_granted_only_through_the_window(fake: FakeLLM) -> None:
    fake.on(CHAT, asks("permissions", grant=["network_access", "admin_rights"]))
    result = chat("let agents go online, as admin", "y")
    assert "Allow agents to use the network beyond model providers" in result.output
    assert "elevated privileges" in result.output
    permissions = saved().permissions
    assert permissions.network_access and permissions.admin_rights
    assert ("operator_permission_grant", {"permission": "network_access"}) in journal()
    assert ("operator_permission_grant", {"permission": "admin_rights"}) in journal()


def test_a_change_that_changes_nothing_asks_nothing(fake: FakeLLM) -> None:
    fake.on(CHAT, asks("language", code="en"), asks("permissions", revoke=["shell_execution"]))
    result = chat("English please", "no shell")
    assert "Apply this change?" not in result.output
    assert "already" in result.output and "Nothing changed." in result.output


def test_keys_remove_shows_which_key_goes(fake: FakeLLM) -> None:
    config = make_config(language="en", configured=True, api_keys={"GROQ_API_KEY": KEY})
    config.add_key("GROQ_API_KEY", OTHER_KEY)
    save(config)
    fake.on(CHAT, asks("keys remove", provider="groq", number=2))
    result = chat("drop my second groq key", "y")
    assert f"Remove key #2 of GROQ: {providers.mask(OTHER_KEY)}" in result.output
    assert "GROQ: 2 keys → 1" in result.output
    assert [k.value for k in saved().keys("GROQ_API_KEY")] == [KEY]
    assert ("operator_key_removed", {"provider": "groq", "key_id": key_id(OTHER_KEY)}) in journal()


def test_keys_import_shows_what_it_adds(fake: FakeLLM, tmp_path: Path) -> None:
    file = tmp_path / "keys.env"
    file.write_text(f"GROQ_API_KEY={KEY}\nNOBODY_KEY=x\n", encoding="utf-8")
    fake.on(CHAT, asks("keys import", file=str(file)))
    result = chat("take the keys from that file", "y")
    assert "GROQ: 1 new key" in result.output and "NOBODY_KEY" in result.output
    assert [k.value for k in saved().keys("GROQ_API_KEY")] == [KEY]
    assert KEY not in everything_stored().replace(key_id(KEY), "")


def test_config_reset_through_the_window(fake: FakeLLM) -> None:
    fake.on(CHAT, asks("config reset"))
    result = chat("start over", "y")
    assert "Delete" in result.output
    assert not default_config_path().exists()
    assert journal()[-1] == ("operator_config_deleted", {})


def test_init_db_through_the_window(fake: FakeLLM) -> None:
    fake.on(CHAT, asks("init-db"))
    result = chat("prepare the database", "y")
    assert f"schema v{db.SCHEMA_VERSION} (0 migrations applied)" in result.output


def test_operator_decisions_from_the_chat(fake: FakeLLM) -> None:
    task_id, subtask_id = escalated_task()
    ref = f"{task_id[:8]}/0"
    fake.on(CHAT, asks("retry", subtask=ref, note="the key is fixed"))
    result = chat("try that again", "y")
    assert "[0] fetch: needs_human → ready" in result.output
    assert "Note: the key is fixed" in result.output
    store = Store(db.open_db())
    try:
        assert store.get_subtask(subtask_id).status is SubtaskStatus.READY
        last = history(store.conn, task_id)[-1]
        assert (last.actor, last.action) == ("human", "operator_retry")
        assert last.details["reason"] == "the key is fixed"
    finally:
        store.conn.close()


def test_cancel_from_the_chat(fake: FakeLLM) -> None:
    task_id, _ = escalated_task()
    fake.on(CHAT, asks("cancel", task=task_id[:8]))
    result = chat("forget it", "y")
    assert "1 unfinished subtask stops." in result.output
    store = Store(db.open_db())
    try:
        assert store.task_status(task_id) is TaskStatus.CANCELLED
    finally:
        store.conn.close()


def test_run_starts_a_task_of_the_session_after_yes(fake: FakeLLM) -> None:
    plan = {"subtasks": [{"description": "Find flights", "acceptance_criteria": "listed"}]}
    fake.on(CHAT, asks("run", goal="Find cheap flights to Lisbon", title="Flights"))
    fake.on("Plan task", json.dumps(plan))
    fake.on("Execute subtask", '{"flights": ["TP 1351"]}')
    fake.on("Review subtask", '{"verdict": "accept"}')
    fake.on(RETELL, '{"text": "Found TP 1351."}')
    result = chat("flights to Lisbon", "y")
    assert "Start a task: Find cheap flights to Lisbon" in result.output
    assert "Found TP 1351." in result.output
    assert any(m.task_id for m in conversation())


def test_resume_continues_a_task_after_yes(fake: FakeLLM) -> None:
    task_id, _ = escalated_task()
    fake.on(CHAT, asks("retry", subtask=f"{task_id[:8]}/0"), asks("resume", task=task_id[:8]))
    fake.on("Execute subtask", '{"ok": true}')
    fake.on("Review subtask", '{"verdict": "accept"}')
    fake.on(RETELL, '{"text": "Done now."}')
    result = chat("retry it", "y", "go on", "y")
    assert f"Continue task {task_id[:8]}" in result.output and "Done now." in result.output


# ── Problems ─────────────────────────────────────────────────────────────────


def test_an_unknown_command_is_told_to_the_interlocutor(fake: FakeLLM) -> None:
    fake.on(CHAT, asks("format disk"))
    result = chat("wipe it")
    assert "DESMAN has no command format disk" in result.output
    assert '"keys add"' in conversation()[-1].text  # the event lists what there is


def test_a_problem_from_the_command_is_shown_and_told(fake: FakeLLM) -> None:
    escalated_task()
    fake.on(CHAT, asks("status", task="nosuchtask"))
    result = chat("status of nosuchtask")
    assert "nosuchtask" in result.output
    assert "did not run" in conversation()[-1].text


# ── Keys ─────────────────────────────────────────────────────────────────────


def test_a_key_is_typed_into_the_hidden_field_only(fake: FakeLLM) -> None:
    fake.on(
        CHAT, asks("keys add", provider="groq", label="work"), '{"type": "reply", "text": "Ok"}'
    )
    result = chat("add my groq key", KEY, "thanks")
    assert [(k.value, k.label) for k in saved().keys("GROQ_API_KEY")] == [(KEY, "work")]
    assert KEY not in result.output
    assert ("operator_key_added", {"provider": "groq", "key_id": key_id(KEY)}) in journal()
    # Neither the history, nor the journals, nor what the model saw hold the key.
    assert KEY not in everything_stored().replace(key_id(KEY), "")
    assert all(KEY not in r.prompt for r in fake.calls)


def test_a_hidden_field_left_empty_adds_nothing(fake: FakeLLM) -> None:
    fake.on(CHAT, asks("keys add", provider="groq"))
    chat("add a key", "")
    assert saved().keys("GROQ_API_KEY") == []
    assert [action for action, _ in journal()] == ["operator_declined"]


@pytest.mark.parametrize(
    "args",
    [
        {"provider": "groq", "label": KEY},
        {"provider": "groq", "key": KEY},
        {"provider": KEY},
    ],
)
def test_a_key_written_by_the_model_changes_nothing(fake: FakeLLM, args: dict[str, Any]) -> None:
    fake.on(CHAT, asks("keys add", **args))
    result = chat("here is my key, add it")
    assert "looked like an API key" in result.output
    assert saved().keys("GROQ_API_KEY") == []
    assert journal() == []
    assert KEY not in everything_stored() and KEY not in result.output


# ── The wizard in the chat ───────────────────────────────────────────────────


def test_a_section_of_the_wizard_from_the_chat(fake: FakeLLM) -> None:
    fake.on(CHAT, asks("config", section="permissions"))
    # Without a terminal the checkbox asks for each permission in turn.
    result = chat("let me set permissions", "y", "n", "n", "y", "n")
    assert "Open it?" in result.output and "Agent permissions" in result.output
    assert saved().permissions.network_access is True
    assert ("operator_permission_grant", {"permission": "network_access"}) in journal()


def test_the_wizard_opens_only_when_the_person_says_so(fake: FakeLLM) -> None:
    fake.on(CHAT, asks("config"))  # a question the model took for a wish to set DESMAN up
    before = saved()
    result = chat("which providers are connected?", "n")
    assert "Open the whole setup wizard" in result.output
    assert "Language" not in result.output and "Model providers" not in result.output
    assert saved() == before
    assert [action for action, _ in journal()] == ["operator_declined"]


def test_leaving_the_wizard_saves_nothing(fake: FakeLLM) -> None:
    fake.on(CHAT, asks("config"))
    before = saved()
    chat("open the settings", "y")  # the input ends inside the wizard
    assert saved() == before
    assert [action for action, _ in journal()] == ["operator_declined"]


# ── desman permissions ───────────────────────────────────────────────────────


def desman(*args: str) -> Result:
    return CliRunner().invoke(cli, list(args))


def test_permissions_command_shows_grants_and_revokes() -> None:
    save(make_config(language="en", configured=True))
    shown = desman("permissions")
    assert shown.exit_code == 0 and "network_access" in shown.output
    granted = desman("permissions", "--grant", "network_access", "--grant", "shell_execution")
    assert granted.exit_code == 0, granted.output
    assert "network_access granted." in granted.output
    assert saved().permissions.network_access and saved().permissions.shell_execution
    revoked = desman("permissions", "--revoke", "shell_execution")
    assert "shell_execution revoked." in revoked.output
    assert not saved().permissions.shell_execution
    assert "Nothing changed." in desman("permissions", "--revoke", "shell_execution").output


def test_permissions_command_refuses_a_contradiction() -> None:
    save(make_config(language="en", configured=True))
    result = desman("permissions", "--grant", "admin_rights", "--revoke", "admin_rights")
    assert result.exit_code == 2 and "both granted and revoked: admin_rights" in result.output
    assert not saved().permissions.admin_rights


def test_the_journal_is_append_only() -> None:
    conn = db.open_db(":memory:")
    operator.record(conn, "operator_command")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM operator_log")
    conn.close()
