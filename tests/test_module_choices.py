"""Scoped human preferences: durable storage, live routing, session isolation and safe UI."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from conftest import FakeClock
from desman import commands, modules_cli, providers, runs
from desman.cli import cli
from desman.config import ApiKey, Config, open_config
from desman.core import calls, choices, db, operator
from desman.core.models import Kind, SubtaskSpec, TaskSpec
from desman.core.sessions import Role as SessionRole
from desman.core.sessions import Sessions
from desman.core.store import Store
from desman.llm import router
from desman.llm.base import Message, Request
from desman.llm.openai_compat import key_id
from helpers import make_config

FIRST = "first-secret-value-111111"
SECOND = "second-secret-value-222222"
REQUEST = Request(messages=[Message(role="user", content="hello")])


def config() -> Config:
    cfg = make_config(language="en", configured=True)
    cfg.api_keys["GROQ_API_KEY"] = [
        ApiKey(value=FIRST, label="main"),
        ApiKey(value=SECOND, label="backup"),
    ]
    return cfg


def task(conn: sqlite3.Connection, kind: Kind = Kind.CODE) -> str:
    return Store(conn).create_task(
        TaskSpec(
            goal="goal",
            subtasks=[SubtaskSpec(description="work", acceptance_criteria="ok", kind=kind)],
        )
    )


def test_upgrade_35_and_choice_constraints(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "old.db")
    assert db.migrate(conn, db.MIGRATIONS[:35]) == 35
    identity = task(conn)
    assert db.migrate(conn) == db.SCHEMA_VERSION - 35
    db.verify_schema(conn)
    choices.set_choice(conn, "task", identity, "code", choices.Choice(model="groq/test"))
    assert db.migrate(conn) == 0
    for scope, role, model, key in [
        ("wrong", "code", "groq/test", None),
        ("task", "executor", "groq/test", None),
        ("task", "code", None, None),
    ]:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO module_choices VALUES (?, ?, ?, ?, ?, 'now')",
                (scope, identity, role, model, key),
            )
    conn.close()


def test_set_clear_validation_and_atomic_journal() -> None:
    conn = db.open_db(":memory:")
    identity = task(conn)
    initial = choices.Choice(model="groq/test", key=key_id(SECOND))
    assert choices.set_choice(conn, "task", identity, "code", initial) == (
        choices.Choice(),
        initial,
    )
    choices.set_choice(conn, "task", identity, "code", initial)  # No-op.
    assert len(operator.entries(conn)) == 1
    assert choices.list_choices(conn, "task", identity) == {"code": initial}
    for role in ("bad", "executor"):
        with pytest.raises(ValueError):
            choices.set_choice(conn, "task", identity, role, initial)
    with pytest.raises(ValueError):
        choices.set_choice(conn, "session", "absent", "code", initial)
    for payload in ({"model": "bad"}, {"key": SECOND}):
        with pytest.raises(ValidationError):
            choices.Choice.model_validate(payload)
    choices.set_choice(conn, "task", identity, "reviewer", initial)
    choices.clear(conn, "task", identity)
    assert choices.list_choices(conn, "task", identity) == {}
    entries = operator.entries(conn)
    assert len(entries) == 4
    assert entries[0].details["new"] == {"model": "groq/test", "key": key_id(SECOND)}
    assert SECOND not in json.dumps([e.model_dump() for e in entries])
    conn.close()


@pytest.mark.parametrize(
    "field,role,kind",
    [
        ("reviewer", calls.Role.REVIEWER, None),
        ("code", calls.Role.EXECUTOR, Kind.CODE),
    ],
)
def test_live_precedence_other_lineup_and_cross_connection(
    tmp_path: Path, clock: FakeClock, field: str, role: calls.Role, kind: Kind | None
) -> None:
    conn = db.open_db(tmp_path / "shared.db")
    editor = db.open_db(tmp_path / "shared.db")
    cfg = config()
    lineup = router.lineup(cfg, role, provider="groq", kind=kind)
    auto, preferred, scoped = [c.label for c in (lineup[0], lineup[1], lineup[-1])]
    setattr(cfg.tuning.models, field, preferred)
    setattr(cfg.tuning.keys, field, key_id(FIRST))
    identity = task(conn)
    seen = []

    def send(request: httpx.Request) -> httpx.Response:
        seen.append((json.loads(request.content)["model"], request.headers["authorization"]))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    models = runs.models(
        cfg,
        conn,
        provider="groq",
        cache=False,
        client=httpx.Client(transport=httpx.MockTransport(send)),
        clock=clock,
        sleep=clock.sleep,
    )
    llm, other = models.executor(kind) if kind else (models.for_(role), None)
    with calls.for_(calls.Purpose(role, task_id=identity)):
        llm.complete(REQUEST)
        choices.set_choice(
            editor, "task", identity, field, choices.Choice(model=scoped, key=key_id(SECOND))
        )
        llm.complete(REQUEST)
        assert models.first(role) is not None
        if other:
            assert isinstance(other, router.Fallback)
            # Alternatives now resolve at call time against the model actually used.
            reply = other.complete(REQUEST)
            assert f"{reply.provider}/{reply.model}" != scoped
            alternate = seen.pop()
            assert alternate[0] != scoped.split("/", 1)[1]
        # Clearing just model inherits the role model, while retaining the scoped key.
        modules_cli.set_module(editor, cfg, "task", identity, field, model="auto")
        llm.complete(REQUEST)
        choices.clear(editor, "task", identity, field)
        llm.complete(REQUEST)
    assert seen == [
        (preferred.split("/", 1)[1], f"Bearer {FIRST}"),
        (scoped.split("/", 1)[1], f"Bearer {SECOND}"),
        (preferred.split("/", 1)[1], f"Bearer {SECOND}"),
        (preferred.split("/", 1)[1], f"Bearer {FIRST}"),
    ]
    assert cfg.tuning.models.model_dump()[field] == preferred
    empty_cfg = config()
    plain = runs.models(empty_cfg, conn, provider="groq", cache=False)
    with calls.for_(calls.Purpose(role, task_id=identity)):
        chain = plain.executor(kind)[0] if kind else plain.for_(role)
        assert isinstance(chain, router.Fallback) and chain.choices[0].label == auto
    editor.close()
    conn.close()


def test_session_scope_and_temporary_erase(clock: FakeClock) -> None:
    conn = db.open_db(":memory:")
    sessions = Sessions(conn, clock=clock)
    one = sessions.add(None, SessionRole.USER, "one", temporary=True).session_id
    two = sessions.add(None, SessionRole.USER, "two").session_id
    cfg = config()
    candidates = router.lineup(cfg, calls.Role.INTERLOCUTOR, provider="groq")
    preferred, selected = candidates[1].label, candidates[-1].label
    cfg.tuning.models.interlocutor = preferred
    choices.set_choice(
        conn, "session", one, "interlocutor", choices.Choice(model=selected, key=key_id(SECOND))
    )
    seen = []

    def send(request: httpx.Request) -> httpx.Response:
        seen.append((json.loads(request.content)["model"], request.headers["authorization"]))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    models = runs.models(
        cfg,
        conn,
        provider="groq",
        cache=False,
        client=httpx.Client(transport=httpx.MockTransport(send)),
        clock=clock,
        sleep=clock.sleep,
    )
    for session in (one, two):
        with calls.for_(calls.Purpose(calls.Role.INTERLOCUTOR, session_id=session)):
            models.for_(calls.Role.INTERLOCUTOR).complete(REQUEST)
    assert seen == [
        (selected.split("/", 1)[1], f"Bearer {SECOND}"),
        (preferred.split("/", 1)[1], f"Bearer {FIRST}"),
    ]
    sessions.erase(one)
    assert choices.list_choices(conn, "session", one) == {}
    choices.set_choice(conn, "session", two, "interlocutor", choices.Choice(model=selected))
    with pytest.raises(sqlite3.IntegrityError):
        sessions.erase(two)
    assert choices.get(conn, "session", two, "interlocutor").model == selected
    conn.close()


def test_network_chains_ignore_scoped_choices(clock: FakeClock) -> None:
    conn = db.open_db(":memory:")
    cfg = config()
    identity = task(conn)
    selected = router.lineup(cfg, calls.Role.REVIEWER, provider="groq")[-1].label
    choices.set_choice(
        conn, "task", identity, "reviewer", choices.Choice(model=selected, key=key_id(SECOND))
    )
    seen = []

    def send(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    models = runs.models(
        cfg,
        conn,
        only={"groq"},
        cache=False,
        client=httpx.Client(transport=httpx.MockTransport(send)),
        clock=clock,
        sleep=clock.sleep,
    )
    with calls.for_(calls.Purpose(calls.Role.REVIEWER, task_id=identity)):
        first = models.first(calls.Role.REVIEWER)
        assert first and first.label != selected
        models.for_(calls.Role.REVIEWER).complete(REQUEST)
    assert seen == [f"Bearer {FIRST}"]
    conn.close()


def test_cli_show_json_partial_updates_clear_and_errors() -> None:
    cfg = config()
    open_config().save(cfg)
    conn = db.open_db()
    identity = task(conn)
    ref = Store(conn).short_id(identity)
    session = Sessions(conn).add(None, SessionRole.USER, "chat").session_id
    selected = router.lineup(cfg, calls.Role.EXECUTOR, kind=Kind.CODE)[-1].label
    runner = CliRunner()
    result = runner.invoke(
        cli, ["modules", "set", "code", "--task", ref, "--model", selected, "--key", "groq:backup"]
    )
    assert result.exit_code == 0, result.output
    assert "→" in result.output and SECOND not in result.output
    data = json.loads(runner.invoke(cli, ["modules", "show", "--task", ref, "--json"]).output)
    assert {row["role"] for row in data} == {"code", "reviewer"}
    code = next(row for row in data if row["role"] == "code")
    assert code["model"] == selected and code["source"] == "task"
    assert code["key_label"] == f"backup {providers.mask(SECOND)}"
    assert code["calls"] == code["tokens"] == 0 and code["state"] == "idle"
    assert FIRST not in json.dumps(data) and SECOND not in json.dumps(data)
    assert json.loads(runner.invoke(cli, ["modules", "--json"]).output) == data
    assert (
        runner.invoke(
            cli,
            ["modules", "set", "interlocutor", "--session", session[:8], "--key", "groq:backup"],
        ).exit_code
        == 0
    )
    assert runner.invoke(cli, ["modules", "clear", "--task", ref, "code"]).exit_code == 0
    assert choices.get(conn, "task", identity, "code") == choices.Choice()
    for args in [
        ["set", "code", "--model", selected],
        ["set", "code", "--task", ref, "--session", session, "--key", "auto"],
        ["set", "code", "--task", "absent", "--key", "auto"],
        ["set", "interlocutor", "--session", "absent", "--key", "auto"],
        ["set", "bad", "--task", ref, "--model", selected],
        ["set", "code", "--task", ref, "--model", "bad"],
        ["set", "code", "--task", ref, "--key", SECOND],
        ["set", "code", "--task", ref],
    ]:
        result = runner.invoke(cli, ["modules", *args])
        assert result.exit_code != 0, args
        assert SECOND not in result.output
    conn.close()


def test_catalog_proposal_uses_current_session_and_waits_for_confirmation() -> None:
    cfg = config()
    open_config().save(cfg)
    conn = db.open_db(":memory:")
    session = Sessions(conn).add(None, SessionRole.USER, "chat").session_id
    ctx = commands.Context(
        Store(conn), open_config(), lambda *_: None, lambda _: None, session_id=lambda: session
    )
    entry = commands.find("modules set")
    assert entry and entry.kind == commands.Kind.CHANGE
    args = entry.args.model_validate({"role": "interlocutor", "key": "groq:backup"})
    proposal = entry.handler(ctx, args)
    assert isinstance(proposal, commands.Proposal)
    assert choices.get(conn, "session", session, "interlocutor") == choices.Choice()
    proposal.apply()
    assert choices.get(conn, "session", session, "interlocutor").key == key_id(SECOND)
    conn.close()


def test_journal_failure_rolls_back_choice() -> None:
    conn = db.open_db(":memory:")
    identity = task(conn)
    conn.execute(
        "CREATE TRIGGER refuse_operator BEFORE INSERT ON operator_log "
        "BEGIN SELECT RAISE(ABORT, 'test journal failure'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        choices.set_choice(conn, "task", identity, "code", choices.Choice(model="groq/test"))
    assert choices.get(conn, "task", identity, "code") == choices.Choice()
    conn.close()


def test_scoped_structured_replies_still_use_cache(clock: FakeClock) -> None:
    from pydantic import BaseModel

    from desman.llm.structured import complete_json

    class Answer(BaseModel):
        answer: str

    conn = db.open_db(":memory:")
    cfg = config()
    identity = task(conn)
    chosen = router.lineup(cfg, calls.Role.REVIEWER, provider="groq")[-1].label
    choices.set_choice(conn, "task", identity, "reviewer", choices.Choice(model=chosen))
    seen = []

    def send(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": "ok"}'}}]})

    models = runs.models(
        cfg,
        conn,
        provider="groq",
        client=httpx.Client(transport=httpx.MockTransport(send)),
        clock=clock,
        sleep=clock.sleep,
    )
    with calls.for_(calls.Purpose(calls.Role.REVIEWER, task_id=identity)):
        for _ in range(2):
            assert (
                complete_json(
                    models.for_(calls.Role.REVIEWER), REQUEST, Answer, heading="Answer"
                ).answer
                == "ok"
            )
    assert len(seen) == 1
    row = next(r for r in modules_cli.modules_data(conn, cfg, identity) if r["role"] == "reviewer")
    assert row["calls"] == 1
    conn.close()


def test_existing_chat_honours_its_session_choice(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    cfg = config()
    open_config().save(cfg)
    conn = db.open_db()
    session = Sessions(conn).add(None, SessionRole.USER, "earlier message").session_id
    chosen = router.lineup(cfg, calls.Role.INTERLOCUTOR)[-1].label
    choices.set_choice(
        conn, "session", session, "interlocutor", choices.Choice(model=chosen, key=key_id(SECOND))
    )
    conn.close()
    seen = []

    def send(request: httpx.Request) -> httpx.Response:
        seen.append((json.loads(request.content)["model"], request.headers["authorization"]))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"type": "reply", "text": "Scoped chat reply."})
                        }
                    }
                ]
            },
        )

    original = runs.models

    def models(config: Config, conn: sqlite3.Connection, *, cache: bool = True) -> runs.Models:
        return original(
            config,
            conn,
            cache=False,
            client=httpx.Client(transport=httpx.MockTransport(send)),
            clock=clock,
            sleep=clock.sleep,
        )

    monkeypatch.setattr(runs, "models", models)
    result = CliRunner().invoke(cli, ["-c"], input="hello\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "Scoped chat reply." in result.output
    assert seen == [(chosen.split("/", 1)[1], f"Bearer {SECOND}")]


@pytest.mark.parametrize(
    "scope,role",
    [("task", "planner"), ("task", "interlocutor"), ("session", "reviewer")],
)
def test_unused_scope_roles_are_refused_without_writes(scope: choices.Scope, role: str) -> None:
    import click

    cfg = config()
    open_config().save(cfg)
    conn = db.open_db()
    task_id = task(conn)
    session = Sessions(conn).add(None, SessionRole.USER, "chat").session_id
    identity = task_id if scope == "task" else session
    ref = Store(conn).short_id(task_id) if scope == "task" else session[:8]
    message = (
        f"{role} cannot be chosen for one {scope}: only "
        f"{', '.join(choices.ROLES_BY_SCOPE[scope])}. "
        "Choose it for the role with `desman tuning`."
    )
    before = operator.entries(conn)

    def assert_no_writes() -> None:
        assert conn.execute("SELECT count(*) FROM module_choices").fetchone()[0] == 0
        assert operator.entries(conn) == before

    result = CliRunner().invoke(cli, ["modules", "set", role, f"--{scope}", ref, "--key", "auto"])
    assert result.exit_code == 1, result.output
    assert message in result.output
    assert_no_writes()

    with pytest.raises(ValueError, match=f"a {scope} choice is not read by the {role}"):
        choices.set_choice(conn, scope, identity, role, choices.Choice(model="groq/test"))
    assert_no_writes()

    ctx = commands.Context(
        Store(conn), open_config(), lambda *_: None, lambda _: None, session_id=lambda: session
    )
    entry = commands.find("modules set")
    assert entry is not None
    args = entry.args.model_validate({"role": role, scope: ref, "key": "auto"})
    with pytest.raises(click.ClickException) as error:
        entry.handler(ctx, args)
    assert str(error.value) == message
    assert_no_writes()
    conn.close()


def test_clearing_a_role_its_scope_does_not_read_is_refused_plainly() -> None:
    import click
    import pytest as _pytest

    from desman import modules_cli

    with _pytest.raises(click.ClickException, match="planner"):
        modules_cli.check_role("task", "planner")
    with _pytest.raises(click.ClickException, match="reviewer"):
        modules_cli.check_role("session", "reviewer")
    modules_cli.check_role("task", None)  # every role of the scope
