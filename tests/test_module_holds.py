"""Human module holds: atomic ownership, call-time fallback, safe page contracts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from conftest import FakeClock
from desman import modules_cli, runs, ui
from desman.config import Config
from desman.core import calls, db, holds, operator
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.sessions import Role, Sessions
from desman.core.store import Store
from desman.gui.module_view import cards
from desman.llm import router
from desman.llm.base import Message, Request
from desman.llm.openai_compat import key_id
from helpers import make_config

SECRET = "module-holds-secret"
REQUEST = Request(messages=[Message(role="user", content="hello")])


def config() -> Config:
    return make_config(language="en", configured=True, api_keys={"GROQ_API_KEY": SECRET})


def task(store: Store, goal: str = "work") -> str:
    return store.create_task(
        TaskSpec(goal=goal, subtasks=[SubtaskSpec(description="work", acceptance_criteria="ok")])
    )


def test_upgrade_42(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "old.db")
    db.migrate(conn, db.MIGRATIONS[:42])
    identity = task(Store(conn))
    assert db.migrate(conn) == 1
    db.verify_schema(conn)
    assert Store(conn).find_task(identity)
    assert holds.listed(conn) == []
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO module_holds VALUES ('a','b','wrong','','stopped','now','{}')")
    conn.close()


def test_stop_continue_preserves_manual_pause_and_journals(clock: FakeClock) -> None:
    conn = db.open_db(":memory:")
    store = Store(conn, clock=clock)
    one, two, other = task(store), task(store), task(store)
    store.pause_task(two)
    h = holds.Hold(source=key_id(SECRET), model="groq/model")
    holds.stop(conn, h, [one, two])
    assert set(holds.listed(conn)[0].paused_tasks) == {one}
    assert store.find_task(one)["paused_at"]
    assert store.find_task(other)["paused_at"] is None
    holds.stop(conn, h, [other])  # Idempotent, no additional pauses.
    assert store.find_task(other)["paused_at"] is None
    holds.continue_(conn, h)
    assert store.find_task(one)["paused_at"] is None
    assert store.find_task(two)["paused_at"]
    assert [e.action for e in operator.entries(conn)] == [
        "operator_module_stop",
        "operator_module_continue",
    ]
    assert SECRET not in json.dumps([e.model_dump() for e in operator.entries(conn)])
    conn.close()


def test_stop_and_journal_rollback_together() -> None:
    conn = db.open_db(":memory:")
    store = Store(conn)
    identity = task(store)
    conn.execute(
        "CREATE TRIGGER reject_hold BEFORE INSERT ON module_holds BEGIN SELECT RAISE(ABORT,'no'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        holds.stop(conn, holds.Hold(source=key_id(SECRET), model="groq/model"), [identity])
    assert store.find_task(identity)["paused_at"] is None
    assert not holds.listed(conn) and not operator.entries(conn)
    conn.close()


def test_remove_releases_stop_and_restore_does_not_resume_manual_pause(clock: FakeClock) -> None:
    conn = db.open_db(":memory:")
    store = Store(conn, clock=clock)
    identity = task(store)
    h = holds.Hold(source=key_id(SECRET), model="groq/model", scope="task", scope_id=identity)
    holds.stop(conn, h, [identity])
    holds.remove(conn, h)
    assert store.find_task(identity)["paused_at"] is None
    store.pause_task(identity)
    holds.restore(conn, h)
    assert store.find_task(identity)["paused_at"]
    conn.close()


def test_scoped_removal_falls_back_live_and_new_work_skips_all(clock: FakeClock) -> None:
    conn = db.open_db(":memory:")
    store = Store(conn, clock=clock)
    one, two = task(store), task(store)
    cfg = config()
    seen = []

    def send(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    models = runs.models(
        cfg,
        conn,
        provider="groq",
        cache=False,
        clock=clock,
        sleep=clock.sleep,
        client=httpx.Client(transport=httpx.MockTransport(send)),
    )
    first = router.lineup(cfg, calls.Role.EXECUTOR, provider="groq")[0]
    h = holds.Hold(source=key_id(SECRET), model=first.label, scope="task", scope_id=one)
    holds.remove(conn, h)
    with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id=one)):
        models.llm.complete(REQUEST)
    with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id=two)):
        models.llm.complete(REQUEST)
    assert seen[0] != first.model and seen[1] == first.model
    holds.remove(conn, h.model_copy(update={"scope": "all", "scope_id": ""}))
    with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id=task(store))):
        models.llm.complete(REQUEST)
    assert seen[-1] != first.model
    conn.close()


def test_empty_lineup_is_limit_wait_without_http(clock: FakeClock) -> None:
    conn = db.open_db(":memory:")
    cfg = config()
    identity = task(Store(conn))
    for choice in router.lineup(cfg, calls.Role.EXECUTOR, provider="groq"):
        holds.remove(conn, holds.Hold(source=key_id(SECRET), model=choice.label))
    models = runs.models(cfg, conn, provider="groq", cache=False, clock=clock, sleep=clock.sleep)
    with (
        calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id=identity)),
        pytest.raises(runs.ModuleWaiting, match="no module left for text") as error,
    ):
        models.llm.complete(REQUEST)
    assert error.value.kind.value == "limit"
    assert conn.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0] == 0
    conn.close()


def test_interlocutor_stop_waits_then_continue_answers(clock: FakeClock) -> None:
    conn = db.open_db(":memory:")
    cfg = config()
    session = Sessions(conn).add(None, Role.USER, "hello").session_id
    first = router.lineup(cfg, calls.Role.INTERLOCUTOR, provider="groq")[0]
    h = holds.Hold(source=key_id(SECRET), model=first.label, scope="session", scope_id=session)
    models = runs.models(
        cfg,
        conn,
        provider="groq",
        cache=False,
        clock=clock,
        sleep=clock.sleep,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})
            )
        ),
    )
    holds.stop(conn, h)
    with calls.for_(calls.Purpose(calls.Role.INTERLOCUTOR, session_id=session)):
        with pytest.raises(runs.ModuleWaiting):
            models.for_(calls.Role.INTERLOCUTOR).complete(REQUEST)
        holds.continue_(conn, h)
        assert models.for_(calls.Role.INTERLOCUTOR).complete(REQUEST).text == "hello"
    conn.close()


def test_snapshot_deduplicates_and_has_short_titles_and_chat() -> None:
    conn = db.open_db(":memory:")
    cfg = config()
    goal = "A very long goal with many details that should stay inside Tasks only"
    identity = task(Store(conn), goal)
    session = Sessions(conn).add(None, Role.USER, "hello").session_id
    snapshot = cards(conn, cfg, modules_cli.modules_data(conn, cfg))
    identities = [(m["module_source"], m["model"]) for m in snapshot]
    assert len(identities) == len(set(identities))
    assert goal not in json.dumps(snapshot)
    work = [w for m in snapshot for w in m["work"]]
    assert any(w["id"] == identity and w["title"].endswith("…") for w in work)
    assert any(w["id"] == session and w["kind"] == "chat" for w in work)
    conn.close()


@pytest.mark.parametrize(
    "extra",
    [{"extra": 1}, {"scope": "task"}, {"source": SECRET}, {"scope": "all", "scope_id": "x"}],
)
def test_page_hold_contract_forbids_invalid_fields(extra: dict[str, object]) -> None:
    payload = {
        "type": "module_hold",
        "operation": "stop",
        "scope": "all",
        "model": "groq/model",
        "source": key_id(SECRET),
    }
    with pytest.raises(ValidationError):
        ui.ACTIONS.validate_python(payload | extra)


def test_empty_lineup_coordinator_pauses_without_spending_attempt(clock: FakeClock) -> None:
    from desman.core.coordinator import Coordinator, Review, Success, Verdict, Work
    from desman.core.models import Subtask
    from desman.core.states import SubtaskStatus

    conn = db.open_db(":memory:")
    cfg = config()
    store = Store(conn, clock=clock)
    identity = task(store)
    for choice in router.lineup(cfg, calls.Role.EXECUTOR, provider="groq"):
        holds.remove(conn, holds.Hold(source=key_id(SECRET), model=choice.label))
    models = runs.models(cfg, conn, provider="groq", cache=False, clock=clock, sleep=clock.sleep)

    def execute(work: Work) -> Success:
        models.llm.complete(REQUEST)
        return Success(content={"ok": True})

    def review(subtask: Subtask, artifact: dict[str, object] | None) -> Review:
        return Review(verdict=Verdict.ACCEPT)

    coordinator = Coordinator(store, execute, review, providers=models.labels, sleep=clock.sleep)
    before = store.list_subtasks(identity)[0].attempts
    assert coordinator.tick(identity, [])
    subtask = store.list_subtasks(identity)[0]
    assert subtask.status is SubtaskStatus.PAUSED
    assert subtask.attempts == before
    conn.close()


def test_stop_body_selects_assigned_tasks_only(clock: FakeClock) -> None:
    from desman.core import choices

    conn = db.open_db(":memory:")
    store = Store(conn, clock=clock)
    one, manual, other = task(store), task(store), task(store)
    cfg = config()
    lineup = router.lineup(cfg, calls.Role.EXECUTOR, provider="groq")
    first, second = lineup[0].label, lineup[1].label
    cfg.tuning.models.reviewer = second
    choices.set_choice(conn, "task", other, "text", choices.Choice(model=second))
    store.pause_task(manual)
    hold = holds.Hold(source=key_id(SECRET), model=first)
    modules_cli.hold_change(conn, cfg, "stop", hold)
    assert set(holds.listed(conn)[0].paused_tasks) == {one}
    assert store.find_task(other)["paused_at"] is None
    modules_cli.hold_change(conn, cfg, "continue", hold)
    assert store.find_task(one)["paused_at"] is None
    assert store.find_task(manual)["paused_at"]
    conn.close()


def test_remove_task_cancels_and_hides_atomically(clock: FakeClock) -> None:
    from desman.core.states import SubtaskStatus

    conn = db.open_db(":memory:")
    store = Store(conn, clock=clock)
    identity = task(store)
    conn.execute(
        "CREATE TRIGGER reject_hide BEFORE UPDATE OF hidden_at ON tasks BEGIN SELECT RAISE(ABORT,'no'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.remove_task(identity)
    assert store.list_subtasks(identity)[0].status is SubtaskStatus.READY
    conn.execute("DROP TRIGGER reject_hide")
    store.remove_task(identity)
    assert store.list_subtasks(identity)[0].status is SubtaskStatus.CANCELLED
    assert store.find_task(identity)["hidden_at"]
    conn.close()


def test_terminal_queues_held_turn_and_retries_with_fake_clock(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman.chat import Chat
    from desman.config import open_config
    from desman.llm.fake import FakeLLM

    conn = db.open_db(":memory:")
    sessions = Sessions(conn, clock=clock)
    open_config().save(config())
    lines = iter(["hello", "/exit"])

    def held(request: Request) -> str:
        raise runs.ModuleWaiting("interlocutor")

    fake = FakeLLM(held, '{"type":"reply","text":"hello back"}')
    monkeypatch.setattr(runs, "models", lambda *a, **kw: runs.Models(fake, None))
    chat = Chat(sessions, read=lambda: next(lines), sleep=clock.sleep)
    chat.run()
    assert chat.session_id
    messages = sessions.messages(chat.session_id)
    assert [m.text for m in messages if m.role is Role.USER] == ["hello"]
    assert [m.text for m in messages if m.role is Role.INTERLOCUTOR] == ["hello back"]
    assert sessions.turn_settled(next(m.seq for m in messages if m.role is Role.USER))
    assert clock.slept == 1
    conn.close()


def test_one_remaining_module_does_not_wait_for_missing_alternate(clock: FakeClock) -> None:
    conn = db.open_db(":memory:")
    cfg = config()
    store = Store(conn, clock=clock)
    identity = task(store)
    lineup = router.lineup(cfg, calls.Role.EXECUTOR, provider="groq")
    for choice in lineup[1:]:
        holds.remove(conn, holds.Hold(source=key_id(SECRET), model=choice.label))
    models = runs.models(cfg, conn, provider="groq", cache=False, clock=clock, sleep=clock.sleep)
    main, alternate = models.labels(store.list_subtasks(identity)[0])
    assert main == lineup[0].label
    assert alternate in {main, None}
    conn.close()


def test_global_stop_queues_existing_chat_but_new_chat_uses_another_module(
    clock: FakeClock,
) -> None:
    conn = db.open_db(":memory:")
    cfg = config()
    sessions = Sessions(conn, clock=clock)
    old = sessions.add(None, Role.USER, "old").session_id
    first = router.lineup(cfg, calls.Role.INTERLOCUTOR, provider="groq")[0]
    h = holds.Hold(source=key_id(SECRET), model=first.label)
    clock.advance(1)
    holds.stop(conn, h, clock=clock)
    clock.advance(1)
    new = sessions.add(None, Role.USER, "new").session_id
    models = runs.models(cfg, conn, provider="groq", cache=False, clock=clock, sleep=clock.sleep)
    with (
        calls.for_(calls.Purpose(calls.Role.INTERLOCUTOR, session_id=old)),
        pytest.raises(runs.ModuleWaiting),
    ):
        models.first(calls.Role.INTERLOCUTOR)
    with calls.for_(calls.Purpose(calls.Role.INTERLOCUTOR, session_id=new)):
        selected = models.first(calls.Role.INTERLOCUTOR)
        assert selected and selected.label != first.label
    conn.close()


def test_page_stop_is_immediate_and_remove_requires_confirmation(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman.chat import Chat
    from desman.config import open_config

    conn = db.open_db(":memory:")
    cfg = config()
    open_config().save(cfg)
    chat = Chat(Sessions(conn, clock=clock))
    identity = task(chat.store)
    model = router.lineup(cfg, calls.Role.EXECUTOR)[0].label
    confirmations = []
    answer = False

    def confirm(question: str, *, default: bool = False) -> bool:
        confirmations.append(question)
        return answer

    monkeypatch.setattr(ui, "confirm", confirm)
    payload = {
        "type": "module_hold",
        "operation": "stop",
        "source": key_id(SECRET),
        "model": model,
        "scope": "task",
        "scope_id": identity,
    }
    chat._action(ui.ACTIONS.validate_python(payload))
    assert not confirmations and chat.store.find_task(identity)["paused_at"]
    chat._action(ui.ACTIONS.validate_python(payload | {"operation": "continue"}))
    assert not confirmations and chat.store.find_task(identity)["paused_at"] is None
    chat._action(ui.ACTIONS.validate_python(payload | {"operation": "remove"}))
    assert len(confirmations) == 1 and not holds.listed(conn)
    assert model in confirmations[0]
    answer = True
    chat._action(ui.ACTIONS.validate_python(payload | {"operation": "remove"}))
    assert holds.listed(conn)[0].state == "removed"
    conn.close()


def test_lent_module_serving_local_work_has_one_card() -> None:
    from desman.config import Donation, Network

    conn = db.open_db(":memory:")
    cfg = make_config(
        settings={"OLLAMA_MODEL": "llama3.2"},
        network=Network(
            enabled=True, donated={"ollama": Donation(calls_per_day=100, tokens_per_day=100_000)}
        ),
    )
    identity = task(Store(conn))
    snapshot = cards(conn, cfg, modules_cli.modules_data(conn, cfg))
    matching = [card for card in snapshot if card["model"] == "ollama/llama3.2"]
    assert len(matching) == 1
    assert matching[0]["section"] == "network"
    assert any(work["id"] == identity and work["kind"] == "task" for work in matching[0]["work"])
    conn.close()


def test_stopping_last_answering_fallback_queues_chat(clock: FakeClock) -> None:
    from test_live import call

    conn = db.open_db(":memory:")
    cfg = config()
    session = Sessions(conn, clock=clock).add(None, Role.USER, "hello").session_id
    fallback = router.lineup(cfg, calls.Role.INTERLOCUTOR, provider="groq")[1]
    with calls.for_(calls.Purpose(calls.Role.INTERLOCUTOR, session_id=session)):
        calls.recorder(conn)(call(provider="groq", model=fallback.model, key_id=key_id(SECRET)))
    snapshot = cards(conn, cfg, [])
    fallback_card = next(card for card in snapshot if card["model"] == fallback.label)
    assert any(work["id"] == session and work["kind"] == "chat" for work in fallback_card["work"])
    holds.stop(
        conn,
        holds.Hold(source=key_id(SECRET), model=fallback.label, scope="session", scope_id=session),
        clock=clock,
    )
    models = runs.models(cfg, conn, provider="groq", cache=False, clock=clock, sleep=clock.sleep)
    with (
        calls.for_(calls.Purpose(calls.Role.INTERLOCUTOR, session_id=session)),
        pytest.raises(runs.ModuleWaiting),
    ):
        models.for_(calls.Role.INTERLOCUTOR).complete(REQUEST)
    conn.close()


def test_continue_preserves_manual_repause_in_same_millisecond(clock: FakeClock) -> None:
    conn = db.open_db(":memory:")
    store = Store(conn, clock=clock)
    identity = task(store)
    hold = holds.Hold(source=key_id(SECRET), model="groq/model")
    holds.stop(conn, hold, [identity], clock=clock)
    first_stamp = store.find_task(identity)["paused_at"]
    store.resume_task(identity)
    store.pause_task(identity)
    assert store.find_task(identity)["paused_at"] == first_stamp
    holds.continue_(conn, hold, clock=clock)
    assert store.find_task(identity)["paused_at"] == first_stamp
    conn.close()
