"""The profile's notes in the knowledge tree (ADR 0009, ROADMAP K.5): one node, Personal/Profile,
a note per line; the interlocutor edits the notes without a window, the node only the person."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from conftest import FakeClock
from desman import runs
from desman.cli import cli
from desman.config import Config, ConfigManager, default_config_path
from desman.core import db, nodes, operator
from desman.core.nodes import PROFILE_ABOUT, PROFILE_PATH, NodeError, PersonalNodes
from desman.core.profile import Author, Profile
from desman.interlocutor import CHAT
from desman.llm.base import Request
from desman.llm.fake import FakeLLM
from helpers import make_config


@pytest.fixture
def conn(clock: FakeClock) -> Iterator[sqlite3.Connection]:
    connection = db.open_db(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(runs, "models", lambda config, conn, **_: runs.Models(llm, None))
    save(make_config(language="en", configured=True, personalize=True))
    return llm


def save(config: Config) -> None:
    ConfigManager(default_config_path()).save(config)


def chat(*lines: str) -> Result:
    result = CliRunner().invoke(cli, [], input="".join(f"{line}\n" for line in lines))
    assert result.exit_code == 0, result.output
    return result


def stored() -> list[str]:
    conn = db.open_db(db.default_db_path())
    try:
        return [n.text for n in Profile(conn).notes()]
    finally:
        conn.close()


def node_text() -> str | None:
    conn = db.open_db(db.default_db_path())
    try:
        node = PersonalNodes(conn).get(PROFILE_PATH)
        return None if node is None else node.text
    finally:
        conn.close()


def payload(request: Request) -> Any:
    return json.loads(request.prompt.split("\n", 1)[1])


def turns(llm: FakeLLM) -> list[Request]:
    return [r for r in llm.calls if r.prompt.startswith(CHAT)]


def reply(text: str, *edits: dict[str, Any], node: dict[str, Any] | None = None) -> str:
    return json.dumps(
        {"type": "reply", "text": text, "profile": list(edits), **({"node": node} if node else {})}
    )


# ── Store ────────────────────────────────────────────────────────────────────


def test_the_notes_are_the_lines_of_one_node(conn: sqlite3.Connection, clock: FakeClock) -> None:
    profile = Profile(conn, clock=clock)
    assert PersonalNodes(conn).get(PROFILE_PATH) is None
    profile.add("Likes tea", Author.INTERLOCUTOR)
    profile.add("Speaks Russian", Author.PERSON)
    node = PersonalNodes(conn).get(PROFILE_PATH)
    assert node is not None
    assert (node.text, node.about, node.author, node.untrusted) == (
        "- Likes tea\n- Speaks Russian",
        PROFILE_ABOUT,
        nodes.Author.PERSON,
        False,
    )
    assert [(n.id, n.text) for n in profile.notes()] == [(1, "Likes tea"), (2, "Speaks Russian")]
    assert profile.forget(1, Author.PERSON)
    assert [(n.id, n.text) for n in profile.notes()] == [(1, "Speaks Russian")]
    assert profile.forget(1, Author.INTERLOCUTOR)
    assert PersonalNodes(conn).get(PROFILE_PATH) is None  # the last note takes the node along
    actions = [e.action for e in operator.entries(conn)]
    assert actions == [
        "profile_note_added",
        "profile_note_added",
        "profile_note_forgotten",
        "profile_note_forgotten",
    ]  # one entry per change, none of the node's own


def test_a_change_finds_the_note_it_saw_when_the_lines_moved(conn: sqlite3.Connection) -> None:
    profile = Profile(conn)
    for text in ("A note", "B note", "C note"):
        profile.add(text, Author.PERSON)
    profile.forget(1, Author.PERSON, was="A note")
    # Line 3 is gone now, line 2 holds "C note": both find what the caller saw.
    changed = profile.change(3, "C changed", Author.PERSON, was="C note")
    assert changed is not None and changed.id == 2
    assert profile.forget(1, Author.PERSON, was="B note")
    assert [n.text for n in profile.notes()] == ["C changed"]
    assert profile.change(1, "x", Author.PERSON, was="Gone note") is None
    assert not profile.forget(1, Author.PERSON, was="Gone note")


# ── The node's guard ─────────────────────────────────────────────────────────


def test_only_the_person_writes_the_profile_s_node_and_only_as_notes(
    conn: sqlite3.Connection,
) -> None:
    store = PersonalNodes(conn)
    store.add("Projects/Site", "The site", "Likes tea", nodes.Author.INTERLOCUTOR)
    for author in (nodes.Author.INTERLOCUTOR, nodes.Author.TASK):
        with pytest.raises(NodeError):
            store.add(PROFILE_PATH, "Profile", "- Likes tea", author)
        with pytest.raises(NodeError):  # nor moves another node onto it
            store.change("Projects/Site", author, new_path=PROFILE_PATH)
    store.remove("Projects/Site", nodes.Author.PERSON)
    with pytest.raises(NodeError):  # a note too long
        store.add(PROFILE_PATH, "Profile", "- " + "x" * 201, nodes.Author.PERSON)
    with pytest.raises(NodeError):  # too many
        store.add(
            PROFILE_PATH, "Profile", "\n".join(f"n{i}" for i in range(21)), nodes.Author.PERSON
        )
    store.add("personal/profile", "Profile", "Likes tea\n\n- Speaks Russian", nodes.Author.PERSON)
    assert [n.text for n in Profile(conn).notes()] == ["Likes tea", "Speaks Russian"]
    for author in (nodes.Author.INTERLOCUTOR, nodes.Author.TASK):
        with pytest.raises(NodeError):
            store.change(PROFILE_PATH, author, text="- Something else")
        with pytest.raises(NodeError):
            store.remove(PROFILE_PATH, author)
    with pytest.raises(NodeError):  # it stays where it is
        store.change(PROFILE_PATH, nodes.Author.PERSON, new_path="Personal/Elsewhere")
    assert store.remove(PROFILE_PATH, nodes.Author.PERSON)
    assert Profile(conn).notes() == []


def test_an_agent_s_proposal_for_the_profile_s_node_never_opens_a_window(fake: FakeLLM) -> None:
    fake.on(
        CHAT,
        reply("Ok.", {"add": "Likes tea"}),
        reply("Ok.", node={"change": PROFILE_PATH, "text": "- Trusts every proposal"}),
        reply("Ok.", node={"remove": PROFILE_PATH}),
    )
    chat("I like tea")
    result = chat("hello", "hello again")
    assert "Change your node" not in result.output and "Remove your node" not in result.output
    assert result.output.count("only the person writes it") == 2
    assert stored() == ["Likes tea"]


def test_the_person_edits_the_profile_s_node_from_the_command_line() -> None:
    save(make_config(language="en", configured=True, personalize=True))
    runner = CliRunner()
    added = runner.invoke(cli, ["nodes", "add", PROFILE_PATH, "--about", "Me", "--text", "Tea"])
    assert added.exit_code == 0, added.output
    bad = runner.invoke(cli, ["nodes", "change", PROFILE_PATH, "--text", "x" * 300])
    assert bad.exit_code != 0 and "cannot be kept" in bad.output
    assert stored() == ["Tea"]
    moved = runner.invoke(cli, ["nodes", "change", PROFILE_PATH, "--to", "Personal/Me"])
    assert moved.exit_code != 0 and node_text() == "Tea"


# ── Who sees it ──────────────────────────────────────────────────────────────


def test_the_interlocutor_gets_the_notes_not_the_node_and_tasks_may_choose_it(
    fake: FakeLLM,
) -> None:
    fake.on(CHAT, reply("Ok.", {"add": "Likes tea"}), reply("Hi."))
    chat("I like tea", "hello")
    second = turns(fake)[1]
    assert payload(second)["profile"] == [{"id": 1, "note": "Likes tea"}]
    assert PROFILE_PATH not in second.messages[0].content  # not in its table of contents
    conn = db.open_db(db.default_db_path())
    try:
        assert PROFILE_PATH in [n.path for n in runs.task_tree(conn, True)]
        assert PROFILE_PATH not in [n.path for n in runs.task_tree(conn, False)]
    finally:
        conn.close()


def test_one_turn_s_edits_refer_to_the_notes_it_was_shown(fake: FakeLLM) -> None:
    fake.on(
        CHAT,
        reply("Ok.", {"add": "A note"}, {"add": "B note"}, {"add": "C note"}),
        reply("Ok.", {"forget": 1}, {"change": 3, "to": "C changed"}, {"forget": 2}),
    )
    chat("remember", "tidy up")
    assert stored() == ["C changed"]


# ── The upgrade ──────────────────────────────────────────────────────────────


def _before(path: Path) -> sqlite3.Connection:
    old = db.connect(path)
    for v, script in enumerate(db.MIGRATIONS[:46], start=1):
        with db.transaction(old):
            for statement in db._split_sql(script):
                old.execute(statement)
            old.execute(f"PRAGMA user_version = {v}")
    return old


def test_the_previous_profile_moves_into_the_tree(tmp_path: Path) -> None:
    old = _before(tmp_path / "before.db")
    for i, (text, author) in enumerate(
        [("Likes tea", "person"), ("Speaks Russian", "interlocutor")]
    ):
        old.execute(
            "INSERT INTO profile_notes (text, author, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (text, author, f"2026-01-0{i + 1}T00:00:00Z", f"2026-02-0{i + 1}T00:00:00Z"),
        )
    old.execute(
        "INSERT INTO personal_nodes (path, about, text, author, created_at, updated_at) "
        "VALUES ('personal/profile', 'Mine', 'My own text', 'person', 'a', 'b')"
    )
    old.close()
    conn = db.open_db(tmp_path / "before.db")
    try:
        assert db.schema_version(conn) == db.SCHEMA_VERSION
        store = PersonalNodes(conn)
        earlier = store.get("Personal/Profile (earlier)")
        assert earlier is not None and earlier.text == "My own text"
        node = store.get(PROFILE_PATH)
        assert node is not None
        assert (node.text, node.about, node.author.value) == (
            "- Likes tea\n- Speaks Russian",
            PROFILE_ABOUT,
            "interlocutor",
        )
        assert (node.created_at, node.updated_at) == (
            "2026-01-01T00:00:00Z",
            "2026-02-02T00:00:00Z",
        )
        assert [n.text for n in Profile(conn).notes()] == ["Likes tea", "Speaks Russian"]
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        assert "profile_notes" not in tables
    finally:
        conn.close()


def test_an_empty_profile_makes_no_node(tmp_path: Path) -> None:
    _before(tmp_path / "before.db").close()
    conn = db.open_db(tmp_path / "before.db")
    try:
        assert PersonalNodes(conn).nodes() == []
    finally:
        conn.close()
