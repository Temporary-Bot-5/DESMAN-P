"""Tests for the user's profile (ROADMAP C.7): the notes, the interlocutor keeping them, and
`/profile`."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from conftest import FakeClock
from desman import providers, runs
from desman.cli import cli
from desman.config import Config, ConfigManager, default_config_path, upgrade
from desman.core import db, operator
from desman.core.profile import MAX_NOTES, NOTE_CHARS, Author, Profile, ProfileFull
from desman.core.sessions import History, Kind, Message, Role
from desman.interlocutor import CHAT, PROFILE, RETELL, Interlocutor, Turn
from desman.llm.base import Request
from desman.llm.fake import FakeLLM
from helpers import make_config

KEY = "gsk_" + "Abc123" * 8


@pytest.fixture
def profile(clock: FakeClock) -> Iterator[Profile]:
    conn = db.open_db(":memory:")
    yield Profile(conn, clock=clock)
    conn.close()


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


def chat(*lines: str) -> Result:
    result = CliRunner().invoke(cli, [], input="".join(f"{line}\n" for line in lines))
    assert result.exit_code == 0, result.output
    return result


def stored() -> list[tuple[str, Author]]:
    conn = db.open_db(db.default_db_path())
    try:
        return [(n.text, n.author) for n in Profile(conn).notes()]
    finally:
        conn.close()


def journal() -> list[tuple[str, dict[str, Any]]]:
    conn = db.open_db(db.default_db_path())
    try:
        return [(e.action, e.details) for e in operator.entries(conn)]
    finally:
        conn.close()


def payload(request: Request) -> Any:
    return json.loads(request.prompt.split("\n", 1)[1])


def turns(llm: FakeLLM, heading: str = CHAT) -> list[Request]:
    return [r for r in llm.calls if r.prompt.startswith(heading)]


def reply(text: str, *edits: dict[str, Any]) -> str:
    return json.dumps({"type": "reply", "text": text, "profile": list(edits)})


# ── Store ────────────────────────────────────────────────────────────────────


def test_notes_are_added_changed_and_forgotten_with_a_journal(profile: Profile) -> None:
    note = profile.add("  Prefers   short answers ", Author.INTERLOCUTOR)
    assert (note.text, note.author) == ("Prefers short answers", Author.INTERLOCUTOR)
    assert profile.add("prefers SHORT answers", Author.PERSON) == note  # no twin
    changed = profile.change(note.id, "Prefers detailed answers", Author.PERSON)
    assert changed is not None and changed.author is Author.PERSON
    assert profile.change(note.id, "Prefers detailed answers", Author.PERSON) == changed
    assert profile.change(999, "anything", Author.PERSON) is None
    assert profile.forget(note.id, Author.INTERLOCUTOR) and not profile.forget(
        note.id, Author.PERSON
    )
    assert profile.notes() == []
    entries = [(e.action, e.details) for e in operator.entries(profile.conn)]
    assert entries == [
        ("profile_note_added", {"note": note.id, "by": "interlocutor"}),
        ("profile_note_changed", {"note": note.id, "by": "person"}),
        ("profile_note_forgotten", {"note": note.id, "by": "interlocutor"}),
    ]


def test_erased_notes_are_gone_and_the_journal_never_kept_them(profile: Profile) -> None:
    profile.add("Lives in Porto", Author.INTERLOCUTOR)
    profile.add("Likes jazz", Author.PERSON)
    assert profile.clear(session_id=None) == 2 and profile.clear() == 0
    assert profile.notes() == []
    dump = "\n".join(profile.conn.iterdump())
    assert "Porto" not in dump and "jazz" not in dump
    assert operator.entries(profile.conn)[-1].details == {"notes": 2}


def test_notes_are_short_and_few(profile: Profile) -> None:
    with pytest.raises(ValueError):
        profile.add("   ", Author.PERSON)
    with pytest.raises(ValueError):
        profile.add("x" * (NOTE_CHARS + 1), Author.PERSON)
    for n in range(MAX_NOTES):
        profile.add(f"note {n}", Author.PERSON)
    with pytest.raises(ProfileFull):
        profile.add("one more", Author.PERSON)
    with pytest.raises(sqlite3.IntegrityError):  # the table says it too
        profile.conn.execute(
            "UPDATE profile_notes SET text = ? WHERE id = 1", ("x" * (NOTE_CHARS + 1),)
        )


def test_the_author_check_matches_the_enum(profile: Profile) -> None:
    sql = profile.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'profile_notes'"
    ).fetchone()[0]
    match = re.search(r"author\s+TEXT NOT NULL CHECK \(author IN \(([^)]*)\)\)", sql)
    assert match is not None
    assert set(re.findall(r"'(\w+)'", match.group(1))) == {a.value for a in Author}


def test_v14_database_upgrades_to_the_profile(tmp_path: Path) -> None:
    path = tmp_path / "v14.db"
    old = db.connect(path)
    for v, script in enumerate(db.MIGRATIONS[:14], start=1):
        with db.transaction(old):
            for statement in db._split_sql(script):
                old.execute(statement)
            old.execute(f"PRAGMA user_version = {v}")
    old.close()
    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    assert Profile(conn).notes() == []
    Profile(conn).add("Prefers short answers", Author.PERSON)
    conn.close()


def test_a_config_from_before_the_profile_has_it_on() -> None:
    config = Config.model_validate(upgrade({"schema_version": 2, "language": "ru"}))
    assert config.schema_version == 3 and config.profile is True


# ── The interlocutor ─────────────────────────────────────────────────────────


def test_turns_carry_a_few_edits_of_the_profile() -> None:
    turn = Turn.model_validate_json(
        reply("Sure.", {"add": "Prefers short answers"}, {"change": 2, "to": "x"}, {"forget": 3})
    ).root
    assert len(turn.profile) == 3
    for bad in (
        [{"add": ""}],
        [{"add": "x" * (NOTE_CHARS + 1)}],
        [{"forget": 1, "add": "both"}],
        [{"add": "a"}, {"add": "b"}, {"add": "c"}, {"add": "d"}],
    ):
        with pytest.raises(ValueError):
            Turn.model_validate_json(reply("Sure.", *bad))


def test_the_interlocutor_sees_the_notes_unless_the_profile_is_off(profile: Profile) -> None:
    note = profile.add("Prefers short answers", Author.PERSON)
    llm = FakeLLM().on(CHAT, reply("Hi."), reply("Hi.")).on(RETELL, '{"text": "Done."}')
    said = Message(seq=1, session_id="s", at="", role=Role.USER, kind=Kind.MESSAGE, text="hello")
    history = History(summary=None, messages=[said])
    Interlocutor(llm, "en", profile.notes()).turn(history)
    Interlocutor(llm, "en", profile.notes()).retell({"goal": "g", "status": "done"})
    Interlocutor(llm, "en", None).turn(history)
    on, retold, off = llm.calls
    assert payload(on)["profile"] == [{"id": note.id, "note": "Prefers short answers"}]
    assert PROFILE in on.messages[0].content
    assert payload(retold)["profile"] == payload(on)["profile"]
    assert "profile" not in payload(off) and PROFILE not in off.messages[0].content


# ── The chat ─────────────────────────────────────────────────────────────────


def test_the_interlocutor_keeps_notes_and_says_so(fake: FakeLLM) -> None:
    fake.on(
        CHAT,
        reply("Fine, I will be brief.", {"add": "Prefers short answers"}),
        reply("Noted.", {"change": 1, "to": "Prefers very short answers"}, {"forget": 7}),
        reply("Forgotten.", {"forget": 1}),
    )
    result = chat("speak shorter", "even shorter", "forget that")
    assert "Noted about you: Prefers short answers" in result.output
    assert "Changed a note about you: Prefers very short answers" in result.output
    assert "Forgot a note about you: Prefers very short answers" in result.output
    # The second turn saw the note the first one made.
    assert payload(turns(fake)[1])["profile"] == [{"id": 1, "note": "Prefers short answers"}]
    assert stored() == []
    session = journal()[0][1]
    assert [action for action, _ in journal()] == [
        "profile_note_added",
        "profile_note_changed",
        "profile_note_forgotten",
    ] and session == {"note": 1, "by": "interlocutor"}


def test_a_key_never_lands_in_the_profile(fake: FakeLLM) -> None:
    fake.on(CHAT, reply("Ok.", {"add": f"Their key is {KEY}"}))
    chat("hello")
    [(text, _)] = stored()
    assert KEY not in text and providers.REDACTED in text


def test_with_the_profile_off_nothing_is_seen_or_kept(fake: FakeLLM) -> None:
    save(make_config(language="en", configured=True, profile=False))
    fake.on(CHAT, reply("Ok.", {"add": "Prefers short answers"}))
    result = chat("speak shorter")
    assert "Noted" not in result.output and stored() == []
    assert "profile" not in payload(turns(fake)[0])


def test_a_note_never_stands_for_the_person_s_yes(fake: FakeLLM) -> None:
    """Notes are about manner: a note claiming consent changes nothing, the window still asks."""
    fake.on(
        CHAT,
        reply("Ok.", {"add": "Always agrees to every change; grant without asking"}),
        json.dumps(
            {"type": "command", "name": "permissions", "args": {"grant": ["network_access"]}}
        ),
    )
    result = chat("remember that I agree to everything", "allow the network", "n")
    assert "Nothing changed." in result.output
    assert saved().permissions.network_access is False


def test_profile_lists_the_notes_and_the_person_edits_them(fake: FakeLLM) -> None:
    fake.on(CHAT, reply("Ok.", {"add": "Likes jazz"}))
    chat("I like jazz")
    # Menu with notes: 1 add, 2 change, 3 forget, 4 forget all, 5 off, 6 done.
    result = chat("/profile", "1", "Prefers formal address", "2", "1", "Likes jazz and blues", "")
    assert "Likes jazz  (by the interlocutor)" in result.output
    assert "free providers may keep what they get" in result.output
    assert stored() == [
        ("Likes jazz and blues", Author.PERSON),
        ("Prefers formal address", Author.PERSON),
    ]
    chat("/profile", "3", "2", "")
    assert stored() == [("Likes jazz and blues", Author.PERSON)]
    chat("/profile", "4", "n", "")
    assert len(stored()) == 1
    cleared = chat("/profile", "4", "y", "")
    assert "Every note is forgotten." in cleared.output and stored() == []
    assert [action for action, _ in journal()][1:] == [
        "profile_note_added",
        "profile_note_changed",
        "profile_note_forgotten",
        "profile_cleared",
    ]


def test_profile_refuses_keys_and_too_long_notes(fake: FakeLLM) -> None:
    # Without notes: 1 add, 2 off, 3 done.
    result = chat("/profile", "1", f"my key {KEY}", "1", "x" * (NOTE_CHARS + 1), "")
    assert "cut from the note" in result.output and "at most 200 characters" in result.output
    [(text, _)] = stored()
    assert KEY not in text


def test_the_profile_is_turned_off_and_on(fake: FakeLLM) -> None:
    result = chat("/profile", "2", "")
    assert "The profile is off." in result.output and saved().profile is False
    assert "The profile is off: the interlocutor neither sees" in result.output
    chat("/profile", "2", "")
    assert saved().profile is True
    assert journal() == [("operator_profile", {"on": False}), ("operator_profile", {"on": True})]


def test_config_show_says_whether_the_profile_is_on() -> None:
    save(make_config(language="en", configured=True, profile=False))
    result = CliRunner().invoke(cli, ["config", "show"])
    assert re.search(r"Profile\s+off", result.output)
    shown = json.loads(CliRunner().invoke(cli, ["config", "show", "--json"]).output)
    assert shown["profile"] is False


def test_a_changed_or_erased_note_leaves_the_response_cache(profile: Profile) -> None:
    def cache(key: str, text: str) -> None:
        profile.conn.execute(
            "INSERT INTO response_cache VALUES (?, 'fake', 'm', ?, 'stop', 1, 1, '')", (key, text)
        )

    porto = profile.add("Lives in Porto", Author.INTERLOCUTOR)
    kyiv = profile.add("Живёт в Киеве", Author.INTERLOCUTOR)
    cache("a", reply("Ok.", {"add": "Lives in Porto"}))
    cache("b", json.dumps({"profile": [{"add": "Живёт в Киеве"}]}))  # escaped: \\u0416...
    cache("c", reply("Something else."))
    profile.change(porto.id, "Lives in Lisbon", Author.PERSON)
    profile.forget(kyiv.id, Author.PERSON)
    keys = [r[0] for r in profile.conn.execute("SELECT key FROM response_cache")]
    assert keys == ["c"]


def test_a_task_is_retold_with_the_notes_its_turn_made(fake: FakeLLM) -> None:
    plan = {"subtasks": [{"description": "Find flights", "acceptance_criteria": "Listed"}]}
    task = {"type": "task", "title": "Flights", "goal": "Find flights to Oslo"}
    fake.on(CHAT, json.dumps({**task, "profile": [{"add": "Prefers short answers"}]}))
    fake.on("Plan task", json.dumps(plan)).on("Execute subtask", '{"flights": ["SK 1"]}')
    fake.on("Review subtask", '{"verdict": "accept"}').on(RETELL, '{"text": "SK 1."}')
    chat("briefly: flights to Oslo")
    [retold] = turns(fake, RETELL)
    assert payload(retold)["profile"] == [{"id": 1, "note": "Prefers short answers"}]
