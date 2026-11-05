"""Human correspondence, inside the route's single write transaction (G.10a)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from uuid import uuid4

from desman.core import db
from desman.net import protocol as p
from desman.registry import store


class Refusal(Exception):
    """A messenger refusal, journaled by the HTTP boundary."""


def _name(conn: sqlite3.Connection, node: str) -> str:
    row = conn.execute("SELECT name FROM messenger_names WHERE node = ?", (node,)).fetchone()
    if row is None:
        raise Refusal("no_name")
    return str(row[0])


def _other(conn: sqlite3.Connection, name: str) -> str:
    row = conn.execute("SELECT node FROM messenger_names WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise Refusal("unknown_name")
    return str(row[0])


def _pair(a: str, b: str) -> tuple[str, str]:
    if a == b:
        raise Refusal("unknown_name")
    return (a, b) if a < b else (b, a)


def _has(conn: sqlite3.Connection, table: str, columns: str, pair: tuple[str, str]) -> bool:
    # table/columns are constants from this module, never request data.
    return conn.execute(f"SELECT 1 FROM {table} WHERE {columns}", pair).fetchone() is not None


def _limit(
    conn: sqlite3.Connection,
    table: str,
    who: str,
    column: str,
    time: str,
    now: datetime,
    maximum: int,
) -> None:
    since = db.to_iso(now - timedelta(days=1))
    count = conn.execute(
        f"SELECT count(*) FROM {table} WHERE {column} = ? AND {time} > ?", (who, since)
    ).fetchone()[0]
    if count >= maximum:
        raise Refusal("messenger_limit")


def set_name(
    conn: sqlite3.Connection, who: str, asked: p.SetName, now: datetime
) -> p.MessengerName:
    name = asked.name
    if name in {"admin", "operator", "registry", "root", "system", "support"} or name.startswith(
        "desman"
    ):
        raise Refusal("name_reserved")
    row = conn.execute("SELECT node FROM messenger_names WHERE name = ?", (name,)).fetchone()
    if row is not None and row[0] != who:
        raise Refusal("name_taken")
    conn.execute(
        "INSERT INTO messenger_names VALUES (?, ?, ?) ON CONFLICT(node) DO UPDATE SET "
        "name = excluded.name, set_at = excluded.set_at",
        (who, name, db.to_iso(now)),
    )
    store.log(conn, "name_set", who, {"name": name})
    return p.MessengerName(name=name)


def contact(
    conn: sqlite3.Connection, who: str, asked: p.ContactAction, now: datetime
) -> p.ContactState:
    _name(conn, who)
    other = _other(conn, asked.name)
    pair = _pair(who, other)
    moment = db.to_iso(now)
    state: p.ContactState = p.ContactState(name=asked.name, state="none")
    if asked.action == "add":
        if _has(conn, "blocks", "node = ? AND blocked = ?", (who, other)):
            raise Refusal("blocked")
        if _has(conn, "contacts", "a = ? AND b = ?", pair):
            return p.ContactState(name=asked.name, state="contact")
        if _has(conn, "blocks", "node = ? AND blocked = ?", (other, who)):
            return p.ContactState(name=asked.name, state="asked")
        if _has(conn, "contact_requests", "from_node = ? AND to_node = ?", (other, who)):
            conn.execute("INSERT INTO contacts VALUES (?, ?, ?)", (*pair, moment))
            conn.execute("INSERT OR IGNORE INTO contact_history VALUES (?, ?)", pair)
            conn.execute(
                "DELETE FROM contact_requests WHERE (from_node = ? AND to_node = ?) OR "
                "(from_node = ? AND to_node = ?)",
                (who, other, other, who),
            )
            return p.ContactState(name=asked.name, state="contact")
        if not _has(conn, "contact_requests", "from_node = ? AND to_node = ?", (who, other)):
            _limit(
                conn,
                "contact_events",
                who,
                "from_node",
                "created_at",
                now,
                p.MAX_CONTACT_REQUESTS_PER_DAY,
            )
            conn.execute("INSERT INTO contact_requests VALUES (?, ?, ?)", (who, other, moment))
            conn.execute("INSERT INTO contact_events VALUES (?, ?, ?)", (who, other, moment))
        return p.ContactState(name=asked.name, state="asked")
    if asked.action in {"remove", "block"}:
        conn.execute("DELETE FROM contacts WHERE a = ? AND b = ?", pair)
        conn.execute(
            "DELETE FROM contact_requests WHERE (from_node = ? AND to_node = ?) OR "
            "(from_node = ? AND to_node = ?)",
            (who, other, other, who),
        )
    if asked.action == "block":
        conn.execute("INSERT OR IGNORE INTO blocks VALUES (?, ?, ?)", (who, other, moment))
        state = p.ContactState(name=asked.name, state="blocked")
    elif asked.action == "unblock":
        conn.execute("DELETE FROM blocks WHERE node = ? AND blocked = ?", (who, other))
    return state


def send(conn: sqlite3.Connection, who: str, asked: p.SendMessage, now: datetime) -> p.MessageSent:
    _name(conn, who)
    other = _other(conn, asked.to)
    if not _has(conn, "contacts", "a = ? AND b = ?", _pair(who, other)):
        raise Refusal("not_contact")
    _limit(conn, "messages", who, "sender", "sent_at", now, p.MAX_MESSAGES_PER_DAY)
    conn.execute(
        "DELETE FROM messages WHERE sent_at < ?", (db.to_iso(now - timedelta(days=p.MESSAGE_DAYS)),)
    )
    cursor = conn.execute(
        "INSERT INTO messages(sender,recipient,text,sent_at) VALUES (?, ?, ?, ?)",
        (who, other, asked.text, db.to_iso(now)),
    )
    assert cursor.lastrowid is not None
    store.log(
        conn, "message_sent", who, {"sender": who, "recipient": other, "id": cursor.lastrowid}
    )
    return p.MessageSent(id=cursor.lastrowid, sent_at=now)


def inbox(conn: sqlite3.Connection, who: str, asked: p.InboxQuery, now: datetime) -> p.Inbox:
    own = conn.execute("SELECT name FROM messenger_names WHERE node = ?", (who,)).fetchone()

    def names(sql: str, *, twice: bool = False) -> list[str]:
        return [r[0] for r in conn.execute(sql, (who, who) if twice else (who,)).fetchall()]

    contacts = names(
        "SELECT name FROM messenger_names WHERE node IN (SELECT b FROM contacts WHERE a = ? UNION SELECT a FROM contacts WHERE b = ?) ORDER BY name",
        twice=True,
    )
    outgoing = names(
        "SELECT name FROM messenger_names JOIN contact_requests ON node = to_node WHERE from_node = ? ORDER BY name"
    )
    incoming = names(
        "SELECT name FROM messenger_names JOIN contact_requests ON node = from_node WHERE to_node = ? ORDER BY name"
    )
    blocked = names(
        "SELECT name FROM messenger_names JOIN blocks ON messenger_names.node = blocked WHERE blocks.node = ? ORDER BY name"
    )
    where = "(m.sender = ? OR m.recipient = ?) AND m.sent_at >= ?"
    params: list[str | int] = [who, who, db.to_iso(now - timedelta(days=p.MESSAGE_DAYS))]
    if asked.with_name is not None:
        other = _other(conn, asked.with_name)
        where += " AND (m.sender = ? OR m.recipient = ?)"
        params.extend([other, other])
    if asked.after is not None:
        where += " AND m.id > ?"
        params.append(asked.after)
    order = "ASC" if asked.after is not None else "DESC"
    rows = conn.execute(
        "SELECT m.id, s.name AS sender, r.name AS recipient, m.text, m.sent_at FROM messages m "
        "JOIN messenger_names s ON s.node = m.sender JOIN messenger_names r ON r.node = m.recipient "
        f"WHERE {where} ORDER BY m.id {order} LIMIT 100",
        params,
    ).fetchall()
    messages = [p.Message.model_validate(dict(r)) for r in rows]
    messages.sort(key=lambda m: m.id)
    return p.Inbox(
        name=own[0] if own else None,
        contacts=contacts,
        asked=outgoing,
        asking=incoming,
        blocked=blocked,
        messages=messages,
    )


def complain(
    conn: sqlite3.Connection, who: str, asked: p.Complain, now: datetime
) -> p.ComplaintReceived:
    _name(conn, who)
    other = _other(conn, asked.name)
    pair = _pair(who, other)
    evidence = None
    if asked.message is not None:
        evidence = conn.execute(
            "SELECT text FROM messages WHERE id = ? AND sender = ? AND recipient = ?",
            (asked.message, other, who),
        ).fetchone()
        if evidence is None:
            raise Refusal("unknown_name")
    known = (
        evidence is not None
        or _has(conn, "contact_history", "a = ? AND b = ?", pair)
        or _has(conn, "contact_events", "from_node = ? AND to_node = ?", (other, who))
        or _has(conn, "messages", "sender = ? AND recipient = ?", (other, who))
    )
    if not known:
        raise Refusal("unknown_name")
    _limit(conn, "complaints", who, "from_node", "created_at", now, p.MAX_COMPLAINTS_PER_DAY)
    id = uuid4().hex
    conn.execute(
        "INSERT INTO complaints(id,from_node,about_node,message_id,message_text,text,created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            id,
            who,
            other,
            asked.message,
            evidence[0] if evidence else None,
            asked.text,
            db.to_iso(now),
        ),
    )
    store.log(conn, "complaint", who, {"id": id, "from_node": who, "about_node": other})
    return p.ComplaintReceived(id=id)
