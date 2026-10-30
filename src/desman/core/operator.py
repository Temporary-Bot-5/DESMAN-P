"""The operator's journal: what the person did outside any task (ROADMAP C.5, ADR 0003 §8).

`audit_log` tells the story of tasks; changes the person makes from the chat (a command they
confirmed or declined, a key added, a permission granted) belong to no task and are written
here. Append-only, like the audit log. A key is recorded by its provider and fingerprint
(`openai_compat.key_id`), never by its value.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from pydantic import BaseModel, ConfigDict

from desman.core import db


class OperatorEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq: int
    ts: str
    session_id: str | None
    action: str
    details: dict[str, Any]


def record(
    conn: sqlite3.Connection,
    action: str,
    details: dict[str, Any] | None = None,
    *,
    session_id: str | None = None,
) -> None:
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO operator_log (session_id, action, details_json) VALUES (?, ?, ?)",
            (session_id, action, json.dumps(details or {}, ensure_ascii=False)),
        )


def entries(conn: sqlite3.Connection) -> list[OperatorEntry]:
    rows = conn.execute("SELECT * FROM operator_log ORDER BY seq").fetchall()
    return [
        OperatorEntry(
            seq=r["seq"],
            ts=r["ts"],
            session_id=r["session_id"],
            action=r["action"],
            details=json.loads(r["details_json"]),
        )
        for r in rows
    ]
