"""The network's journal (ROADMAP G.3, ADR 0004 decision 12): what desman-global did.

Joining and leaving, modules lent and taken back; from G.5 on, every record sent to the registry
or received from it and every one refused. Append-only, like `operator_log`. Keys appear by
fingerprint only, and no entry holds a secret or the text of what was sent.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from pydantic import BaseModel, ConfigDict

from desman.core import db


class NetworkEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq: int
    ts: str
    action: str
    details: dict[str, Any]


def record(conn: sqlite3.Connection, action: str, details: dict[str, Any] | None = None) -> None:
    with db.transaction(conn):
        write(conn, action, details)


def write(conn: sqlite3.Connection, action: str, details: dict[str, Any] | None = None) -> None:
    """`record` inside the caller's transaction: the entry and its change commit together."""
    conn.execute(
        "INSERT INTO network_log (action, details_json) VALUES (?, ?)",
        (action, json.dumps(details or {}, ensure_ascii=False, sort_keys=True)),
    )


def entries(conn: sqlite3.Connection) -> list[NetworkEntry]:
    rows = conn.execute("SELECT * FROM network_log ORDER BY seq").fetchall()
    return [
        NetworkEntry(
            seq=r["seq"], ts=r["ts"], action=r["action"], details=json.loads(r["details_json"])
        )
        for r in rows
    ]
