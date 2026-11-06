"""Read-only panel snapshots; no application authority is held by the HTTP thread."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from desman import providers
from desman.config import ConfigError, open_config
from desman.core import calls, db
from desman.core.report import inbox, task_report, task_summaries
from desman.core.sessions import Sessions
from desman.core.store import Store
from desman.gui.front import WebFront
from desman.live import Progress, key_labels


def snapshot(path: Path, kind: str, front: WebFront) -> dict[str, Any]:
    """Open an existing WAL database read-only, independently of the chat connection."""
    if not path.exists():
        return (
            {
                "current": None,
                "temporary": False,
                "sessions": [],
                "terminal": {"id": None, "unread": 0},
            }
            if kind == "sessions"
            else {"tasks": []}
        )
    try:
        config = open_config().load()
    except (ConfigError, OSError):
        config = None

    def redact(text: str) -> str:
        return providers.redact_keys(text, config)[0]

    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {db.BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA query_only = ON")
    try:
        with db.read_snapshot(conn):
            if kind == "sessions":
                with front.condition:
                    current, temporary = front.current, front.temporary
                    seen = front.terminal_seq
                terminal = conn.execute("SELECT id FROM sessions WHERE terminal = 1").fetchone()
                terminal_id = terminal["id"] if terminal else None
                unread = (
                    conn.execute(
                        "SELECT COUNT(*) FROM messages WHERE session_id = ? AND kind = 'event' AND seq > ?",
                        (terminal_id, seen),
                    ).fetchone()[0]
                    if terminal_id != current
                    else 0
                )
                return {
                    "terminal": {"id": terminal_id, "unread": unread},
                    "current": current,
                    "temporary": temporary,
                    "sessions": [
                        {
                            "id": s.id,
                            "name": redact(s.name),
                            "last_at": s.last_at,
                            "messages": s.messages,
                            "pinned": s.pinned,
                        }
                        for s in Sessions(conn).recent(limit=-1)
                    ],
                }
            store = Store(conn)
            waiting = {item.subtask_id: item for item in inbox(store)}
            labels = key_labels(config) if config else {}
            tasks = []
            ids = {summary.id for summary in task_summaries(store, limit=30)}
            ids.update(item.task_id for item in waiting.values())
            placeholders = ", ".join("?" for _ in ids)
            rows = conn.execute(
                f"SELECT id FROM tasks WHERE id IN ({placeholders}) "
                "ORDER BY created_at DESC, rowid DESC",
                tuple(ids),
            )
            for row in rows:
                report = task_report(store, row["id"])
                short = store.short_id(report.id)
                subtasks = []
                for sub in report.subtasks:
                    item = waiting.get(sub.id)
                    subtasks.append(
                        {
                            "seq": sub.seq,
                            "status": sub.status.value,
                            "description": redact(sub.description),
                            "reason": redact(sub.reason or "") if item else None,
                            "ref": item.ref if item else None,
                            "version": item.version if item else None,
                        }
                    )
                modules = calls.modules(conn, report.id)
                # Reuse the live view's interpretation of the latest module's state.
                progress = Progress(store, report.id, config) if config else None
                by_id = {s.id: s for s in report.subtasks}
                tasks.append(
                    {
                        "id": report.id,
                        "ref": short,
                        "goal": redact(report.goal),
                        "status": report.status.value,
                        "done": report.done,
                        "total": report.total,
                        "created_at": report.created_at,
                        "needs_human": any(s.id in waiting for s in report.subtasks),
                        "subtasks": subtasks,
                        "modules": [
                            {
                                "role": m.role.value if m.role else "",
                                "provider": redact(m.provider),
                                "model": redact(m.model),
                                "key": redact(labels.get(m.key_id or "", "")),
                                "calls": m.calls,
                                "prompt_tokens": m.prompt_tokens,
                                "completion_tokens": m.completion_tokens,
                                "subtask": by_id[m.subtask_id].seq
                                if m.subtask_id in by_id
                                else None,
                                "state": progress._state(m, by_id, modules)[0] if progress else "",
                            }
                            for m in modules
                        ],
                    }
                )
            return {"tasks": tasks}
    finally:
        conn.close()
