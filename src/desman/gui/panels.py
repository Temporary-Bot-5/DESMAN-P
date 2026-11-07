"""Read-only panel snapshots; no application authority is held by the HTTP thread."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from desman import modules_cli, providers, tuning_cli
from desman.config import Config, ConfigError, open_config
from desman.core import calls, choices, db
from desman.core.report import inbox, task_report, task_summaries
from desman.core.sessions import Sessions
from desman.core.store import Store
from desman.gui.front import WebFront
from desman.i18n import t
from desman.live import Progress, key_labels
from desman.llm import router


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
            else {"modules": [], "roles": [], "interlocutor": None}
            if kind == "modules"
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
            config = config or Config()
            if kind == "modules":
                module_rows = modules_cli.modules_data(conn, config)
                for row in module_rows:
                    row["goal"] = conn.execute(
                        "SELECT goal FROM tasks WHERE id = ?", (row["task_id"],)
                    ).fetchone()[0]
                    decorate(conn, config, row, "task", row["task_id"])
                with front.condition:
                    current = front.current
                session = conn.execute(
                    "SELECT id FROM sessions WHERE id = ? AND terminal = 0 AND deleted_at IS NULL",
                    (current,),
                ).fetchone()
                chat = interlocutor(conn, config, session["id"] if session else None)
                result = {
                    "modules": module_rows,
                    "roles": tuning_cli.role_rows(config),
                    "interlocutor": chat,
                }
                # Redact every string, including user-controlled labels and stored model names.
                return redact_snapshot(result, redact)
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
                cards = modules_cli.modules_data(conn, config, report.id)
                for card in cards:
                    decorate(conn, config, card, "task", report.id)
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
                        "cards": cards,
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
            return redact_snapshot({"tasks": tasks}, redact)
    finally:
        conn.close()


def decorate(
    conn: sqlite3.Connection,
    config: Config,
    row: dict[str, Any],
    scope: choices.Scope,
    identity: str,
) -> None:
    if scope == "task":
        report = task_report(Store(conn), identity)
        if any(
            sub.status.value == "needs_human"
            and (row["role"] not in ("text", "code", "analysis") or sub.kind.value == row["role"])
            for sub in report.subtasks
        ):
            row["state"] = t("Needs human")
    own = choices.get(conn, scope, identity, row["role"])
    row.update(
        icon="module",
        scope=scope,
        scope_id=identity,
        choice_model=own.model,
        choice_key=own.key,
        state_style="good"
        if row["state"] == t("working")
        else "warn"
        if row["state"] in (t("waiting for a limit"), t("Needs human"))
        else "bad"
        if row["state"] == t("quota used up")
        else "muted",
    )


def interlocutor(conn: sqlite3.Connection, config: Config, identity: str | None) -> dict[str, Any]:
    own = choices.get(conn, "session", identity, "interlocutor") if identity else choices.Choice()
    cfg = choices.effective(config, "interlocutor", own)
    try:
        lineup = router.lineup(cfg, calls.Role.INTERLOCUTOR)
    except router.NoProvider:
        lineup = []
    first = lineup[0] if lineup else None
    fingerprint = router.preferred_key(cfg, calls.Role.INTERLOCUTOR)
    key = next(
        (
            k
            for k in tuning_cli.key_choices(config)
            if k["fingerprint"] == fingerprint and first and k["provider"] == first.provider.id
        ),
        None,
    )
    usage = conn.execute(
        "SELECT COUNT(*) AS calls, COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS tokens "
        "FROM model_calls WHERE session_id = ? AND role = 'interlocutor'",
        (identity,),
    ).fetchone()
    return {
        "role": "interlocutor",
        "icon": "module",
        "scope": "session",
        "scope_id": identity,
        "model": first.label if first else None,
        "provider": first.provider.id if first else None,
        "key_label": key["label"] if key else "auto",
        "key_fingerprint": fingerprint if key else None,
        "choice_model": own.model,
        "choice_key": own.key,
        "source": "chat"
        if own != choices.Choice()
        else "role"
        if config.tuning.models.interlocutor or config.tuning.keys.interlocutor
        else "auto",
        "calls": usage["calls"],
        "tokens": usage["tokens"],
        "state": t("idle"),
    }


def redact_snapshot(data: dict[str, Any], redact: Callable[[str], str]) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, str):
            return redact(value)
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return {k: clean(v) for k, v in data.items()}
