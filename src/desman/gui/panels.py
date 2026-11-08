"""Read-only panel snapshots; no application authority is held by the HTTP thread."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from html import escape
from pathlib import Path
from typing import Any

from desman import modules_cli, providers, tuning_cli
from desman.config import Config, ConfigError, open_config
from desman.core import calls, choices, db, holds
from desman.core.report import inbox, short_description, task_report, task_summaries
from desman.core.sessions import Sessions
from desman.core.store import Store
from desman.gui import module_view
from desman.gui.front import WebFront
from desman.i18n import t
from desman.live import Progress, key_labels
from desman.llm import router
from desman.net.node import active_capabilities


def snapshot(path: Path, kind: str, front: WebFront) -> dict[str, Any]:
    """Open an existing WAL database read-only, independently of the chat connection."""
    if kind != "settings" and not path.exists():
        return (
            {
                "current": None,
                "temporary": False,
                "sessions": [],
                "terminal": {"id": None, "unread": 0},
            }
            if kind == "sessions"
            else {
                "modules": [],
                "roles": [],
                "roles_by_scope": choices.ROLES_BY_SCOPE,
                "interlocutor": None,
            }
            if kind == "modules"
            else {"tasks": []}
        )
    try:
        config = open_config().load()
    except (ConfigError, OSError):
        if kind == "settings":
            raise
        config = None

    # Labels/settings may themselves contain escaped copies of a configured secret.
    forms = {
        form
        for keys in (config.api_keys.values() if config else [])
        for key in keys
        for form in (escape(key.value), json.dumps(key.value)[1:-1])
        if form
    }

    def redact(text: str) -> str:
        for form in sorted(forms, key=len, reverse=True):
            text = text.replace(form, providers.REDACTED)
        return providers.redact_keys(text, config)[0]

    if kind == "settings" and not path.exists():
        from desman.gui.settings import data

        return redact_snapshot(data(None, config or Config(), front), redact)

    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {db.BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA query_only = ON")
    try:
        with db.read_snapshot(conn):
            if kind == "settings":
                from desman.gui.settings import data

                return redact_snapshot(data(conn, config or Config(), front), redact)
            if kind == "sessions":
                with front.condition:
                    current, temporary = front.current, front.temporary
                    seen = front.terminal_seq
                terminal = conn.execute(
                    "SELECT id, cleared_seq FROM sessions WHERE terminal = 1"
                ).fetchone()
                terminal_id = terminal["id"] if terminal else None
                seen = max(seen, terminal["cleared_seq"] if terminal else 0)
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
                            **(front.runners.status(s.id) if front.runners is not None else {}),
                            "unread": s.id != current
                            and bool(
                                conn.execute(
                                    "SELECT 1 FROM messages WHERE session_id = ? AND role != 'user' "
                                    "AND seq > ? LIMIT 1",
                                    (
                                        s.id,
                                        conn.execute(
                                            "SELECT read_seq FROM sessions WHERE id = ?", (s.id,)
                                        ).fetchone()[0],
                                    ),
                                ).fetchone()
                            ),
                        }
                        for s in Sessions(conn).recent(limit=-1)
                    ],
                }
            config = config or Config()
            if kind == "modules":
                module_rows = modules_cli.modules_data(conn, config)
                with front.condition:
                    current = front.current
                    pending = front.pending_choice
                session = conn.execute(
                    "SELECT id FROM sessions WHERE id = ? AND terminal = 0 AND deleted_at IS NULL",
                    (current,),
                ).fetchone()
                chat = interlocutor(
                    conn,
                    config,
                    session["id"] if session else None,
                    choices.Choice.model_validate(pending) if pending and current is None else None,
                )
                grouped = module_view.cards(conn, config, module_rows)
                if front.runners is not None:
                    for card in grouped:
                        if card.get("hold_state") in {"stopped", "removed"}:
                            continue
                        statuses = [
                            front.runners.status(w["id"])
                            for w in card["work"]
                            if w["kind"] == "chat"
                        ]
                        if any(status["state"] == "answering" for status in statuses):
                            card["state"] = t("working")
                        elif any(status.get("reason") for status in statuses):
                            card["state"] = next(
                                status["reason"] for status in statuses if status.get("reason")
                            )
                    status = front.runners.status(current or "")
                    if not chat.get("hold_state"):
                        if status.get("reason"):
                            chat["state"] = status["reason"]
                        elif status["state"] == "answering":
                            chat["state"] = t("working")
                result = {
                    "modules": grouped,
                    "removed": module_view.removed_cards(conn, grouped),
                    "network_visible": config.network.enabled or bool(config.network.donated),
                    "network_suspended": config.network.enabled
                    and not active_capabilities(config, conn),
                    "roles": tuning_cli.role_rows(config),
                    "roles_by_scope": choices.ROLES_BY_SCOPE,
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
                "AND hidden_at IS NULL ORDER BY pinned DESC, created_at DESC, rowid DESC",
                tuple(ids),
            )
            busy = (
                conn.execute(
                    "SELECT COUNT(DISTINCT owner) FROM ("
                    "SELECT lease_owner AS owner FROM subtasks WHERE lease_until > ? "
                    "UNION SELECT review_owner AS owner FROM subtasks WHERE review_until > ?)",
                    (db.now_iso(), db.now_iso()),
                ).fetchone()[0]
                >= front.worker_count
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
                            "title": short_description(redact(sub.description)),
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
                        "pinned": report.pinned,
                        "paused_at": report.paused_at,
                        "ref": short,
                        "goal": redact(report.goal),
                        "status": report.status.value,
                        "queued": busy
                        and report.status.value == "active"
                        and report.paused_at is None
                        and any(s.status.value == "ready" for s in report.subtasks)
                        and not any(
                            s.status.value in ("running", "review") for s in report.subtasks
                        ),
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
    if scope == "task":
        row["pinned"] = bool(
            conn.execute("SELECT pinned FROM tasks WHERE id = ?", (identity,)).fetchone()[0]
        )
    role = "executor" if row["role"] in {"text", "code", "analysis"} else row["role"]
    kind_filter = (
        "AND subtask_id IN (SELECT id FROM subtasks WHERE kind = ?)" if role == "executor" else ""
    )
    latest = conn.execute(
        f"SELECT provider, model FROM model_calls WHERE {'task_id' if scope == 'task' else 'session_id'} = ? AND role = ? AND error_kind IS NULL {kind_filter} ORDER BY seq DESC LIMIT 1",
        (identity, role, *([row["role"]] if kind_filter else [])),
    ).fetchone()
    row["last_model"] = f"{latest['provider']}/{latest['model']}" if latest else None
    own = choices.get(conn, scope, identity, row["role"])
    row["module_source"] = modules_cli.module_source(config, row)
    matching = [
        h
        for h in holds.active(conn, scope, identity)
        if h.source == row["module_source"] and h.model == row["model"]
    ]
    if matching:
        row["hold_state"] = matching[0].state
        row["hold_scope"] = matching[0].scope
        row["hold_scope_id"] = matching[0].scope_id
        row["state"] = t("stopped") if matching[0].state == "stopped" else t("removed")
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


def interlocutor(
    conn: sqlite3.Connection,
    config: Config,
    identity: str | None,
    pending: choices.Choice | None = None,
) -> dict[str, Any]:
    own = choices.get(conn, "session", identity, "interlocutor") if identity else choices.Choice()
    own = pending or own
    cfg = choices.effective(config, "interlocutor", own)
    try:
        lineup = router.lineup(cfg, calls.Role.INTERLOCUTOR)
    except router.NoProvider:
        lineup = []
    created = conn.execute("SELECT created_at FROM sessions WHERE id=?", (identity,)).fetchone()
    removed = {
        (h.source, h.model)
        for h in holds.active(conn, "session", identity or "")
        if h.state == "removed"
        or h.state == "stopped"
        and created is not None
        and created[0] > h.at
    }
    lineup = [
        c
        for c in lineup
        if any((source, c.label) not in removed for source in router.sources(cfg, c))
    ]
    first = lineup[0] if lineup else None
    latest = conn.execute(
        "SELECT provider, model, key_id FROM model_calls WHERE session_id = ? AND role = 'interlocutor' "
        "AND error_kind IS NULL ORDER BY seq DESC LIMIT 1",
        (identity,),
    ).fetchone()
    if latest:
        first = router.Choice(providers.find(latest["provider"]), latest["model"])
    elif (
        router.preferred(cfg, calls.Role.INTERLOCUTOR) is None
        and router.preferred_key(cfg, calls.Role.INTERLOCUTOR) is None
    ):
        first = None
    fingerprint = latest["key_id"] if latest else router.preferred_key(cfg, calls.Role.INTERLOCUTOR)
    if first:
        sources = [
            source for source in router.sources(cfg, first) if (source, first.label) not in removed
        ]
        if fingerprint not in sources:
            fingerprint = next(iter(sources), None)
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
    model_usage = conn.execute(
        "SELECT COUNT(*) AS calls, COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS tokens "
        "FROM model_calls WHERE session_id = ? AND role = 'interlocutor' "
        "AND provider = ? AND model = ?",
        (identity, first.provider.id if first else None, first.model if first else None),
    ).fetchone()
    source = modules_cli.module_source(
        config,
        {
            "provider": first.provider.id if first else None,
            "key_fingerprint": fingerprint if key else None,
        },
    )
    held = next(
        (
            h
            for h in holds.active(conn, "session", identity or "")
            if h.source == source and first and h.model == first.label
        ),
        None,
    )
    return {
        "hold_state": held.state if held else None,
        "hold_scope": held.scope if held else None,
        "hold_scope_id": held.scope_id if held else None,
        "module_source": modules_cli.module_source(
            config,
            {
                "provider": first.provider.id if first else None,
                "key_fingerprint": fingerprint if key else None,
            },
        ),
        "last_model": f"{latest[0]}/{latest[1]}" if latest else None,
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
        "model_calls": model_usage["calls"],
        "model_tokens": model_usage["tokens"],
        "calls": usage["calls"],
        "tokens": usage["tokens"],
        "state": t("stopped") if held and held.state == "stopped" else t("idle"),
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
