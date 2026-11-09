"""One safe card per section/source/model, with short work references (ADR 0008)."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from desman import modules_cli, providers
from desman.config import Config
from desman.core import calls, choices, db, holds
from desman.core.store import Store
from desman.i18n import t
from desman.llm import router
from desman.llm.openai_compat import key_id
from desman.net.node import capabilities


def short_title(title: str) -> str:
    title = " ".join(title.split())
    return title if len(title) <= 30 else title[:30].rstrip() + "…"


def module_id(source: str | None, model: str | None) -> str | None:
    """A module's short name: the same source (key fingerprint or local server) and model give
    the same id in every card and on every page load. A fingerprint is no secret."""
    if not source or not model:
        return None
    return hashlib.sha256(f"{source}\0{model}".encode()).hexdigest()[:6]


def key_labels(config: Config) -> dict[str, str]:
    """Short configured labels for page cards; no masked-value suffix."""
    return {
        key_id(k.value): k.label or f"#{i}"
        for p in providers.PROVIDERS
        for i, k in enumerate(config.keys(p.key_env or ""), 1)
    }


def cards(
    conn: sqlite3.Connection, config: Config, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    found: dict[tuple[str, str, str], dict[str, Any]] = {}
    current_holds = holds.listed(conn)
    keys = key_labels(config)
    task_states: dict[str, bool] = {}

    def active_task(identity: str) -> bool:
        if identity not in task_states:
            task_states[identity] = Store(conn).task_status(identity).value in {
                "active",
                "needs_human",
            }
        return task_states[identity]

    def add(
        source: str, model: str | None, section: str, role: str = "text"
    ) -> dict[str, Any] | None:
        if not source or not model:
            return None
        identity = (section, source, model)
        if identity not in found:
            usage = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(prompt_tokens+completion_tokens),0) FROM model_calls "
                "WHERE provider=? AND model=? AND (key_id=? OR (? IN ('ollama','llamacpp') AND key_id IS NULL))",
                (*model.split("/", 1), source, source),
            ).fetchone()
            found[identity] = {
                "module_id": module_id(source, model),
                "module_source": source,
                "model": model,
                "provider": model.split("/", 1)[0],
                "provider_name": providers.find(model.split("/", 1)[0]).name,
                "key_fingerprint": source if source not in {"ollama", "llamacpp"} else None,
                "key_label": keys.get(source)
                or (
                    "Ollama"
                    if source == "ollama"
                    else "llama.cpp"
                    if source == "llamacpp"
                    else source
                ),
                "label": model,
                "role": role,
                "source": "auto",
                "scope": "network" if section == "network" else "all",
                "scope_id": "",
                "section": section,
                "work": [],
                "state": t("idle"),
                "calls": usage[0],
                "tokens": usage[1],
                "model_calls": usage[0],
                "model_tokens": usage[1],
                "icon": "module",
                "choice_model": None,
                "choice_key": None,
            }
        return found[identity]

    for row in rows:
        row = modules_cli.assigned_module(conn, config, row)
        task = conn.execute(
            "SELECT goal, origin FROM tasks WHERE id=?", (row["task_id"],)
        ).fetchone()
        network = task["origin"] == "network"
        card = add(
            modules_cli.module_source(config, row),
            row["model"],
            "network" if network else "local",
            row["role"],
        )
        if card and active_task(row["task_id"]):
            card["work"].append(
                {
                    "kind": "network job" if network else "task",
                    "id": row["task_id"],
                    "title": short_title(task["goal"]),
                    "role": row["role"],
                }
            )
            if row["state"] != t("idle"):
                card["state"] = row["state"]
    cutoff = db.to_iso(datetime.now(UTC) - timedelta(days=1))
    for session in conn.execute(
        "SELECT s.id, s.created_at, COALESCE(s.title, (SELECT text FROM messages WHERE session_id=s.id ORDER BY seq LIMIT 1)) AS name FROM sessions s WHERE deleted_at IS NULL AND terminal=0 AND EXISTS "
        "(SELECT 1 FROM messages m WHERE m.session_id=s.id AND m.at >= ?)",
        (cutoff,),
    ):
        cfg = choices.effective(
            config, "interlocutor", choices.get(conn, "session", session["id"], "interlocutor")
        )
        try:
            lineup = router.lineup(cfg, calls.Role.INTERLOCUTOR)
        except router.NoProvider:
            lineup = []
        removed = {
            (h.source, h.model)
            for h in holds.active(conn, "session", session["id"])
            if h.state == "removed" or h.state == "stopped" and session["created_at"] > h.at
        }
        lineup = [
            c
            for c in lineup
            if any((source, c.label) not in removed for source in router.sources(cfg, c))
        ]
        latest = conn.execute(
            "SELECT provider, model, key_id FROM model_calls WHERE session_id=? "
            "AND role='interlocutor' AND error_kind IS NULL ORDER BY seq DESC LIMIT 1",
            (session["id"],),
        ).fetchone()
        if (
            latest
            and (latest["key_id"] or latest["provider"], f"{latest['provider']}/{latest['model']}")
            not in removed
        ):
            actual = router.Choice(providers.find(latest["provider"]), latest["model"])
            lineup = [actual, *(c for c in lineup if c.label != actual.label)]
        if lineup:
            first = lineup[0]
            preferred = (
                latest["key_id"]
                if latest and first.label == f"{latest['provider']}/{latest['model']}"
                else router.preferred_key(cfg, calls.Role.INTERLOCUTOR)
            )
            sources = [
                source
                for source in router.sources(cfg, first)
                if (source, first.label) not in removed
            ]
            source = preferred if preferred in sources else next(iter(sources), "")
            card = add(source, first.label, "local", "interlocutor")
            if card:
                card["work"].append(
                    {
                        "kind": "chat",
                        "id": session["id"],
                        "title": short_title(session["name"] or session["id"][:8]),
                        "role": "interlocutor",
                    }
                )
    # A module is listed while it has work: a live task, a chat, a model lent to desman-global,
    # or a stop the person may lift. One whose tasks are done or failed is gone.
    kept: set[tuple[str, str, str]] = set()
    for cap in capabilities(config):
        if add(cap.key or cap.provider, f"{cap.provider}/{cap.model}", "network"):
            kept.add(("network", cap.key or cap.provider, f"{cap.provider}/{cap.model}"))
    hour = db.to_iso(datetime.now(UTC) - timedelta(hours=1))
    for row in conn.execute(
        "SELECT c.*, t.goal, t.origin, s.kind, s.status FROM model_calls c "
        "LEFT JOIN tasks t ON t.id=c.task_id LEFT JOIN subtasks s ON s.id=c.subtask_id "
        "WHERE c.at >= ? AND c.seq IN (SELECT MAX(seq) FROM model_calls GROUP BY provider, model, key_id, task_id, session_id)",
        (hour,),
    ):
        section = "network" if row["origin"] == "network" else "local"
        card = add(row["key_id"] or row["provider"], f"{row['provider']}/{row['model']}", section)
        if not card:
            continue
        if row["error_type"] == "QuotaExhausted":
            card["state"] = t("quota used up")
        elif row["error_type"] == "RateLimited":
            card["state"] = t("waiting for a limit")
        historically_removed = any(
            h.state == "removed"
            and h.source == (row["key_id"] or row["provider"])
            and h.model == f"{row['provider']}/{row['model']}"
            and (
                h.scope == "all"
                or h.scope == "network"
                and section == "network"
                or h.scope == "task"
                and h.scope_id == row["task_id"]
                or h.scope == "session"
                and h.scope_id == row["session_id"]
            )
            for h in current_holds
        )
        # Keep recent usage, but do not reassign work to a removed historical module.
        if historically_removed:
            continue
        if (
            row["task_id"]
            and active_task(row["task_id"])
            and row["status"]
            in {
                "pending",
                "ready",
                "running",
                "review",
                "needs_human",
                "paused",
            }
        ):
            role = row["kind"] if row["role"] == "executor" else row["role"]
            work = {
                "kind": "network job" if section == "network" else "task",
                "id": row["task_id"],
                "title": short_title(row["goal"]),
                "role": role,
            }
            if work not in card["work"]:
                card["work"].append(work)
            if row["status"] in {"running", "review"} and row["error_type"] is None:
                card["state"] = t("working")
        if row["session_id"] and row["role"] == "interlocutor":
            session = conn.execute(
                "SELECT COALESCE(title, (SELECT text FROM messages WHERE session_id=s.id ORDER BY seq LIMIT 1)) AS name "
                "FROM sessions s WHERE id=? AND deleted_at IS NULL AND terminal=0",
                (row["session_id"],),
            ).fetchone()
            latest = conn.execute(
                "SELECT MAX(seq) FROM model_calls WHERE session_id=? AND role='interlocutor'",
                (row["session_id"],),
            ).fetchone()[0]
            if session is not None and latest == row["seq"]:
                chat_work = {
                    "kind": "chat",
                    "id": row["session_id"],
                    "title": short_title(session["name"] or row["session_id"][:8]),
                    "role": "interlocutor",
                }
                if chat_work not in card["work"]:
                    card["work"].append(chat_work)
    for hold in current_holds:
        section = "network" if hold.scope == "network" else "local"
        card = add(hold.source, hold.model, section)
        if card:
            card.setdefault("holds", []).append(hold.model_dump(exclude={"paused_tasks"}))
            # Stopped everywhere: kept so it can be continued. A stop on one task or chat
            # shows on that card; its module is not kept here once that work is over.
            if hold.state == "stopped" and hold.scope in {"all", "network"}:
                kept.add((section, hold.source, hold.model))
        if hold.scope in {"all", "network"}:
            for card in found.values():
                if card["module_source"] != hold.source or card["model"] != hold.model:
                    continue
                if hold.scope == "network" and card["section"] != "network":
                    continue
                card["state"] = t("stopped") if hold.state == "stopped" else t("removed")
                card["hold_state"] = hold.state
                card["scope"] = hold.scope
                card["scope_id"] = hold.scope_id
    return [
        card
        for identity, card in found.items()
        if card["work"] or identity in kept or card.get("hold_state") == "stopped"
    ]
