"""Read-only settings data and translation of human fields into existing command bodies."""

from __future__ import annotations

import sqlite3
from typing import Any

import click

from desman import commands, global_cli, i18n, protected_cli, providers, runs, tuning_cli, wizard
from desman.config import TUNING_BOUNDS, Config
from desman.core import protected
from desman.core.profile import Profile
from desman.gui.front import WebFront
from desman.i18n import t
from desman.llm.openai_compat import key_id
from desman.settings_actions import (
    Behaviour,
    General,
    Keys,
    Limits,
    Models,
    Network,
    Permission,
    Personalize,
    Protected,
    Roles,
    Setting,
    Wizard,
)


def data(conn: sqlite3.Connection | None, config: Config, front: WebFront) -> dict[str, Any]:
    """Only explicit public fields leave the server; never serialize Config or APIKey."""
    provider_rows = []
    for p in providers.PROVIDERS + providers.SERVICES:
        provider_rows.append(
            {
                "id": p.id,
                "name": p.name,
                "url": p.url,
                "ready": p.is_ready(config),
                "keyed": p.key_env is not None,
                "keys": [
                    {
                        "fingerprint": key_id(k.value),
                        "label": providers.label(k.label),
                        "mask": providers.mask(k.value),
                    }
                    for k in config.keys(p.key_env or "")
                ],
                "settings": [
                    {
                        "name": s.env,
                        "label": wizard.provider_setting_label(s),
                        "value": p.setting(config, s.env),
                    }
                    for s in p.settings
                ],
                "lendable": providers.can_donate(p),
            }
        )
    with front.condition:
        outcome = dict(front.settings_outcome) if front.settings_outcome else None
    # changes() can record an initial release approval; HTTP snapshots must never do that.
    try:
        changes = protected._state(conn, protected.here())[1] if conn is not None else []
        protection = {"blocked": bool(changes), "lines": protected_cli.change_lines(changes)}
    except protected.ProtectedError as e:
        protection = {
            "blocked": True,
            "lines": [t("Cannot check the protected files: {error}", error=e)],
        }
    return {
        "general": {
            "language": config.language,
            "languages": [
                {"value": code, "label": cat.name} for code, cat in i18n.CATALOGS.items()
            ],
        },
        "providers": provider_rows,
        "roles": tuning_cli.role_rows(config),
        "permissions": [
            {"name": n, "label": label, "on": getattr(config.permissions, n)}
            for n, label in wizard.permission_labels().items()
        ],
        "personalize": {
            "on": config.personalize,
            "consent": wizard.personalize_consent(),
            "notes": [{"id": n.id, "text": n.text} for n in Profile(conn).notes()]
            if conn is not None and config.personalize
            else [],
        },
        "behaviour": {
            "values": config.behaviour.model_dump(),
            "labels": wizard.behaviour_labels(),
            "choices": {
                part: [{"value": v or "auto", "label": label} for v, label in opts.items()]
                for part, opts in wizard.behaviour_values().items()
            },
        },
        "limits": [
            {
                "name": n,
                "label": tuning_cli.labels()[n],
                "value": getattr(config.tuning, n),
                "min": bounds[0],
                "max": bounds[1],
                "default": runs.TUNING_DEFAULTS[n],
            }
            for n, bounds in TUNING_BOUNDS.items()
        ],
        "network": {
            **global_cli.status(config).as_json(),
            "default_quota": global_cli.DEFAULT_DONATION.model_dump(),
        },
        "protected": protection,
        "result": outcome,
    }


def command(change: Setting, config: Config) -> tuple[commands.Entry, commands.Args]:
    """Resolve fingerprints against current config, so stale indexes cannot remove another key."""
    name: str
    args: dict[str, Any]
    if isinstance(change, General):
        name, args = "language", {"code": change.language}
    elif isinstance(change, Wizard):
        name, args = "config", {}
    elif isinstance(change, Keys):
        name, args = f"keys {change.operation}", {"provider": change.provider}
        if change.operation == "add":
            args["label"] = change.label
        elif change.operation == "remove":
            p = providers.find(change.provider)
            number = next(
                (
                    i
                    for i, k in enumerate(config.keys(p.key_env or ""), 1)
                    if key_id(k.value) == change.key
                ),
                None,
            )
            if number is None:
                raise click.ClickException(t("Please check your answer and try again."))
            args["number"] = number
    elif isinstance(change, Roles):
        row = next(r for r in tuning_cli.role_rows(config) if r["role"] == change.role)
        if change.model is not None and change.model not in [
            *row["candidate_models"],
            row["model"],
        ]:
            raise click.ClickException(t("Invalid module choice."))
        selected = next((k for k in row["candidate_keys"] if k["fingerprint"] == change.key), None)
        # An unavailable saved key may be kept; changing the model must not silently clear it.
        if change.key is not None and selected is None:
            if change.key != row["key_fingerprint"]:
                raise click.ClickException(t("Invalid module choice."))
            keys = {}
        else:
            provider = (change.model or next(iter(row["candidate_models"]), "")).split("/", 1)[0]
            if selected and selected["provider"] != provider:
                raise click.ClickException(t("Invalid module choice."))
            keys = {change.role: selected["selector"] if selected else "auto"}
        name, args = "tuning", {"models": {change.role: change.model or "auto"}, "keys": keys}
    elif isinstance(change, Permission):
        name, args = "permissions", {"grant" if change.on else "revoke": [change.name]}
    elif isinstance(change, Personalize):
        name, args = "personalize", {"on": change.on}
    elif isinstance(change, Behaviour):
        name, args = "behaviour", change.model_dump(exclude={"section"})
    elif isinstance(change, Limits):
        name, args = "tuning", {change.name: change.value}
    elif isinstance(change, Models):
        name, args = f"models {change.operation}", {"model": change.model}
        if change.operation == "add":
            args["server"] = change.server
    elif isinstance(change, Network):
        name, args = f"global {change.operation}", {}
        if change.operation == "join":
            args["registry"] = change.registry
        if change.provider:
            args["provider"] = change.provider
        if change.operation in ("donate", "quota"):
            args.update(calls=change.calls, tokens=change.tokens)
    elif isinstance(change, Protected):
        name, args = f"protected {change.operation}", {}
        if change.operation == "show":
            args["diff"] = True
    else:
        raise ValueError("not a catalog field")
    entry = commands.find(name)
    assert entry is not None
    return entry, commands.parse(entry, args, config)
