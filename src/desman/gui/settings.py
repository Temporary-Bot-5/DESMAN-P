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


def role_change(change: Roles, config: Config) -> tuning_cli.TuningChange:
    """Keep page key identities separate from label-sensitive CLI selectors."""
    fingerprints: dict[str, str | None]
    row = next(r for r in tuning_cli.role_rows(config) if r["role"] == change.role)
    if change.model is not None and change.model not in [
        *row["candidate_models"],
        row["model"],
    ]:
        raise click.ClickException(t("Invalid module choice."))
    if change.key is not None and change.key == row["key_fingerprint"]:
        # Like a CLI model edit, retain the independent saved key.
        fingerprints = {}
    else:
        fingerprints = {
            change.role: tuning_cli.resolved_key(config, change.role, change.model, change.key)
        }
    return tuning_cli.TuningChange(
        models={change.role: change.model or "auto"}, fingerprints=fingerprints
    )


def local_search(config: Config, query: str) -> dict[str, Any]:
    """Read-only local/HF choices. External failures are ordinary translated status notes."""
    import httpx

    from desman import local_models
    from desman.llm import hub

    tags, reachable = local_models.installed(config)
    selected = providers.find("ollama").model(config)
    rows: list[dict[str, Any]] = [
        {
            "value": m.name,
            "label": m.name,
            "group": t("Downloaded"),
            "hint": hub.size_text(m.size),
            "downloaded": True,
            "selected": m.name == selected,
        }
        for m in tags
    ]
    notes = [] if reachable else [t("Ollama cannot be reached right now.")]
    try:
        with httpx.Client() as client:
            found = hub.search(
                providers.redact_keys(query, config)[0], client, local_models.token(config)
            )
        machine = hub.memory()
        for model in found:
            known = model.gguf()
            params = model.metadata.total if model.metadata else None
            if params:
                known = hub.Gguf(
                    hub.Ref(model.id, hub.DEFAULT_QUANT),
                    (),
                    int(params * 0.6),
                    False,
                    (hub.DEFAULT_QUANT,),
                )
            size = (
                f"{params / 1e9:.1f}B · "
                + t("Approximately {size} (Q4_K_M)", size=hub.size_text(known.size))
                if params and known
                else hub.size_text(known.size)
                if known
                else t("Size unknown")
            )
            fits = hub.fits(known, machine) if known else None
            hint = size + (
                " · " + (t("Fits in memory") if fits else t("Does not fit in memory"))
                if fits is not None
                else ""
            )
            if model.downloads is not None:
                hint += " · " + t("Downloads: {count}", count=model.downloads)
            rows.append(
                {
                    "value": model.id,
                    "label": model.id,
                    "group": "Hugging Face",
                    "hint": hint,
                    "downloaded": False,
                }
            )
    except (httpx.HTTPError, ValueError):
        notes.append(t("Hugging Face cannot be reached right now."))
    return {
        "options": rows,
        "notes": notes,
        "reachable": reachable,
        "host": providers.find("ollama").setting(config, "OLLAMA_HOST"),
        "selected": selected,
    }
