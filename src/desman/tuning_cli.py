"""`desman tuning`: how the work is done, as the person tunes it (ROADMAP 5.4).

The model and key each role asks first, and the limits of the workflow: subtasks in a plan, parts of a
split subtask, the dependency results an executor sees, the retry budgets of a planned task.
Unset means DESMAN's default (`runs.TUNING_DEFAULTS`, its own order of models). The
Improvement Analyst may propose a change here, as `desman tuning` arguments; only the person
makes it. Non-interactive; the chat calls the same functions after the person's "yes".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import click
from rich.text import Text

from desman import local_models, providers, runs, ui
from desman.config import TUNING_LIMITS, Config, ConfigManager, ModelChoices, Tuning
from desman.config_cli import json_option, load_config
from desman.i18n import t
from desman.llm import catalog, router
from desman.llm.openai_compat import key_id

AUTO = "auto"
"""On the command line: back to DESMAN's default."""
ROLES = tuple(ModelChoices.model_fields)
"""Role preference fields, including executor kinds and the interlocutor."""


@click.command("tuning")
@click.option("--max-subtasks", metavar="N|auto", help="The most subtasks in a plan (1–12).")
@click.option(
    "--max-parts", metavar="N|auto", help="The most parts a failing subtask is split into (2–6)."
)
@click.option(
    "--context-tokens",
    metavar="N|auto",
    help="Tokens of dependency results an executor is shown (1000–16000).",
)
@click.option(
    "--max-attempts",
    metavar="N|auto",
    help="Attempts at a subtask before you decide (2–8), for tasks planned from now on.",
)
@click.option(
    "--max-transient",
    metavar="N|auto",
    help="Interruptions of a subtask before you decide (2–10), for tasks planned from now on.",
)
@click.option(
    "--model",
    "models",
    multiple=True,
    metavar="ROLE=PROVIDER/MODEL|auto",
    help="The model a role asks first: planner, reviewer, decomposer, or the executor's text, "
    "code, analysis, interlocutor. Repeatable.",
)
@click.option(
    "--key",
    "keys",
    multiple=True,
    metavar="ROLE=PROVIDER:KEY|auto",
    help="The key a role asks first: planner, reviewer, decomposer, text, code, analysis, "
    "interlocutor. Use a label or number. Repeatable.",
)
@click.option("--reset", is_flag=True, help="Everything back to DESMAN's defaults.")
@json_option
def tuning(
    max_subtasks: str | None,
    max_parts: str | None,
    context_tokens: str | None,
    max_attempts: str | None,
    max_transient: str | None,
    models: tuple[str, ...],
    keys: tuple[str, ...],
    reset: bool,
    as_json: bool,
) -> None:
    """How the work is done: the model each role asks first and the limits of the workflow.
    Show, or change."""
    manager = load_config()
    numbers = {
        "max_subtasks": max_subtasks,
        "max_parts": max_parts,
        "context_tokens": context_tokens,
        "max_attempts": max_attempts,
        "max_transient": max_transient,
    }
    change = TuningChange(
        limits={k: v for k, v in numbers.items() if v is not None},
        models=parse_models(models),
        keys=parse_keys(keys),
        reset=reset,
    )
    if change.empty():
        current = manager.load()
        if as_json:
            click.echo(
                json.dumps(tuning_json(current.tuning, current), ensure_ascii=False, indent=2)
            )
            return
        print_tuning(current.tuning, current)
        return
    old, new, config = set_tuning(manager, change)
    if as_json:
        click.echo(json.dumps(tuning_json(new, config), ensure_ascii=False, indent=2))
        return
    print_change(old, new, config)


def parse_models(given: tuple[str, ...]) -> dict[str, str]:
    """`ROLE=PROVIDER/MODEL` arguments as {role: label}."""
    out = {}
    for item in given:
        role, sep, label = item.partition("=")
        role, label = role.strip(), label.strip()
        if not sep or role not in ROLES or not label:
            raise click.UsageError(
                t("--model takes ROLE=PROVIDER/MODEL, ROLE one of: {roles}", roles=", ".join(ROLES))
            )
        out[role] = label
    return out


def parse_keys(given: tuple[str, ...]) -> dict[str, str]:
    out = {}
    for item in given:
        role, sep, value = item.partition("=")
        if not sep or role.strip() not in ROLES or not value.strip():
            raise click.UsageError(
                t(
                    "--key takes ROLE=PROVIDER:KEY|auto, ROLE one of: {roles}",
                    roles=", ".join(ROLES),
                )
            )
        out[role.strip()] = value.strip()
    return out


def key_choices(config: Config) -> list[dict[str, str]]:
    """Configured cloud keys, with only safe public identifiers and display labels."""
    return [
        {
            "provider": p.id,
            "fingerprint": key_id(k.value),
            "selector": f"{p.id}:#{i}",
            "label": f"{providers.label(k.label) if k.label else f'#{i}'} {providers.mask(k.value)}",
        }
        for p in providers.CLOUD
        for i, k in enumerate(config.keys(p.key_env or ""), 1)
    ]


def resolve_key(config: Config, selector: str) -> str | None:
    if selector == AUTO:
        return None
    provider, sep, name = selector.partition(":")
    choices = key_choices(config)
    try:
        p = providers.find(provider)
    except providers.UnknownProvider:
        p = None
    if p is not None and sep and name:
        keys = config.keys(p.key_env or "")
        # Labels take precedence over numeric selectors, as in keys remove.
        matches = [k for k in keys if k.label == name]
        if len(matches) == 1:
            return key_id(matches[0].value)
        for i, key in enumerate(keys, 1):
            if name in (str(i), f"#{i}"):
                return key_id(key.value)
    available = ", ".join(f"{k['selector']} {k['label']}" for k in choices) or t("no key")
    # Do not echo the selector: a person might accidentally paste a secret there.
    raise click.ClickException(
        t("Unknown provider or key. Available: {choices}", choices=available)
    )


def resolved_key(
    config: Config, role: str, model: str | None, fingerprint: str | None
) -> str | None:
    """Validate a page's public key identity without interpreting CLI selectors."""
    if fingerprint is None:
        return None
    if not any(
        k["fingerprint"] == fingerprint
        and (model is None or k["provider"] == model.split("/", 1)[0])
        for k in key_choices(config)
    ):
        raise click.ClickException(t("Invalid module choice."))
    return fingerprint


def role_rows(config: Config) -> list[dict[str, Any]]:
    """Safe role data shared by the CLI, wizard and settings page."""
    keys = key_choices(config)
    role_labels = {
        "planner": t("Planner"),
        "reviewer": t("Reviewer"),
        "decomposer": t("Decomposer"),
        "interlocutor": t("Interlocutor"),
        **{kind: f"{t('Executor')} ({kind})" for kind in ("text", "code", "analysis")},
    }
    ollama = providers.find("ollama")
    installed = local_models.installed(config, cached_only=True)[0]
    out = []
    for role in ROLES:
        fingerprint = getattr(config.tuning.keys, role)
        found = next((k for k in keys if k["fingerprint"] == fingerprint), None)
        candidates = router.candidates(config, role)
        available: list[dict[str, Any]] = []
        for provider in (*providers.CLOUD, ollama):
            provider_keys = [dict(k) for k in keys if k["provider"] == provider.id]
            if (provider_keys and not provider.is_ready(config)) or (
                not provider_keys
                and not (provider.id == "ollama" and (provider.is_ready(config) or installed))
            ):
                continue
            models = [m for m in candidates if m.startswith(provider.id + "/")]
            if not provider.local:
                models = list(
                    dict.fromkeys(
                        [*(provider.id + "/" + m.id for m in catalog.models_of(provider)), *models]
                    )
                )
            if provider.id == "ollama":
                models = list(dict.fromkeys([*models, *("ollama/" + m.name for m in installed)]))
            available.append(
                {"id": provider.id, "name": provider.name, "models": models, "keys": provider_keys}
            )
        candidates = list(
            dict.fromkeys([*candidates, *(m for p in available for m in p["models"])])
        )
        out.append(
            {
                "role": role,
                "providers": available,
                "label": role_labels[role],
                "model": getattr(config.tuning.models, role),
                "key_fingerprint": fingerprint,
                "key_label": found["label"] if found else t("missing") if fingerprint else AUTO,
                "candidate_models": candidates,
                "candidate_keys": [dict(k) for k in keys],
            }
        )
    return out


@dataclass(frozen=True)
class TuningChange:
    """Numbers, models and key selectors to change (`auto`: the default), or reset all."""

    limits: dict[str, str | int] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    keys: dict[str, str] = field(default_factory=dict)
    reset: bool = False
    fingerprints: dict[str, str | None] = field(default_factory=dict)

    def empty(self) -> bool:
        return not (self.limits or self.models or self.keys or self.fingerprints or self.reset)

    def applied(self, current: Tuning, config: Config | None = None) -> Tuning:
        data = Tuning().model_dump() if self.reset else current.model_dump()
        for name, value in self.limits.items():
            if value == AUTO:
                data[name] = None
            elif isinstance(value, int):
                data[name] = value
            else:
                try:
                    data[name] = int(value)
                except ValueError:
                    raise click.UsageError(
                        t("--{name} takes a whole number or auto", name=name.replace("_", "-"))
                    ) from None
        for role, label in self.models.items():
            data["models"][role] = None if label == AUTO else label
        for role, selector in self.keys.items():
            if config is None:
                raise ValueError("key choices require config")
            data["keys"][role] = resolve_key(config, selector)
        for role, fingerprint in self.fingerprints.items():
            if config is None:
                raise ValueError("key choices require config")
            data["keys"][role] = resolved_key(config, role, data["models"][role], fingerprint)
        try:
            return Tuning.model_validate(data)
        except ValueError as e:
            raise click.UsageError(str(e)) from e


def changed_tuning(config: Config, change: TuningChange) -> tuple[Tuning, Tuning, Config]:
    """Pure preview shared by settings surfaces; the input config remains untouched."""
    old = config.tuning
    new = change.applied(old, config)
    return old, new, config.model_copy(update={"tuning": new})


def set_tuning(manager: ConfigManager, change: TuningChange) -> tuple[Tuning, Tuning, Config]:
    """(before, after, the config), changed under the config's lock."""
    with manager.edit() as config:
        old, new, _ = changed_tuning(config, change)
        config.tuning = new
    return old, config.tuning, config


# ── Output ───────────────────────────────────────────────────────────────────


def labels() -> dict[str, str]:
    return {
        "max_subtasks": t("most subtasks in a plan"),
        "max_parts": t("most parts of a split subtask"),
        "context_tokens": t("tokens of dependency results shown"),
        "max_attempts": t("attempts before you decide"),
        "max_transient": t("interruptions before you decide"),
    }


def shown(tuning: Tuning, name: str) -> str:
    """A number or a model as set, or what the default is."""
    if name in TUNING_LIMITS:
        value = getattr(tuning, name)
        return (
            t("default ({value})", value=runs.TUNING_DEFAULTS[name])
            if value is None
            else str(value)
        )
    label = getattr(tuning.models, name)
    return t("DESMAN's choice") if label is None else str(label)


def rows(tuning: Tuning) -> list[tuple[str, str]]:
    names = labels()
    out = [(names[n], shown(tuning, n)) for n in TUNING_LIMITS]
    out += [(t("model: {role}", role=role), shown(tuning, role)) for role in ROLES]
    return out


def lines(old: Tuning, new: Tuning, config: Config | None = None) -> list[str]:
    """What changes, a line each: "most subtasks in a plan: default (8) → 6"."""
    names = labels()
    out = [
        t("{what}: {old} → {new}", what=names[n], old=shown(old, n), new=shown(new, n))
        for n in TUNING_LIMITS
        if getattr(old, n) != getattr(new, n)
    ]
    out += [
        t(
            "{what}: {old} → {new}",
            what=t("model: {role}", role=role),
            old=shown(old, role),
            new=shown(new, role),
        )
        for role in ROLES
        if getattr(old.models, role) != getattr(new.models, role)
    ]
    for role in ROLES:
        if getattr(old.keys, role) != getattr(new.keys, role):

            def key_shown(tuning: Tuning, role: str = role) -> str:
                fp = getattr(tuning.keys, role)
                choices = key_choices(config) if config else []
                return next(
                    (k["label"] for k in choices if k["fingerprint"] == fp),
                    t("missing") if fp else AUTO,
                )

            out.append(
                t(
                    "{what}: {old} → {new}",
                    what=t("key: {role}", role=role),
                    old=key_shown(old),
                    new=key_shown(new),
                )
            )
    return out


def unready(config: Config, tuning: Tuning) -> list[tuple[str, str]]:
    """(role, model) chosen but not among the models that role may ask now."""
    return [
        (role, label)
        for role in ROLES
        if (label := getattr(tuning.models, role)) is not None
        and label not in router.candidates(config, role)
    ]


def print_tuning(tuning: Tuning, config: Config | None = None) -> None:
    table = ui.table("", "", title=t("How the work is done"))
    table.show_header = False
    for what, value in rows(tuning):
        table.add_row(Text(what, style="muted"), value)
    if config is not None:
        for row in role_rows(config):
            table.add_row(
                Text(t("key: {role}", role=row["role"]), style="muted"), Text(row["key_label"])
            )
    ui.out(table)
    ui.note(
        t(
            "Change with `desman tuning --max-subtasks N|auto` (and --max-parts, "
            "--context-tokens, --max-attempts, --max-transient), `--model ROLE=PROVIDER/MODEL|auto`, `--key ROLE=PROVIDER:KEY|auto`, "
            "or `--reset`."
        )
    )


def print_change(old: Tuning, new: Tuning, config: Config) -> None:
    changed = lines(old, new, config)
    if not changed:
        ui.note(t("Nothing changed."))
    for line in changed:
        ui.ok(line)
    for role, label in unready(config, new):
        ui.warn(
            t(
                "{model} is not among the models {role} can use now (a ready free provider): it "
                "is kept, and goes first once it is.",
                model=label,
                role=role,
            )
        )
    for row in role_rows(config):
        fp, model = row["key_fingerprint"], row["model"]
        key = next((k for k in key_choices(config) if k["fingerprint"] == fp), None)
        if key and model and model.split("/", 1)[0] != key["provider"]:
            ui.warn(
                t(
                    "The key for {role} only applies to models from {provider}.",
                    role=row["role"],
                    provider=key["provider"],
                )
            )


def tuning_json(tuning: Tuning, config: Config | None = None) -> dict[str, Any]:
    result = {**tuning.model_dump(), "defaults": dict(runs.TUNING_DEFAULTS)}
    if config is not None:
        result["roles"] = role_rows(config)
    return result
