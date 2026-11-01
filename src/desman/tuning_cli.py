"""`desman tuning`: how the work is done, as the person tunes it (ROADMAP 5.4).

The model each role asks first, and the limits of the workflow: subtasks in a plan, parts of a
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

from desman import runs, ui
from desman.config import TUNING_LIMITS, Config, ConfigManager, ModelChoices, Tuning
from desman.config_cli import json_option, load_config
from desman.i18n import t
from desman.llm import router

AUTO = "auto"
"""On the command line: back to DESMAN's default."""
ROLES = tuple(ModelChoices.model_fields)
"""planner, reviewer, decomposer, and the executor's text, code, analysis."""


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
    "code, analysis. Repeatable.",
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
        reset=reset,
    )
    if change.empty():
        current = manager.load()
        if as_json:
            click.echo(json.dumps(tuning_json(current.tuning), ensure_ascii=False, indent=2))
            return
        print_tuning(current.tuning)
        return
    old, new, config = set_tuning(manager, change)
    if as_json:
        click.echo(json.dumps(tuning_json(new), ensure_ascii=False, indent=2))
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


@dataclass(frozen=True)
class TuningChange:
    """What to change: numbers and models given (`auto`: back to the default), or all."""

    limits: dict[str, str | int] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    reset: bool = False

    def empty(self) -> bool:
        return not (self.limits or self.models or self.reset)

    def applied(self, current: Tuning) -> Tuning:
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
        try:
            return Tuning.model_validate(data)
        except ValueError as e:
            raise click.UsageError(str(e)) from e


def set_tuning(manager: ConfigManager, change: TuningChange) -> tuple[Tuning, Tuning, Config]:
    """(before, after, the config), changed under the config's lock."""
    with manager.edit() as config:
        old = config.tuning
        config.tuning = change.applied(old)
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


def lines(old: Tuning, new: Tuning) -> list[str]:
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
    return out


def unready(config: Config, tuning: Tuning) -> list[tuple[str, str]]:
    """(role, model) chosen but not among the models that role may ask now."""
    return [
        (role, label)
        for role in ROLES
        if (label := getattr(tuning.models, role)) is not None
        and label not in router.candidates(config, role)
    ]


def print_tuning(tuning: Tuning) -> None:
    table = ui.table("", "", title=t("How the work is done"))
    table.show_header = False
    for what, value in rows(tuning):
        table.add_row(Text(what, style="muted"), value)
    ui.out(table)
    ui.note(
        t(
            "Change with `desman tuning --max-subtasks N|auto` (and --max-parts, "
            "--context-tokens, --max-attempts, --max-transient), `--model ROLE=PROVIDER/MODEL|auto`, "
            "or `--reset`."
        )
    )


def print_change(old: Tuning, new: Tuning, config: Config) -> None:
    changed = lines(old, new)
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


def tuning_json(tuning: Tuning) -> dict[str, Any]:
    return {**tuning.model_dump(), "defaults": dict(runs.TUNING_DEFAULTS)}
