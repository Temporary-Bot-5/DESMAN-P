"""The catalog of DESMAN's commands for the chat (ROADMAP C.5, ADR 0003 §3–4).

Every command of the CLI has an entry here (a test walks the CLI and the catalog both ways):
a description for the interlocutor, its arguments as a Pydantic model and its class. The class
is decided by the call, not by the command: `language` without a code reads, with one changes.

- **read**: runs at once; the person sees the result and the interlocutor gets it as data;
- **change**: the handler checks the request and returns a `Proposal`, the exact change worded
  by code. The chat shows it in a window and only the person's "yes" there applies it;
- **input**: the person types what is needed in a window (a hidden field for a key, a section
  of the wizard). Nothing the model wrote can fill it.

Handlers call the same functions as the CLI (`desman.actions`, `desman.config_cli`) and report
problems as `click.ClickException`, like the CLI. The interlocutor only names a command and its
arguments; arguments are validated here and never run as a command line. What it gets back
never holds a key: `for_model()` cuts anything key-shaped, and key values are left out of the
data in the first place.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import click
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, ValidationError
from rich.text import Text

from desman import actions, config_cli, i18n, providers, ui, wizard
from desman.config import BEHAVIOUR_CHARS, Behaviour, Config, ConfigError, ConfigManager
from desman.core import db
from desman.core.report import inbox, task_report, task_summaries
from desman.core.states import SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.i18n import t, tn
from desman.llm.openai_compat import key_id


class Kind(StrEnum):
    READ = "read"
    CHANGE = "change"
    INPUT = "input"


class Args(BaseModel):
    """A command's arguments: what the interlocutor may pass, nothing else."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


@dataclass(frozen=True)
class Outcome:
    """What a command did. The person has seen it; `data` is for the interlocutor (None: the
    conversation already tells it, e.g. a task that was retold)."""

    data: Any
    done: bool = True
    """False: the person left the window empty or cancelled it; nothing changed."""


@dataclass(frozen=True)
class Proposal:
    """A change waiting for the person's "yes": what it does, worded by code, and the doing."""

    lines: list[str]
    apply: Callable[[], Outcome]
    warnings: list[str] = field(default_factory=list)
    question: str | None = None
    """Asked instead of "Apply this change?" (opening the wizard changes nothing yet)."""


@dataclass
class Context:
    """What handlers work with: the chat's database and session, the config, and the chat's own
    way of running a task (with live progress and a retelling)."""

    store: Store
    manager: ConfigManager
    start_task: Callable[[str, str], None]
    """(title, goal): plan a task of the session and run it."""
    resume_task: Callable[[str], None]
    """(task id): run the task further."""

    def config(self) -> Config:
        try:
            return self.manager.load()
        except ConfigError as e:
            raise click.ClickException(str(e)) from e


Handler = Callable[[Context, Any], "Outcome | Proposal"]


@dataclass(frozen=True)
class Entry:
    name: str
    """As typed after `desman`, e.g. "keys add"."""
    about: str
    """For the interlocutor (a prompt to a model: English)."""
    args: type[Args]
    kind: Kind | Callable[[Any], Kind]
    handler: Handler
    aliases: tuple[str, ...] = ()
    """Other CLI names of the same command (`keys` lists keys like `keys list`)."""

    def kind_of(self, args: Args) -> Kind:
        return self.kind if isinstance(self.kind, Kind) else self.kind(args)


class BadArgs(Exception):
    """The interlocutor's arguments do not fit the command. The message never quotes them."""


# ── Arguments ────────────────────────────────────────────────────────────────

Ref = Annotated[str, StringConstraints(min_length=1, max_length=100)]
Note = Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class NoArgs(Args):
    pass


class TaskArgs(Args):
    task: Ref = Field(description="task id or a unique prefix of it")


class ListArgs(Args):
    status: Literal["active", "done", "needs_human", "failed", "cancelled"] | None = None
    limit: Annotated[int, Field(ge=1, le=100)] = 20


class DecisionArgs(Args):
    subtask: Ref = Field(description='"<task>/<number>" as status and inbox show it')
    note: Note | None = Field(None, description="guidance for the next attempt, or why")


class FailArgs(Args):
    subtask: Ref
    note: Note = Field(description="why it failed (required)")


class CancelArgs(Args):
    task: Ref
    note: Note = "cancelled by operator"


class RunArgs(Args):
    goal: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    title: Annotated[str, StringConstraints(min_length=1, max_length=120)] | None = None


class UsageArgs(Args):
    since: Annotated[str, StringConstraints(pattern=r"^(all|[1-9]\d*[mhd])$")] = "24h"


class LanguageArgs(Args):
    code: Annotated[str, StringConstraints(min_length=1, max_length=40)] | None = Field(
        None, description="a language code such as ru, or its name"
    )


class ProviderArgs(Args):
    provider: Annotated[str, StringConstraints(min_length=1, max_length=40)] | None = None


class KeyAddArgs(Args):
    provider: Annotated[str, StringConstraints(min_length=1, max_length=40)]
    label: Annotated[str, StringConstraints(max_length=60)] = ""


class KeyRemoveArgs(Args):
    provider: Annotated[str, StringConstraints(min_length=1, max_length=40)]
    number: Annotated[int, Field(ge=1)] | None = Field(None, description="as keys list numbers it")
    all: bool = False


class ImportArgs(Args):
    file: Annotated[str, StringConstraints(min_length=1, max_length=500)]


class PersonalizeArgs(Args):
    on: bool | None = Field(None, description="true: turn on, false: off; none: show")


class BehaviourArgs(Args):
    length: Literal["short", "detailed", "auto"] | None = None
    tone: Literal["formal", "informal", "auto"] | None = None
    style: Literal["simple", "technical", "auto"] | None = None
    custom: Annotated[str, StringConstraints(max_length=BEHAVIOUR_CHARS)] | None = Field(
        None, description="anything else, in the person's words; empty clears it"
    )
    reset: bool = False

    def change(self) -> config_cli.BehaviourChange:
        return config_cli.BehaviourChange(**self.model_dump())


class ConfigArgs(Args):
    section: Literal["providers", "permissions", "personalize", "behaviour"] | None = Field(
        None, description="one section of the setup wizard; none: the whole wizard"
    )


Permission = Literal["admin_rights", "fs_outside_access", "network_access", "shell_execution"]


class PermissionArgs(Args):
    grant: list[Permission] = []
    revoke: list[Permission] = []


# ── Read ─────────────────────────────────────────────────────────────────────


def _help(ctx: Context, args: NoArgs) -> Outcome:
    table = ui.table("", "")
    table.show_header = False
    for entry in CATALOG:
        table.add_row(Text(entry.name, style="key"), Text(_short_help(entry), style="muted"))
    ui.out(table)
    ui.note(t("Ask for any of them in your own words."))
    return Outcome({"commands": [e.name for e in CATALOG]})


def _short_help(entry: Entry) -> str:
    """The CLI's own help line for the command, in the person's language."""
    from desman.cli import cli  # here: the CLI imports the chat, which imports this catalog

    command = cli_command(cli, entry.name)
    return "" if command is None else command.get_short_help_str(limit=90)


def _status(ctx: Context, args: TaskArgs) -> Outcome:
    report = task_report(ctx.store, actions.find_task(ctx.store, args.task))
    actions.print_report(report)
    return Outcome(report.model_dump(mode="json"))


def _list(ctx: Context, args: ListArgs) -> Outcome:
    wanted = TaskStatus(args.status) if args.status else None
    summaries = task_summaries(ctx.store, status=wanted, limit=args.limit)
    actions.print_summaries(summaries)
    if not summaries:
        ui.note(t("No tasks."))
    return Outcome([s.model_dump(mode="json", exclude={"subtasks"}) for s in summaries])


def _audit(ctx: Context, args: TaskArgs) -> Outcome:
    found = actions.task_history(ctx.store, args.task)
    actions.print_history(found)
    return Outcome(found.as_json())


def _inbox(ctx: Context, args: NoArgs) -> Outcome:
    items = inbox(ctx.store)
    actions.print_inbox(items)
    return Outcome([i.model_dump(mode="json") for i in items])


def _usage(ctx: Context, args: UsageArgs) -> Outcome:
    config = ctx.config()
    rows = config_cli.usage_since(args.since)
    config_cli.print_usage(config, rows, config_cli.parse_window(args.since), args.since)
    return Outcome(
        [{k: v for k, v in r.items() if k != "key"} for r in config_cli.usage_json(config, rows)]
    )


def _keys_list(ctx: Context, args: NoArgs) -> Outcome:
    rows = config_cli.key_rows(ctx.config())
    config_cli.print_key_rows(rows)
    return Outcome(
        [
            {**r, "keys": [{k: v for k, v in key.items() if k != "value"} for key in r["keys"]]}
            for r in rows
            if r["keys"]
        ]
    )


def _keys_check(ctx: Context, args: ProviderArgs) -> Outcome:
    targets, results = config_cli.check_keys(ctx.config(), args.provider)
    config_cli.print_checks(targets, results)
    return Outcome(json.loads(json.dumps([asdict(r) for r in results], default=str)))


def _config_show(ctx: Context, args: NoArgs) -> Outcome:
    out = providers.describe(ctx.config())
    config_cli.print_config(ctx.manager.config_path, out)
    for row in out["providers"]:  # labels and how many, no masked values either
        if "keys" in row:
            row["keys"] = [k["label"] for k in row["keys"]]
    out["unknown_keys"] = {env: len(keys) for env, keys in out["unknown_keys"].items()}
    out["unknown_settings"] = sorted(out["unknown_settings"])
    return Outcome(out)


def _language_kind(args: LanguageArgs) -> Kind:
    return Kind.READ if args.code is None else Kind.CHANGE


def _language(ctx: Context, args: LanguageArgs) -> Outcome | Proposal:
    current = ctx.config().language
    if args.code is None:
        config_cli.print_languages(current)
        return Outcome({"current": current, "languages": list(i18n.CATALOGS)})
    code = config_cli.language_code(args.code)
    if code == current:
        ui.note(t("The language is {name} already.", name=i18n.CATALOGS[code].name))
        return Outcome({"language": code, "changed": False})

    def apply() -> Outcome:
        config_cli.set_language(ctx.manager, code)
        return Outcome({"language": code, "changed": True})

    old, new = i18n.CATALOGS[current].name, i18n.CATALOGS[code].name
    return Proposal([t("Language: {old} → {new}", old=old, new=new)], apply)


def _personalize_kind(args: PersonalizeArgs) -> Kind:
    return Kind.READ if args.on is None else Kind.CHANGE


def _personalize(ctx: Context, args: PersonalizeArgs) -> Outcome | Proposal:
    current = ctx.config().personalize
    if args.on is None or args.on == current:
        config_cli.print_personalize(current)
        return Outcome({"on": current, "changed": False})
    on = args.on

    def apply() -> Outcome:
        config_cli.set_personalize(ctx.manager, on)
        return Outcome({"on": on, "changed": True})

    if on:
        line = t("Let the interlocutor study your conversations and keep notes about you.")
        warnings = [t("The notes go to the model providers with every message.")]
    else:
        line = t("Stop the interlocutor from studying your conversations and keeping notes.")
        warnings = []
    return Proposal([line], apply, warnings)


def _behaviour_kind(args: BehaviourArgs) -> Kind:
    return Kind.READ if args.change().empty() else Kind.CHANGE


def _behaviour(ctx: Context, args: BehaviourArgs) -> Outcome | Proposal:
    config = ctx.config()
    current = config.behaviour
    change = args.change()
    if change.empty():
        config_cli.print_behaviour(current)
        return Outcome(current.model_dump())
    new = change.applied(current, config)
    lines = config_cli.behaviour_lines(current, new)
    if not lines:
        ui.note(t("Nothing changed."))
        return Outcome({"changed": False, "behaviour": current.model_dump()})

    def apply() -> Outcome:
        old, now = config_cli.set_behaviour(ctx.manager, change)
        config_cli.print_behaviour_change(old, now)
        return Outcome({"changed": True, "behaviour": now.model_dump()})

    return Proposal([t("How the interlocutor talks to you:")] + lines, apply)


def _permissions_kind(args: PermissionArgs) -> Kind:
    return Kind.READ if not args.grant and not args.revoke else Kind.CHANGE


def _permissions(ctx: Context, args: PermissionArgs) -> Outcome | Proposal:
    current = ctx.config().permissions
    if not args.grant and not args.revoke:
        config_cli.print_permissions(current)
        return Outcome(current.model_dump())
    changes = config_cli.permission_changes(current, args.grant, args.revoke)
    if not changes:
        ui.note(t("Nothing changed."))
        return Outcome({"changed": [], "permissions": current.model_dump()})
    labels = wizard.permission_labels()
    lines = [
        t("Allow agents to {what} ({name})", what=labels[name], name=name)
        if now
        else t("No longer allow agents to {what} ({name})", what=labels[name], name=name)
        for name, _, now in changes
    ]
    warnings = []
    if any(name == "admin_rights" and now for name, _, now in changes):
        warnings.append(t("admin_rights: agents may act with elevated privileges."))

    def apply() -> Outcome:
        done = config_cli.set_permissions(ctx.manager, args.grant, args.revoke)
        config_cli.print_permission_changes(done)
        return Outcome(
            {
                "changed": {name: now for name, _, now in done},
                "permissions": ctx.config().permissions.model_dump(),
            }
        )

    return Proposal(lines, apply, warnings)


# ── Tasks ────────────────────────────────────────────────────────────────────


def _run(ctx: Context, args: RunArgs) -> Proposal:
    title = args.title or actions.one_line(args.goal, 60)

    def apply() -> Outcome:
        ctx.start_task(title, args.goal)
        return Outcome(None)

    return Proposal([t("Start a task: {goal}", goal=actions.one_line(args.goal, 300))], apply)


def _resume(ctx: Context, args: TaskArgs) -> Proposal:
    task_id = actions.find_task(ctx.store, args.task)
    report = task_report(ctx.store, task_id)

    def apply() -> Outcome:
        ctx.resume_task(task_id)
        return Outcome(None)

    line = t(
        "Continue task {ref} ({status}, {done}/{total} done): {goal}",
        ref=ctx.store.short_id(task_id),
        status=report.status,
        done=report.done,
        total=report.total,
        goal=actions.one_line(report.goal, 200),
    )
    return Proposal([line], apply)


_DECISIONS = {
    "retry": (SubtaskStatus.READY, "operator_retry", "retry requested by operator"),
    "accept": (SubtaskStatus.DONE, "operator_accept", "accepted by operator"),
    "fail": (SubtaskStatus.FAILED, "operator_fail", ""),
}
"""As the CLI's `retry`, `accept` and `fail` make them: target, audit action, default reason."""


def _decision(name: str) -> Callable[[Context, DecisionArgs | FailArgs], Proposal]:
    target, action, default = _DECISIONS[name]

    def handler(ctx: Context, args: DecisionArgs | FailArgs) -> Proposal:
        pending = actions.prepare_decision(ctx.store, args.subtask, target)
        note = (args.note or "").strip() or default
        sub = pending.sub

        def apply() -> Outcome:
            decision = actions.apply_decision(ctx.store, pending, note, action)
            actions.print_decision(decision)
            return Outcome(decision.as_json())

        lines = [f"[{sub.seq}] {actions.one_line(sub.description, 80)}: {sub.status} → {target}"]
        if args.note:
            lines.append(t("Note: {note}", note=actions.one_line(note, 300)))
        return Proposal(lines, apply)

    return handler


def _cancel(ctx: Context, args: CancelArgs) -> Outcome | Proposal:
    task_id = actions.find_task(ctx.store, args.task)
    report = task_report(ctx.store, task_id)
    finished = {SubtaskStatus.DONE, SubtaskStatus.CANCELLED}
    left = sum(s.status not in finished for s in report.subtasks)

    def apply() -> Outcome:
        result = actions.cancel_task(ctx.store, task_id, args.note)
        actions.print_cancelled(result)
        return Outcome(result.as_json())

    lines = [
        t(
            "Cancel task {ref}: {goal}",
            ref=ctx.store.short_id(task_id),
            goal=actions.one_line(report.goal, 200),
        ),
        tn(left, "{n} unfinished subtask stops.", "{n} unfinished subtasks stop."),
    ]
    return Proposal(lines, apply)


def _init_db(ctx: Context, args: NoArgs) -> Proposal:
    path = db.default_db_path()

    def apply() -> Outcome:
        applied = actions.init_database(path)
        actions.print_init(path, applied)
        return Outcome({"schema_version": db.SCHEMA_VERSION, "migrations_applied": applied})

    lines = [
        t("Create the database or apply pending migrations. Safe to re-run."),
        str(path),
    ]
    return Proposal(lines, apply)


# ── Keys and the configuration ───────────────────────────────────────────────


def _keys_add(ctx: Context, args: KeyAddArgs) -> Outcome:
    p = config_cli.keyed_provider(args.provider)
    value = config_cli.ask_key(p)  # the hidden field: nothing the model wrote reaches it
    if not value:
        ui.note(t("No key given; nothing changed."))
        return Outcome({"provider": p.id, "added": False}, done=False)
    result = config_cli.add_key(ctx.manager, p, value, args.label)
    config_cli.print_added(result)
    return Outcome(
        {
            "provider": p.id,
            "added": result.added,
            "key_id": key_id(value),
            "keys": result.number,
        }
    )


def _keys_remove(ctx: Context, args: KeyRemoveArgs) -> Proposal:
    p = config_cli.keyed_provider(args.provider)
    if args.all == (args.number is not None):
        raise click.UsageError(t("give a key NUMBER or --all"))
    planned = config_cli.removal(ctx.config(), p, args.number)

    def apply() -> Outcome:
        done = config_cli.remove_keys(ctx.manager, p, args.number)
        config_cli.print_removed(done)
        return Outcome(
            {"provider": p.id, "removed": len(done.removed), "left": done.count - len(done.removed)}
        )

    lines = [
        t("Remove key #{number} of {name}: {key}", number=n, name=p.name, key=masked)
        + (f" ({providers.label(label)})" if label else "")
        for n, masked, label in planned.removed
    ]
    lines.append(
        t(
            "{name}: {before} keys → {after}",
            name=p.name,
            before=planned.count,
            after=planned.count - len(planned.removed),
        )
    )
    return Proposal(lines, apply)


def _keys_import(ctx: Context, args: ImportArgs) -> Outcome | Proposal:
    file = Path(args.file).expanduser()
    pairs = config_cli.read_key_file(file)
    preview = config_cli.import_keys(ctx.manager, pairs, dry_run=True)
    if not preview.added:
        ui.note(t("No new keys in {file}; nothing changes.", file=file))
        return Outcome({"added": {}, "already": preview.already, "unknown": preview.unknown})

    def apply() -> Outcome:
        result = config_cli.import_keys(ctx.manager, pairs)
        config_cli.print_imported(ctx.manager, file, result)
        return Outcome(
            {"added": result.added, "already": result.already, "unknown": result.unknown}
        )

    lines = [t("Add the keys in {file} to the configuration:", file=file)]
    lines += [
        tn(n, "{name}: {n} new key", "{name}: {n} new keys", name=name)
        for name, n in sorted(preview.added.items())
    ]
    warnings = []
    if preview.unknown:
        warnings.append(
            t("skipped names no provider uses: {names}", names=", ".join(preview.unknown))
        )
    return Proposal(lines, apply, warnings)


def _config(ctx: Context, args: ConfigArgs) -> Proposal:
    """The wizard takes over the screen, so the person first says whether to open it: a
    question the model misread must not drop them into the language menu."""

    def apply() -> Outcome:
        try:
            if args.section is None:
                wizard.run(ctx.manager)
            else:
                wizard.run_section(ctx.manager, args.section)
        except ConfigError as e:
            raise click.ClickException(
                t("config {error}; run `desman config` again", error=e)
            ) from e
        except (ui.Cancelled, click.Abort):
            ui.out("")
            ui.warn(t("Configuration cancelled; nothing was saved."))
            return Outcome({"saved": False}, done=False)
        return Outcome({"saved": True})

    if args.section == "providers":
        line = t("Open the setup wizard at its model providers: their keys and settings.")
    elif args.section == "permissions":
        line = t("Open the setup wizard at the agents' permissions.")
    elif args.section == "personalize":
        line = t("Open the setup wizard at personalisation: whether the agent keeps notes.")
    elif args.section == "behaviour":
        line = t("Open the setup wizard at how the interlocutor talks.")
    else:
        line = t("Open the whole setup wizard: language, model providers and keys, permissions.")
    return Proposal([line], apply, question=t("Open it?"))


def _config_reset(ctx: Context, args: NoArgs) -> Outcome | Proposal:
    path = ctx.manager.config_path
    if not (path.exists() or path.is_symlink()):
        ui.note(t("No configuration at {path}; nothing to delete.", path=path))
        return Outcome({"deleted": False})

    def apply() -> Outcome:
        config_cli.reset_config(ctx.manager)
        return Outcome({"deleted": True})

    return Proposal([config_cli.reset_question(ctx.manager)], apply)


# ── The catalog ──────────────────────────────────────────────────────────────

_KEYED = [p.id for p in providers.CLOUD + providers.SEARCH if p.key_env is not None]

CATALOG: tuple[Entry, ...] = (
    Entry("help", "List DESMAN's commands for the person.", NoArgs, Kind.READ, _help),
    Entry("status", "Show a task and its subtasks.", TaskArgs, Kind.READ, _status),
    Entry("list", "List tasks, newest first.", ListArgs, Kind.READ, _list),
    Entry("audit", "Show the full history of a task.", TaskArgs, Kind.READ, _audit),
    Entry(
        "inbox",
        "Subtasks waiting for the person's decision (retry, accept or fail), with the reason.",
        NoArgs,
        Kind.READ,
        _inbox,
    ),
    Entry(
        "usage",
        'Model calls and tokens spent per provider and key; since e.g. "24h", "7d", "all".',
        UsageArgs,
        Kind.READ,
        _usage,
    ),
    Entry(
        "run",
        "Start a task for a goal (the same as a task turn). The person confirms.",
        RunArgs,
        Kind.CHANGE,
        _run,
    ),
    Entry(
        "resume",
        "Continue a task where it stopped. The person confirms.",
        TaskArgs,
        Kind.CHANGE,
        _resume,
    ),
    Entry(
        "retry",
        "Send a subtask waiting for the person back to work. The person confirms.",
        DecisionArgs,
        Kind.CHANGE,
        _decision("retry"),
    ),
    Entry(
        "accept",
        "Mark a subtask waiting for the person done as it is. The person confirms.",
        DecisionArgs,
        Kind.CHANGE,
        _decision("accept"),
    ),
    Entry(
        "fail",
        "Give up on a subtask; what depends on it is blocked. The person confirms.",
        FailArgs,
        Kind.CHANGE,
        _decision("fail"),
    ),
    Entry(
        "cancel",
        "Cancel a task: its unfinished subtasks stop. The person confirms.",
        CancelArgs,
        Kind.CHANGE,
        _cancel,
    ),
    Entry(
        "language",
        "Without a code: list the interface languages. With one: switch to it (the person "
        "confirms).",
        LanguageArgs,
        _language_kind,
        _language,
    ),
    Entry(
        "permissions",
        "Without arguments: show what agents may do. With grant/revoke: change it (the person "
        "confirms).",
        PermissionArgs,
        _permissions_kind,
        _permissions,
    ),
    Entry(
        "personalize",
        "Without arguments: show whether the interlocutor studies the conversations and keeps "
        "notes about the person. With on: turn that on or off (the person confirms).",
        PersonalizeArgs,
        _personalize_kind,
        _personalize,
    ),
    Entry(
        "behaviour",
        "Without arguments: show how the interlocutor talks (length of answers, tone, style, "
        "the person's own words). With any of them: change it (the person confirms); auto "
        "gives a part back to the interlocutor's judgement. Holds without personalisation.",
        BehaviourArgs,
        _behaviour_kind,
        _behaviour,
    ),
    Entry(
        "keys list",
        "How many API keys each provider has and what each spent in the last 24 hours.",
        NoArgs,
        Kind.READ,
        _keys_list,
        aliases=("keys",),
    ),
    Entry(
        "keys check",
        "Ask providers whether their keys work and what limits they report (no tokens spent).",
        ProviderArgs,
        Kind.READ,
        _keys_check,
    ),
    Entry(
        "keys add",
        "Add an API key for a provider: the person types it into a hidden field. Never ask "
        f"for the key in the chat. Providers: {', '.join(_KEYED)}.",
        KeyAddArgs,
        Kind.INPUT,
        _keys_add,
    ),
    Entry(
        "keys remove",
        "Remove a provider's key by its number in keys list, or all of them. The person confirms.",
        KeyRemoveArgs,
        Kind.CHANGE,
        _keys_remove,
    ),
    Entry(
        "keys import",
        "Add the keys in a file (JSON or NAME=value lines). The person confirms.",
        ImportArgs,
        Kind.CHANGE,
        _keys_import,
    ),
    Entry(
        "config",
        "Open the setup wizard, or one section of it (providers: keys and settings; "
        "permissions; personalize; behaviour), for the person to fill in; only when they ask to set DESMAN up or to "
        "change settings. To show providers, keys or settings use keys list or config show.",
        ConfigArgs,
        Kind.INPUT,
        _config,
    ),
    Entry(
        "config show",
        "Show the configuration: language, permissions, providers (keys never shown).",
        NoArgs,
        Kind.READ,
        _config_show,
    ),
    Entry(
        "config reset",
        "Delete the configuration file, API keys included. The person confirms.",
        NoArgs,
        Kind.CHANGE,
        _config_reset,
    ),
    Entry(
        "init-db",
        "Create the database or apply pending migrations. The person confirms.",
        NoArgs,
        Kind.CHANGE,
        _init_db,
    ),
)


def _folded(name: str) -> str:
    """`keys_add`, `/keys-add`, `desman keys add` are all `keys add`; `init db` is `init-db`."""
    name = re.sub(r"[\s_\-/]+", " ", name.casefold()).strip()
    return name.removeprefix("desman ").strip()


_BY_NAME = {_folded(n): e for e in CATALOG for n in (e.name, *e.aliases)}


def find(name: str) -> Entry | None:
    return _BY_NAME.get(_folded(name))


def cli_command(root: click.Group, name: str) -> click.Command | None:
    """The CLI command `name` ("keys add") under `root`, or None."""
    command: click.Command | None = root
    for word in name.split():
        if not isinstance(command, click.Group):
            return None
        command = command.commands.get(word)
    return command


def parse(entry: Entry, raw: dict[str, JsonValue], config: Config | None) -> Args:
    """The interlocutor's arguments for `entry`, validated. An argument that looks like an API
    key is refused before anything keeps it: a key is only ever typed into the hidden field."""
    if providers.redact_keys(json.dumps(raw, ensure_ascii=False), config)[1]:
        raise BadArgs(t("an argument looked like an API key, so the command was dropped"))
    try:
        return entry.args.model_validate(raw)
    except ValidationError as e:
        problems = [
            f"{'.'.join(str(part) for part in error['loc']) or '-'}: {error['msg']}"
            for error in e.errors(include_input=False, include_url=False, include_context=False)
        ]
        raise BadArgs("; ".join(problems)) from None


def shown(entry: Entry, args: Args) -> str:
    """The command as the person and the interlocutor see it: name and the arguments given."""
    given = args.model_dump(exclude_defaults=True)
    return entry.name + (" " + json.dumps(given, ensure_ascii=False) if given else "")


def for_model() -> str:
    """The catalog as the interlocutor's prompt lists it."""
    lines = []
    for entry in CATALOG:
        lines.append(f"- {entry.name} {_signature(entry.args)}: {entry.about}")
    return "\n".join(lines)


def _signature(args: type[Args]) -> str:
    schema = args.model_json_schema()
    required = set(schema.get("required", []))
    parts = []
    for name, prop in schema.get("properties", {}).items():
        parts.append(f'"{name}"{"" if name in required else "?"}: {_type(prop)}')
    return "{" + ", ".join(parts) + "}"


def _type(prop: dict[str, Any]) -> str:
    if "enum" in prop:
        return "|".join(json.dumps(v) for v in prop["enum"])
    if "const" in prop:
        return json.dumps(prop["const"])
    if "anyOf" in prop:
        return "|".join(_type(p) for p in prop["anyOf"] if p.get("type") != "null")
    if prop.get("type") == "array":
        return f"[{_type(prop.get('items', {}))}]"
    return str(prop.get("type", "any"))


# ── What the chat records ────────────────────────────────────────────────────


def config_changes(before: Config | None, after: Config | None) -> list[tuple[str, dict[str, Any]]]:
    """(action, details) for the operator's journal: every difference that matters between two
    configs. Keys by provider and fingerprint, settings by name only (a setting may be a
    secret). None stands for no config file."""
    old, new = before or Config(), after or Config()
    changes: list[tuple[str, dict[str, Any]]] = []
    for name in type(old.permissions).model_fields:
        was, now = getattr(old.permissions, name), getattr(new.permissions, name)
        if was != now:
            action = "operator_permission_grant" if now else "operator_permission_revoke"
            changes.append((action, {"permission": name}))
    for env in sorted(set(old.api_keys) | set(new.api_keys)):
        had = {key_id(k.value) for k in old.keys(env)}
        has = {key_id(k.value) for k in new.keys(env)}
        provider = providers.by_key_env(env)
        name = provider.id if provider is not None else env
        changes += [
            ("operator_key_added", {"provider": name, "key_id": k}) for k in sorted(has - had)
        ]
        changes += [
            ("operator_key_removed", {"provider": name, "key_id": k}) for k in sorted(had - has)
        ]
    if old.language != new.language:
        changes.append(("operator_language", {"from": old.language, "to": new.language}))
    for env in sorted(set(old.settings) | set(new.settings)):
        if old.settings.get(env) != new.settings.get(env):
            changes.append(("operator_setting", {"name": env}))
    if old.personalize != new.personalize:
        changes.append(("operator_personalize", {"on": new.personalize}))
    if old.behaviour != new.behaviour:  # the parts, not the person's own words
        parts = [
            p
            for p in Behaviour.model_fields
            if getattr(old.behaviour, p) != getattr(new.behaviour, p)
        ]
        changes.append(("operator_behaviour", {"changed": parts}))
    if before is not None and after is None:
        changes.append(("operator_config_deleted", {}))
    return changes


def for_interlocutor(data: Any, configs: list[Config | None], budget_chars: int) -> str:
    """`data` as JSON text for the interlocutor: cut to `budget_chars`, and with anything that
    looks like a key (or is one of the keys in `configs`) removed."""
    text = json.dumps(data, ensure_ascii=False)
    for config in configs:
        text = providers.redact_keys(text, config)[0]
    return text if len(text) <= budget_chars else text[:budget_chars] + " …(cut)"
