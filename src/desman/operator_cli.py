"""Operator commands shared by the terminal and command catalog."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, TypeVar, get_args

import click
from pydantic import BaseModel, TypeAdapter, ValidationError
from rich.text import Text

from desman import net, ui
from desman.config import ConfigError, ConfigManager, RegistryUrl
from desman.config_cli import load_config
from desman.core import db, netlog
from desman.core.models import Kind
from desman.global_cli import journal
from desman.i18n import t
from desman.net import operator, operator_records, sign
from desman.net import protocol as p

_KEY_HELP = "desman operator key import FILE | desman operator key create"


def current_key() -> sign.NodeKey | None:
    try:
        return operator.load_key()
    except (OSError, sign.KeyFileError) as e:
        raise click.ClickException(t("The operator key cannot be used.")) from e


def fingerprint(key: sign.NodeKey | None) -> str | None:
    return None if key is None else key.fingerprint


def show_status(
    manager: ConfigManager, conn: sqlite3.Connection | None = None, *, for_chat: bool = False
) -> dict[str, object]:
    key = current_key()
    if key is None:
        ui.note(t("No operator key. Use {command}.", command=_KEY_HELP))
        return {"fingerprint": None}
    public = sign.encode_public(key.public)
    ui.out(Text(t("Operator key: {fingerprint}", fingerprint=key.fingerprint)))
    ui.out(Text(public))
    try:
        address = net.operator_registry(manager.load())
    except ConfigError as e:
        raise click.ClickException(str(e)) from e
    if address is None:
        ui.note(t("No registry address. Set it with --registry URL."))
        return {"fingerprint": key.fingerprint, "public_key": public, "registry": None}
    ui.out(Text(t("Registry: {address}", address=address)))
    try:
        identity = call(
            operator.OperatorClient(address, key),
            lambda client: client.whoami(),
        )
    except click.ClickException as e:
        if not for_chat:
            raise
        message = e.format_message()
        ui.error(message)
        return {
            "fingerprint": key.fingerprint,
            "public_key": public,
            "registry": address,
            "error": message,
        }
    ui.out(
        Text(
            t(
                "Operator {fingerprint} assigned ({scopes}) until {date}.",
                fingerprint=identity.fingerprint,
                scopes=", ".join(identity.scopes),
                date=ui.when(identity.until.isoformat()),
            )
        )
    )
    with journal(conn) as opened:
        netlog.record(opened, "operator_status", {"fingerprint": identity.fingerprint})
    return {**identity.model_dump(mode="json"), "public_key": public, "registry": address}


@dataclass(frozen=True)
class KeyChange:
    action: Literal["import", "create", "remove"]
    before: str | None
    key: sign.NodeKey | None
    registry: str | None = None

    @property
    def lines(self) -> list[str]:
        if self.action == "remove":
            if self.before is None:
                return [t("No operator key. Use {command}.", command=_KEY_HELP)]
            return [t("Remove operator key {fingerprint}.", fingerprint=self.before)]
        assert self.key is not None
        lines = [t("Install operator key {fingerprint}.", fingerprint=self.key.fingerprint)]
        if self.before is not None and self.before != self.key.fingerprint:
            lines.append(t("Remove operator key {fingerprint}.", fingerprint=self.before))
        if self.registry is not None:
            lines.append(t("Registry: {address}", address=self.registry))
        return lines

    def apply(self, manager: ConfigManager, conn: sqlite3.Connection | None = None) -> str | None:
        if self.action == "remove" and self.before is None:
            return None
        try:
            with journal(conn) as opened, manager.locked(), db.transaction(opened):
                current = current_key()
                if fingerprint(current) != self.before:
                    raise click.ClickException(
                        t("The operator key changed. Run the command again.")
                    )
                config = manager.load() if self.registry is not None else None
                logged = self.before if self.key is None else self.key.fingerprint
                netlog.write(opened, f"operator_key_{self.action}", {"fingerprint": logged})
                if self.key is None:
                    operator.remove_key()
                else:
                    operator.install_key(self.key, replace=self.before not in (None, logged))
                if config is not None:
                    try:
                        config.network.registry = self.registry
                        manager.save(config)
                    except (ConfigError, OSError):
                        if current is None:
                            operator.remove_key()
                        else:
                            operator.install_key(current, replace=True)
                        raise
        except (OSError, sign.KeyFileError) as e:
            raise click.ClickException(t("The operator key cannot be used.")) from e
        except ConfigError as e:
            raise click.ClickException(str(e)) from e
        return logged


def show_created_key(key: sign.NodeKey) -> dict[str, str]:
    public = sign.encode_public(key.public)
    command = f"desman-registry assign {public}"
    ui.out(Text(t("Operator key: {fingerprint}", fingerprint=key.fingerprint)))
    ui.out(Text(public))
    ui.out(Text(command))
    return {"fingerprint": key.fingerprint, "public_key": public, "assign_command": command}


def prepare_import(source: Path, registry: str | None, *, replace: bool) -> KeyChange:
    try:
        key = operator.read_key(source)
    except (OSError, sign.KeyFileError) as e:
        raise click.ClickException(t("The operator key cannot be used.")) from e
    before = fingerprint(current_key())
    if before not in (None, key.fingerprint) and not replace:
        raise click.ClickException(
            t("An operator key already exists. Use --replace to replace it.")
        )
    return KeyChange("import", before, key, registry)


def prepare_create(registry: str | None) -> KeyChange:
    if current_key() is not None:
        raise click.ClickException(t("An operator key already exists."))
    return KeyChange("create", None, sign.NodeKey.generate(), registry)


def prepare_remove() -> KeyChange:
    return KeyChange("remove", fingerprint(current_key()), None)


def _registry(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return TypeAdapter(RegistryUrl).validate_python(value)
    except ValidationError as e:
        raise click.ClickException(t("Invalid registry URL.")) from e


@click.group("operator")
def operator_group() -> None:
    """Manage access to the registry as an operator."""


@operator_group.command("status")
def status_command() -> None:
    """Show operator access and its term."""
    show_status(load_config())


@operator_group.group("key")
def key_group() -> None:
    """Manage this machine's operator key."""


@key_group.command("import")
@click.argument("file", type=click.Path(path_type=Path))
@click.option("--registry", metavar="URL")
@click.option("--replace", is_flag=True, help="Replace the existing operator key.")
def import_command(file: Path, registry: str | None, replace: bool) -> None:
    """Import an operator key from a PEM file."""
    change = prepare_import(file, _registry(registry), replace=replace)
    value = change.apply(load_config())
    ui.out(Text(t("Operator key: {fingerprint}", fingerprint=value)))


@key_group.command("create")
@click.option("--registry", metavar="URL")
def create_command(registry: str | None) -> None:
    """Create an operator key and show its public key."""
    change = prepare_create(_registry(registry))
    change.apply(load_config())
    assert change.key is not None
    show_created_key(change.key)


@key_group.command("remove")
@click.option("--yes", is_flag=True, help="Confirm removal.")
def remove_command(yes: bool) -> None:
    """Remove this machine's operator key."""
    change = prepare_remove()
    for line in change.lines:
        ui.out(Text(line))
    if yes:
        change.apply(load_config())


M = TypeVar("M", bound=BaseModel)
TASKS = sign.Scope.TASKS
MODERATION = sign.Scope.MODERATION


def connect(manager: ConfigManager) -> operator.OperatorClient:
    key = current_key()
    if key is None:
        raise click.ClickException(t("No operator key. Use {command}.", command=_KEY_HELP))
    try:
        address = net.operator_registry(manager.load())
    except ConfigError as e:
        raise click.ClickException(str(e)) from e
    if address is None:
        raise click.ClickException(t("No registry address. Set it with --registry URL."))
    return operator.OperatorClient(address, key)


def call(
    client: operator.OperatorClient,
    action: Callable[[operator.OperatorClient], M],
    scope: sign.Scope | None = None,
    *,
    show_assignment: bool = True,
) -> M:
    try:
        return action(client)
    except ValidationError as e:
        raise click.ClickException(t("Invalid operator request.")) from e
    except operator.OperatorError as e:
        match e.reason:
            case "not_operator" | "revoked":
                # The command result sent back to the interlocutor contains only the message.
                if show_assignment:
                    ui.out(
                        Text(f"desman-registry assign {sign.encode_public(client.key.public)} …")
                    )
                message = t(
                    "This key has no current operator assignment. Ask the root owner to run "
                    "desman-registry assign with the public key."
                )
            case "scope":
                message = t("The operator assignment needs the {scope} scope.", scope=scope)
            case "already_settled":
                message = t("This item has already been settled.")
            case "cannot_settle":
                message = t("This item is not waiting for a decision.")
            case "not_found":
                message = t("No matching item was found.")
            case "ambiguous":
                message = t("More than one item matches. Give more of the ID.")
            case "malformed" | "invalid_decision" | "replay" | "forged" | "not_yet" | "conflict":
                message = t("Invalid operator request.")
            case "expired":
                message = t("The operator assignment or record has expired.")
            case "bad_reply" | "large_reply":
                message = t("The registry sent an invalid reply.")
            case reason if "Timeout" in reason:
                message = t("The registry request timed out.")
            case _:
                message = t("The registry could not be reached.")
        raise click.ClickException(message) from e


@dataclass(frozen=True)
class Change:
    lines: list[str]
    apply: Callable[[sqlite3.Connection | None], BaseModel]


def _change(
    client: operator.OperatorClient,
    action: Callable[[operator.OperatorClient], M],
    scope: sign.Scope,
    lines: list[str],
    event: str,
    target: Callable[[M], str],
) -> Change:
    def apply(conn: sqlite3.Connection | None) -> M:
        if fingerprint(current_key()) != client.key.fingerprint:
            raise click.ClickException(t("The operator key changed. Run the command again."))
        with journal(conn) as opened:
            result = call(client, action, scope)
            netlog.record(
                opened,
                f"operator_{event}",
                {"fingerprint": client.key.fingerprint, "id": target(result)},
            )
        return result

    return Change([ui.clean(line) for line in lines], apply)


def _inputs(given: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in given:
        name, sep, value = item.partition("=")
        if not sep or name in result:
            raise click.BadParameter(t("Invalid operator request."), param_hint="--input")
        result[name] = value
    return result


def _field(label: str, value: object) -> str:
    return f"{label}: {value}"


def _task_lines(body: p.Job | p.ClusterTask) -> list[str]:
    if isinstance(body, p.Job):
        lines = [
            _field(t("Description"), body.description),
            _field(t("Kind"), body.kind.value),
            _field(t("runs code"), t("yes") if body.runs_code else t("no")),
        ]
    else:
        lines = [_field(t("Goal"), body.goal)]
    return [
        *lines,
        _field(t("Acceptance criteria"), body.acceptance_criteria),
        *(_field(f"{t('Input')} {name}", value) for name, value in body.inputs.items()),
    ]


def _until(client: operator.OperatorClient, scope: sign.Scope) -> datetime:
    identity = call(client, lambda c: c.whoami())
    if scope not in identity.scopes:
        raise click.ClickException(
            t("The operator assignment needs the {scope} scope.", scope=scope)
        )
    if identity.until <= client.clock().replace(microsecond=0):
        raise click.ClickException(t("The operator assignment or record has expired."))
    return identity.until


def job_add(
    client: operator.OperatorClient,
    description: str,
    acceptance: str,
    kind: Kind,
    runs_code: bool,
    inputs: Iterable[str],
    days: int,
) -> Change:
    body = p.Job(
        description=description,
        acceptance_criteria=acceptance,
        kind=kind,
        runs_code=runs_code,
        inputs=_inputs(inputs),
    )
    until = _until(client, TASKS)
    record = operator_records.job(client.key, body, now=client.clock(), days=days, until=until)
    return _change(
        client,
        lambda c: c.job_add(record),
        TASKS,
        [t("Offer a job to nodes."), *_task_lines(body), _term(record)],
        "job_add",
        lambda reply: reply.job,
    )


def cluster_add(
    client: operator.OperatorClient,
    goal: str,
    acceptance: str,
    inputs: Iterable[str],
    days: int,
) -> Change:
    body = p.ClusterTask(goal=goal, acceptance_criteria=acceptance, inputs=_inputs(inputs))
    until = _until(client, TASKS)
    record = operator_records.cluster_task(
        client.key, body, now=client.clock(), days=days, until=until
    )
    return _change(
        client,
        lambda c: c.cluster_add(record),
        TASKS,
        [t("Offer a cluster task to nodes."), *_task_lines(body), _term(record)],
        "cluster_add",
        lambda reply: reply.task,
    )


def _term(record: sign.Signed) -> str:
    # The same expiration is displayed and sent, even if confirmation takes time.
    return _field(t("Until"), record.record.expires_at.astimezone().isoformat(timespec="seconds"))


def job_settle(
    client: operator.OperatorClient,
    id: str,
    decision: p.OperatorDecision,
) -> Change:
    # Resolve a prefix before the confirmation so it cannot later select a different job.
    found = call(client, lambda c: c.job(id))
    return _change(
        client,
        lambda c: c.job_settle(found.job, decision),
        TASKS,
        [_field(t("Job"), found.job), _field(t("Decision"), decision)],
        "job_settle",
        lambda reply: reply.job,
    )


def cluster_settle(
    client: operator.OperatorClient,
    id: str,
    decision: p.OperatorDecision,
    note: str | None,
) -> Change:
    found = call(client, lambda c: c.cluster(id))
    request = p.OperatorClusterSettle(id=found.task, decision=decision, note=note)
    lines = [_field(t("Cluster task"), found.task), _field(t("Decision"), decision)]
    if note is not None:
        lines.append(_field(t("Note"), note))
    return _change(
        client,
        lambda c: c.cluster_settle(request.id, request.decision, request.note),
        TASKS,
        lines,
        "cluster_settle",
        lambda reply: reply.task,
    )


def complaint_dismiss(
    client: operator.OperatorClient,
    id: str,
) -> Change:
    rows = call(client, lambda c: c.complaints(all=True), MODERATION)
    target = _resolve(id, (row.id for row in rows.complaints))
    return _change(
        client,
        lambda c: c.complaint_dismiss(target),
        MODERATION,
        [t("Dismiss complaint {id}.", id=target)],
        "complaint_dismiss",
        lambda reply: reply.id,
    )


def _resolve(ref: str, ids: Iterable[str]) -> str:
    ref = TypeAdapter(p.OperatorPrefix).validate_python(ref)
    matches = [value for value in ids if value.startswith(ref)]
    if not matches:
        raise click.ClickException(t("No matching item was found."))
    if len(matches) > 1:
        raise click.ClickException(t("More than one item matches. Give more of the ID."))
    return matches[0]


def moderate(
    client: operator.OperatorClient,
    action: Literal["suspend", "lift"],
    node: str,
    reason: str,
    days: int = p.MODERATION_DAYS,
    complaints: Iterable[str] = (),
) -> Change:
    until = _until(client, MODERATION)
    nodes = call(client, lambda c: c.nodes())
    node = _resolve(node, (row.node for row in nodes.nodes))
    now = client.clock()
    lifts = None
    lifetime = timedelta(days=days)
    refs = tuple(complaints)
    ids = []
    if refs:
        rows = call(client, lambda c: c.complaints(all=True), MODERATION)
        ids = [_resolve(ref, (row.id for row in rows.complaints)) for ref in refs]
    if action == "lift":
        suspensions = call(client, lambda c: c.suspensions(), MODERATION)
        active = next(
            (
                row
                for row in suspensions.suspensions
                if row.node == node
                and row.action == "suspend"
                and not row.lifted
                and datetime.fromisoformat(row.expires_at) > now
            ),
            None,
        )
        if active is None:
            raise click.ClickException(t("No suspension is in effect for {id}.", id=node))
        lifts = active.id
        lifetime = datetime.fromisoformat(active.expires_at) - now.replace(microsecond=0)
    body = p.Moderation(node=node, action=action, reason=reason, complaints=ids, lifts=lifts)
    record = operator_records.moderation(client.key, body, now=now, lifetime=lifetime, until=until)
    lines = [
        _field(t("Action"), action),
        _field(t("Node"), node),
        _field(t("Reason"), body.reason),
        _term(record),
    ]
    if lifts is not None:
        lines.append(_field(t("ID"), lifts))
    lines.extend(_field(t("Complaint"), id) for id in ids)
    return _change(
        client,
        lambda c: c.moderation(record),
        MODERATION,
        lines,
        action,
        lambda reply: reply.id,
    )


def owner(
    client: operator.OperatorClient,
    node: str,
    name: str | None,
    clear: bool,
) -> Change:
    if (name is None) == (not clear):
        raise click.UsageError(t("Give a NAME or --clear, not both."))
    if name is not None and not 1 <= len(name) <= 64:
        raise click.BadParameter(t("At most 64 characters."), param_hint="NAME")
    nodes = call(client, lambda c: c.nodes())
    target = _resolve(node, (row.node for row in nodes.nodes))
    return _change(
        client,
        lambda c: c.owner(target, name),
        TASKS,
        [t("Node {key} is its own owner.", key=target)]
        if clear
        else [t("Node {key} belongs to {owner}.", key=target, owner=name)],
        "owner",
        lambda reply: reply.node,
    )


def prepare(
    client: operator.OperatorClient, build: Callable[[operator.OperatorClient], Change]
) -> Change:
    try:
        return build(client)
    except ValidationError as e:
        raise click.ClickException(t("Invalid operator request.")) from e
    except operator_records.AssignmentEnded as e:
        raise click.ClickException(t("The operator assignment or record has expired.")) from e


def _lines(*lines: str) -> None:
    for line in lines:
        ui.out(Text(ui.clean(line)))


def _table(columns: tuple[str, ...], rows: Iterable[tuple[str, ...]]) -> None:
    table = ui.table(*columns)
    for row in rows:
        table.add_row(*(Text(ui.clean(cell)) for cell in row))
    ui.out(table)


def _status(status: p.JobStatus | p.ClusterStatus) -> str:
    return {
        "execute": t("to do"),
        "recheck": t("to recheck"),
        "waiting": t("waits for a cluster"),
        "running": t("a cluster works on it"),
        "operator": t("waits for the operator"),
        "accepted": t("accepted"),
        "dropped": t("dropped"),
        "expired": t("expired"),
    }[status]


def _job_reason(reason: str | None) -> str:
    return {
        "disputed": t("the recheck rejected the result"),
        "failed": t("nodes could not do it"),
        "leases_lost": t("nodes took it and fell silent"),
    }.get(reason or "", reason or "—")


def _cluster_reason(reason: str | None) -> str:
    return {
        "result": t("its result came"),
        "job_given_up": t("a job gave up"),
        "operators_lost": t("its operators were lost"),
    }.get(reason or "", reason or "—")


def _node_status(status: str) -> str:
    return {"active": t("taking part"), "left": t("has left"), "revoked": t("revoked")}[status]


def _print_jobs(rows: list[p.OperatorJob]) -> None:
    _table(
        (t("Job"), t("Status"), t("Description"), t("Until")),
        ((r.job, _status(r.status), r.task.description, ui.when(r.expires_at)) for r in rows),
    )


def _print_clusters(rows: list[p.OperatorCluster]) -> None:
    _table(
        (t("Cluster task"), t("Status"), t("Goal"), t("Members"), t("Accepted"), t("Until")),
        (
            (
                r.task,
                _status(r.status),
                r.goal.goal,
                str(r.members),
                f"{r.accepted}/{r.jobs}",
                ui.when(r.expires_at),
            )
            for r in rows
        ),
    )


def _print_job(reply: p.OperatorJobDetail) -> None:
    _lines(
        _field(t("Job"), reply.job),
        _field(t("Status"), _status(reply.status)),
        _field(t("Reason"), _job_reason(reply.reason)),
        _field(t("Until"), ui.when(reply.expires_at)),
        *_task_lines(reply.task),
    )
    if not reply.handed_in:
        ui.note(t("Nothing handed in yet."))
    for item in reply.handed_in:
        _lines(
            f"{ui.when(item.received_at)}  {item.kind}  {item.outcome}  {item.node}",
            str(item.body.get("output") or item.body.get("reason") or ""),
        )
        for file in item.body.get("files") or []:
            _lines(f"{file['path']} ({t('characters')}: {len(str(file.get('text', '')))})")


def _print_cluster(reply: p.OperatorClusterDetail) -> None:
    _lines(
        _field(t("Cluster task"), reply.task),
        _field(t("Cluster"), reply.cluster),
        _field(t("Status"), _status(reply.status)),
        _field(t("Reason"), _cluster_reason(reply.why)),
        _field(t("Until"), ui.when(reply.expires_at)),
        *_task_lines(reply.goal),
    )
    if reply.operator is not None:
        _lines(
            _field(t("Operator"), reply.operator),
            _field(t("Members"), ", ".join(reply.members_list)),
        )
    if reply.note:
        _lines(_field(t("Note"), reply.note))
    if reply.plan:
        _lines(_field(t("Plan"), reply.plan.version))
        for index, node in enumerate(reply.plan.nodes):
            _lines(f"{index}. {node.description} ← {', '.join(map(str, node.depends_on))}")
    else:
        ui.note(t("No plan yet."))
    if reply.job_list:
        _print_jobs(reply.job_list)
    if reply.result is not None:
        _lines(_field(t("Result"), reply.result.summary))


def _print_complaints(rows: Iterable[p.ComplaintRow]) -> None:
    _table(
        (t("ID"), t("Created"), t("From"), t("About"), t("Count"), t("Status")),
        (
            (
                r.id,
                ui.when(r.created_at),
                r.from_name or r.from_node,
                r.about_name or r.about_node,
                str(r.count),
                r.status,
            )
            for r in rows
        ),
    )


def print_reply(reply: BaseModel, *, as_json: bool = False) -> None:
    if as_json:
        encoded = json.dumps(reply.model_dump(mode="json"), ensure_ascii=False)
        if ui.is_terminal_front():
            click.echo(encoded)
        else:
            ui.out(Text(encoded))
        return
    match reply:
        case p.OperatorInbox():
            if not reply.jobs and not reply.clusters:
                ui.note(t("Nothing waits for the operator."))
            if reply.jobs:
                _print_jobs(reply.jobs)
            if reply.clusters:
                _print_clusters(reply.clusters)
        case p.OperatorJobs():
            _print_jobs(reply.jobs)
        case p.OperatorClusters():
            _print_clusters(reply.clusters)
        case p.OperatorJobDetail():
            _print_job(reply)
        case p.OperatorClusterDetail():
            _print_cluster(reply)
        case p.OperatorComplaints():
            _print_complaints(reply.complaints)
        case p.ComplaintDetail():
            _print_complaints([reply])
            _lines(t("Quoted complaint:"), reply.text)
            if reply.message_text is not None:
                _lines(t("Quoted message:"), reply.message_text)
            for row in reply.decisions:
                _lines(f"{ui.when(row.decided_at)}  {row.decision}  {row.record or '—'}")
                if row.note:
                    _lines(row.note)
        case p.OperatorSuspensions():
            _table(
                (t("ID"), t("Action"), t("Node"), t("Until"), t("Lifted")),
                (
                    (
                        r.id,
                        r.action,
                        r.node,
                        ui.when(r.expires_at),
                        t("yes") if r.lifted else t("no"),
                    )
                    for r in reply.suspensions
                ),
            )
        case p.OperatorReports():
            _table(
                (t("ID"), t("Kind"), t("Node"), t("Created")),
                ((r.id, r.kind, r.node, ui.when(r.created_at)) for r in reply.reports),
            )
        case p.OperatorReportDetail():
            _lines(
                _field(t("ID"), reply.id),
                _field(t("Node"), reply.node),
                _field(t("Kind"), reply.kind),
                _field(t("Created"), ui.when(reply.created_at)),
                json.dumps(
                    reply.report.report.model_dump(mode="json"), ensure_ascii=False, indent=2
                ),
            )
        case p.OperatorNodes():
            _table(
                (t("Node"), t("Owner"), t("Status"), t("Lends"), t("Last seen")),
                (
                    (
                        r.node,
                        r.owner or "—",
                        _node_status(r.status),
                        ", ".join(
                            c.provider if c.model is None else f"{c.provider}/{c.model}"
                            for c in r.capabilities
                        )
                        or "—",
                        ui.when(r.last_seen),
                    )
                    for r in reply.nodes
                ),
            )
        case p.OperatorLog():
            _table(
                (t("Time"), t("Action"), t("Node"), t("Details")),
                (
                    (
                        ui.when(r.ts),
                        r.action,
                        r.node or "—",
                        json.dumps(r.details, ensure_ascii=False, sort_keys=True),
                    )
                    for r in reply.entries
                ),
            )


def print_change(reply: BaseModel) -> dict[str, str]:
    match reply:
        case p.OperatorJob():
            ui.ok(t("Job {id} is now: {status}", id=reply.job, status=_status(reply.status)))
            return {"id": reply.job, "status": reply.status}
        case p.OperatorCluster():
            ui.ok(
                t("Cluster task {id} is now: {status}", id=reply.task, status=_status(reply.status))
            )
            return {"id": reply.task, "status": reply.status}
        case p.ModerationRow():
            ui.ok(t("Moderation recorded as {id}.", id=reply.id))
            return {"id": reply.id}
        case p.OperatorRef():
            ui.ok(t("Complaint dismissed."))
            return {"id": reply.id}
        case p.OperatorNode():
            if reply.owner is None:
                ui.ok(t("Node {key} is its own owner.", key=reply.node))
            else:
                ui.ok(t("Node {key} belongs to {owner}.", key=reply.node, owner=reply.owner))
            return {"id": reply.node}
    raise AssertionError(type(reply))


def _read_cli(
    action: Callable[[operator.OperatorClient], BaseModel],
    as_json: bool,
    scope: sign.Scope | None = None,
) -> None:
    print_reply(call(connect(load_config()), action, scope), as_json=as_json)


def _change_cli(build: Callable[[operator.OperatorClient], Change]) -> None:
    change = prepare(connect(load_config()), build)
    print_change(change.apply(None))


@operator_group.group("job")
def job_group() -> None:
    """The operator's jobs: add, show, settle."""


@operator_group.group("cluster")
def cluster_group() -> None:
    """A goal for a cluster of nodes: DESMAN appoints its operator (ADR 0006)."""


@operator_group.group("complaint")
def complaint_group() -> None:
    """Inspect and settle messenger complaints."""


@operator_group.group("report")
def report_group() -> None:
    """Inspect reports submitted by network nodes."""


@operator_group.command("inbox")
@click.option("--json", "as_json", is_flag=True)
def inbox_command(as_json: bool) -> None:
    """Jobs and cluster tasks waiting for a decision."""
    _read_cli(lambda c: c.inbox(), as_json)


@operator_group.command("jobs")
@click.option("--status", type=click.Choice(get_args(p.JobStatus)))
@click.option("--json", "as_json", is_flag=True)
def jobs_command(status: p.JobStatus | None, as_json: bool) -> None:
    """List jobs with their status."""
    _read_cli(lambda c: c.jobs(status), as_json)


@job_group.command("show")
@click.argument("id")
@click.option("--json", "as_json", is_flag=True)
def job_show_command(id: str, as_json: bool) -> None:
    """A job, and every result and verdict nodes handed in for it."""
    _read_cli(lambda c: c.job(id), as_json)


@operator_group.command("clusters")
@click.option("--status", type=click.Choice(get_args(p.ClusterStatus)))
@click.option("--json", "as_json", is_flag=True)
def clusters_command(status: p.ClusterStatus | None, as_json: bool) -> None:
    """List cluster tasks with their status."""
    _read_cli(lambda c: c.clusters(status), as_json)


@cluster_group.command("show")
@click.argument("id")
@click.option("--json", "as_json", is_flag=True)
def cluster_show_command(id: str, as_json: bool) -> None:
    """A cluster task: its cluster, its plan, its jobs and where each is."""
    _read_cli(lambda c: c.cluster(id), as_json)


@operator_group.command("complaints")
@click.option("--all", "include_settled", is_flag=True, help="Include settled complaints.")
@click.option("--json", "as_json", is_flag=True)
def complaints_command(include_settled: bool, as_json: bool) -> None:
    """List messenger complaints."""
    _read_cli(lambda c: c.complaints(all=include_settled), as_json, MODERATION)


@complaint_group.command("show")
@click.argument("id")
@click.option("--json", "as_json", is_flag=True)
def complaint_show_command(id: str, as_json: bool) -> None:
    """Show a complaint and its quoted message."""
    _read_cli(lambda c: c.complaint(id), as_json, MODERATION)


@operator_group.command("suspensions")
@click.option("--json", "as_json", is_flag=True)
def suspensions_command(as_json: bool) -> None:
    """List signed moderation records."""
    _read_cli(lambda c: c.suspensions(), as_json, MODERATION)


@operator_group.command("reports")
@click.option("--kind", type=click.Choice(get_args(p.ReportKind)))
@click.option("--json", "as_json", is_flag=True)
def reports_command(kind: p.ReportKind | None, as_json: bool) -> None:
    """List bug reports and shared memory."""
    _read_cli(lambda c: c.reports(kind), as_json)


@report_group.command("show")
@click.argument("id")
@click.option("--json", "as_json", is_flag=True)
def report_show_command(id: str, as_json: bool) -> None:
    """Show the exact submitted report."""
    _read_cli(lambda c: c.report(id), as_json)


@operator_group.command("nodes")
@click.option("--json", "as_json", is_flag=True)
def nodes_command(as_json: bool) -> None:
    """The nodes that registered: status, what they lend, when seen last."""
    _read_cli(lambda c: c.nodes(), as_json)


@operator_group.command("log")
@click.option(
    "--limit",
    default=100,
    show_default=True,
    type=click.IntRange(1, 500),
    help="How many of the latest entries.",
)
@click.option("--json", "as_json", is_flag=True)
def log_command(limit: int, as_json: bool) -> None:
    """The registry's journal: registrations, departures, assignments, revocations, refusals."""
    _read_cli(lambda c: c.log(limit), as_json)


@job_group.command("add")
@click.option("--description", required=True, help="What to do (at most 4000 characters).")
@click.option(
    "--acceptance", required=True, help="How the result is judged (at most 2000 characters)."
)
@click.option(
    "--kind",
    type=click.Choice([kind.value for kind in Kind]),
    default=Kind.TEXT.value,
    show_default=True,
    help="What sort of work it is.",
)
@click.option("--runs-code", is_flag=True, help="Done only by running code in the sandbox.")
@click.option("--input", "inputs", multiple=True, metavar="NAME=TEXT")
@click.option("--days", default=7, show_default=True, type=click.IntRange(1, 90))
def job_add_command(
    description: str,
    acceptance: str,
    kind: str,
    runs_code: bool,
    inputs: tuple[str, ...],
    days: int,
) -> None:
    """Sign a job with the operator's key and offer it to nodes."""
    _change_cli(lambda c: job_add(c, description, acceptance, Kind(kind), runs_code, inputs, days))


@job_group.command("settle")
@click.argument("id")
@click.argument("decision", type=click.Choice(get_args(p.OperatorDecision)))
def job_settle_command(id: str, decision: p.OperatorDecision) -> None:
    """Decide a job: accept its result, lend it again from the start, or drop it."""
    _change_cli(lambda c: job_settle(c, id, decision))


@cluster_group.command("add")
@click.option("--goal", required=True, help="What the cluster is to do (at most 4000 characters).")
@click.option(
    "--acceptance", required=True, help="How the result is judged (at most 2000 characters)."
)
@click.option("--input", "inputs", multiple=True, metavar="NAME=TEXT")
@click.option(
    "--days", default=7, show_default=True, type=click.IntRange(1, operator_records.CLUSTER_DAYS)
)
def cluster_add_command(goal: str, acceptance: str, inputs: tuple[str, ...], days: int) -> None:
    """Sign a goal with the operator's key; a cluster is formed when enough nodes are free."""
    _change_cli(lambda c: cluster_add(c, goal, acceptance, inputs, days))


@cluster_group.command("settle")
@click.argument("id")
@click.argument("decision", type=click.Choice(get_args(p.OperatorDecision)))
@click.option("--note", help="What to tell the cluster (with again).")
def cluster_settle_command(id: str, decision: p.OperatorDecision, note: str | None) -> None:
    """Decide a cluster task: accept its result, send it back, or drop it."""
    _change_cli(lambda c: cluster_settle(c, id, decision, note))


@complaint_group.command("dismiss")
@click.argument("id")
def complaint_dismiss_command(id: str) -> None:
    """Dismiss a messenger complaint."""
    _change_cli(lambda c: complaint_dismiss(c, id))


@operator_group.command("suspend")
@click.argument("node")
@click.option("--reason", required=True)
@click.option(
    "--days",
    default=p.MODERATION_DAYS,
    show_default=True,
    type=click.IntRange(1, p.MAX_MODERATION_DAYS),
)
@click.option("--complaint", "complaints", multiple=True)
def suspend_command(node: str, reason: str, days: int, complaints: tuple[str, ...]) -> None:
    """Sign a suspension of a node's messaging."""
    _change_cli(lambda c: moderate(c, "suspend", node, reason, days, complaints))


@operator_group.command("lift")
@click.argument("node")
@click.option("--reason", required=True)
def lift_command(node: str, reason: str) -> None:
    """Sign the lifting of a messaging suspension."""
    _change_cli(lambda c: moderate(c, "lift", node, reason))


@operator_group.command("owner")
@click.argument("node")
@click.argument("name", required=False)
@click.option("--clear", is_flag=True, help="The node is its own owner again.")
def owner_command(node: str, name: str | None, clear: bool) -> None:
    """Say whose node NODE is: nodes of one owner never recheck each other's work."""
    _change_cli(lambda c: owner(c, node, name, clear))
