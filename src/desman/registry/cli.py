"""`desman-registry`: the registry on the owner's server (ADR 0004; ROADMAP G.4, G.6, G.7b).

`serve` answers nodes; `assign` and `revoke` sign records with the root key; `backup` copies
the database; `nodes` and `log` show what the registry knows and did. The operator's work
(G.6): `job add` signs a job with the operator's key, `jobs`, `job show` and `inbox` show them,
`job settle` decides one that waits for a person, `owner` says whose node is whose (a result is
rechecked by another owner's node). `trust sign` signs the trust list with the root (G.6c):
without one in its term nodes take no work, so sign it again before it ends (a timer). Non-interactive, like every subcommand. It reads no `config.json`: its directory is `--dir` (or
$DESMAN_REGISTRY_DIR), its language $DESMAN_REGISTRY_LANGUAGE.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import click
from pydantic import ValidationError
from rich.text import Text

from desman import i18n, ui
from desman.core.models import Kind
from desman.i18n import t
from desman.net import protocol, sign
from desman.registry import admin, checks, releases, server, store, work

LANGUAGE_ENV: Final = "DESMAN_REGISTRY_LANGUAGE"
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


class RegistryGroup(click.Group):
    """Switches to $DESMAN_REGISTRY_LANGUAGE before Click reads the arguments."""

    def main(self, *args: Any, **kwargs: Any) -> Any:
        code = os.environ.get(LANGUAGE_ENV, "en")
        i18n.set_language(code if code in i18n.CATALOGS else "en")
        i18n.translate_commands(self)
        return super().main(*args, **kwargs)


def _now() -> datetime:
    return datetime.now(UTC)


def _dir(ctx: click.Context) -> Path:
    found: Path = ctx.obj
    return found


def _fail(e: admin.AdminError) -> click.ClickException:
    return click.ClickException(str(e))


@click.group(cls=RegistryGroup)
@click.option(
    "--dir",
    "directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="The registry's directory: database and keys (default: $DESMAN_REGISTRY_DIR or ~/.local/share/desman-registry).",
)
@click.pass_context
def cli(ctx: click.Context, directory: Path | None) -> None:
    """desman-registry: the registry of desman-global, run on the owner's server."""
    ctx.obj = store.default_dir() if directory is None else directory


@cli.command()
@click.option(
    "--host",
    default=server.DEFAULT_HOST,
    show_default=True,
    help="Address to listen on; keep 127.0.0.1 behind the reverse proxy that gives TLS.",
)
@click.option(
    "--port",
    default=server.DEFAULT_PORT,
    show_default=True,
    type=click.IntRange(0, 65535),
    help="Port to listen on.",
)
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """Answer nodes until stopped (plain HTTP: TLS is the reverse proxy's)."""
    path = _dir(ctx) / store.DB_NAME
    store.open_registry(path).close()
    httpd = server.make_server(server.Registry(path), host, port)
    address = f"http://{host}:{httpd.server_port}"
    ui.ok(t("desman-registry listens on {address} (Ctrl+C stops it).", address=address))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


@cli.command()
@click.argument("dest", type=click.Path(dir_okay=False, path_type=Path))
@click.pass_context
def backup(ctx: click.Context, dest: Path) -> None:
    """Copy the database to DEST while the server runs."""
    try:
        admin.backup(_dir(ctx) / store.DB_NAME, dest)
    except admin.AdminError as e:
        raise _fail(e) from e
    ui.ok(t("Backup written: {path}", path=str(dest)))


def _root_option(command: Any) -> Any:
    return click.option(
        "--root-key",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="The root's private key (default: root.key in the registry's directory).",
    )(command)


def _open(ctx: click.Context, root_key: Path | None) -> tuple[Any, sign.NodeKey]:
    directory = _dir(ctx)
    root = admin.load_root_key(directory / store.ROOT_KEY if root_key is None else root_key)
    return store.open_registry(directory / store.DB_NAME), root


@cli.command()
@click.argument(
    "public_key", required=False, type=click.Path(dir_okay=False, exists=True, path_type=Path)
)
@click.option(
    "--scope",
    "scopes",
    multiple=True,
    type=click.Choice([s.value for s in sign.Scope]),
    help="What the operator may sign; repeat for several (default: all).",
)
@click.option(
    "--days",
    default=admin.ASSIGNMENT_DAYS,
    show_default=True,
    type=click.IntRange(min=1),
    help="How long the assignment lasts.",
)
@_root_option
@click.pass_context
def assign(
    ctx: click.Context,
    public_key: Path | None,
    scopes: tuple[str, ...],
    days: int,
    root_key: Path | None,
) -> None:
    """Assign an operator, signed by the root: the key in PUBLIC_KEY (a PEM file), or the
    registry's own operator key (created on first use)."""
    try:
        conn, root = _open(ctx, root_key)
        try:
            operator = (
                admin.operator_key(_dir(ctx) / store.OPERATOR_KEY)
                if public_key is None
                else admin.public_key(public_key)
            )
            wanted = {sign.Scope(s) for s in scopes} or set(sign.Scope)
            done = admin.assign(conn, root, operator, wanted, days=days, now=_now())
        finally:
            conn.close()
    except admin.AdminError as e:
        raise _fail(e) from e
    ui.ok(
        t(
            "Operator {fingerprint} assigned ({scopes}) until {date}.",
            fingerprint=done.operator,
            scopes=", ".join(done.scopes),
            date=done.expires_at.date().isoformat(),
        )
    )


@cli.command()
@click.argument("fingerprint")
@click.option("--reason", default="", help="Why, kept in the record (at most 500 characters).")
@_root_option
@click.pass_context
def revoke(ctx: click.Context, fingerprint: str, reason: str, root_key: Path | None) -> None:
    """Revoke a key (a node's or an operator's), signed by the root."""
    if not _FINGERPRINT.match(fingerprint):
        raise click.BadParameter(
            t("Not a key's fingerprint (64 hex digits): {key}", key=fingerprint),
            param_hint="FINGERPRINT",
        )
    if len(reason) > 500:
        raise click.BadParameter(
            t("The reason is longer than 500 characters."), param_hint="--reason"
        )
    try:
        conn, root = _open(ctx, root_key)
        try:
            admin.revoke(conn, root, fingerprint, reason, now=_now())
        finally:
            conn.close()
    except admin.AdminError as e:
        raise _fail(e) from e
    ui.ok(
        t("Revoked {fingerprint}: nodes refuse it from their next poll.", fingerprint=fingerprint)
    )


def _json_option(command: Any) -> Any:
    return click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")(command)


def _status(found: store.Node) -> str:
    if found.revoked:
        return t("revoked")
    return t("taking part") if found.status == "active" else t("has left")


@cli.command()
@_json_option
@click.pass_context
def nodes(ctx: click.Context, as_json: bool) -> None:
    """The nodes that registered: status, what they lend, when seen last."""
    conn = store.open_registry(_dir(ctx) / store.DB_NAME)
    try:
        found = store.nodes(conn)
    finally:
        conn.close()
    if as_json:
        click.echo(json.dumps([n.as_json() for n in found], ensure_ascii=False, indent=2))
        return
    if not found:
        ui.note(t("No node has registered yet."))
        return
    table = ui.table(t("Node"), t("Owner"), t("Status"), t("Lends"), t("Last seen"))
    for n in found:
        lends = ", ".join(
            c.provider if c.model is None else f"{c.provider}/{c.model}" for c in n.capabilities
        )
        table.add_row(n.fingerprint[:16], n.owner or "—", _status(n), lends or "—", n.last_seen)
    ui.out(table)


@cli.command("log")
@click.option(
    "--limit",
    default=50,
    show_default=True,
    type=click.IntRange(min=1),
    help="How many of the latest entries.",
)
@_json_option
@click.pass_context
def log_command(ctx: click.Context, limit: int, as_json: bool) -> None:
    """The registry's journal: registrations, departures, assignments, revocations, refusals."""
    conn = store.open_registry(_dir(ctx) / store.DB_NAME)
    try:
        found = store.entries(conn, limit)
    finally:
        conn.close()
    if as_json:
        click.echo(
            json.dumps(
                [
                    {"seq": e.seq, "ts": e.ts, "action": e.action, "node": e.node, **e.details}
                    for e in found
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not found:
        ui.note(t("The journal is empty."))
        return
    table = ui.table(t("Time"), t("Action"), t("Node"), t("Details"))
    for e in found:
        details = json.dumps(e.details, ensure_ascii=False, sort_keys=True)
        table.add_row(e.ts, e.action, (e.node or "—")[:16], details)
    ui.out(table)


# ── Work (G.6) ───────────────────────────────────────────────────────────────


def _job_status(found: work.JobRow) -> str:
    names = {
        "execute": t("to do"),
        "recheck": t("to recheck"),
        "accepted": t("accepted"),
        "operator": t("waits for the operator"),
        "dropped": t("dropped"),
        "expired": t("expired"),
    }
    return names.get(found.status, found.status)


def _job_reason(found: work.JobRow) -> str:
    names = {
        "disputed": t("the recheck rejected the result"),
        "failed": t("nodes could not do it"),
        "leases_lost": t("nodes took it and fell silent"),
    }
    return "" if found.reason is None else names.get(found.reason, found.reason)


def _open_db(ctx: click.Context) -> Any:
    return store.open_registry(_dir(ctx) / store.DB_NAME)


@cli.group()
def job() -> None:
    """The operator's jobs: add, show, settle."""


@job.command("add")
@click.option("--description", required=True, help="What to do (at most 4000 characters).")
@click.option(
    "--acceptance",
    required=True,
    help="How the result is judged (at most 2000 characters).",
)
@click.option(
    "--kind",
    type=click.Choice([k.value for k in Kind]),
    default=Kind.TEXT.value,
    show_default=True,
    help="What sort of work it is.",
)
@click.option("--runs-code", is_flag=True, help="Done only by running code in the sandbox.")
@click.option(
    "--input",
    "inputs",
    multiple=True,
    metavar="NAME=TEXT",
    help="Material the job works on; repeat for several.",
)
@click.option(
    "--days",
    default=admin.JOB_DAYS,
    show_default=True,
    type=click.IntRange(1, 90),
    help="How long nodes may take it.",
)
@click.option(
    "--operator-key",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="The operator's private key (default: operator.key in the registry's directory).",
)
@click.pass_context
def job_add(
    ctx: click.Context,
    description: str,
    acceptance: str,
    kind: str,
    runs_code: bool,
    inputs: tuple[str, ...],
    days: int,
    operator_key: Path | None,
) -> None:
    """Sign a job with the operator's key and offer it to nodes."""
    given: dict[str, str] = {}
    for item in inputs:
        name, sep, text = item.partition("=")
        if not sep or name in given:
            raise click.BadParameter(
                t("Expected NAME=TEXT, each name once: {value}", value=item[:80]),
                param_hint="--input",
            )
        given[name] = text
    try:
        spec = protocol.Job(
            description=description,
            acceptance_criteria=acceptance,
            inputs=given,
            kind=Kind(kind),
            runs_code=runs_code,
        )
    except ValidationError as e:
        raise click.ClickException(
            t("The job is not valid: {error}", error=f"{e.error_count()} problem(s)")
        ) from e
    directory = _dir(ctx)
    try:
        key = admin.load_operator_key(
            directory / store.OPERATOR_KEY if operator_key is None else operator_key
        )
        conn = _open_db(ctx)
        try:
            added = admin.add_job(conn, key, spec, days=days, now=_now())
        finally:
            conn.close()
    except admin.AdminError as e:
        raise _fail(e) from e
    ui.ok(
        t(
            "Job {id} offered to nodes until {date}.",
            id=added.id,
            date=added.signed.record.expires_at.date().isoformat(),
        )
    )


def _jobs_table(found: list[work.JobRow]) -> None:
    table = ui.table(t("Job"), t("Status"), t("Kind"), t("Description"), t("Added"))
    for j in found:
        status = _job_status(j)
        if j.reason:
            status += f" ({_job_reason(j)})"
        spec = j.job
        table.add_row(j.id[:8], status, spec.kind.value, Text(spec.description[:60]), j.created_at)
    ui.out(table)


@cli.command()
@click.option(
    "--status",
    type=click.Choice(["execute", "recheck", "accepted", "operator", "dropped", "expired"]),
    default=None,
    help="Only jobs in this state.",
)
@_json_option
@click.pass_context
def jobs(ctx: click.Context, status: str | None, as_json: bool) -> None:
    """The operator's jobs and where each one is."""
    conn = _open_db(ctx)
    try:
        work.expire(conn, _now())  # leases nodes left to lapse, even with no node calling
        found = work.jobs(conn, status)
    finally:
        conn.close()
    if as_json:
        click.echo(json.dumps([j.as_json() for j in found], ensure_ascii=False, indent=2))
        return
    if not found:
        ui.note(t("No jobs."))
        return
    _jobs_table(found)


@cli.group()
def proposal() -> None:
    """The Improvement Analyst's proposals submitted by network nodes."""


@proposal.command("list")
@click.option(
    "--status",
    type=click.Choice(["submitted", "reviewing", "reviewed", "accepted", "rejected"]),
    default=None,
    help="Only proposals in this state.",
)
@_json_option
@click.pass_context
def proposal_list(ctx: click.Context, status: str | None, as_json: bool) -> None:
    """List proposal submissions for operator review."""
    conn = _open_db(ctx)
    try:
        work.expire(conn, _now())
        found = store.proposals(conn, status)
    finally:
        conn.close()
    if as_json:
        click.echo(
            json.dumps(
                [
                    {
                        "id": item.id,
                        "node": item.node,
                        "proposal": item.submission.proposal,
                        "evaluation": item.submission.evaluation,
                        "status": item.status,
                        "created_at": item.created_at,
                        "summary": item.submission.summary,
                        "verdict": item.submission.verdict,
                    }
                    for item in found
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not found:
        ui.note(t("No proposals have been submitted."))
        return
    table = ui.table(t("ID"), t("Proposal"), t("Node"), t("Status"), t("Benchmark"))
    for item in found:
        table.add_row(
            item.id[:12],
            str(item.submission.proposal),
            item.node[:16],
            item.status,
            f"{item.submission.mode}: {item.submission.verdict}",
        )
    ui.out(table)
    ui.note(t("Inspect one with: desman-registry proposal show ID"))


@proposal.command("show")
@click.argument("proposal_id", metavar="ID")
@_json_option
@click.pass_context
def proposal_show(ctx: click.Context, proposal_id: str, as_json: bool) -> None:
    """Show a submitted proposal, its diff and both local benchmark results."""
    conn = _open_db(ctx)
    try:
        matches = store.find_proposal(conn, proposal_id)
        if len(matches) != 1:
            raise click.ClickException(t("No unique proposal matches {id}.", id=proposal_id))
        work.expire(conn, _now())
        found = store.proposal(conn, matches[0])
        checked = checks.results(conn, matches[0])
        released = conn.execute(
            "SELECT number FROM releases WHERE proposal = ?", (matches[0],)
        ).fetchone()
    finally:
        conn.close()
    assert found is not None
    data = {
        "id": found.id,
        "node": found.node,
        "status": found.status,
        "created_at": found.created_at,
        "submission": found.submission.model_dump(mode="json"),
        "reason": found.reason,
        "checks": checked,
        "release": None if released is None else released["number"],
    }
    if as_json:
        click.echo(json.dumps(data, ensure_ascii=False, indent=2))
        return
    item = found.submission
    for line in (
        f"{t('Proposal submission')} {found.id}",
        f"{t('Node')}: {found.node}",
        f"{t('Status')}: {found.status}",
        f"{t('Base commit')}: {item.base_commit}",
        f"{t('Benchmark')}: {item.mode} ({item.verdict})",
    ):
        ui.out(Text(line))
    if released is not None:
        ui.out(Text(f"{t('Release')}: {released['number']}"))
    _show_snapshots(item.base, item.candidate)
    if found.reason:
        ui.out(Text(f"{t('Reason')}: {found.reason}"))
    for check in checked:
        ui.out(Text(f"{t('Node')}: {check['node']}"))
        ui.out(Text(f"{t('Outcome')}: {check['outcome']}"))
        ui.out(Text(f"{t('Verdict')}: {check['verdict']}"))
        ui.out(Text(f"{t('Matches submission')}: {check['matches_submission']}"))
        if check["reason"]:
            ui.out(Text(f"{t('Reason')}: {check['reason']}"))
        _show_snapshots(
            None
            if check["base"] is None
            else protocol.BenchmarkSnapshot.model_validate(check["base"]),
            None
            if check["candidate"] is None
            else protocol.BenchmarkSnapshot.model_validate(check["candidate"]),
        )
    ui.out(Text(item.summary))
    if item.diff:
        ui.out(Text(item.diff))


@cli.command()
@_json_option
@click.pass_context
def inbox(ctx: click.Context, as_json: bool) -> None:
    """Jobs waiting for the operator's word (`job settle`)."""
    conn = _open_db(ctx)
    try:
        work.expire(conn, _now())  # a job that lost its last lease belongs here already
        found = work.jobs(conn, "operator")
    finally:
        conn.close()
    if as_json:
        click.echo(json.dumps([j.as_json() for j in found], ensure_ascii=False, indent=2))
        return
    if not found:
        ui.note(t("Nothing waits for the operator."))
        return
    _jobs_table(found)
    ui.note(t("Decide each with: desman-registry job settle JOB accept|again|drop"))


@job.command("show")
@click.argument("job_id", metavar="JOB")
@_json_option
@click.pass_context
def job_show(ctx: click.Context, job_id: str, as_json: bool) -> None:
    """A job, and every result and verdict nodes handed in for it."""
    conn = _open_db(ctx)
    try:
        work.expire(conn, _now())
        try:
            found = work.job(conn, admin.find_job(conn, job_id))
        except admin.AdminError as e:
            raise _fail(e) from e
        assert found is not None
        handed = work.handed_in(conn, found.id)
    finally:
        conn.close()
    if as_json:
        click.echo(
            json.dumps(
                found.as_json() | {"handed_in": [h.as_json() for h in handed]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    spec = found.job
    status = _job_status(found) + (f" ({_job_reason(found)})" if found.reason else "")
    kind = spec.kind.value + (f", {t('runs code')}" if spec.runs_code else "")
    # Text, never markup: what nodes (and the operator) wrote is shown as it is.
    lines = [
        f"{t('Job')} {found.id}",
        f"{t('Status')}: {status}",
        f"{t('Kind')}: {kind}",
        f"{t('Description')}: {spec.description}",
        f"{t('Acceptance criteria')}: {spec.acceptance_criteria}",
        *(f"{t('Input')} {name}: {text[:200]}" for name, text in spec.inputs.items()),
    ]
    for line in lines:
        ui.out(Text(line))
    if not handed:
        ui.note(t("Nothing handed in yet."))
    for h in handed:
        body = h.signed.record.body
        ui.out("")
        ui.out(Text(f"{h.received_at}  {h.kind}  {h.outcome}  {t('node')} {h.node[:16]}"))
        said = str(body.get("output") or body.get("reason") or "")
        if said:
            ui.out(Text(said))
        for f in body.get("files") or []:
            size = len(str(f.get("text", "")))
            ui.out(Text(f"— {f.get('path')} ({t('characters')}: {size})"))


@job.command("settle")
@click.argument("job_id", metavar="JOB")
@click.argument("decision", type=click.Choice(["accept", "again", "drop"]))
@click.pass_context
def job_settle(ctx: click.Context, job_id: str, decision: str) -> None:
    """Decide a job: accept its result, lend it again from the start, or drop it."""
    chosen: work.Decision = (
        "accept" if decision == "accept" else "again" if decision == "again" else "drop"
    )
    conn = _open_db(ctx)
    try:
        settled = admin.settle(conn, job_id, chosen, now=_now())
    except admin.AdminError as e:
        raise _fail(e) from e
    finally:
        conn.close()
    ui.ok(t("Job {id} is now: {status}", id=settled.id, status=_job_status(settled)))


@cli.command()
@click.argument("node_key", metavar="NODE")
@click.argument("name", required=False)
@click.option("--clear", is_flag=True, help="The node is its own owner again.")
@click.pass_context
def owner(ctx: click.Context, node_key: str, name: str | None, clear: bool) -> None:
    """Say whose node NODE is: nodes of one owner never recheck each other's work."""
    if (name is None) == (not clear):
        raise click.UsageError(t("Give a NAME or --clear, not both."))
    if name is not None and not 1 <= len(name) <= 64:
        raise click.BadParameter(t("At most 64 characters."), param_hint="NAME")
    conn = _open_db(ctx)
    try:
        fingerprint = admin.set_owner(conn, node_key, None if clear else name)
    except admin.AdminError as e:
        raise _fail(e) from e
    finally:
        conn.close()
    if clear:
        ui.ok(t("Node {key} is its own owner.", key=fingerprint[:16]))
    else:
        ui.ok(t("Node {key} belongs to {owner}.", key=fingerprint[:16], owner=str(name)))


@cli.group()
def trust() -> None:
    """The trust list: signed by the root, without a fresh one nodes take no work."""


@trust.command("sign")
@click.option(
    "--days",
    default=admin.TRUST_DAYS,
    show_default=True,
    type=click.IntRange(1, admin.MAX_TRUST_DAYS),
    help="How long the list lasts; sign the next one before it ends.",
)
@_root_option
@click.pass_context
def trust_sign(ctx: click.Context, days: int, root_key: Path | None) -> None:
    """Sign a new trust list naming every assignment and revocation in effect."""
    try:
        conn, root = _open(ctx, root_key)
        try:
            done = admin.sign_trust_list(conn, root, days=days, now=_now())
        finally:
            conn.close()
    except admin.AdminError as e:
        raise _fail(e) from e
    ui.ok(
        t(
            "Trust list signed until {date} (records named: {count}).",
            count=done.records,
            date=done.expires_at.isoformat(),
        )
    )


@trust.command("show")
@_json_option
@click.pass_context
def trust_show(ctx: click.Context, as_json: bool) -> None:
    """The trust list handed out now."""
    conn = _open_db(ctx)
    try:
        found = admin.trust_list(conn, _now())
    finally:
        conn.close()
    if as_json:
        click.echo(
            json.dumps(
                None
                if found is None
                else {
                    "id": found.id,
                    "records": found.records,
                    "issued_at": found.issued_at.isoformat(),
                    "expires_at": found.expires_at.isoformat(),
                },
                indent=2,
            )
        )
        return
    if found is None:
        ui.warn(
            t(
                "No trust list in its term: nodes take no work. Sign one with: desman-registry trust sign"
            )
        )
        return
    if _now() >= found.issued_at + sign.TRUST_LIST_MAX_AGE:
        ui.warn(t("The trust list is more than 2 days old: nodes take no work. Sign a new list."))
    ui.ok(
        t(
            "Trust list {id}: signed {issued}, until {date}; records named: {count}.",
            id=found.id[:16],
            count=found.records,
            issued=found.issued_at.isoformat(),
            date=found.expires_at.isoformat(),
        )
    )


def _show_snapshots(
    base: protocol.BenchmarkSnapshot | None, candidate: protocol.BenchmarkSnapshot | None
) -> None:
    """Both local and independent benchmark snapshots (G.7b)."""
    for label, snapshot in (
        (t("Current version"), base),
        (t("Proposed version"), candidate),
    ):
        if snapshot is None:
            continue
        ui.out(Text(label))
        for case in snapshot.cases:
            ui.out(
                Text(
                    t(
                        "{id}: {status} (subtasks {n})",
                        id=case.id,
                        status=case.status,
                        n=case.subtasks,
                    )
                )
            )
        roles = snapshot.stats.get("roles")
        if isinstance(roles, list):
            for key, label in (
                ("calls", t("Calls")),
                ("prompt_tokens", t("Prompt tokens")),
                ("completion_tokens", t("Completion tokens")),
            ):
                values = [
                    row[key]
                    for row in roles
                    if isinstance(row, dict) and isinstance(row.get(key), int)
                ]
                if values:
                    ui.out(Text(f"{label}: {sum(values)}"))


@proposal.command("accept")
@click.argument("proposal_id", metavar="ID")
@click.option(
    "--repo",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="The owner's checkout to release into.",
)
@click.option("--confirm", help="Confirm the exact release preview hash.")
@_root_option
@_json_option
@click.pass_context
def proposal_accept(
    ctx: click.Context,
    proposal_id: str,
    repo: Path,
    confirm: str | None,
    root_key: Path | None,
    as_json: bool,
) -> None:
    """Preview or confirm a root-signed release of a checked submission."""
    conn = _open_db(ctx)
    try:
        if confirm is None:
            result = releases.plan(conn, proposal_id, repo)
        else:
            result = releases.accept(
                conn, proposal_id, repo, confirm, directory=_dir(ctx), root_key=root_key, now=_now()
            )
    except admin.AdminError as e:
        raise _fail(e) from e
    finally:
        conn.close()
    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if confirm is None:
        for label, value in (
            (t("Proposal submission"), result["proposal"]),
            (t("Node"), result["node"]),
            (t("Summary"), result["summary"]),
            (t("Repository"), result["repo"]),
            (t("Branch"), result["branch"]),
            (t("Base commit"), result["base_commit"]),
            ("HEAD", result["head"]),
            (t("Files"), ", ".join(result["files"])),
        ):
            ui.out(Text(f"{label}: {value}"))
        for check in result["check_results"]:
            ui.out(
                Text(
                    f"{t('Node')}: {check['node']}  {t('Verdict')}: {check['verdict']}  {t('Matches submission')}: {check['matches_submission']}"
                )
            )
        ui.out(Text(result["diff"]))
        ui.note(t("Nothing was done. Repeat with --confirm {hash}.", hash=result["confirmation"]))
    else:
        ui.ok(
            t(
                "Release {number}: {commit} on {branch}.",
                number=result["number"],
                commit=result["commit"],
                branch=result["branch"],
            )
        )
        ui.note(
            t(
                "Push it yourself: {command}",
                command=f"git -C {shlex.quote(str(repo.resolve()))} push origin {shlex.quote(result['branch'])}",
            )
        )


@proposal.command("reject")
@click.argument("proposal_id", metavar="ID")
@click.option(
    "--reason", required=True, help="Why the submission is rejected (1 to 1000 characters)."
)
@click.pass_context
def proposal_reject(ctx: click.Context, proposal_id: str, reason: str) -> None:
    """Reject a submission with a recorded reason."""
    conn = _open_db(ctx)
    try:
        pid = releases.reject(conn, proposal_id, reason)
    except admin.AdminError as e:
        raise _fail(e) from e
    finally:
        conn.close()
    ui.ok(t("Proposal {id} rejected.", id=pid))


@cli.group()
def release() -> None:
    """Root-signed releases accepted by the operator."""


@release.command("list")
@_json_option
@click.pass_context
def release_list(ctx: click.Context, as_json: bool) -> None:
    """List the registry's releases."""
    conn = _open_db(ctx)
    try:
        found = releases.listing(conn)
    finally:
        conn.close()
    if as_json:
        click.echo(json.dumps(found, ensure_ascii=False, indent=2))
        return
    for row in found:
        ui.out(
            Text(
                f"{row['number']}  {row['commit'][:12]}  {row['branch']}  {row['proposal'][:12]}  {row['created_at']}"
            )
        )


@release.command("show")
@click.argument("number", type=click.IntRange(min=1))
@_json_option
@click.pass_context
def release_show(ctx: click.Context, number: int, as_json: bool) -> None:
    """Show a release and its root-signed record."""
    conn = _open_db(ctx)
    try:
        found = releases.show(conn, number)
    except admin.AdminError as e:
        raise _fail(e) from e
    finally:
        conn.close()
    if as_json:
        click.echo(json.dumps(found, ensure_ascii=False, indent=2))
    else:
        ui.out(
            Text(
                t(
                    "Release {number}: {commit} on {branch}.",
                    number=number,
                    commit=found["commit"],
                    branch=found["branch"],
                )
            )
        )
        ui.out(Text(json.dumps(found, ensure_ascii=False, indent=2)))
