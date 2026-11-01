"""`desman protected`: the protected files and the person's approval of a change (ROADMAP 5.3).

`show` says whether every protected file is as approved (with `--diff`, what changed); while one
is not, nothing runs. `approve` is the person's approval of the files as they are now. Both are
non-interactive; the chat calls the same functions, and approves only after the person's "yes"
in its window.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click
from rich.text import Text

from desman import ui
from desman.config_cli import json_option
from desman.core import db, protected
from desman.core.protected import Change, Kind
from desman.i18n import t, tn

F = TypeVar("F", bound=Callable[..., Any])


def db_option(command: F) -> F:
    return click.option(
        "--db",
        "db_path",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="Database file (default: the usual one).",
    )(command)


def connect(db_path: Path | None) -> sqlite3.Connection:
    try:
        return db.open_db(db_path or db.default_db_path())
    except (db.SchemaVersionError, db.SchemaMismatchError) as e:
        raise click.ClickException(str(e)) from e


def current(conn: sqlite3.Connection) -> list[Change]:
    try:
        return protected.changes(conn)
    except protected.ProtectedError as e:
        raise click.ClickException(t("Cannot check the protected files: {error}", error=e)) from e


@click.group(invoke_without_command=True)
@click.pass_context
def protected_group(ctx: click.Context) -> None:
    """Protected files: the tool policy, the reviewer, the sandbox, the approval rules, the
    test harness. A change nobody approved stops every run."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(show)


@protected_group.command("show")
@click.option("--diff", "with_diff", is_flag=True, help="Show what changed in each file.")
@db_option
@json_option
def show(with_diff: bool = False, db_path: Path | None = None, as_json: bool = False) -> None:
    """Whether every protected file is as you approved it; tasks run only then."""
    conn = connect(db_path)
    try:
        changes = current(conn)
        if as_json:
            click.echo(json.dumps(state_json(conn, changes), ensure_ascii=False, indent=2))
            return
        print_state(conn, changes, with_diff=with_diff)
    finally:
        conn.close()


@protected_group.command("approve")
@click.option("--note", default=None, help="Why: kept with the approval.")
@db_option
@json_option
def approve(note: str | None = None, db_path: Path | None = None, as_json: bool = False) -> None:
    """Approve the protected files as they are now, so tasks run again. Look at the change
    first: desman protected show --diff."""
    conn = connect(db_path)
    try:
        done = approve_now(conn, note)
    finally:
        conn.close()
    if as_json:
        out = {"approved": [c.model_dump(mode="json") for c in done]}
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
        return
    print_approved(done)


def approve_now(
    conn: sqlite3.Connection,
    note: str | None,
    *,
    expected: list[Change] | None = None,
    session_id: str | None = None,
) -> list[Change]:
    try:
        return protected.approve(conn, note, expected=expected, session_id=session_id)
    except protected.ProtectedError as e:
        raise click.ClickException(t("Cannot check the protected files: {error}", error=e)) from e


def state_json(conn: sqlite3.Connection, changes: list[Change]) -> dict[str, Any]:
    last = protected.last_approval(conn)
    return {
        "approved": None
        if last is None
        else {"id": last.id, "ts": last.ts, "source": last.source, "note": last.note},
        "changes": [c.model_dump(mode="json") for c in changes],
    }


def kind_label(kind: Kind) -> str:
    return {
        Kind.CHANGED: t("changed"),
        Kind.MISSING: t("missing"),
        Kind.ADDED: t("put on the list"),
        Kind.REMOVED: t("taken off the list"),
    }[kind]


def change_lines(changes: list[Change]) -> list[str]:
    return [f"{c.path}: {kind_label(c.kind)}" for c in changes]


def print_state(conn: sqlite3.Connection, changes: list[Change], *, with_diff: bool) -> None:
    last = protected.last_approval(conn)
    if last is not None:
        when = db.parse_iso(last.ts).astimezone().strftime("%Y-%m-%d %H:%M")  # local time
        if last.source == "release":
            ui.note(t("Approved: the release as shipped ({when}).", when=when))
        elif last.note:
            ui.note(t("Approved by you on {when}: {note}", when=when, note=last.note))
        else:
            ui.note(t("Approved by you on {when}.", when=when))
    if not changes:
        ui.ok(t("Every protected file is as approved."))
        return
    ui.warn(
        tn(
            len(changes),
            "{n} protected file changed without your approval: nothing runs until you approve it.",
            "{n} protected files changed without your approval: nothing runs until you approve "
            "them.",
        )
    )
    for line in change_lines(changes):
        ui.out(Text("  " + line))
    if with_diff:
        for change in changes:
            text = protected.diff(conn, change)
            if text is None:
                ui.note(
                    t(
                        "What was approved is not known yet (nothing was approved on this "
                        "database): compare with the release in git (git diff)."
                    )
                )
                break
            ui.out(Text(text or t("{path}: no difference in the text.", path=change.path)))
    else:
        ui.note(t("See what changed: desman protected show --diff"))
    ui.note(t("Approve: desman protected approve [--note TEXT]"))


def print_approved(done: list[Change]) -> None:
    if not done:
        ui.ok(t("Nothing to approve: every protected file is as approved."))
        return
    ui.ok(
        tn(
            len(done),
            "Approved {n} change: tasks run again.",
            "Approved {n} changes: tasks run again.",
        )
    )
    for line in change_lines(done):
        ui.out(Text("  " + line))
