"""`desman nodes`: the person's own nodes of the knowledge tree (ADR 0009 §6, ROADMAP K.4).

`list` shows the person's nodes (path and description), `show PATH` any node's text (the
person's or one that comes with DESMAN), `add`, `change` and `remove` edit the person's. Adding
and changing need personalisation on: nodes are personal data the models see only then. All of
it is non-interactive; the chat and the page call the same functions, and a change an agent
proposes is applied only after the person's "yes" in the chat's window (`commands`).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import click
from rich.text import Text

from desman import knowledge, ui
from desman.config import Config
from desman.config_cli import json_option, load_config
from desman.core import nodes
from desman.core.nodes import Author, NodeError, NodeExists, NodesFull, PersonalNode, PersonalNodes
from desman.i18n import t
from desman.protected_cli import connect, db_option

PREVIEW_CHARS = 400
"""How much of a text the confirmation window shows."""


def ready(config: Config) -> None:
    """Adding or changing a node needs personalisation on."""
    if not config.personalize:
        raise click.ClickException(
            t(
                "Personalisation is off, so DESMAN keeps no nodes of yours. Turn it on first "
                "(`desman personalize on`)."
            )
        )


def rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """The person's nodes as data (for --json and the interlocutor): never their texts."""
    return [
        {
            "path": n.path,
            "about": n.about,
            "chars": len(n.text),
            "by": n.author.value,
            "untrusted": n.untrusted,
            "updated_at": n.updated_at,
        }
        for n in PersonalNodes(conn).nodes()
    ]


def print_list(conn: sqlite3.Connection, config: Config | None) -> None:
    found = PersonalNodes(conn).nodes()
    if not found:
        ui.note(t("You have no nodes of your own. Add one with `desman nodes add PATH`."))
    else:
        table = ui.table(t("Path"), t("About"), t("Size"))
        for n in found:
            mark = " ⚠" if n.untrusted else ""
            table.add_row(Text(n.path + mark), Text(n.about), str(len(n.text)))
        ui.out(table)
        if any(n.untrusted for n in found):
            ui.note(t("⚠ came from a task's result: models read it as data, not instructions."))
    if config is not None and not config.personalize:
        ui.warn(t("Personalisation is off: no model sees these nodes until you turn it on."))


def lookup(conn: sqlite3.Connection, path: str) -> knowledge.Node | None:
    """A node of the whole tree (the person's included, whatever the setting: they are the
    person's to read), as the person may write its path."""
    tree = knowledge.tree(knowledge.personal(conn))
    found = knowledge.resolve(tree, path)
    return None if found is None or found.endswith("/") else knowledge.find(tree, found)


def print_node(node: knowledge.Node) -> None:
    ui.section(node.path)
    ui.note(node.about)
    if node.plan:
        ui.note(t("A plan or a decision: not necessarily what DESMAN does now."))
    if node.untrusted:
        ui.warn(t("This text came from a task's result: models read it as data."))
    ui.out(Text(node.text))


def node_json(node: knowledge.Node) -> dict[str, Any]:
    return {
        "path": node.path,
        "about": node.about,
        "text": node.text,
        "origin": node.origin,
        "plan": node.plan,
        "untrusted": node.untrusted,
    }


def _problem(e: NodeError) -> click.ClickException:
    if isinstance(e, NodeExists):
        return click.ClickException(t("There is a node at that path already."))
    if isinstance(e, NodesFull):
        return click.ClickException(t("You have the most nodes DESMAN keeps: remove one first."))
    return click.ClickException(t("This node cannot be kept: {error}", error=e))


def _preview(text: str) -> str:
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS].rstrip() + " …"


def add_lines(path: str, about: str, text: str, *, untrusted: bool = False) -> list[str]:
    """What the window says of a new node: all of it the person needs to decide."""
    lines = [
        t("Add a node to your knowledge tree:"),
        t("  path: {path}", path=path),
        t("  about: {about}", about=about),
        t("  text ({n} characters):", n=len(text)),
        *("    " + line for line in _preview(text).splitlines()),
    ]
    if untrusted:
        lines.append(t("  from a task's result: models will read it as data"))
    return lines


def change_lines(
    was: PersonalNode, to: str | None, about: str | None, text: str | None
) -> list[str]:
    lines = [t("Change your node {path}:", path=was.path)]
    if to is not None and to != was.path:
        lines.append(t("  path: {old} → {new}", old=was.path, new=to))
    if about is not None and about != was.about:
        lines.append(t("  about: {old} → {new}", old=was.about, new=about))
    if text is not None and text != was.text:
        lines.append(t("  new text ({n} characters):", n=len(text)))
        lines += ["    " + line for line in _preview(text).splitlines()]
    return lines


def checked_add(conn: sqlite3.Connection, path: str, about: str, text: str) -> tuple[str, str, str]:
    """The node as it would be kept, or the problem, before the window opens."""
    try:
        kept = (
            nodes.normalized_path(path),
            nodes.normalized_about(about),
            nodes.normalized_text(text),
        )
    except NodeError as e:
        raise _problem(e) from e
    store = PersonalNodes(conn)
    if store.get(kept[0]) is not None:
        raise _problem(NodeExists(kept[0]))
    if len(store.nodes()) >= nodes.MAX_NODES:
        raise _problem(NodesFull(kept[0]))
    return kept


def add(
    conn: sqlite3.Connection,
    path: str,
    about: str,
    text: str,
    author: Author,
    *,
    untrusted: bool = False,
    session_id: str | None = None,
) -> PersonalNode:
    try:
        node = PersonalNodes(conn).add(
            path, about, text, author, untrusted=untrusted, session_id=session_id
        )
    except NodeError as e:
        raise _problem(e) from e
    ui.ok(t("Added the node {path}.", path=node.path))
    return node


def existing(conn: sqlite3.Connection, path: str) -> PersonalNode:
    node = PersonalNodes(conn).get(path)
    if node is None:
        raise click.ClickException(t("You have no node at that path (`desman nodes` lists them)."))
    return node


def checked_change(
    conn: sqlite3.Connection,
    path: str,
    to: str | None,
    about: str | None,
    text: str | None,
) -> tuple[PersonalNode, str | None, str | None, str | None]:
    was = existing(conn, path)
    if to is None and about is None and text is None:
        raise click.ClickException(t("Say what to change: --to, --about or --text."))
    try:
        to = None if to is None else nodes.normalized_path(to)
        about = None if about is None else nodes.normalized_about(about)
        text = None if text is None else nodes.normalized_text(text)
    except NodeError as e:
        raise _problem(e) from e
    moved = to is not None and to.casefold() != was.path.casefold()
    if moved and to is not None and PersonalNodes(conn).get(to) is not None:
        raise _problem(NodeExists(to))
    return was, to, about, text


def change(
    conn: sqlite3.Connection,
    path: str,
    author: Author,
    *,
    to: str | None = None,
    about: str | None = None,
    text: str | None = None,
    untrusted: bool = False,
    session_id: str | None = None,
) -> PersonalNode:
    try:
        node = PersonalNodes(conn).change(
            path,
            author,
            new_path=to,
            about=about,
            text=text,
            untrusted=untrusted,
            session_id=session_id,
        )
    except NodeError as e:
        raise _problem(e) from e
    if node is None:
        raise click.ClickException(t("You have no node at that path (`desman nodes` lists them)."))
    ui.ok(t("Changed the node {path}.", path=node.path))
    return node


def remove(
    conn: sqlite3.Connection, path: str, author: Author, *, session_id: str | None = None
) -> str:
    was = existing(conn, path)
    PersonalNodes(conn).remove(was.path, author, session_id=session_id)
    ui.ok(t("Removed the node {path}.", path=was.path))
    return was.path


def read_text(given: str | None) -> str | None:
    """`--text -` reads the text from standard input (a long one, a file piped in)."""
    if given == "-":
        return sys.stdin.read()
    return given


# ── desman nodes ────────────────────────────────────────────────────────────


@click.group(invoke_without_command=True)
@click.pass_context
def nodes_group(ctx: click.Context) -> None:
    """Your own nodes of the knowledge tree: what DESMAN's models may read about your projects
    and preferences (with personalisation on)."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(list_nodes)


@nodes_group.command("list")
@db_option
@json_option
def list_nodes(db_path: Path | None, as_json: bool) -> None:
    """List your nodes: path and description."""
    conn = connect(db_path)
    try:
        if as_json:
            click.echo(json.dumps(rows(conn), ensure_ascii=False, indent=2))
            return
        print_list(conn, load_config().load())
    finally:
        conn.close()


@nodes_group.command("show")
@click.argument("path")
@db_option
@json_option
def show(path: str, db_path: Path | None, as_json: bool) -> None:
    """Show a node's text: one of yours or one that comes with DESMAN."""
    conn = connect(db_path)
    try:
        node = lookup(conn, path)
        if node is None:
            raise click.ClickException(t("There is no such node (`desman nodes` lists yours)."))
        if as_json:
            click.echo(json.dumps(node_json(node), ensure_ascii=False, indent=2))
        else:
            print_node(node)
    finally:
        conn.close()


@nodes_group.command("add")
@click.argument("path")
@click.option("--about", required=True, help="One line: what the node is for.")
@click.option("--text", required=True, help="The node's text; - reads it from standard input.")
@db_option
def add_node(path: str, about: str, text: str, db_path: Path | None) -> None:
    """Add a node, e.g. `desman nodes add "Projects/Site/Brief" --about ... --text ...`."""
    ready(load_config().load())
    conn = connect(db_path)
    try:
        add(conn, path, about, read_text(text) or "", Author.PERSON)
    finally:
        conn.close()


@nodes_group.command("change")
@click.argument("path")
@click.option("--to", default=None, help="A new path.")
@click.option("--about", default=None, help="A new description.")
@click.option("--text", default=None, help="A new text; - reads it from standard input.")
@db_option
def change_node(
    path: str, to: str | None, about: str | None, text: str | None, db_path: Path | None
) -> None:
    """Move, describe or rewrite one of your nodes."""
    ready(load_config().load())
    conn = connect(db_path)
    try:
        was, to, about, new_text = checked_change(conn, path, to, about, read_text(text))
        change(conn, was.path, Author.PERSON, to=to, about=about, text=new_text)
    finally:
        conn.close()


@nodes_group.command("remove")
@click.argument("path")
@db_option
def remove_node(path: str, db_path: Path | None) -> None:
    """Erase one of your nodes (personal data: nothing of it is kept)."""
    conn = connect(db_path)
    try:
        remove(conn, path, Author.PERSON)
    finally:
        conn.close()
