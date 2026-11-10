"""The knowledge tree (ADR 0009): what DESMAN knows, as sections and nodes a model chooses from
by their table of contents, so that only what a piece of work needs becomes its context.

K.1 builds the part that comes with DESMAN (`release`), from the code and the repository, at
run time; nothing of it is stored, and only people change it, through git:

- `Commands/GROUP`: the chat's commands of a group (`keys`, `global`, `task`, ...), each with
  its arguments and what it does, as the interlocutor's catalog says them.
- `DESMAN/Manual/TOPIC`: the interlocutor's manual (what DESMAN does now).
- `DESMAN/Docs/...`: the repository's documents split by their `##` headings. PROJECT_PLAN,
  ROADMAP and the ADRs describe intentions: their nodes are marked `plan`, and a model must tell
  them as plans, never as what DESMAN does. Without a repository checkout (an installed package)
  there are no Docs nodes.

K.4 adds the person's own nodes (`core.nodes`, stored in the database): `personal()` reads them
as nodes of origin `person`, and `tree()` / `for_tasks()` put them after the release part. They
are given only while the person agreed to personalisation, and never to a temporary chat or a
desman-global job. K.5: one of them, `Personal/Profile`, holds the profile's notes; the chat's
interlocutor gets those with every message, so its table of contents leaves that node out.

Texts are prompts' material: English where the source is (the manual, the catalog), the docs as
they are written. Tokens are estimated as characters / 4, as everywhere else.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Final, Literal

from desman.core import nodes as personal_store
from desman.core.context import Knowledge, KnowledgeText

NODE_CHARS: Final = personal_store.TEXT_CHARS
"""The longest a node's text may be (~1.5k tokens): longer sections are split into parts."""

ABOUT_CHARS: Final = personal_store.ABOUT_CHARS
"""A node's description is one line for choosing, at most this long."""

CONTENTS_TOKENS: Final = 800
"""The table of contents is shown whole while it fits this; otherwise only its sections."""

CHARS_PER_TOKEN: Final = 4

Origin = Literal["release", "person"]


@dataclass(frozen=True)
class Node:
    path: str
    """Sections and the node's name, joined by "/": `Commands/keys`, `DESMAN/Manual/tasks`."""
    about: str
    text: str
    origin: Origin = "release"
    plan: bool = False
    """It describes intentions (PROJECT_PLAN, ROADMAP, ADRs), not what DESMAN does now."""
    untrusted: bool = False
    """A personal node whose text came from a task's result: data, never instructions."""


def _one_line(text: str, limit: int = ABOUT_CHARS) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def _parts(text: str, limit: int = NODE_CHARS) -> list[str]:
    """Split a text at paragraph (then line, then hard) boundaries into pieces ≤ `limit`. Only
    blank lines around a piece are dropped: indentation (code, nested lists) is kept."""
    pieces: list[str] = []
    current = ""
    for block in re.split(r"(\n\s*\n)", text):
        while len(block) > limit:  # one paragraph longer than a node: lines, then hard cuts
            cut = block.rfind("\n", 0, limit)
            cut = cut if cut > 0 else limit
            if current:
                pieces.append(current)
                current = ""
            pieces.append(block[:cut])
            block = block[cut:]
        if len(current) + len(block) > limit:
            pieces.append(current)
            current = ""
        current += block
    pieces.append(current)
    return [p.strip("\n").rstrip() for p in pieces if p.strip()]


def _numbered(path: str, about: str, texts: list[str], plan: bool = False) -> list[Node]:
    """One node per text; several get "(part N of M)", so every path names one node."""
    if len(texts) == 1:
        return [Node(path, _one_line(about), texts[0], plan=plan)]
    return [
        Node(
            f"{path} (part {n} of {len(texts)})", _one_line(f"{about} (part {n})"), text, plan=plan
        )
        for n, text in enumerate(texts, 1)
    ]


def command_nodes() -> list[Node]:
    from desman import commands

    groups: dict[str, list[str]] = {}
    names: dict[str, list[str]] = {}
    for line, entry in zip(commands.for_model().split("\n"), commands.CATALOG, strict=True):
        group = entry.name.split()[0]
        groups.setdefault(group, []).append(line)
        names.setdefault(group, []).append(entry.name)
    return [
        node
        for group, lines in groups.items()
        for node in _numbered(
            f"Commands/{group}", "commands: " + ", ".join(names[group]), _parts("\n".join(lines))
        )
    ]


def manual_nodes() -> list[Node]:
    from desman import manual

    return [
        Node(f"DESMAN/Manual/{topic.name}", _one_line(topic.about), topic.text)
        for topic in manual.TOPICS
    ]


def repository() -> Path | None:
    """The checkout DESMAN runs from (src/desman/ → the root), if it is one."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "docs").is_dir() and (root / "ROADMAP.md").is_file() else None


PLANS: Final = ("PROJECT_PLAN.md", "ROADMAP.md", "docs/adr/")
"""Documents of intentions: their nodes are marked `plan`."""


def _documents(root: Path) -> list[Path]:
    found = [root / "PROJECT_PLAN.md", root / "ROADMAP.md"]
    found += sorted((root / "docs").glob("*.md")) + sorted((root / "docs" / "adr").glob("*.md"))
    return [p for p in found if p.is_file()]


def _name(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return relative.removesuffix(".md").removeprefix("docs/").replace("adr/", "ADR/")


def _title(line: str) -> str:
    """A heading as a path's part: plain characters, never "/" (it separates sections)."""
    return line.lstrip("#").strip().replace("/", "-")


def _sections(text: str) -> tuple[str, list[tuple[str | None, str]]]:
    """The document's title and its sections by `##` heading (the preamble's is None). A
    heading inside a fenced code block is code, not a heading."""
    title = ""
    heading: str | None = None
    sections: list[tuple[str | None, str]] = []
    current: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        elif not fenced and line.startswith("# ") and not title:
            title = _title(line)
        elif not fenced and line.startswith("## "):
            sections.append((heading, "\n".join(current)))
            heading, current = _title(line), [line]
            continue
        current.append(line)
    sections.append((heading, "\n".join(current)))
    return title, sections


def _first_sentence(body: str) -> str:
    """The section's first line of prose: not a heading, not inside or opening a code block."""
    fenced = False
    for line in body.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced or line.startswith("#"):
            continue
        plain = re.sub(r"[*_`>|]", "", line).strip(" -")
        if plain:
            return plain
    return ""


def doc_nodes(root: Path | None = None) -> list[Node]:
    root = root if root is not None else repository()
    if root is None:
        return []
    nodes: list[Node] = []
    for path in _documents(root):
        plan = path.relative_to(root).as_posix().startswith(PLANS)
        base = f"DESMAN/Docs/{_name(path, root)}"
        title, sections = _sections(path.read_text(encoding="utf-8"))
        seen: dict[str, int] = {}
        for heading, body in sections:
            if not body.strip():
                continue
            node_path = base if heading is None else f"{base}/{heading}"
            seen[node_path] = seen.get(node_path, 0) + 1
            if seen[node_path] > 1:  # the same heading twice in one document
                node_path += f" ({seen[node_path]})"
            # The path names the document and the heading; the description says what it is.
            if heading is None:
                about = title or _name(path, root)
            else:
                about = _first_sentence(body) or f"{title}: {heading}"
            nodes += _numbered(node_path, about, _parts(body), plan)
    return nodes


@cache
def release_tree() -> tuple[Node, ...]:
    """Everything that comes with DESMAN, in a fixed order: commands, manual, documents. Built
    once per process: it changes only with the code (git)."""
    return (*command_nodes(), *manual_nodes(), *doc_nodes())


@cache
def task_tree() -> tuple[Node, ...]:
    """What tasks may be given (K.3): the tree without `Commands/`, which only the chat runs."""
    return tuple(n for n in release_tree() if not n.path.startswith("Commands/"))


def personal(conn: sqlite3.Connection) -> tuple[Node, ...]:
    """The person's own nodes (K.4), as the tree holds them. The caller decides whether they
    may be seen (`Config.personalize`, not a temporary chat)."""
    return tuple(
        Node(n.path, n.about, n.text, origin="person", untrusted=n.untrusted)
        for n in personal_store.PersonalNodes(conn).nodes()
    )


def tree(own: Iterable[Node] = ()) -> tuple[Node, ...]:
    """The whole tree the chat chooses from: the release part, then the person's nodes."""
    return (*release_tree(), *own)


def for_tasks(own: Iterable[Node] = ()) -> tuple[Node, ...]:
    """The tree tasks are given: `task_tree()`, then the person's nodes."""
    return (*task_tree(), *own)


def resolve(nodes: Iterable[Node], given: str) -> str | None:
    """A path of the tree as a model may write it, in any case: a node's path, a section's
    (ending in "/": a document's section, not its preamble node of the same name), or just a
    manual topic's name. None when there is no such thing."""
    paths, sections = _index(tuple(nodes))
    written = given.strip().strip("`")
    path = written.rstrip("/").casefold()
    if written.endswith("/") and path in sections:
        return sections[path] + "/"
    if path in paths:
        return paths[path]
    if path in sections:
        return sections[path] + "/"
    return paths.get(f"DESMAN/Manual/{path}".casefold())


@lru_cache(maxsize=8)
def _index(nodes: tuple[Node, ...]) -> tuple[dict[str, str], dict[str, str]]:
    """Every node's path and every section's, by their case-folded spelling (per tree: the
    release and task trees are built once)."""
    paths: dict[str, str] = {}
    sections: dict[str, str] = {}
    for node in nodes:
        paths[node.path.casefold()] = node.path
        parts = node.path.split("/")
        for depth in range(1, len(parts)):
            section = "/".join(parts[:depth])
            sections[section.casefold()] = section
    return paths, sections


def read(nodes: Iterable[Node], given: str) -> KnowledgeText | None:
    """What a model reading `given` gets: a node's text, or a section's table of contents
    ("" or "/": the whole tree's). None when there is no such node or section."""
    nodes = tuple(nodes)
    if not given.strip().strip("/"):
        return KnowledgeText(path="/", text=contents(nodes))
    path = resolve(nodes, given)
    if path is None:
        return None
    node = find(nodes, path)
    if node is None:
        return KnowledgeText(path=path, text=contents(nodes, path.rstrip("/")))
    return KnowledgeText(path=node.path, text=node.text, plan=node.plan, untrusted=node.untrusted)


def reader(tree: Callable[[], Iterable[Node]] = task_tree) -> Knowledge:
    """Looks paths up in `tree`, asked at every lookup (the release part is built once per
    process; the person's nodes are read fresh by the function given here: `runs`)."""
    return lambda path: read(tree(), path)


def find(nodes: Iterable[Node], path: str) -> Node | None:
    return next((n for n in nodes if n.path == path), None)


def under(nodes: Iterable[Node], section: str) -> list[Node]:
    """The section's own node (a document's preamble), if any, and everything below it."""
    section = section.rstrip("/")
    return [n for n in nodes if n.path == section or n.path.startswith(section + "/")]


def _tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


Entry = Node | tuple[str, int]
"""A line of the table of contents: a node, or a section with how many nodes it holds."""


def contents(
    nodes: Iterable[Node], section: str | None = None, budget_tokens: int = CONTENTS_TOKENS
) -> str:
    """The table of contents a model chooses from: every node of `section` (all of them when
    None) with its description, while that fits the budget. Otherwise the most detailed
    listing that fits, deeper subsections shown with how many nodes they hold, to be opened one
    at a time; failing that, paths alone, as many as fit. Never the texts."""
    shown = under(nodes, section) if section else list(nodes)
    whole = _render(shown)
    if _tokens(whole) <= budget_tokens:
        return whole
    first = section.rstrip("/").count("/") + 2 if section else 1
    deepest = max(n.path.count("/") for n in shown) + 1 if shown else first
    entries = _collapsed(shown, first)
    for depth in range(first + 1, deepest):
        deeper = _collapsed(shown, depth)
        if _tokens(_render(deeper)) > budget_tokens:
            break
        entries = deeper
    listing = _render(entries)
    if _tokens(listing) <= budget_tokens:
        return listing
    kept: list[str] = []
    lines = _render(entries, about=False).split("\n")
    for n, line in enumerate(lines):
        if _tokens("\n".join([*kept, line])) > budget_tokens - 10:
            kept.append(f"- … and {len(lines) - n} more")
            break
        kept.append(line)
    return "\n".join(kept)


def _collapsed(shown: list[Node], depth: int) -> list[Entry]:
    """Nodes up to `depth` parts as themselves; deeper ones as their section, counted."""
    entries: list[Entry] = []
    counts: dict[str, int] = {}
    for node in shown:
        parts = node.path.split("/")
        if len(parts) <= depth:
            entries.append(node)
            continue
        group = "/".join(parts[:depth])
        if group not in counts:
            entries.append((group, 0))
        counts[group] = counts.get(group, 0) + 1
    return [e if isinstance(e, Node) else (e[0], counts[e[0]]) for e in entries]


def _render(entries: Iterable[Entry], about: bool = True) -> str:
    lines = []
    for entry in entries:
        if isinstance(entry, Node):
            mark = " [plan]" if entry.plan else " [untrusted]" if entry.untrusted else ""
            lines.append(
                f"- {entry.path}{mark}: {entry.about}" if about else f"- {entry.path}{mark}"
            )
        else:
            lines.append(f"- {entry[0]}/ ({entry[1]} nodes)")
    return "\n".join(lines)
