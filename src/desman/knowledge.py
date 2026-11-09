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

Texts are prompts' material: English where the source is (the manual, the catalog), the docs as
they are written. Tokens are estimated as characters / 4, as everywhere else.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

NODE_CHARS: Final = 6000
"""The longest a node's text may be (~1.5k tokens): longer sections are split into parts."""

ABOUT_CHARS: Final = 100
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

    @property
    def section(self) -> str:
        return self.path.rsplit("/", 1)[0] if "/" in self.path else ""


def _one_line(text: str, limit: int = ABOUT_CHARS) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def _parts(text: str, limit: int = NODE_CHARS) -> list[str]:
    """Split a text at paragraph (then line, then hard) boundaries into pieces ≤ `limit`."""
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
    if current.strip():
        pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]


def command_nodes() -> list[Node]:
    from desman import commands

    groups: dict[str, list[str]] = {}
    names: dict[str, list[str]] = {}
    for line, entry in zip(commands.for_model().split("\n"), commands.CATALOG, strict=True):
        group = entry.name.split()[0]
        groups.setdefault(group, []).append(line)
        names.setdefault(group, []).append(entry.name)
    return [
        Node(f"Commands/{group}", _one_line("commands: " + ", ".join(names[group])), text)
        for group, lines in groups.items()
        for text in _parts("\n".join(lines))
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
    return line.lstrip("#").strip().replace("/", "∕")


def doc_nodes(root: Path | None = None) -> list[Node]:
    root = root if root is not None else repository()
    if root is None:
        return []
    nodes: list[Node] = []
    for path in _documents(root):
        relative = path.relative_to(root).as_posix()
        plan = relative.startswith(PLANS)
        base = f"DESMAN/Docs/{_name(path, root)}"
        text = path.read_text(encoding="utf-8")
        heading = None
        title = ""
        sections: list[tuple[str | None, str]] = []
        current: list[str] = []
        for line in text.splitlines():
            if line.startswith("# ") and not title:
                title = _title(line)
            if line.startswith("## "):
                sections.append((heading, "\n".join(current)))
                heading, current = _title(line), [line]
            else:
                current.append(line)
        sections.append((heading, "\n".join(current)))
        for heading, body in sections:
            if not body.strip():
                continue
            node_path = base if heading is None else f"{base}/{heading}"
            parts = _parts(body)
            first = next(
                (
                    plain
                    for line in body.splitlines()
                    if not line.startswith("#")
                    and (plain := re.sub(r"[*_`>|]", "", line).strip(" -"))
                ),
                "",
            )
            for n, part in enumerate(parts, 1):
                # The path names the document and the heading; the description says what it is.
                about = (first or f"{title}: {heading}") if heading else title or _name(path, root)
                suffix = f" ({n}/{len(parts)})" if len(parts) > 1 else ""
                nodes.append(
                    Node(
                        node_path + suffix.replace("/", "∕"),
                        _one_line(about + suffix),
                        part,
                        plan=plan,
                    )
                )
    return nodes


def release_tree() -> list[Node]:
    """Everything that comes with DESMAN, in a fixed order: commands, manual, documents."""
    return [*command_nodes(), *manual_nodes(), *doc_nodes()]


def find(nodes: Iterable[Node], path: str) -> Node | None:
    return next((n for n in nodes if n.path == path), None)


def under(nodes: Iterable[Node], section: str) -> list[Node]:
    prefix = section.rstrip("/") + "/"
    return [n for n in nodes if n.path.startswith(prefix)]


def _tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _line(node: Node) -> str:
    return f"- {node.path}{' [plan]' if node.plan else ''}: {node.about}"


def contents(
    nodes: list[Node], section: str | None = None, budget_tokens: int = CONTENTS_TOKENS
) -> str:
    """The table of contents a model chooses from: every node of `section` (all of them when
    None) with its description, while that fits the budget. Otherwise the subsections, each with
    how many nodes it holds, to be opened one at a time; never the texts."""
    shown = under(nodes, section) if section else list(nodes)
    whole = "\n".join(_line(n) for n in shown)
    if _tokens(whole) <= budget_tokens:
        return whole
    first = section.rstrip("/").count("/") + 2 if section else 1
    deepest = max(n.path.count("/") for n in shown) + 1 if shown else first
    listing = _collapsed(shown, first)
    for depth in range(first + 1, deepest):  # the most detailed listing that still fits
        deeper = _collapsed(shown, depth)
        if _tokens(deeper) > budget_tokens:
            break
        listing = deeper
    if _tokens(listing) <= budget_tokens:
        return listing
    # Only nodes and no deeper sections, still too many: their paths alone, then as many as fit.
    bare = [
        line.split(": ", 1)[0] if not line.endswith("nodes)") else line
        for line in listing.split("\n")
    ]
    kept: list[str] = []
    for n, line in enumerate(bare):
        if _tokens("\n".join([*kept, line])) > budget_tokens - 10:
            kept.append(f"- … and {len(bare) - n} more")
            break
        kept.append(line)
    return "\n".join(kept)


def _collapsed(shown: list[Node], depth: int) -> str:
    """Nodes up to `depth` parts listed as themselves; deeper ones as their section, counted."""
    lines: list[str] = []
    counts: dict[str, int] = {}
    for node in shown:
        parts = node.path.split("/")
        if len(parts) <= depth:  # a node right in the section is listed as itself
            lines.append(_line(node))
            continue
        group = "/".join(parts[:depth])
        if group not in counts:
            lines.append(group)
        counts[group] = counts.get(group, 0) + 1
    return "\n".join(
        f"- {line}/ ({counts[line]} nodes)" if line in counts else line for line in lines
    )
