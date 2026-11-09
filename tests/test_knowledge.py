"""The knowledge tree's release part (ADR 0009, K.1): nothing of the catalog, the manual or the
documents is left out, every node fits, the table of contents fits its budget."""

from __future__ import annotations

from pathlib import Path

import pytest

from desman import commands, knowledge, manual
from desman.knowledge import Node


@pytest.fixture(scope="module")
def tree() -> list[Node]:
    return list(knowledge.release_tree())


def test_every_node_fits_and_is_named_once(tree: list[Node]) -> None:
    paths = [n.path for n in tree]
    assert len(paths) == len(set(paths))
    for node in tree:
        assert node.text.strip() and len(node.text) <= knowledge.NODE_CHARS, node.path
        assert "\n" not in node.about and 0 < len(node.about) <= knowledge.ABOUT_CHARS, node.path
        assert node.origin == "release"


def test_every_command_is_in_its_group(tree: list[Node]) -> None:
    for entry in commands.CATALOG:
        group = entry.name.split()[0]
        texts = [n.text for n in knowledge.under(tree, "Commands") if n.path == f"Commands/{group}"]
        assert any(
            line.startswith(f"- {entry.name}") and entry.about in line
            for text in texts
            for line in text.splitlines()
        ), entry.name


def test_every_manual_topic_is_a_node(tree: list[Node]) -> None:
    for topic in manual.TOPICS:
        node = knowledge.find(tree, f"DESMAN/Manual/{topic.name}")
        assert node is not None and node.text == topic.text and not node.plan


def test_documents_are_split_without_losing_a_line(tree: list[Node]) -> None:
    root = knowledge.repository()
    assert root is not None  # the tests run from a checkout
    for name, file, plan in [
        ("ROADMAP", "ROADMAP.md", True),
        ("PROJECT_PLAN", "PROJECT_PLAN.md", True),
        ("ADR/0009-knowledge-tree", "docs/adr/0009-knowledge-tree.md", True),
        ("install", "docs/install.md", False),
    ]:
        nodes = [
            n
            for n in tree
            if n.path == f"DESMAN/Docs/{name}" or n.path.startswith(f"DESMAN/Docs/{name}/")
        ]
        assert nodes and all(n.plan is plan for n in nodes), name
        kept = "\n".join(n.text for n in nodes)
        for line in (root / file).read_text(encoding="utf-8").splitlines():
            assert line.strip() in kept, (name, line)


def test_no_documents_outside_a_checkout(tmp_path: Path) -> None:
    assert knowledge.doc_nodes(tmp_path) == []  # a folder without documents has none
    (tmp_path / "docs").mkdir()
    (tmp_path / "ROADMAP.md").write_text("# Plan\n\n## Phase\n\ntext\n", encoding="utf-8")
    [whole, phase] = knowledge.doc_nodes(tmp_path)
    assert (whole.path, phase.path) == ("DESMAN/Docs/ROADMAP", "DESMAN/Docs/ROADMAP/Phase")
    assert phase.plan and phase.about == "text" and whole.about == "Plan"


def test_a_long_section_is_split_into_numbered_parts(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    long = "\n\n".join(f"paragraph {i} " + "x" * 900 for i in range(20))
    (tmp_path / "docs" / "guide.md").write_text(f"# Guide\n\n## Long\n\n{long}\n", encoding="utf-8")
    (tmp_path / "ROADMAP.md").write_text("# R\n", encoding="utf-8")
    parts = [n for n in knowledge.doc_nodes(tmp_path) if "Long" in n.path]
    assert len(parts) > 1 and all(len(n.text) <= knowledge.NODE_CHARS for n in parts)
    assert parts[0].path.endswith(f"(part 1 of {len(parts)})") and not parts[0].plan
    assert all(f"paragraph {i} " in "".join(n.text for n in parts) for i in range(20))


def test_contents_fit_their_budget_and_never_show_texts(tree: list[Node]) -> None:
    for section in (
        None,
        "Commands",
        "DESMAN",
        "DESMAN/Manual",
        "DESMAN/Docs",
        "DESMAN/Docs/ROADMAP",
    ):
        listing = knowledge.contents(tree, section)
        assert listing and len(listing) // knowledge.CHARS_PER_TOKEN <= knowledge.CONTENTS_TOKENS
        assert not any(n.text in listing for n in tree if len(n.text) > 200), section
    top = knowledge.contents(tree)
    assert "- DESMAN/Docs/ (" in top and "nodes)" in top  # collapsed, counted
    small = [Node("A/x", "first", "t"), Node("A/y", "second", "t", plan=True)]
    assert knowledge.contents(small) == "- A/x: first\n- A/y [plan]: second"
    assert knowledge.contents(small, budget_tokens=5) == "- A/ (2 nodes)"
    many = [Node(f"B/n{i}", "about " * 10, "t") for i in range(50)]
    cut = knowledge.contents(many, "B", budget_tokens=100).splitlines()
    assert cut[0] == "- B/n0" and cut[-1].startswith("- … and ")  # paths alone, then cut


def test_code_blocks_headings_and_repeated_sections(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "ROADMAP.md").write_text("# R\n", encoding="utf-8")
    (tmp_path / "docs" / "how.md").write_text(
        "# How\n\nIntro.\n\n## Run / stop\n\n```bash\n## not a heading\n    indented\n```\n"
        "Then it runs.\n\n## Notes\n\nOne.\n\n## Notes\n\nTwo.\n",
        encoding="utf-8",
    )
    nodes = [n for n in knowledge.doc_nodes(tmp_path) if "/how" in n.path]
    assert [n.path for n in nodes] == [
        "DESMAN/Docs/how",
        "DESMAN/Docs/how/Run - stop",
        "DESMAN/Docs/how/Notes",
        "DESMAN/Docs/how/Notes (2)",
    ]
    run = nodes[1]
    assert "## not a heading\n    indented" in run.text  # code kept whole, indentation too
    assert run.about == "Then it runs."  # prose, not the code block's language
    # Opening the document's section offers its preamble too.
    assert knowledge.contents(nodes, "DESMAN/Docs/how").startswith("- DESMAN/Docs/how: How")


def test_a_long_command_group_gets_numbered_parts() -> None:
    nodes = knowledge._numbered("Commands/global", "commands: many", ["a", "b"])
    assert [n.path for n in nodes] == [
        "Commands/global (part 1 of 2)",
        "Commands/global (part 2 of 2)",
    ]


def test_bare_paths_are_real_paths() -> None:
    many = [Node(f"B/Phase {i}: start", "about " * 10, "t") for i in range(50)]
    lines = knowledge.contents(many, "B", budget_tokens=100).splitlines()
    assert lines[0] == "- B/Phase 0: start"  # a ": " in the path is kept
    assert all(knowledge.find(many, line[2:]) for line in lines[:-1])
