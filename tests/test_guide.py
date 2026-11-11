"""The participants' guide (T.6) names only commands that exist, with arguments they accept."""

import re
import shlex
from pathlib import Path

import click

from desman import chat
from desman.cli import cli

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "guide-ru.md"


def _named() -> list[str]:
    text = GUIDE.read_text(encoding="utf-8")
    return re.findall(r"`(/[a-z][^`]*)`", text)


def test_the_guide_is_linked_from_the_site() -> None:
    html = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    assert "blob/master/docs/guide-ru.md" in html


def test_every_command_in_the_guide_exists() -> None:
    named = _named()
    assert len(named) > 10
    slash = chat._commands()
    for line in named:
        words = shlex.split(line.replace("НОМЕР", "1"))
        if words[0] in slash:
            continue
        path, rest = [words[0][1:]], words[1:]
        command = cli.commands[path[0]]
        while isinstance(command, click.Group) and rest and rest[0] in command.commands:
            path.append(rest[0])
            command = command.commands[rest.pop(0)]
        assert not isinstance(command, click.Group), line
        # Parses the arguments as the chat would hand them to the CLI body (no callback runs).
        command.make_context(" ".join(path), rest)
