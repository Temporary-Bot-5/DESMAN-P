"""The participants' guides (T.6, T.6s): one per language of the site, each naming only commands
that exist, with arguments they accept."""

import json
import re
import shlex
from pathlib import Path

import click
import pytest

from desman import chat
from desman.cli import cli

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DOCS = ROOT / "docs"
LANGUAGES = ["en", "ru", "uk", "de", "fr", "es", "pt", "pl", "tr", "zh", "ja", "ko", "hi"]
PLACEHOLDER = re.compile(r"(--(?:message|task|apply)) \S+")
"""A guide's placeholder after these options is a word of its language (NUMBER, НОМЕР, …)."""


def _site_words() -> dict[str, dict[str, str]]:
    text = (SITE / "i18n.js").read_text(encoding="utf-8")
    body = text.removeprefix("window.I18N = ").strip().removesuffix(";")
    words: dict[str, dict[str, str]] = json.loads(body)
    return words


def _named(language: str) -> list[str]:
    text = (DOCS / f"guide-{language}.md").read_text(encoding="utf-8")
    return re.findall(r"`(/[a-z][^`]*)`", text)


def _command(line: str) -> tuple[click.Command | None, list[str], list[str]]:
    """The CLI command a guide's line names (None: a chat command), its path and arguments."""
    words = shlex.split(PLACEHOLDER.sub(r"\1 1", line))
    if words[0] in chat._commands():
        return None, [words[0]], words[1:]
    path, rest = [words[0][1:]], words[1:]
    command = cli.commands[path[0]]
    while isinstance(command, click.Group) and rest and rest[0] in command.commands:
        path.append(rest[0])
        command = command.commands[rest.pop(0)]
    return command, path, rest


def test_every_language_of_the_site_has_its_words_and_its_guide() -> None:
    html = (SITE / "index.html").read_text(encoding="utf-8")
    assert re.findall(r'<option value="([a-z]+)"', html) == LANGUAGES
    words = _site_words()
    assert list(words) == LANGUAGES
    keys = set(re.findall(r'data-i18n="([a-z0-9_]+)"', html)) | {"copied", "copy_failed"}
    for language in LANGUAGES:
        assert set(words[language]) == keys, language
        assert (DOCS / f"guide-{language}.md").is_file(), language
    assert "blob/master/docs/guide-" in (SITE / "app.js").read_text(encoding="utf-8")


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_guide_has_the_same_sections_and_commands(language: str) -> None:
    text = (DOCS / f"guide-{language}.md").read_text(encoding="utf-8")
    assert len(re.findall(r"^## ", text, re.MULTILINE)) == 10

    def shape(line: str) -> list[str]:
        _, path, rest = _command(line)
        return path + [word for word in rest if word.startswith("-")]

    assert [shape(line) for line in _named(language)] == [shape(line) for line in _named("en")]

    def literal(name: str) -> set[str]:
        """Code that is not a chat command: install paths, uninstall commands, answers."""
        text = (DOCS / f"guide-{name}.md").read_text(encoding="utf-8")
        spans = re.findall(r"`([^`]*)`", text)
        return {PLACEHOLDER.sub(r"\1 1", span) for span in spans if not span.startswith("/")}

    assert literal("en") <= literal(language)


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_command_in_the_guide_exists(language: str) -> None:
    named = _named(language)
    assert len(named) > 10
    for line in named:
        command, path, rest = _command(line)
        if command is None:
            continue
        assert not isinstance(command, click.Group), line
        # Parses the arguments as the chat would hand them to the CLI body (no callback runs).
        command.make_context(" ".join(path), rest)
