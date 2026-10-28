"""Tests for translations: every message has one, the CLI follows the configured language."""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import json
import string
import subprocess
import sys
from pathlib import Path
from typing import get_args

import pytest
from click.testing import CliRunner, Result

from desman import i18n, providers
from desman.cli import cli
from desman.config import Config, open_config
from desman.locales import CLICK_DEVELOPER_ONLY, ru

SOURCE = Path(__file__).parent.parent / "src" / "desman"


@functools.cache
def source_messages() -> dict[str, str | None]:
    """Every message the source passes to t() or tn() (singular → plural), plus command help."""
    found: dict[str, str | None] = {}
    for path in sorted(SOURCE.rglob("*.py")):
        if path.name == "i18n.py":  # translates help texts collected below, not literals
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            args = node.args
            if node.func.id == "t":
                message = args[0]
                assert isinstance(message, ast.Constant) and isinstance(message.value, str), (
                    f"{path.name}:{node.lineno}: t(literal)"
                )
                found[message.value] = None
            elif node.func.id == "tn":
                singular, plural = args[1], args[2]
                assert (
                    isinstance(singular, ast.Constant)
                    and isinstance(singular.value, str)
                    and isinstance(plural, ast.Constant)
                    and isinstance(plural.value, str)
                ), f"{path.name}:{node.lineno}: tn(n, literal, literal)"
                found[singular.value] = plural.value
    for obj, attr in i18n._help_fields(cli):
        text = i18n._ORIGINAL.get((id(obj), attr), getattr(obj, attr))
        if text:
            found.setdefault(i18n.help_key(text), None)
    return found


def fields(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def desman(*args: str) -> Result:
    return CliRunner().invoke(cli, list(args))


def configure(language: str) -> None:
    open_config().save(Config(language=language, configured=True))  # type: ignore[arg-type]


# ── The catalog ──────────────────────────────────────────────────────────────


TRANSLATED = {code: c for code, c in i18n.CATALOGS.items() if code != "en"}


def test_every_language_is_a_config_choice() -> None:
    assert set(get_args(Config.model_fields["language"].annotation)) == set(i18n.CATALOGS)


@pytest.mark.parametrize("code", TRANSLATED)
def test_every_message_has_a_translation(code: str) -> None:
    assert sorted(set(source_messages()) - set(TRANSLATED[code].messages)) == []


@pytest.mark.parametrize("code", TRANSLATED)
def test_no_translation_is_left_over(code: str) -> None:
    assert sorted(set(TRANSLATED[code].messages) - set(source_messages())) == []


@functools.cache
def click_messages() -> dict[str, str | None]:
    """What the installed Click passes to gettext/ngettext in the modules i18n patches."""
    found: dict[str, str | None] = {}
    for name in i18n.CLICK_MODULES:
        tree = ast.parse(inspect.getsource(importlib.import_module(name)))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            args = [a.value if isinstance(a, ast.Constant) else None for a in node.args]
            if node.func.id == "_" and args and isinstance(args[0], str):
                found[args[0]] = None
            elif node.func.id == "ngettext" and len(args) > 1 and isinstance(args[0], str):
                found[args[0]] = args[1] if isinstance(args[1], str) else None
    return found


@pytest.mark.parametrize("code", TRANSLATED)
@pytest.mark.parametrize("which", ["messages", "click"])
def test_translations_keep_the_fields(code: str, which: str) -> None:
    catalog = TRANSLATED[code]
    plural_of = source_messages() if which == "messages" else click_messages()
    for source, translated in getattr(catalog, which).items():
        forms = (translated,) if isinstance(translated, str) else translated
        expected = fields(source) | fields(plural_of.get(source) or "")
        for form in forms:
            # A form may leave a field out ("pick 1" needs no {n}), never invent one; a
            # language with one form uses the plural's fields (Click passes both).
            assert fields(form) <= expected | {"n"}, source


@pytest.mark.parametrize("code", TRANSLATED)
def test_everything_click_says_is_translated(code: str) -> None:
    # Fails when a Click release adds or rewords a message: translate the new text (keep the
    # old one too, older Click versions still use it). Mistakes in command definitions are
    # left to the reference catalog (ru).
    needed = set(click_messages())
    if code != "ru":
        needed -= CLICK_DEVELOPER_ONLY
    assert sorted(needed - set(TRANSLATED[code].click)) == []


@pytest.mark.parametrize("code", TRANSLATED)
def test_plural_messages_have_every_form_of_the_language(code: str) -> None:
    catalog = TRANSLATED[code]
    plurals = {**source_messages(), **click_messages()}
    for table in (catalog.messages, catalog.click):
        for singular, translated in table.items():
            if plurals.get(singular) is not None:
                assert isinstance(translated, tuple), (code, singular)
                assert len(translated) == catalog.forms, (code, singular)
            else:
                assert isinstance(translated, str), (code, singular)


@pytest.mark.parametrize("code", TRANSLATED)
def test_plural_rule_picks_an_existing_form(code: str) -> None:
    catalog = TRANSLATED[code]
    assert {catalog.plural(n) for n in range(200)} == set(range(catalog.forms))


@pytest.mark.parametrize("code", TRANSLATED)
def test_no_translation_is_empty(code: str) -> None:
    catalog = TRANSLATED[code]
    for table in (catalog.messages, catalog.click):
        for source, translated in table.items():
            forms = (translated,) if isinstance(translated, str) else translated
            assert all(form.strip() for form in forms), (code, source)


@pytest.mark.parametrize(
    ("n", "form"),
    [
        (1, 0),
        (2, 1),
        (4, 1),
        (5, 2),
        (11, 2),
        (12, 2),
        (14, 2),
        (21, 0),
        (22, 1),
        (25, 2),
        (111, 2),
        (0, 2),
    ],
)
def test_russian_plural_rule(n: int, form: int) -> None:
    assert ru.plural(n) == form


def test_plural_and_fallback() -> None:
    i18n.set_language("ru")
    assert i18n.tn(21, "{n} key", "{n} keys") == "21 ключ"
    assert i18n.tn(3, "{n} key", "{n} keys") == "3 ключа"
    assert i18n.tn(11, "{n} key", "{n} keys") == "11 ключей"
    assert i18n.t("not in any catalog {x}", x=1) == "not in any catalog 1"  # English, not a key
    i18n.set_language("xx")  # unknown: English
    assert i18n.tn(1, "{n} key", "{n} keys") == "1 key"
    assert i18n.language() == "en"


# ── The CLI ──────────────────────────────────────────────────────────────────


def test_help_follows_the_configured_language() -> None:
    configure("ru")
    help_ru = desman("--help").output
    assert "Использование:" in help_ru and "Опции:" in help_ru and "Команды:" in help_ru
    assert "Показать эту справку и выйти." in help_ru  # Click's own text
    assert "Список задач, новые сверху." in help_ru  # a command's short help
    assert "Вывод для программ (JSON)." in desman("list", "--help").output  # an option
    configure("en")
    help_en = desman("--help").output
    assert "Usage:" in help_en and "List tasks, newest first." in help_en
    assert "Список" not in help_en


def test_click_errors_and_messages_are_translated() -> None:
    configure("ru")
    result = desman("statuss")
    assert result.exit_code == 2
    assert "Ошибка: Нет такой команды 'statuss'. Может быть, 'status'?" in result.output
    assert "укажите либо GOAL, либо --spec" in desman("run").output
    assert "нет базы данных" in desman("list").output
    cloud = len(providers.CLOUD)
    assert f"0 ключей у 0 из {cloud} провайдеров" in desman("keys").output


def test_json_output_does_not_depend_on_the_language() -> None:
    configure("en")
    english = json.loads(desman("config", "show", "--json").output)
    configure("ru")
    russian = json.loads(desman("config", "show", "--json").output)
    assert {**russian, "language": "en"} == english


def test_a_broken_config_shows_english_and_the_error() -> None:
    path = open_config().config_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert "Usage:" in desman("--help").output
    result = desman("keys")
    assert result.exit_code == 1 and "desman config reset" in result.output


def test_the_wizard_switches_the_language_for_the_rest_of_the_command() -> None:
    # Language 2 (Русский), providers menu: Done, then the four permission questions.
    done = str(len(providers.PROVIDERS) + 1)
    result = CliRunner().invoke(cli, ["config"], input=f"2\n{done}\n\n\n\n\n")
    assert result.exit_code == 0, result.output
    assert "Настройки сохранены в" in result.output
    assert "DESMAN(P) — Configuration" in result.output  # asked in English before the choice


def test_version_follows_the_language() -> None:
    configure("ru")
    assert ", версия " in desman("--version").output
    configure("en")
    assert ", version " in desman("--version").output


def test_bad_options_and_choices_are_reported_in_russian() -> None:
    configure("ru")
    bogus = desman("--bogus").output
    assert "Ошибка: Нет такой опции" in bogus and "No such" not in bogus
    choice = desman("run", "x", "--priority", "invalid").output
    assert "Недопустимое значение" in choice and "'invalid' — не одно из:" in choice


@pytest.mark.parametrize("module", ["desman.locales.ru", "desman.i18n", "desman.cli"])
def test_each_module_imports_first_in_a_fresh_process(module: str) -> None:
    subprocess.run([sys.executable, "-c", f"import {module}"], check=True, timeout=60)
