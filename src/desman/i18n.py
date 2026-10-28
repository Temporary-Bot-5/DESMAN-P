"""Everything a person reads, in the language chosen in `desman config`.

The English text in the source is the key: `t("Task {id} created", id=...)` looks it up in the
current language's catalog (`desman/locales/<code>.py`) and falls back to the English text, so
a missing translation shows English, never a key. Plurals go through `tn()` with the catalog's
plural rule (Russian has three forms).

Translated, besides our own messages:
- command and option help (`translate_commands()`, applied when the CLI starts, before Click
  parses anything, so `--help` is translated too);
- Click's own text ("Usage:", "Options", "Missing argument"...): Click asks `gettext` for it,
  and `install_click()` points those lookups at our catalog.

Not translated: `--json` output, the audit log and model prompts. They are data, read by
programs and by models, and stay the same whatever the operator's language.

Adding a language: a `locales/<code>.py` with `CATALOG`, its code in `CATALOGS` and in
`Config.language`. A test checks that every message in the source has a translation.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Iterator

import click

from desman.locales import Catalog, Forms, english_plural, ru

__all__ = ["CATALOGS", "Catalog", "Forms", "language", "set_language", "t", "tn"]

ENGLISH = Catalog("English", english_plural, {})


CATALOGS: dict[str, Catalog] = {"en": ENGLISH, "ru": ru.CATALOG}
_current: Catalog = ENGLISH


def set_language(code: str) -> None:
    """Switch every later message; an unknown code means English."""
    global _current
    _current = CATALOGS.get(code, ENGLISH)


def language() -> str:
    return next(code for code, catalog in CATALOGS.items() if catalog is _current)


def t(message: str, /, **fields: object) -> str:
    """`message` in the current language, with `{fields}` filled in."""
    translated = _current.messages.get(message, message)
    text = translated if isinstance(translated, str) else translated[0]
    return text.format(**fields) if fields else text


def tn(n: int, singular: str, plural: str, /, **fields: object) -> str:
    """The form of a message that fits count `n` (available as `{n}`)."""
    translated = _current.messages.get(singular)
    if translated is None:
        text = singular if n == 1 else plural
    else:
        forms = (translated,) if isinstance(translated, str) else translated
        text = forms[min(_current.plural(n), len(forms) - 1)]
    return text.format(n=n, **fields)


# ── Command help ─────────────────────────────────────────────────────────────

_ORIGINAL: dict[tuple[int, str], str | None] = {}
"""English help of every translated command and option, so switching back works."""


def translate_commands(root: click.Command) -> None:
    """Put the current language's help texts into the command tree under `root`."""
    for obj, attr in _help_fields(root):
        key = (id(obj), attr)
        if key not in _ORIGINAL:
            _ORIGINAL[key] = getattr(obj, attr)
        original = _ORIGINAL[key]
        setattr(obj, attr, None if original is None else t(help_key(original)))
        if isinstance(obj, click.Command) and hasattr(obj, "_help_option"):
            # Click builds --help once per command, with the text of the language it was
            # built in; drop it, so the next help page builds it in the current one.
            obj._help_option = None


def help_key(text: str) -> str:
    """Docstrings are indented: the catalog holds them as Click shows them."""
    return inspect.cleandoc(text)


def _help_fields(root: click.Command) -> Iterator[tuple[object, str]]:
    seen: set[int] = set()
    stack = [root]
    while stack:
        command = stack.pop()
        if id(command) in seen:
            continue
        seen.add(id(command))
        yield from ((command, attr) for attr in ("help", "short_help", "epilog"))
        yield from ((p, "help") for p in command.params if isinstance(p, click.Option))
        if isinstance(command, click.Group):
            stack.extend(command.commands.values())


# ── Click's own messages ─────────────────────────────────────────────────────

CLICK_MODULES = (
    "click.core",
    "click.decorators",
    "click.exceptions",
    "click.formatting",
    "click.parser",
    "click.termui",
    "click.types",
    "click.utils",
)


def _click_gettext(message: str) -> str:
    translated = _current.click.get(message, message)
    return translated if isinstance(translated, str) else translated[0]


def _click_ngettext(singular: str, plural: str, n: int) -> str:
    translated = _current.click.get(singular)
    if translated is None:
        return singular if n == 1 else plural
    forms = (translated,) if isinstance(translated, str) else translated
    return forms[min(_current.plural(n), len(forms) - 1)]


def install_click() -> None:
    """Point Click's message lookups at our catalogs.

    Click translates through `gettext` (`from gettext import gettext as _` in each module), which
    only reads compiled .mo files chosen by environment variables. Rebinding those module names
    is the one way to feed it a catalog from the config. A Click release that renames them
    leaves its text in English; `test_i18n` notices.
    """
    for name in CLICK_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        if hasattr(module, "_"):
            module._ = _click_gettext  # type: ignore[attr-defined]
        if hasattr(module, "ngettext"):
            module.ngettext = _click_ngettext  # type: ignore[attr-defined]


install_click()
