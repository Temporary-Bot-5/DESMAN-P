"""Translations of DESMAN's messages, one module per language (see desman.i18n).

`Catalog` lives here, not in `desman.i18n`, so a catalog module can be imported on its own:
i18n imports the catalogs, the catalogs import nothing of i18n.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

Forms = str | tuple[str, ...]


@dataclass(frozen=True)
class Catalog:
    name: str
    """The language's own name, for the language menu."""
    plural: Callable[[int], int]
    """Which plural form (index into a tuple in `messages`) a count takes."""
    messages: Mapping[str, Forms]
    """Our messages: English text (singular form for plurals) → translation."""
    click: Mapping[str, Forms] = field(default_factory=dict)
    """Click's own messages, by Click's English text."""


def english_plural(n: int) -> int:
    return 0 if n == 1 else 1
