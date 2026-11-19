"""Check quoted fragments against the text they were copied from."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator

_QUOTE = re.compile(r'«([^»]*)»|“([^”]*)”|„([^“]*)“|"([^"]*)"')
_TRANSLATION = str.maketrans({"ё": "е", "\u0301": None, **dict.fromkeys('«»“”„"‘’‚‹›')})


def string_values(value: object) -> Iterator[str]:
    """Yield the text of each JSON string value without JSON delimiters or escapes."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from string_values(item)


def quotes(text: str) -> list[str]:
    """Return quoted fragments of at least twelve characters, in text order."""
    fragments = (
        next(group for group in match.groups() if group is not None).strip()
        for match in _QUOTE.finditer(text)
    )
    return [fragment for fragment in fragments if len(fragment) >= 12]


def normalize(s: str) -> str:
    text = s.casefold().translate(_TRANSLATION)
    text = " ".join(re.sub(r"[—–‑-]+", "-", text).split())
    start, end = 0, len(text)
    while start < end and _edge(text[start]):
        start += 1
    while end > start and _edge(text[end - 1]):
        end -= 1
    return text[start:end]


def _edge(char: str) -> bool:
    return char.isspace() or unicodedata.category(char).startswith("P")


def missing(result_quotes: Iterable[str], sources: Iterable[str]) -> list[str]:
    normalized = [normalize(source) for source in sources]
    absent = []
    for quote in result_quotes:
        pieces = [normalize(piece) for piece in re.split(r"…|\.{3,}", quote)]
        substantial = [piece for piece in pieces if len(piece) >= 8]
        if not substantial or any(
            not any(piece in source for source in normalized) for piece in substantial
        ):
            absent.append(quote)
    return absent
