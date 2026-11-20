"""Check quoted fragments against the text they were copied from."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator

_QUOTE = re.compile(r'«([^»]*)»|“([^”]*)”|„([^“]*)“|(?<![^\s(\[])"(?=\S)([^"\r\n{}`]+)"')
_CODE_START = re.compile(r"[`{\[]")
_JSON = json.JSONDecoder()
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
    code = iter(_code_spans(text))
    start, end = next(code, (len(text), len(text)))
    fragments = []
    for match in _QUOTE.finditer(text):
        if match[4] is not None:
            while end <= match.start():
                start, end = next(code, (len(text), len(text)))
            if start < match.end() or (match.end() < len(text) and not _edge(text[match.end()])):
                continue
        fragment = next(group for group in match.groups() if group is not None).strip()
        if len(fragment) >= 12:
            fragments.append(fragment)
    return fragments


def _code_spans(text: str) -> Iterator[tuple[int, int]]:
    end = 0
    for match in _CODE_START.finditer(text):
        start = match.start()
        if start < end:
            continue
        if match[0] == "`":
            fence = "```" if text.startswith("```", start) else "`"
            close = text.find(fence, start + len(fence))
            if close < 0:
                continue
            end = close + len(fence)
        else:
            try:
                _, end = _JSON.raw_decode(text, start)
            except (ValueError, RecursionError):
                continue
        yield start, end


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


def _script(text: str) -> str | None:
    """The writing system most of the letters of `text` belong to (LATIN, CYRILLIC, ...)."""
    scripts = Counter(
        unicodedata.name(char, "?").split(" ")[0] for char in text[:20_000] if char.isalpha()
    )
    return scripts.most_common(1)[0][0] if scripts else None


def missing(result_quotes: Iterable[str], sources: Iterable[str]) -> list[str]:
    texts = list(sources)
    normalized = [normalize(source) for source in texts]
    script = _script("".join(texts))
    absent = []
    for quote in result_quotes:
        if script is not None and _script(quote) not in (script, None):
            continue  # a title or term given in another language, not a quote from the text
        pieces = [normalize(piece) for piece in re.split(r"…|\.{3,}", quote)]
        substantial = [piece for piece in pieces if len(piece) >= 8]
        if not substantial or any(
            not any(piece in source or _dialogue(piece, source) for source in normalized)
            for piece in substantial
        ):
            absent.append(quote)
    return absent


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
DIALOGUE_GAP = 150
"""Most characters between two sentences of one quote: the author's words between replies."""


def _dialogue(piece: str, source: str) -> bool:
    """A quote of replies with the author's words between them left out, as quotes of dialogue
    are usually written: every sentence word for word, in order, each close to the one before."""
    sentences = [normalize(s) for s in _SENTENCE_END.split(piece)]
    sentences = [s for s in sentences if s]
    if len(sentences) < 2:
        return False
    start = source.find(sentences[0])
    while start >= 0:
        end = start + len(sentences[0])
        for sentence in sentences[1:]:
            found = source.find(sentence, end)
            if found < 0 or found - end > DIALOGUE_GAP:
                break
            end = found + len(sentence)
        else:
            return True
        start = source.find(sentences[0], start + 1)
    return False


def nearest(quote: str, sources: Iterable[str], *, extra: int = 40) -> str | None:
    """Where the text has the passage a quote that is not word for word seems to copy: found by
    its first words (any spacing, punctuation and case between them), so the model can see
    what the text really says ("по нем", not "по нему")."""
    words = re.findall(r"\w+", quote)
    for count in (5, 4, 3):
        if len(words) < count:
            continue
        pattern = re.compile(r"\W+".join(map(re.escape, words[:count])), re.IGNORECASE)
        for source in sources:
            found = pattern.search(source)
            if found is not None:
                passage = source[found.start() : found.start() + len(quote) + extra]
                return " ".join(passage.split())
    return None
