"""T.6i: findings from using DESMAN as a person (fifth wave)."""

from __future__ import annotations

from collections.abc import Iterator

import click
import pytest

from desman import i18n, ui


@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    given: list[str] = []
    monkeypatch.setattr(ui, "interactive", lambda: False)
    monkeypatch.setattr(click, "prompt", lambda *a, **k: given.pop(0))
    i18n.set_language("en")
    yield given
    i18n.set_language("en")


@pytest.mark.parametrize("word", ["да", "Да", "ja", "oui", "sí", "tak", "evet", "так", "是"])
def test_yes_in_the_persons_language_with_an_english_interface(
    answers: list[str], word: str
) -> None:
    answers.append(word)
    assert ui._terminal_confirm("Apply?") is True


@pytest.mark.parametrize("word", ["нет", "nein", "non", "nie", "hayır", "HAYIR", "ні", "否"])
def test_no_in_the_persons_language_with_an_english_interface(
    answers: list[str], word: str
) -> None:
    answers.append(word)
    assert ui._terminal_confirm("Apply?", default=True) is False


def test_no_word_means_yes_in_one_language_and_no_in_another() -> None:
    assert not ui._every_language("yes") & ui._every_language("no")


def test_first_letters_only_of_the_interface_language(answers: list[str]) -> None:
    answers.extend(["д", "y"])  # "д" is not an English answer: asked again
    assert ui._terminal_confirm("Apply?") is True
    assert answers == []
