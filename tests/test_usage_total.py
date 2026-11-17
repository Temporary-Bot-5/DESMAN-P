"""F72: window totals in printed usage and the chat's command event."""

from __future__ import annotations

import pytest

from desman import config_cli
from desman.config import Config
from desman.core.calls import KeyUsage
from desman.i18n import set_language


def rows(failed: int = 3, limited: int = 1) -> list[KeyUsage]:
    return [
        KeyUsage("groq", "old", 21, failed, limited, 120000, 8900, None, None, {}),
        KeyUsage("nvidia", "other", 21, 0, 0, 345, 10, None, None, {}),
    ]


def test_usage_total(capsys: pytest.CaptureFixture[str]) -> None:
    config_cli.print_usage(Config(), rows(), None, "all")
    text = " ".join(capsys.readouterr().out.split())
    assert (
        "Total: 42 calls (3 failed, 1 limited), 120,345 tokens in, 8,910 tokens out, 2 providers."
        in text
    )
    assert text.index("Total:") < text.index('"removed":')


def test_usage_total_without_failures(capsys: pytest.CaptureFixture[str]) -> None:
    config_cli.print_usage(Config(), rows(0, 0), None, "all")
    text = " ".join(capsys.readouterr().out.split())
    assert "Total: 42 calls, 120,345 tokens in, 8,910 tokens out, 2 providers." in text


def test_empty_usage(capsys: pytest.CaptureFixture[str]) -> None:
    config_cli.print_usage(Config(), [], None, "all")
    text = capsys.readouterr().out
    assert "No model calls yet." in text and "Total:" not in text


@pytest.mark.parametrize(("count", "word"), [(1, "вызов"), (2, "вызова"), (5, "вызовов")])
def test_russian_usage_plural(count: int, word: str, capsys: pytest.CaptureFixture[str]) -> None:
    set_language("ru")
    try:
        config_cli.print_usage(
            Config(), [KeyUsage("groq", None, count, 0, 0, 1, 2, None, None, {})], None, "all"
        )
        text = " ".join(capsys.readouterr().out.split())
        assert f"Итого: {count} {word}," in text
        assert "1 провайдер." in text
    finally:
        set_language("en")


def test_totals_count_providers_once_and_skip_unused_rows() -> None:
    used = rows(0, 0)
    used.extend(
        [
            KeyUsage("groq", "second", 1, 0, 0, 10, 5, None, None, {}),
            KeyUsage("unused", "idle", 0, 0, 0, 0, 0, None, None, {}),
        ]
    )
    assert config_cli.usage_totals(used) == {
        "calls": 43,
        "failed": 0,
        "limited": 0,
        "prompt_tokens": 120355,
        "completion_tokens": 8915,
        "providers": 2,
    }
