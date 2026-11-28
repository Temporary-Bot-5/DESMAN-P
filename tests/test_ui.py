import pytest

from desman import ui


@pytest.mark.parametrize(
    ("hint", "shown"),
    [
        ('Try `desman run "GOAL"`.', 'Try `/run "GOAL"`.'),
        (
            "Try desman resume TASK after the limit resets.",
            "Try /resume TASK after the limit resets.",
        ),
        ("See one: desman proposals show NUMBER", "See one: /proposals show NUMBER"),
        ("Use `desman global join --registry URL`.", "Use `/global join --registry URL`."),
        (
            "Try `desman global report I can't open the page`.",
            "Try `/global report I can't open the page`.",
        ),
        ("Try `desman run --provider openai`.", None),
        ('Try `desman run "GOAL" --provider openai`.', None),
        ("Try desman run --provider openai.", None),
        ('Try desman run "GOAL" --provider openai.', None),
        ("Try desman resume --wait TASK.", None),
        ("Try `desman gui --port 1234`.", None),
        ("Try ``desman list``.", "Try ``/list``."),
        ("Try desman list|status.", None),
        ("Try desman protected approve [--note TEXT].", None),
        (
            "> ```sh\n> desman list\n> ```\nTry desman list.",
            "> ```sh\n> desman list\n> ```\nTry /list.",
        ),
        ("Try `desman list --unknown`.", None),
        ("Try `desman status`.", None),
        ("Try `desman worker` or desman worker.", None),
        ("Try `desman worker --help`.", None),
        ("```sh\ndesman list\n```\nTry `desman list`.", "```sh\ndesman list\n```\nTry `/list`."),
        ("~~~~ sh\ndesman list\n~~~\ndesman list\n~~~~", None),
    ],
)
def test_chat_rewrites_only_runnable_command_hints(hint: str, shown: str | None) -> None:
    assert ui.chat_hint(hint) == hint
    with ui.chat_output():
        assert ui.chat_hint(hint) == (shown or hint)
