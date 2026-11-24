from click.testing import CliRunner

from desman import ui
from desman.config import open_config
from desman.info import print_project_info


def test_banner_uses_commands_for_the_current_interface() -> None:
    runner = CliRunner()
    with runner.isolation() as (output, *_):
        print_project_info(open_config())
        cli = output.getvalue().decode()
    with runner.isolation() as (output, *_), ui.chat_output():
        print_project_info(open_config())
        chat = output.getvalue().decode()

    assert "desman -c / -t" in cli and "desman list / status" in cli
    assert "/sessions / /temporary" in chat and "/list / /status" in chat
    assert '/run "GOAL"' in chat
