import pytest
from click.testing import CliRunner

from desman import providers, ui
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


@pytest.mark.parametrize("width", [80, 120])
def test_banner_keeps_all_ready_providers_inside_panel(
    width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = [providers.Provider(str(i), f"Provider-{i:02d}", "") for i in range(18)]
    monkeypatch.setattr(providers, "ready", lambda _: ready)
    monkeypatch.setenv("COLUMNS", str(width))
    with CliRunner().isolation() as (output, *_), ui.chat_output():
        print_project_info(open_config())
        banner = output.getvalue().decode()
    rows = [line for line in banner.splitlines() if "Provider-" in line]
    assert len(rows) > 1
    assert all(row.startswith("│") and row.endswith("│") and len(row) <= width for row in rows)
    assert all(provider.name in banner for provider in ready)
