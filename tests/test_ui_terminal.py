"""Render the terminal chat's completion menu through prompt_toolkit."""

import pytest
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.buffer import CompletionState
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.layout.screen import Screen
from prompt_toolkit.output import DummyOutput

from desman import ui


@pytest.mark.parametrize("typed", ["/c", "/co"])
def test_completion_menu_aligns_with_input(typed: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui, "_prompt", None)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        prompt = ui._chat_prompt()
        app = prompt.app
        buffer = prompt.default_buffer
        document = Document(typed)
        completer = ui._completer({"/compact": "Summarize", "/config": "Configure"})

        def prepare() -> None:
            buffer.reset(document=document)
            buffer.complete_state = CompletionState(
                document, list(completer.get_completions(document, None))
            )

        screens: list[Screen | None] = []
        app.after_render += lambda _: screens.append(app.renderer.last_rendered_screen)
        pipe.send_text("\r")
        assert prompt.prompt(completer=completer, pre_run=prepare) == typed
        screen = screens[0]
        assert screen is not None
        input_row, first_suggestion = screen.data_buffer[0], screen.data_buffer[1]
        input_start = next(x for x, cell in input_row.items() if cell.char == "/")
        menu_start = min(
            x for x, cell in first_suggestion.items() if "completion-menu" in cell.style
        )
        assert menu_start == input_start - 1
        assert first_suggestion[input_start].char == "/"
