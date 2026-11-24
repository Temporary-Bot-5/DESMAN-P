"""Run the page dispatcher, session runners and supervised task workers."""

from __future__ import annotations

import errno
import threading
import time
import webbrowser
from collections.abc import Callable

import click

from desman import runs, ui
from desman.actions import open_user_db
from desman.config_cli import load_config, run_wizard
from desman.core import db
from desman.core.sessions import Sessions
from desman.gui.front import WebFront
from desman.gui.runners import PageChat, Runners
from desman.gui.server import Server, http_server
from desman.gui.workers import Pool
from desman.i18n import t
from desman.info import print_project_info


class ServerStartError(click.ClickException):
    """The local page server could not start."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail if detail is not None else message


def run(
    resume: bool,
    temporary: bool,
    port: int,
    open_browser: bool,
    *,
    session_id: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Keep setup and navigation on the main thread; model turns own their sessions."""
    front = WebFront()
    app = Server(front, port)
    try:
        server = http_server(app)
    except OSError as exc:
        # Windows: 10048 in use; 10013 when the port is held by another socket (or reserved).
        if exc.errno != errno.EADDRINUSE and getattr(exc, "winerror", None) not in {10048, 10013}:
            raise
        raise ServerStartError(
            t(
                "Port {port} is busy (is another `desman gui` open?). Use another --port, "
                "or leave it out to pick a free one.",
                port=port,
            ),
            detail=t("Port {port} is busy.", port=port) if port else (exc.strerror or str(exc)),
        ) from exc
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    try:
        thread.start()
    except RuntimeError as exc:
        server.server_close()
        raise ServerStartError(str(exc)) from exc
    try:
        ui.out(app.url)
        if open_browser:
            try:
                opened = webbrowser.open(app.url)
            except webbrowser.Error:
                opened = False
            if not opened:
                ui.note(t("Open this address in your browser: {url}", url=app.url))
        with ui.use(front), ui.chat_output(), front.capture():
            try:
                manager = load_config()
                if not manager.is_configured():
                    run_wizard(manager)
                with ui.greeting():
                    print_project_info(manager)
                conn = open_user_db()
                try:
                    sessions = Sessions(conn)
                    front.worker_count = runs.tuned(manager.load().tuning, "workers")
                    pool = Pool(
                        db.default_db_path(),
                        manager.config_path.parent,
                        front.worker_count,
                    )

                    runners = Runners(
                        db.default_db_path(),
                        front,
                        cap=runs.tuned(manager.load().tuning, "chat_turns"),
                    )

                    chat = PageChat(
                        sessions, runners, resume=resume, temporary=temporary, session_id=session_id
                    )

                    def idle() -> None:
                        while pool.messages:
                            text = pool.messages.popleft()
                            chat.worker_notice(text, blocked=pool.blocked)
                        chat.background_outcomes(
                            lambda id, temporary: runners.submit(id, None, temporary=temporary)
                        )

                    front.idle = idle
                    try:
                        runners.recover(sessions)
                        pool.start()
                        chat.run()
                    finally:
                        front.idle = None
                        try:
                            runners.close()
                        finally:
                            pool.stop()
                finally:
                    conn.close()
            except (KeyboardInterrupt, ui.Cancelled):
                pass
            except click.ClickException as e:  # a broken config or database: say so on the page
                ui.error(e.format_message())
                raise
            finally:
                front.end()
                sleep(0.5)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
