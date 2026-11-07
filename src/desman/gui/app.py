"""Run the page dispatcher, session runners and supervised task workers."""

from __future__ import annotations

import threading
import time
import webbrowser
from collections.abc import Callable

import click

from desman import runs, ui
from desman.config_cli import load_config, run_wizard
from desman.core import db
from desman.core.sessions import Sessions
from desman.gui.front import WebFront
from desman.gui.runners import PageChat, Runners
from desman.gui.server import Server, http_server
from desman.gui.workers import Pool
from desman.info import print_project_info


def run(
    resume: bool,
    temporary: bool,
    port: int,
    open_browser: bool,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Keep setup and navigation on the main thread; model turns own their sessions."""
    front = WebFront()
    app = Server(front, port)
    server = http_server(app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        ui.out(app.url)
        if open_browser:
            webbrowser.open(app.url)
        with ui.use(front), front.capture():
            try:
                manager = load_config()
                if not manager.is_configured():
                    run_wizard(manager)
                with ui.greeting():
                    print_project_info(manager)
                try:
                    conn = db.open_db()
                except (db.SchemaVersionError, db.SchemaMismatchError) as exc:
                    raise click.ClickException(str(exc)) from exc
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

                    chat = PageChat(sessions, runners, resume=resume, temporary=temporary)

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
