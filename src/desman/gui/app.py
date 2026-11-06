"""Run the synchronous chat with the browser front described in ADR 0005."""

from __future__ import annotations

import threading
import time
import webbrowser
from collections.abc import Callable

import click

from desman import ui
from desman.chat import Chat
from desman.config_cli import load_config, run_wizard
from desman.core import db
from desman.core.sessions import Sessions
from desman.gui.front import WebFront
from desman.gui.server import Server, http_server
from desman.info import print_project_info


def run(
    resume: bool,
    temporary: bool,
    port: int,
    open_browser: bool,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run setup and chat on the main thread so Stop follows normal cancellation."""
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
                    Chat(Sessions(conn), resume=resume, temporary=temporary).run()
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
