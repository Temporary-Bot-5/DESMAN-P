"""Authenticated loopback transport for ADR 0005; request handling needs no sockets."""

from __future__ import annotations

import hmac
import json
import secrets
import sqlite3
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from socket import socket
from urllib.parse import parse_qs, urlsplit

from pydantic import ValidationError

from desman import ui
from desman.config import ConfigError
from desman.core import db
from desman.gui.front import WebFront
from desman.gui.panels import snapshot
from desman.i18n import language, t

MAX_BODY = 64 * 1024
SECURITY = {
    "Content-Security-Policy": "default-src 'self'; style-src 'self'; style-src-attr 'unsafe-inline'; frame-ancestors 'none'",
    "X-Frame-Options": "DENY",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
}


def strings() -> dict[str, str]:
    return {
        "lang": language(),
        "settings_general": t("General"),
        "settings_keys": t("Keys"),
        "settings_roles": t("Role models"),
        "other_providers": t("Other providers"),
        "get_key": t("Get a key"),
        "advanced": t("Advanced"),
        "saved": t("Saved"),
        "settings_permissions": t("Permissions"),
        "settings_personalize": t("Personalisation"),
        "settings_behaviour": t("Conversation"),
        "download": t("Download"),
        "settings_limits": t("Work limits"),
        "settings_models": t("Local models"),
        "settings_network": t("desman-global"),
        "settings_protected": t("Protected files"),
        "save": t("Save"),
        "close": t("Close"),
        "language": t("Language"),
        "setup_wizard": t("Run the setup wizard"),
        "missing": t("missing"),
        "provider_site": t("Provider website"),
        "key_label": t("Key label"),
        "add_key": t("Add key"),
        "check": t("Check"),
        "forget": t("Forget"),
        "reset": t("Reset"),
        "provider": t("provider"),
        "show": t("Show"),
        "add": t("Add"),
        "approve": t("Approve"),
        "off": t("off"),
        "registry": t("Registry"),
        "join": t("Join"),
        "leave": t("Leave"),
        "calls_daily": t("Calls per day"),
        "tokens_daily": t("Tokens per day"),
        "lend": t("Lend"),
        "unlend": t("Take back"),
        "change": t("Change"),
        "only_here": t("Only here"),
        "scope_unavailable": t(
            "This role does not read choices here: choose it for the role everywhere."
        ),
        "everywhere": t("For the role everywhere"),
        "apply": t("Apply"),
        "scope": t("Scope"),
        "auto": t("Automatic"),
        "model": t("Model"),
        "key": t("Key"),
        "source_task": t("Task"),
        "source_chat": t("Chat"),
        "source_role": t("Role"),
        "source_auto": t("Automatic"),
        "role_text": t("Executor (text)"),
        "role_code": t("Executor (code)"),
        "role_analysis": t("Executor (analysis)"),
        "terminal_input": t("Type a command here"),
        "pinned": t("Pinned"),
        "modules": t("Modules"),
        "model_usage": t("This model: {calls} calls · {tokens} tokens"),
        "role_usage_task": t("Role totals in this task: {calls} calls · {tokens} tokens"),
        "role_usage_chat": t("Role totals in this chat: {calls} calls · {tokens} tokens"),
        "calls": t("Calls"),
        "tokens": t("Tokens"),
        "status_pending": t("Pending"),
        "status_ready": t("Ready"),
        "status_running": t("Running"),
        "status_review": t("Review"),
        "status_done": t("Done"),
        "status_paused": t("Paused"),
        "status_needs_human": t("Needs human"),
        "status_failed": t("Failed"),
        "status_blocked": t("Blocked"),
        "status_cancelled": t("Cancelled"),
        "status_active": t("Active"),
        "role_planner": t("Planner"),
        "role_executor": t("Executor"),
        "role_reviewer": t("Reviewer"),
        "role_decomposer": t("Decomposer"),
        "role_interlocutor": t("Interlocutor"),
        "send": t("Send"),
        "new_chat": t("New chat"),
        "temporary_chat": t("Temporary chat"),
        "delete_chats": t("Delete all chats"),
        "settings": t("Settings"),
        "messenger": t("Messenger"),
        "report_problem": t("Report a problem"),
        "sessions": t("Chats"),
        "tasks": t("Tasks"),
        "pin": t("Pin"),
        "unpin": t("Unpin"),
        "delete": t("Delete"),
        "answer": t("Your answer"),
        "retry": t("Retry with this answer"),
        "accept": t("Accept as is"),
        "fail": t("Fail"),
        "wait": t("Available when the chat is ready for your next message."),
        "unavailable": t("This session is no longer available."),
        "cancel": t("Cancel"),
        "yes": t("yes"),
        "no": t("no"),
        "ok": t("OK"),
        "stop": t("Stop"),
        "ended": t("DESMAN finished, you can close this tab"),
        "offline": t("DESMAN is not running"),
        "open": t("Open the page with desman gui in a terminal."),
        "invalid": t("Please check your answer and try again."),
    }


@dataclass
class Response:
    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.headers = {**SECURITY, **self.headers}


class Server:
    """Authenticate local browser requests before exposing the synchronous UI front."""

    def __init__(
        self, front: WebFront, port: int, *, poll_timeout: float = 25, database: Path | None = None
    ) -> None:
        self.database = database or db.default_db_path()
        self.front = front
        self.port = port
        self.token = secrets.token_urlsafe(32)
        self.cookie = secrets.token_urlsafe(32)
        self.poll_timeout = min(max(poll_timeout, 0), 25)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/?token={self.token}"

    def handle(
        self, method: str, path: str, headers: Mapping[str, str], body: bytes = b""
    ) -> Response:
        """Validate transport input before serving assets, state or human answers."""
        headers = {k.lower(): v for k, v in headers.items()}
        host = headers.get("host", "")
        if host not in {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}:
            return Response(403)
        url = urlsplit(path)
        query = parse_qs(url.query)
        if method == "GET" and url.path == "/" and "token" in query:
            if len(query["token"]) != 1 or not hmac.compare_digest(
                query["token"][0].encode(), self.token.encode()
            ):
                return Response(403)
            return Response(
                303,
                headers={
                    "Location": "/",
                    "Set-Cookie": f"desman={self.cookie}; HttpOnly; SameSite=Strict; Path=/",
                },
            )
        try:
            cookies = SimpleCookie()
            cookies.load(headers.get("cookie", ""))
            cookie = cookies.get("desman")
            authenticated = cookie is not None and hmac.compare_digest(
                cookie.value.encode(), self.cookie.encode()
            )
        except CookieError:
            authenticated = False
        if not authenticated:
            return Response(403)
        if method == "POST":
            if headers.get("origin") != f"http://{host}":
                return Response(403)
            if (
                headers.get("content-type", "").split(";", 1)[0].strip().lower()
                != "application/json"
            ):
                return Response(415)
            if len(body) > MAX_BODY:
                return Response(413)
            try:
                data = json.loads(body)
            except (ValueError, UnicodeError, RecursionError):
                return Response(400)
            if not isinstance(data, dict):
                return Response(400)
            if url.path == "/action":
                try:
                    action = ui.ACTIONS.validate_python(data)
                except ValidationError:
                    return Response(400)
                return Response(200 if self.front.action(action) else 409)
            if url.path == "/answer":
                return Response(200 if self.front.answer(data) else 400)
            if url.path == "/stop":
                self.front.stop()
                return Response(200)
        if method == "GET":
            assets = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/app.css": ("app.css", "text/css; charset=utf-8"),
            }
            for name in ("logo", "banner", "module"):
                assets[f"/img/{name}.png"] = (f"img/{name}.png", "image/png")
            if url.path in assets:
                name, content_type = assets[url.path]
                return Response(
                    200,
                    files("desman.gui").joinpath("static", name).read_bytes(),
                    {"Content-Type": content_type},
                )
            if url.path in {"/sessions", "/tasks", "/modules", "/settings"}:
                try:
                    return self._json(snapshot(self.database, url.path[1:], self.front))
                except (sqlite3.Error, ConfigError, OSError):
                    return Response(503)
            if url.path == "/strings.json":
                return self._json(strings())
            if url.path == "/events":
                try:
                    after = int(query.get("after", ["0"])[0])
                    state = int(query["state"][0]) if "state" in query else None
                    if after < 0:
                        return Response(400)
                except ValueError:
                    return Response(400)
                return self._json(self.front.poll(after, self.poll_timeout, state))
        return Response(404)

    @staticmethod
    def _json(data: object) -> Response:
        return Response(
            200,
            json.dumps(data, ensure_ascii=False).encode("utf-8"),
            {"Content-Type": "application/json; charset=utf-8"},
        )


class QuietHTTPServer(ThreadingHTTPServer):
    """Keep transport failures out of the chat's captured stdout and stderr."""

    def handle_error(self, request: socket | tuple[bytes, socket], client_address: object) -> None:
        # ADR 0005: no server logs, including tracebacks and request details.
        pass


def http_server(app: Server) -> ThreadingHTTPServer:
    """Bind only loopback and adapt HTTP requests to the authenticated handler."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            self.respond()

        do_POST = do_GET
        do_HEAD = do_GET
        do_OPTIONS = do_GET
        do_PUT = do_GET
        do_DELETE = do_GET
        do_PATCH = do_GET

        def end_headers(self) -> None:
            # Also cover errors produced by the stdlib request parser.
            for key, value in SECURITY.items():
                self.send_header(key, value)
            super().end_headers()

        def handle(self) -> None:
            # Reloading or closing a tab can abandon a pending long poll.
            with suppress(BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                super().handle()

        def respond(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length < 0:
                response = Response(400)
            elif length > MAX_BODY:
                response = Response(413)
            else:
                response = app.handle(
                    self.command,
                    self.path,
                    dict(self.headers.items()),
                    self.rfile.read(length) if length else b"",
                )
            self.send_response(response.status)
            for key, value in response.headers.items():
                if key not in SECURITY:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(response.body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(response.body)

    server = QuietHTTPServer(("127.0.0.1", app.port), Handler)
    app.port = server.server_port
    return server
