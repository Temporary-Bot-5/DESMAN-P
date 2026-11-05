"""Authenticated loopback transport for ADR 0005; request handling needs no sockets."""

from __future__ import annotations

import hmac
import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import parse_qs, urlsplit

from desman.gui.front import WebFront
from desman.i18n import t

MAX_BODY = 64 * 1024
SECURITY = {
    "Content-Security-Policy": "default-src 'self'; style-src 'self'; style-src-attr 'unsafe-inline'; frame-ancestors 'none'",
    "X-Frame-Options": "DENY",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
}


def strings() -> dict[str, str]:
    return {
        "send": t("Send"),
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

    def __init__(self, front: WebFront, port: int, *, poll_timeout: float = 25) -> None:
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
            if url.path in assets:
                name, content_type = assets[url.path]
                return Response(
                    200,
                    files("desman.gui").joinpath("static", name).read_bytes(),
                    {"Content-Type": content_type},
                )
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

    server = ThreadingHTTPServer(("127.0.0.1", app.port), Handler)
    app.port = server.server_port
    return server
