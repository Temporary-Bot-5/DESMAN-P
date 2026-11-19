"""Manage DESMAN's local SearXNG container (ADR 0011)."""

from __future__ import annotations

import json
import secrets
import socket
import subprocess
import time
from collections.abc import Callable
from typing import Any

import click

from desman import ui
from desman.config import ConfigManager
from desman.core import search
from desman.core.sandbox import find_docker
from desman.i18n import t

IMAGE = "searxng/searxng:2026.11.15-2d500cb6b"
CONTAINER = "desman-searxng"


def docker(args: list[str], timeout: float = 60) -> str:
    """One bounded subprocess boundary; tests replace this helper."""
    try:
        result = subprocess.run(
            [find_docker() or "docker", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise click.ClickException(t("Docker failed: {error}", error=str(e))) from e
    if result.returncode:
        raise click.ClickException(
            t("Docker failed: {error}", error=result.stderr.strip() or result.stdout.strip())
        )
    return result.stdout.strip()


def available() -> None:
    if find_docker() is None:
        raise click.ClickException(
            t(
                "Install or start Docker with `desman docker install`, or use Tavily in `desman config`."
            )
        )
    try:
        docker(["info"])
    except click.ClickException as e:
        raise click.ClickException(
            str(e)
            + "\n"
            + t(
                "Install or start Docker with `desman docker install`, or use Tavily in `desman config`."
            )
        ) from e


def inspect() -> dict[str, Any] | None:
    names = docker(["ps", "-a", "--filter", f"name=^/{CONTAINER}$", "--format", "{{.Names}}"])
    if CONTAINER not in names.splitlines():
        return None
    try:
        data: dict[str, Any] = json.loads(docker(["inspect", CONTAINER]))[0]
        return data
    except (ValueError, IndexError, TypeError) as e:
        raise click.ClickException(t("Docker failed: {error}", error=str(e))) from e


def free_port(port: int | None = None, *, start: int = 8888) -> int:
    for candidate in range(port or start, (port + 1) if port else 65536):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    raise click.ClickException(t("No free local port for SearXNG."))


def setup(
    manager: ConfigManager,
    port: int | None = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    save_url: bool = True,
) -> str:
    available()
    existing = inspect()
    folder = manager.config_path.parent / "searxng"
    settings = folder / "settings.yml"
    try:
        folder.mkdir(mode=0o700, exist_ok=True)
        # JSON is also YAML; avoid a YAML dependency and quote every value safely.
        secret = secrets.token_hex(32)
        if settings.exists():
            try:
                old = json.loads(settings.read_text(encoding="utf-8"))
                secret = old.get("server", {}).get("secret_key") or secret
            except (ValueError, AttributeError):
                pass  # Repair an invalid settings file with the tested settings.
        settings.write_text(
            json.dumps(
                {
                    "use_default_settings": True,
                    "general": {"contact_url": "https://github.com/Temporary-Bot-5/DESMAN-P"},
                    "server": {"secret_key": secret, "limiter": False},
                    "search": {"formats": ["html", "json"]},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        settings.chmod(0o644)
    except OSError as e:
        raise click.ClickException(
            t(
                "Cannot read or write {file}: {error}. An older setup may have left the folder owned by another user. Remove that folder with Docker, then run `desman search setup` again: {command}",
                file=str(settings),
                error=str(e),
                command=f'docker run --rm --user root --entrypoint rm -v "{folder.parent.resolve()}:/x" {IMAGE} -rf /x/searxng',
            )
        ) from e
    bindings = existing.get("HostConfig", {}).get("PortBindings", {}) if existing else {}
    ports = bindings.get("8080/tcp") or []
    reuse = bool(
        existing
        and existing.get("Config", {}).get("Image") == IMAGE
        and len(ports) == 1
        and ports[0].get("HostIp") == "127.0.0.1"
        and (port is None or str(port) == ports[0].get("HostPort"))
        and existing.get("HostConfig", {}).get("RestartPolicy", {}).get("Name") == "unless-stopped"
        and any(
            m.get("Source") == str(settings.resolve())
            and m.get("Destination") == "/etc/searxng/settings.yml"
            and m.get("RW") is False
            for m in existing.get("Mounts", [])
        )
    )
    if reuse:
        chosen = int(ports[0]["HostPort"])
        docker(["restart", CONTAINER])
    else:
        docker(["pull", IMAGE], timeout=600)
        if existing:
            docker(["rm", "-f", "-v", CONTAINER])
        candidate = port
        for attempt in range(5):
            chosen = free_port(candidate)
            try:
                docker(
                    [
                        "run",
                        "-d",
                        "--name",
                        CONTAINER,
                        "--restart",
                        "unless-stopped",
                        "-p",
                        f"127.0.0.1:{chosen}:8080",
                        "-v",
                        f"{settings.resolve()}:/etc/searxng/settings.yml:ro",
                        IMAGE,
                    ]
                )
                break
            except click.ClickException as e:
                message = str(e).lower()
                if (
                    attempt == 4
                    or chosen == 65535
                    or not any(
                        text in message
                        for text in (
                            "port is already allocated",
                            "address already in use",
                            "port is in use",
                        )
                    )
                ):
                    raise
                # Docker may have created a stopped container before failing to bind.
                docker(["rm", "-f", "-v", CONTAINER])
                candidate = free_port(start=chosen + 1)
    url = f"http://127.0.0.1:{chosen}"
    problem: str | None = None
    with search.client() as http:
        for attempt in range(30):
            problem = search.check_searxng(url, http)
            if problem is None:
                break
            if attempt < 29:
                sleep(2)
    if problem is not None:
        raise click.ClickException(t("SearXNG did not answer: {error}", error=problem))
    if save_url:
        with manager.edit() as config:
            config.settings["SEARXNG_URL"] = url
    ui.ok(t("SearXNG: {url}", url=url))
    return url


def status(manager: ConfigManager, *, display: bool = True) -> dict[str, object]:
    container = None
    unknown = False
    try:
        available()
        container = inspect()
    except click.ClickException:
        unknown = True
    url = manager.load().settings.get("SEARXNG_URL", "")
    problem = None
    if url:
        with search.client() as http:
            problem = search.check_searxng(url, http)
    result = {
        "container": "unknown"
        if unknown
        else container.get("State", {}).get("Status")
        if container
        else None,
        "url": url,
        "answers": bool(url and problem is None),
        "error": problem,
    }
    if not display:
        return result
    ui.note(t("SearXNG container: {state}", state=result["container"] or t("Not configured")))
    if url:
        ui.note(t("SearXNG: {url}", url=url))
        if problem is None:
            ui.ok(t("SearXNG answers: web search works."))
        else:
            ui.note(t("SearXNG at {url} cannot be used yet: {problem}", url=url, problem=problem))
    else:
        ui.note(t("SearXNG URL is not configured. Run `desman search setup`."))
    return result


def operate(manager: ConfigManager, action: str) -> None:
    available()
    if action == "remove":
        existing = inspect()
        bindings = existing.get("HostConfig", {}).get("PortBindings", {}) if existing else {}
        ports = bindings.get("8080/tcp") or []
        owned_urls = {
            f"http://127.0.0.1:{p['HostPort']}" for p in ports if p.get("HostIp") == "127.0.0.1"
        }
        if existing:
            docker(["rm", "-f", "-v", CONTAINER])
        with manager.edit() as config:
            url = config.settings.get("SEARXNG_URL", "")
            if url.rstrip("/") in owned_urls:
                config.settings.pop("SEARXNG_URL", None)
    else:
        docker([action, CONTAINER])
    ui.ok(t("Done."))


def consent() -> str:
    return t("Set up SearXNG on this computer (Docker, about 100 MB)?")
