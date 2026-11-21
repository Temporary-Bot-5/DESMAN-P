"""Manage DESMAN's local SearXNG installation (ADRs 0011 and 0013)."""

from __future__ import annotations

import json
import logging
import secrets
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import click
import httpx

from desman import searxng_native as native
from desman import ui
from desman.config import Config, ConfigManager
from desman.core import search
from desman.core.sandbox import find_docker
from desman.i18n import t

IMAGE = "searxng/searxng:2026.11.15-2d500cb6b"
CONTAINER = "desman-searxng"
DOCKER_URL = "SEARXNG_DOCKER_URL"
START_RETRY_SECONDS = 600


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


def available(*, timeout: float = 5) -> None:
    if find_docker() is None:
        raise click.ClickException(
            t(
                "Install or start Docker with `desman docker install`, or use Tavily in `desman config`."
            )
        )
    try:
        docker(["info"], timeout=timeout)
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
    force_native: bool = False,
    save_url: bool = True,
    staged: Config | None = None,
) -> str:
    use_docker = not force_native and find_docker() is not None
    folder = native.directory()
    if use_docker:
        try:
            available(timeout=60)
        except click.ClickException as exc:
            raise click.ClickException(f"{exc}\n{_docker_setup_hint()}") from exc
        if _native_installed(folder):
            with native.locked(folder):
                native.stop(folder, sleep=sleep)
                url = _setup_docker(manager, port, sleep)
        else:
            url = _setup_docker(manager, port, sleep)
    else:
        config = staged if staged is not None else manager.load()
        if not force_native and config.settings.get(DOCKER_URL):
            raise click.ClickException(_docker_setup_hint())
        try:
            url = native.setup(folder, port, free_port, sleep=sleep)
        except (OSError, httpx.HTTPError, subprocess.TimeoutExpired, tarfile.TarError) as e:
            raise click.ClickException(t("Native SearXNG failed: {error}", error=str(e))) from e
    if staged is not None:
        _save_url(staged, url, use_docker)
    elif save_url:
        with manager.edit() as config:
            _save_url(config, url, use_docker)
    ui.ok(t("SearXNG: {url}", url=url))
    return url


def _docker_setup_hint() -> str:
    return t("Start Docker and run setup again, or use `desman search setup --native`.")


def _save_url(config: Config, url: str, use_docker: bool) -> None:
    config.settings["SEARXNG_URL"] = url
    if use_docker:
        config.settings[DOCKER_URL] = url
    elif config.settings.get(DOCKER_URL) == url:
        config.settings.pop(DOCKER_URL)


def uses_docker(config: Config) -> bool:
    url = config.settings.get("SEARXNG_URL", "").rstrip("/")
    return bool(url and url == config.settings.get(DOCKER_URL))


def _setup_docker(manager: ConfigManager, port: int | None, sleep: Callable[[float], None]) -> str:
    existing = inspect()
    folder = manager.config_path.parent / "searxng"
    if folder.resolve() == native.directory():
        folder = folder.with_name("searxng-docker")
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
                command=f'docker run --rm --user root --entrypoint rm -v "{folder.parent.resolve()}:/x" {IMAGE} -rf /x/{folder.name}',
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
    return url


def ensure_native(url: str, *, clock: Callable[[], datetime]) -> bool:
    """Check the local service when searched, sharing failed starts across workers."""
    try:
        folder = native.directory()
        address = url.rstrip("/")
        if not native.commit(folder) or native.url(folder) != address:
            return True
        failed = folder / "start-failed"
        with native.locked(folder, timeout=40):
            if not native.commit(folder) or native.url(folder) != address:
                return False
            if native.running(folder) and native.listening(address):
                return True
            try:
                last_failure = float(failed.read_text(encoding="ascii"))
            except (FileNotFoundError, ValueError):
                pass
            else:
                if clock().timestamp() - last_failure < START_RETRY_SECONDS:
                    return False
            try:
                native.start(folder)
            except (click.ClickException, OSError, ValueError):
                failed.write_text(str(clock().timestamp()), encoding="ascii")
                raise
            failed.unlink(missing_ok=True)
        return True
    except (click.ClickException, OSError, ValueError) as exc:
        logging.getLogger(__name__).warning("Could not start native SearXNG: %s", exc)
        return False


def _native_installed(folder: Path) -> bool:
    return bool(native.commit(folder) or (folder / "src").is_dir())


def _uses_native(config: Config, folder: Path) -> bool:
    configured = config.settings.get("SEARXNG_URL", "").rstrip("/")
    return (
        not uses_docker(config)
        and _native_installed(folder)
        and (not configured or native.url(folder) == configured)
    )


def engine(config: Config, http: httpx.Client, clock: Callable[[], datetime]) -> search.SearXNG:
    url = config.settings["SEARXNG_URL"]
    before_search = None if uses_docker(config) else partial(ensure_native, url, clock=clock)
    return search.SearXNG(url, http, before_search=before_search)


def managed_search(config: Config) -> bool:
    """Whether the configured URL belongs to a DESMAN-managed installation."""
    return bool(config.settings.get("SEARXNG_URL")) and (
        uses_docker(config) or _uses_native(config, native.directory())
    )


def _container_urls(container: dict[str, Any] | None) -> set[str]:
    bindings = container.get("HostConfig", {}).get("PortBindings", {}) if container else {}
    return {
        f"http://127.0.0.1:{p['HostPort']}"
        for p in bindings.get("8080/tcp") or []
        if p.get("HostIp") == "127.0.0.1"
    }


def status(manager: ConfigManager, *, display: bool = True) -> dict[str, object]:
    config = manager.load()
    url = config.settings.get("SEARXNG_URL", "")
    folder = native.directory()
    unknown = False
    use_native = _uses_native(config, folder)
    container = None
    if not use_native:
        try:
            available()
            container = inspect()
        except click.ClickException:
            unknown = True
    kind = "native" if use_native else "docker"
    if (
        not use_native
        and url
        and not uses_docker(config)
        and url.rstrip("/") not in _container_urls(container)
    ):
        kind = "external"
        container = None
        unknown = False
    if use_native:
        url = native.url(folder)
        commit = native.commit(folder)
    else:
        commit = (
            native.COMMIT
            if container and container.get("Config", {}).get("Image") == IMAGE
            else None
        )
    state = container.get("State", {}).get("Status") if container else None
    running = native.running(folder) if use_native else state == "running"
    problem = None
    if url:
        if use_native and not running:
            problem = t("stopped")
        else:
            with search.client() as http:
                problem = search.check_searxng(url, http)
    if kind == "external":
        running = problem is None
    result = {
        "kind": kind,
        "running": running,
        "commit": commit,
        "container": "unknown" if unknown else state,
        "url": url,
        "answers": bool(url and problem is None),
        "error": problem,
    }
    if not display:
        return result
    ui.note(
        t(
            "SearXNG installation: {kind}, commit {commit}",
            kind={"docker": t("Docker"), "native": t("Python program"), "external": t("external")}[
                kind
            ],
            commit=commit or t("Not configured"),
        )
    )
    if use_native:
        ui.note(t("SearXNG service running: {running}", running=t("yes") if running else t("no")))
    elif kind == "docker":
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
    config = manager.load()
    folder = native.directory()
    use_native = _uses_native(config, folder)
    if action == "start":
        if use_native:
            _operate_native(folder, action)
        else:
            available()
            docker(["start", CONTAINER])
    else:
        if _native_installed(folder):
            owned_url = _operate_native(folder, action)
            if owned_url and not uses_docker(config):
                _clear_url(manager, {owned_url})
        if find_docker() is not None or config.settings.get(DOCKER_URL):
            available()
            existing = inspect()
            if action == "remove":
                if existing:
                    docker(["rm", "-f", "-v", CONTAINER])
                urls = _container_urls(existing)
                if saved := config.settings.get(DOCKER_URL):
                    urls.add(saved)
                _clear_url(manager, urls, clear_docker=True)
            elif existing:
                docker(["stop", CONTAINER])
    ui.ok(t("Done."))


def _operate_native(folder: Path, action: str) -> str:
    try:
        with native.locked(folder):
            if action == "start":
                native.start(folder)
            else:
                native.stop(folder)
                if action == "remove":
                    owned_url = native.url(folder)
                    shutil.rmtree(folder)
                    return owned_url
    except OSError as exc:
        raise click.ClickException(t("Native SearXNG failed: {error}", error=str(exc))) from exc
    return ""


def _clear_url(manager: ConfigManager, owned_urls: set[str], *, clear_docker: bool = False) -> None:
    with manager.edit() as config:
        url = config.settings.get("SEARXNG_URL", "")
        if url.rstrip("/") in owned_urls:
            config.settings.pop("SEARXNG_URL", None)
        if clear_docker:
            config.settings.pop(DOCKER_URL, None)


def consent(force_native: bool = False) -> str:
    if force_native or find_docker() is None:
        plan = t(
            "Set up local SearXNG as a Python program? Downloads {size} bytes of source and installs dependencies in a virtual environment.",
            size=native.SIZE,
        )
    else:
        return t("Set up local SearXNG with Docker (about 100 MB)?")
    if sys.platform in ("win32", "darwin"):
        plan += " " + t(
            "SearXNG officially supports Linux only. Native setup worked in our Windows test. "
            "If it fails, DESMAN uses the other search engines."
        )
    return plan
