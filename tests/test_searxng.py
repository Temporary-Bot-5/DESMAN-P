"""Local search administration: fake Docker and HTTP only."""

from __future__ import annotations

import json
import shutil
import socket
from pathlib import Path
from typing import Any

import click
import httpx
import pytest
from click.testing import CliRunner

from desman import commands, searxng
from desman.cli import cli
from desman.config import ConfigManager
from desman.core import search


@pytest.fixture
def manager(tmp_path: Path) -> ConfigManager:
    return ConfigManager(tmp_path / "config.json")


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/docker")

    def docker(args: list[str], timeout: float = 60) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(searxng, "docker", docker)
    monkeypatch.setattr(search, "check_searxng", lambda url, http: None)
    monkeypatch.setattr(searxng, "free_port", lambda port=None: port or 8888)
    return calls


def test_setup_writes_settings_and_fixed_argv(
    manager: ConfigManager, calls: list[list[str]]
) -> None:
    searxng.setup(manager)
    settings = json.loads(
        (manager.config_path.parent / "searxng/settings.yml").read_text(encoding="utf-8")
    )
    assert settings["search"]["formats"] == ["html", "json"]
    assert len(settings["server"]["secret_key"]) == 64
    assert settings["server"]["limiter"] is False
    assert settings["general"]["contact_url"].startswith("https://")
    assert ["pull", searxng.IMAGE] in calls
    assert calls[-1] == [
        "run",
        "-d",
        "--name",
        "desman-searxng",
        "--restart",
        "unless-stopped",
        "-p",
        "127.0.0.1:8888:8080",
        "-v",
        f"{manager.config_path.parent / 'searxng/settings.yml'}:/etc/searxng/settings.yml:ro",
        searxng.IMAGE,
    ]
    assert manager.load().settings["SEARXNG_URL"] == "http://127.0.0.1:8888"


def test_no_docker_changes_nothing(manager: ConfigManager, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(click.ClickException, match="Install Docker"):
        searxng.setup(manager)
    assert not manager.config_path.exists()
    assert not (manager.config_path.parent / "searxng").exists()


def test_failed_probe_never_saves_url(
    manager: ConfigManager, calls: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search, "check_searxng", lambda url, http: "not ready")
    sleeps: list[float] = []
    with pytest.raises(click.ClickException, match="not ready"):
        searxng.setup(manager, sleep=sleeps.append)
    assert not manager.load().settings.get("SEARXNG_URL")
    assert sleeps == [2] * 29


@pytest.mark.parametrize("reuse", [True, False])
def test_existing_container_reused_or_recreated(
    manager: ConfigManager, calls: list[list[str]], monkeypatch: pytest.MonkeyPatch, reuse: bool
) -> None:
    existing: dict[str, Any] = {
        "Config": {"Image": searxng.IMAGE if reuse else "old"},
        "HostConfig": {
            "RestartPolicy": {"Name": "unless-stopped"},
            "PortBindings": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8889"}]},
        },
        "Mounts": [
            {
                "Source": str(manager.config_path.parent / "searxng/settings.yml"),
                "Destination": "/etc/searxng/settings.yml",
                "RW": False,
            }
        ],
    }
    monkeypatch.setattr(searxng, "inspect", lambda: existing)
    searxng.setup(manager)
    if reuse:
        assert calls == [["info"], ["restart", searxng.CONTAINER]]
        assert manager.load().settings["SEARXNG_URL"].endswith(":8889")
    else:
        assert ["rm", "-f", "-v", searxng.CONTAINER] in calls
        assert ["pull", searxng.IMAGE] in calls


@pytest.mark.parametrize("action", ["setup", "remove"])
def test_consent_before_mutation(action: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("declined command must not act")

    monkeypatch.setattr(searxng, "setup", forbidden)
    monkeypatch.setattr(searxng, "operate", forbidden)
    result = CliRunner().invoke(cli, ["search", action], input="n\n")
    assert result.exit_code == 0, result.output
    assert "Nothing changed." in result.output


def test_port_selection_skips_busy_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    class Socket:
        def __enter__(self) -> Socket:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def bind(self, address: tuple[str, int]) -> None:
            assert address[0] == "127.0.0.1"
            seen.append(address[1])
            if address[1] == 8888:
                raise OSError("busy")

    monkeypatch.setattr(socket, "socket", Socket)
    assert searxng.free_port() == 8889
    assert seen == [8888, 8889]
    with pytest.raises(click.ClickException, match="No free"):
        searxng.free_port(8888)


def test_malformed_host_dropped_before_result_limit() -> None:
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "results": [
                        {"title": "broken", "url": "http:///go/v1/no"},
                        {"title": "valid", "url": "https://example.com"},
                    ]
                },
            )
        )
    ) as http:
        assert [r.title for r in search.SearXNG("http://local", http).search("q", 1)] == ["valid"]


def test_keys_rows_and_check(manager: ConfigManager, monkeypatch: pytest.MonkeyPatch) -> None:
    from desman import config_cli

    config = manager.load()
    config.settings["SEARXNG_URL"] = "http://127.0.0.1:8888"
    monkeypatch.setattr(config_cli, "_usage", lambda *a: [])
    rows = config_cli.key_rows(config)
    row = next(r for r in rows if r["provider"] == "searxng")
    assert row["url"] == config.settings["SEARXNG_URL"]
    monkeypatch.setattr(search, "check_searxng", lambda url, http: None)
    targets, checks = config_cli.check_keys(config, "searxng")
    assert targets[0].id == "searxng" and checks[0].ok
    monkeypatch.setattr(search, "check_searxng", lambda url, http: "offline")
    assert not config_cli.check_keys(config, "searxng")[1][0].ok


def test_search_catalog_matches_cli() -> None:
    for action in ("setup", "status", "start", "stop", "remove"):
        entry = commands.find(f"search {action}")
        assert entry is not None
        assert entry.kind == (commands.Kind.READ if action == "status" else commands.Kind.CHANGE)
        assert commands.cli_command(cli, entry.name) is not None


def test_wizard_setup_uses_staged_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from test_wizard import DONE, KEEP_PERMISSIONS, KEEP_PERSONAL, saved, wizard

    seen: list[bool] = []

    def setup(manager: ConfigManager, port: int | None = None, *, save_url: bool = True) -> str:
        seen.append(save_url)
        return "http://127.0.0.1:8889"

    monkeypatch.setattr(searxng, "setup", setup)
    from desman import providers

    result = wizard(
        "1", str(len(providers.PROVIDERS) + 1), "y", DONE, *KEEP_PERMISSIONS, *KEEP_PERSONAL
    )
    assert result.exit_code == 0, result.output
    assert seen == [False]
    assert saved().load().settings["SEARXNG_URL"] == "http://127.0.0.1:8889"


def test_docker_helper_reports_failure_and_uses_no_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    seen: list[list[str]] = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == 60
        assert not kwargs.get("shell")
        seen.append(args)
        return subprocess.CompletedProcess(args, 1, "", "daemon unavailable")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(click.ClickException, match="daemon unavailable"):
        searxng.docker(["info"])
    assert seen == [["docker", "info"]]


def test_remove_keeps_external_url_and_settings(
    manager: ConfigManager, calls: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    with manager.edit() as config:
        config.settings["SEARXNG_URL"] = "https://search.example.com"
    monkeypatch.setattr(searxng, "inspect", lambda: {"State": {"Status": "running"}})
    searxng.operate(manager, "remove")
    assert manager.load().settings["SEARXNG_URL"] == "https://search.example.com"
    assert ["rm", "-f", "-v", searxng.CONTAINER] in calls


def test_setup_repairs_settings_and_retains_secret(
    manager: ConfigManager, calls: list[list[str]]
) -> None:
    folder = manager.config_path.parent / "searxng"
    folder.mkdir()
    settings = folder / "settings.yml"
    settings.write_text(
        json.dumps({"server": {"secret_key": "existing-secret"}, "search": {"formats": ["html"]}}),
        encoding="utf-8",
    )
    searxng.setup(manager)
    repaired = json.loads(settings.read_text(encoding="utf-8"))
    assert repaired["server"]["secret_key"] == "existing-secret"
    assert "json" in repaired["search"]["formats"]


def test_daemon_failure_does_not_write_settings(
    manager: ConfigManager, calls: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(args: list[str], timeout: float = 60) -> str:
        raise click.ClickException("daemon unavailable")

    monkeypatch.setattr(searxng, "docker", unavailable)
    with pytest.raises(click.ClickException, match="get-docker"):
        searxng.setup(manager)
    assert not (manager.config_path.parent / "searxng").exists()


@pytest.mark.parametrize("operation", ["mkdir", "read_text", "write_text", "chmod"])
def test_settings_oserror_is_actionable(
    manager: ConfigManager, calls: list[list[str]], monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    folder = manager.config_path.parent / "searxng"
    folder.mkdir()
    settings = folder / "settings.yml"
    settings.write_text("{}", encoding="utf-8")
    original = getattr(Path, operation)

    def fail(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path in (folder, settings):
            raise PermissionError("owned by another user")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, operation, fail)
    with pytest.raises(click.ClickException) as caught:
        searxng.setup(manager)
    message = str(caught.value)
    assert str(settings) in message
    assert "docker run --rm" in message
    assert "rm -rf /x/searxng" in message
    assert "desman search setup" in message
    assert not any(call[0] in ("pull", "run", "rm") for call in calls)


@pytest.mark.parametrize("mount", ["directory", "writable"])
def test_unsafe_existing_mount_recreated(
    manager: ConfigManager, calls: list[list[str]], monkeypatch: pytest.MonkeyPatch, mount: str
) -> None:
    folder = manager.config_path.parent / "searxng"
    monkeypatch.setattr(
        searxng,
        "inspect",
        lambda: {
            "Config": {"Image": searxng.IMAGE},
            "HostConfig": {
                "RestartPolicy": {"Name": "unless-stopped"},
                "PortBindings": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8889"}]},
            },
            "Mounts": [
                {
                    "Source": str(folder if mount == "directory" else folder / "settings.yml"),
                    "Destination": "/etc/searxng"
                    if mount == "directory"
                    else "/etc/searxng/settings.yml",
                    "RW": mount == "writable",
                }
            ],
        },
    )
    searxng.setup(manager)
    assert ["rm", "-f", "-v", searxng.CONTAINER] in calls
    assert calls[-1][-2].endswith("/settings.yml:/etc/searxng/settings.yml:ro")


def test_settings_container_readable(manager: ConfigManager, calls: list[list[str]]) -> None:
    import os
    import stat

    searxng.setup(manager)
    if os.name != "nt":
        assert (
            stat.S_IMODE((manager.config_path.parent / "searxng/settings.yml").stat().st_mode)
            == 0o644
        )
        assert stat.S_IMODE((manager.config_path.parent / "searxng").stat().st_mode) == 0o700


@pytest.mark.parametrize(
    "url,problem",
    [("", None), ("http://127.0.0.1:8888", None), ("http://127.0.0.1:8888", "offline")],
)
def test_status_readable_and_json(
    calls: list[list[str]], monkeypatch: pytest.MonkeyPatch, url: str, problem: str | None
) -> None:
    from desman.config_cli import load_config

    with load_config().edit() as config:
        if url:
            config.settings["SEARXNG_URL"] = url
    monkeypatch.setattr(searxng, "inspect", lambda: {"State": {"Status": "running"}})
    monkeypatch.setattr(search, "check_searxng", lambda url, http: problem)
    result = CliRunner().invoke(cli, ["search", "status"])
    assert result.exit_code == 0, result.output
    assert "SearXNG container: running" in result.output
    assert "{'container'" not in result.output
    assert (problem or ("web search works" if url else "desman search setup")) in result.output
    result = CliRunner().invoke(cli, ["search", "status", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data == {
        "container": "running",
        "url": url,
        "answers": bool(url and not problem),
        "error": problem,
    }
