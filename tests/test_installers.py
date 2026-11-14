"""Installer contract tests: fake tools, isolated home, no network."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / "install/install.sh"


@pytest.fixture
def installer(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    home = tmp_path / "home"
    home.mkdir()
    tools = tmp_path / "bin"
    tools.mkdir()
    log = tmp_path / "calls"
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("DESMAN_", "XDG_")):
            del env[key]
    env.update(HOME=str(home), PATH=f"{tools}:/usr/bin:/bin", LOG=str(log))
    scripts = {
        "git": r"""echo "git $*" >> "$LOG"
case "$*" in
    *'remote get-url origin') echo https://github.com/Temporary-Bot-5/DESMAN-P.git ;;
    'show-ref --verify --quiet refs/tags/master') exit 1 ;;
    'ls-remote '*) if [ "${NO_TAGS:-0}" != 1 ]; then printf 'abc\trefs/tags/v1.2.0\n'; fi ;;
    clone*) mkdir -p "$3/.git" "$3/src/desman"; touch "$3/src/desman/cli.py" ;;
esac
""",
        "uv": r"""echo "uv $*" >> "$LOG"
if [ "$1" = venv ]; then
    mkdir -p .venv/bin
    cat > .venv/bin/desman <<'SCRIPT'
#!/bin/sh
echo "desman $*" >> "$LOG"
if [ "$*" = 'gui --shortcut' ]; then
    mkdir -p "$HOME/.local/share/applications" "$HOME/Desktop"
    for shortcut in "$HOME/.local/share/applications/desman.desktop" "$HOME/Desktop/desman.desktop"; do
        printf '[Desktop Entry]\nExec="%s/.venv/bin/desman" gui\n' "$PWD" > "$shortcut"
    done
fi
SCRIPT
    chmod +x .venv/bin/desman
fi
""",
        "curl": r"""echo "curl $*" >> "$LOG"
# Official installer stand-in: place uv on the path that the installer refreshes.
for output; do :; done
cat > "$output" <<'SCRIPT'
mkdir -p "$HOME/.local/bin"
cp "$UV_STUB" "$HOME/.local/bin/uv"
SCRIPT
""",
    }
    for name, body in scripts.items():
        path = tools / name
        path.write_text("#!/bin/sh\nset -eu\n" + body)
        path.chmod(0o755)
    env["DESMAN_TTY"] = str(tmp_path / "unavailable" / "tty")
    env["UV_STUB"] = str(tools / "uv")
    return home, env


def run_installer(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/sh", str(SHELL), *args], env=env, input="", capture_output=True, text=True
    )


linux = pytest.mark.skipif(sys.platform == "win32", reason="POSIX installer")


@linux
def test_uv_confirmation_and_noninteractive_refusal(installer: tuple[Path, dict[str, str]]) -> None:
    home, env = installer
    tools = Path(env["UV_STUB"]).parent
    # Remove fake uv from PATH but keep it available for the fake official installer.
    stub = tools / "uv-source"
    (tools / "uv").rename(stub)
    env["UV_STUB"] = str(stub)
    # An actual uv on the test runner must not bypass the missing-uv branch.
    assert not (home / ".local/bin/uv").exists()
    result = run_installer(env, "--no-start")
    assert result.returncode == 1
    assert "--yes" in result.stderr
    assert not Path(env["LOG"]).exists()
    result = run_installer(env, "--yes", "--no-start")
    assert result.returncode == 0, result.stderr
    calls = Path(env["LOG"]).read_text(encoding="utf-8").splitlines()
    assert calls[0].startswith("curl -LsSf https://astral.sh/uv/install.sh")
    assert calls.index("uv venv -p 3.12") < calls.index(
        "uv pip install --python .venv/bin/python -e ."
    )
    assert calls[-1] == "desman gui --shortcut"


@linux
def test_install_rerun_uninstall_and_config_preserved(
    installer: tuple[Path, dict[str, str]],
) -> None:
    home, env = installer
    for _ in range(2):
        result = run_installer(env, "--yes", "--no-start")
        assert result.returncode == 0, result.stderr
    calls = Path(env["LOG"]).read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("git clone ") for line in calls) == 1
    assert calls.count("git fetch origin --tags") == 2
    assert calls.count("git checkout -B desman-installed refs/tags/v1.2.0") == 2
    assert calls.count("uv venv -p 3.12") == 1
    app = home / ".local/share/desman-app"
    assert (home / ".local/bin/desman").resolve() == app / ".venv/bin/desman"
    (app / "config.json").write_text("secret")
    data = home / ".local/share/desman"
    data.mkdir()
    (data / "desman.db").touch()
    result = run_installer(env, "--uninstall", "--yes")
    assert result.returncode == 0, result.stderr
    assert data.is_dir()
    assert (home / ".config/desman/config.json").read_text(encoding="utf-8") == "secret"
    for path in (
        app,
        home / ".local/bin/desman",
        home / "Desktop/desman.desktop",
        home / ".local/share/applications/desman.desktop",
    ):
        assert not path.exists()
    result = run_installer(env, "--uninstall", "--purge", "--yes")
    assert result.returncode == 0, result.stderr
    assert "nothing removed" in result.stdout
    assert data.exists()
    assert (home / ".config/desman").exists()


@linux
def test_foreign_directory_refused(installer: tuple[Path, dict[str, str]]) -> None:
    home, env = installer
    app = home / ".local/share/desman-app"
    app.mkdir(parents=True)
    (app / "keep").touch()
    assert run_installer(env, "--yes").returncode == 1
    assert (app / "keep").exists()


@linux
def test_fallback_and_start(installer: tuple[Path, dict[str, str]]) -> None:
    _, env = installer
    env["NO_TAGS"] = "1"
    result = run_installer(env, "--yes")
    assert result.returncode == 0, result.stderr
    assert "No v* tag" in result.stdout
    calls = Path(env["LOG"]).read_text(encoding="utf-8")
    assert "checkout -B desman-installed refs/remotes/origin/master" in calls
    assert calls.endswith("desman gui\n")


@linux
def test_missing_git_hint(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    apt = tools / "apt-get"
    apt.write_text("#!/bin/sh\nexit 0\n")
    apt.chmod(0o755)
    env = {"PATH": str(tools), "HOME": str(tmp_path)}
    result = run_installer(env, "--yes")
    assert result.returncode == 1
    assert "sudo apt-get install git" in result.stderr


def test_windows_wrapper_and_evaluation_source() -> None:
    script = (ROOT / "install/install.ps1").read_text(encoding="utf-8")
    evaluations = [
        line.strip()
        for line in script.splitlines()
        if "iex" in line.lower() or "invoke-expression" in line.lower()
    ]
    assert evaluations == ["irm https://astral.sh/uv/install.ps1 | iex"]
    wrapper = (ROOT / "install/install.bat").read_text(encoding="utf-8")
    assert 'set "DESMAN_INSTALLER_REF=master"' in wrapper
    assert 'set "DESMAN_INSTALLER=%~dp0install.ps1"' in wrapper
    assert 'if exist "%DESMAN_INSTALLER%" goto run' in wrapper
    assert "%TEMP%\\desman-install-%RANDOM%-%RANDOM%.ps1" in wrapper
    assert (
        "https://raw.githubusercontent.com/Temporary-Bot-5/DESMAN-P/%DESMAN_INSTALLER_REF%/install/install.ps1"
        in wrapper
    )
    assert "-OutFile $env:DESMAN_INSTALLER" in wrapper
    assert '-File "%DESMAN_INSTALLER%" %*' in wrapper
    assert wrapper.index("echo Downloading") < wrapper.index("Invoke-WebRequest")
    assert wrapper.index("echo Running") < wrapper.index("-File ")
    assert "if errorlevel 1 goto failed" in wrapper  # the download
    # -Uninstall deletes the batch file it runs from: the run is the last line read.
    run = next(line for line in wrapper.splitlines() if '-File "%DESMAN_INSTALLER%"' in line)
    assert run.endswith("exit /b 1)")
    assert wrapper.index(run) < wrapper.index(":failed")


@pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe batch files")
@pytest.mark.parametrize(("deletes", "code"), [(True, 0), (True, 3), (False, 0), (False, 3)])
def test_the_windows_wrapper_ends_cleanly(tmp_path: Path, deletes: bool, code: int) -> None:
    """`install.bat -Uninstall` deletes its own folder: no "cannot find the path" after it,
    and the exit code is the installer's (0, or 1 after the pause on a failure)."""
    folder = tmp_path / "app" / "install"
    folder.mkdir(parents=True)
    shutil.copy(ROOT / "install/install.bat", folder / "install.bat")
    remove = f"Remove-Item -LiteralPath '{tmp_path / 'app'}' -Recurse -Force; " if deletes else ""
    (folder / "install.ps1").write_text(f"{remove}exit {code}\n", encoding="utf-8")
    result = subprocess.run(
        ["cmd", "/c", str(folder / "install.bat")],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
    )
    output = result.stdout + result.stderr
    assert "cannot find" not in output.lower(), output
    assert result.returncode == (0 if code == 0 else 1), output
    assert (folder / "install.bat").exists() is not deletes


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell dry-run")
def test_windows_dry_run(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell not installed")
    env = {key: value for key, value in os.environ.items() if not key.startswith("DESMAN_")}
    shell_data = tmp_path_factory.mktemp("powershell")
    env.update(
        LOCALAPPDATA=str(shell_data),
        APPDATA=str(shell_data),
        PSModuleAnalysisCachePath=str(shell_data / "ModuleAnalysisCache"),
        USERPROFILE=str(tmp_path),
    )
    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",  # as install.bat runs it: scripts are disabled by default
            "Bypass",
            "-File",
            str(ROOT / "install/install.ps1"),
            "-DryRun",
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    for expected in (
        "uv venv -p 3.12",
        "winget install",
        "MinGit",
        "git checkout -B",
        "desman gui --shortcut",
        "desman global update",
    ):
        assert expected in result.stdout
    assert not (shell_data / "DESMAN").exists()
    assert not (tmp_path / ".config" / "desman").exists()
    assert not (tmp_path / ".local" / "share" / "desman").exists()
    assert not (tmp_path / "Desktop" / "DESMAN.lnk").exists()


class Assets(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        path = (
            values.get("src")
            if tag in ("img", "script")
            else (values.get("href") if tag == "link" else None)
        )
        if path:
            self.paths.append(path)
        assert not any(name.startswith("on") for name in values)


def test_site_assets_and_languages() -> None:
    html = (ROOT / "site/index.html").read_text(encoding="utf-8")
    parser = Assets()
    parser.feed(html)
    assert parser.paths
    for path in parser.paths:
        assert ":" not in path and not path.startswith("/")
        assert (ROOT / "site" / path).is_file()
    # A subshell: its status is the installer's (or curl's), and the trap removes the file.
    command = '(f="$(mktemp)" &amp;&amp; trap \'rm -f "$f"\' EXIT &amp;&amp; curl -fsSL '
    command += (
        "https://raw.githubusercontent.com/Temporary-Bot-5/DESMAN-P/master/install/install.sh"
    )
    command += ' -o "$f" &amp;&amp; sh "$f")'
    assert command in html
    for name in ("README.md", "docs/install.md"):
        assert command.replace("&amp;", "&") in (ROOT / name).read_text(encoding="utf-8")
    assert "| sh" not in html
    assert "install.ps1" not in html
    assert "MIT" not in html
    assert "On Windows the agents do not run code because there is no sandbox." in html
    assert "На Windows агенты не запускают код: там нет песочницы." in html
    assert 'data-en="Install"' in html
    assert 'data-ru="Установите"' in html
    assert "navigator.language" in (ROOT / "site/app.js").read_text(encoding="utf-8")
    css = (ROOT / "site/style.css").read_text(encoding="utf-8")
    assert "@import" not in css and "url(" not in css
    for name in ("banner.png", "logo.png", "module.png"):
        assert (ROOT / "site/img" / name).read_bytes() == (
            ROOT / "src/desman/gui/static/img" / name
        ).read_bytes()


@linux
def test_ref_override_and_command_order(installer: tuple[Path, dict[str, str]]) -> None:
    _, env = installer
    env["DESMAN_REF"] = "v2.0.0"
    result = run_installer(env, "--yes", "--no-start")
    assert result.returncode == 0, result.stderr
    calls = Path(env["LOG"]).read_text(encoding="utf-8").splitlines()
    assert not any("ls-remote" in line for line in calls)
    fetch = calls.index("git fetch origin --tags")
    checkout = calls.index("git checkout -B desman-installed refs/tags/v2.0.0")
    venv = calls.index("uv venv -p 3.12")
    pip = calls.index("uv pip install --python .venv/bin/python -e .")
    shortcut = calls.index("desman gui --shortcut")
    clone = next(i for i, line in enumerate(calls) if line.startswith("git clone "))
    assert clone < fetch < checkout < venv < pip < shortcut


@linux
def test_uninstall_refuses_without_confirmation_and_config_collision(
    installer: tuple[Path, dict[str, str]],
) -> None:
    home, env = installer
    assert run_installer(env, "--yes", "--no-start").returncode == 0
    app = home / ".local/share/desman-app"
    assert run_installer(env, "--uninstall").returncode == 1
    assert app.exists()
    (app / "config.json").write_text("new")
    fallback = home / ".config/desman/config.json"
    fallback.parent.mkdir(parents=True)
    fallback.write_text("old")
    result = run_installer(env, "--yes", "--uninstall")
    assert result.returncode == 1
    assert "backup exists" in result.stderr
    assert fallback.read_text(encoding="utf-8") == "old"
    assert (app / "config.json").read_text(encoding="utf-8") == "new"


@linux
def test_custom_directories_and_shortcuts_purged(installer: tuple[Path, dict[str, str]]) -> None:
    home, env = installer
    app = home / "custom-app"
    data = home / "custom-data"
    config = home / "custom-config"
    desktop = home / "Work Desktop"
    env.update(
        DESMAN_HOME=str(app),
        DESMAN_DATA_DIR=str(data),
        DESMAN_CONFIG_DIR=str(config),
        XDG_DESKTOP_DIR=str(desktop),
    )
    assert run_installer(env, "--yes", "--no-start").returncode == 0
    for folder in (data, config, desktop):
        folder.mkdir()
    (desktop / "desman.desktop").write_text(f'Exec="{app}/.venv/bin/desman" gui\n')
    result = run_installer(env, "--yes", "--uninstall", "--purge")
    assert result.returncode == 0, result.stderr
    for path in (app, data, config):
        assert str(path) in result.stdout
        assert not path.exists()
    assert not (desktop / "desman.desktop").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell dry-run")
def test_windows_uninstall_dry_run(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell not installed")
    env = {key: value for key, value in os.environ.items() if not key.startswith("DESMAN_")}
    shell_data = tmp_path_factory.mktemp("powershell")
    env.update(
        LOCALAPPDATA=str(shell_data),
        APPDATA=str(shell_data),
        PSModuleAnalysisCachePath=str(shell_data / "ModuleAnalysisCache"),
        USERPROFILE=str(tmp_path),
    )
    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "install/install.ps1"),
            "-DryRun",
            "-Uninstall",
            "-Purge",
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert "if absent remove nothing" in result.stdout
    assert "TargetPath is under App" in result.stdout
    assert "Separate purge confirmation:" in result.stdout
    assert str(tmp_path / ".local" / "share" / "desman") in result.stdout
    assert not (shell_data / "DESMAN").exists()
    assert not (tmp_path / ".config" / "desman").exists()
    assert not (tmp_path / ".local" / "share" / "desman").exists()
    assert not (tmp_path / "Desktop" / "DESMAN.lnk").exists()


if sys.platform != "win32":

    @linux
    @pytest.mark.parametrize("answer, expected", [(b"y\n", 0), (b"n\n", 1)])
    def test_piped_install_asks_controlling_terminal(
        installer: tuple[Path, dict[str, str]], answer: bytes, expected: int
    ) -> None:
        _, env = installer
        tools = Path(env["UV_STUB"]).parent
        stub = tools / "uv-source"
        (tools / "uv").rename(stub)
        env["UV_STUB"] = str(stub)
        master, slave = os.openpty()
        env["DESMAN_TTY"] = os.ttyname(slave)
        questions: list[bytes] = []

        def respond() -> None:
            output = b""
            while b"Continue? (y/N) " not in output:
                output += os.read(master, 4096)
            questions.append(output)
            os.write(master, answer)

        responder = threading.Thread(target=respond, daemon=True)
        responder.start()
        try:
            result = run_installer(env, "--no-start")
            responder.join(timeout=5)
            assert result.returncode == expected, result.stderr
            assert questions and b"Install uv" in questions[0]
            assert b"Continue? (y/N) " in questions[0]
            assert Path(env["LOG"]).exists() == (expected == 0)
        finally:
            os.close(slave)
            os.close(master)


@linux
@pytest.mark.parametrize("app_present", [True, False])
def test_uninstall_preserves_foreign_launchers(
    installer: tuple[Path, dict[str, str]], app_present: bool
) -> None:
    home, env = installer
    assert run_installer(env, "--yes", "--no-start").returncode == 0
    app = home / ".local/share/desman-app"
    launcher = home / ".local/bin/desman"
    launcher.unlink()
    foreign = home / "pipx/desman"
    foreign.parent.mkdir()
    foreign.write_text("foreign")
    launcher.symlink_to(foreign)
    shortcuts = [home / "Desktop/desman.desktop", home / ".local/share/applications/desman.desktop"]
    for shortcut in shortcuts:
        shortcut.write_text(f'Exec="{foreign}" gui\n')
    data = home / ".local/share/desman"
    data.mkdir()
    (data / "keep").write_text("history")
    if not app_present:
        shutil.rmtree(app)
    result = run_installer(env, "--yes", "--uninstall", "--purge")
    assert result.returncode == 0, result.stderr
    assert launcher.resolve() == foreign
    for shortcut in shortcuts:
        assert shortcut.read_text(encoding="utf-8") == f'Exec="{foreign}" gui\n'
    if not app_present:
        assert "nothing removed" in result.stdout
        assert (data / "keep").read_text(encoding="utf-8") == "history"
