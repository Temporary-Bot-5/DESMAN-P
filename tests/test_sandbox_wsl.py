"""Real WSL runs in the dedicated DESMAN distribution on Windows."""

import re
import sys
import time
from pathlib import Path

import pytest

from desman.core.sandbox import WSL_DISTRO, Limits, Sandbox, SandboxUnavailable

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform != "win32", reason="WSL needs Windows"),
]


@pytest.fixture(scope="module")
def wsl() -> Sandbox:
    box = Sandbox(docker="")
    try:
        present = box.wsl_distribution_present()
    except SandboxUnavailable as exc:
        pytest.skip(str(exc))
    if not present:
        pytest.skip("desman-sandbox distribution is missing")
    assert box.unavailable() is None and box.backend == "wsl"
    return box


def test_python_and_workspace(wsl: Sandbox, tmp_path: Path) -> None:
    code = "from pathlib import Path; Path('result.txt').write_text('landed'); print('héllo')"
    result = wsl.run(["python3", "-c", code], tmp_path)
    assert result.exit_code == 0 and result.stdout.strip() == "héllo", result
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "landed"


@pytest.mark.parametrize(("check", "code"), [("True", 0), ("False", 1)])
def test_pytest(wsl: Sandbox, tmp_path: Path, check: str, code: int) -> None:
    (tmp_path / "test_example.py").write_text(f"def test_it(): assert {check}\n", encoding="utf-8")
    assert wsl.run(["python3", "-m", "pytest", "-q"], tmp_path).exit_code == code


def test_network_blocked(wsl: Sandbox, tmp_path: Path) -> None:
    code = "import socket; socket.create_connection(('192.0.2.1', 80), timeout=1)"
    result = wsl.run(["python3", "-c", code], tmp_path)
    assert result.exit_code == 1 and "[Errno 101]" in result.stderr, result


def test_timeout_leaves_no_python(wsl: Sandbox, tmp_path: Path) -> None:
    result = wsl.run(
        ["python3", "-c", "import time; print('started', flush=True); time.sleep(30)"],
        tmp_path,
        timeout=2,
    )
    assert result.timed_out and result.exit_code is None and "started" in result.stdout, result
    for attempt in range(4):
        processes = wsl.wsl_command(
            ["-d", WSL_DISTRO, "-u", "root", "--exec", "/bin/busybox", "ps"]
        )
        assert processes.returncode == 0
        if not re.search(rb"\bpython3?\b", processes.stdout):
            break
        if attempt < 3:
            time.sleep(1)
    else:
        pytest.fail(processes.stdout.decode("utf-8", errors="replace"))


def test_output_limits(wsl: Sandbox, tmp_path: Path) -> None:
    box = Sandbox(Limits(stdout_chars=25, stderr_chars=25), docker="", wsl=wsl.wsl)
    result = box.run(
        [
            "python3",
            "-c",
            "import sys; print('A' * 100_000); print('B' * 100_000, file=sys.stderr)",
        ],
        tmp_path,
    )
    assert result.exit_code == 0 and result.stdout_cut and result.stderr_cut, result
    assert result.stdout == "A" * 25
    assert len(result.stderr) == 25
    assert result.stderr in {"B" * 24 + "\n", "B" * 23 + "\r\n"}
