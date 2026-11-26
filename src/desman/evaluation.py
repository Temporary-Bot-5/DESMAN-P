"""Evaluating a proposal (ROADMAP 5.5): the benchmark (`desman.bench`) run on the current
version of DESMAN and on it with the proposal, each in a git worktree of its own, and compared.

Both runs start from the same commit (HEAD of the checkout DESMAN runs from): the current
version is that commit as it is; the proposed one is that commit with the proposal's change
made and its settings changed. The proposal must have been made from the files that commit
has (`details.base`, their SHA-256), or it answers another text and is not evaluated. Each run
is a process of its own (`python -m desman.bench`) with the worktree's code first on its path,
a database of its own in a temporary folder, and the config only in real mode (read, never
written). The worktrees are removed afterwards. G.7b shares validation and runs through `check()`.

Each run is confined (ROADMAP S.3): under bubblewrap (`Sandbox.program`), it sees the system's
/usr, the Python DESMAN runs on (its interpreter and library folders only, read-only), its
worktree (read-only) and one folder of its own in memory, of a bounded size (its data, home and
temporary files: nothing reaches the disk); never the person's home, the checkout, the real
database or config. No network with the fake; in real mode the host's
network (its local services too: the person asked for that run), its proxy and certificates,
and a config of its own holding only what the benchmark may use: the free providers' keys and
the one named by `--provider`, the settings, the language, the model choices (never
permissions, desman-global, the profile). Where bubblewrap cannot run (Windows, macOS, a
system forbidding it) nothing is evaluated, and a sandbox failing to start a run records nothing.

The proposed version is made by code, never by applying the diff (a patch can do more than it
shows): the targets' new texts are checked again and put into the commit's files the way the
diff was built, and the diff that makes must be the proposal's exactly. Then git must see only
those files changed, and in them only the targets' texts. Nothing else of the proposed version
can differ from the current one.

What was found is recorded once (`core.proposals.record_evaluation`): both runs' results and
the verdict, or why the evaluation failed. Nothing is applied (5.6): the person reads it.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import functools
import getpass
import hashlib
import json
import os
import shutil
import site
import sqlite3
import subprocess
import sys
import sysconfig
import tempfile
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from desman import bench, improve, providers
from desman.config import (
    Config,
    ConfigError,
    ConfigManager,
    Tuning,
    default_config_path,
    open_config,
)
from desman.core import proposals
from desman.core.db import default_db_path
from desman.core.proposals import Evaluation, Proposal, Status
from desman.core.sandbox import SANDBOX_PATH, Limits, Sandbox, SandboxUnavailable

Side = Literal["base", "candidate"]

TIMEOUT: dict[bench.Mode, float] = {"fake": 600.0, "real": 3600.0}
"""Seconds one run of the benchmark may take before it is stopped."""
GIT_TIMEOUT = 120.0


class EvaluationError(Exception):
    """The proposal cannot be evaluated (nothing is recorded): the reason, for the person."""


class Unconfined(EvaluationError):
    """No version can be run confined here (no bubblewrap): nothing is evaluated."""


REPROBE_AFTER = 60.0
"""Seconds after which a sandbox found unavailable is probed again (bwrap installed since,
or a probe that failed while the machine was busy)."""


@functools.cache
def _sandbox() -> Sandbox:
    return Sandbox()


_failed_at: weakref.WeakKeyDictionary[Sandbox, float] = weakref.WeakKeyDictionary()
"""When a sandbox was first found unavailable."""
_broken: weakref.WeakKeyDictionary[Sandbox, tuple[str, float]] = weakref.WeakKeyDictionary()
"""A sandbox that passed its probe, then failed to run: why, and when."""
_monotonic = time.monotonic


def isolation(box: Sandbox | None = None) -> str | None:
    """Why a run of the benchmark cannot be confined here, or None when it can. A sandbox found
    unavailable, or failing a run, is probed again after `REPROBE_AFTER`."""
    default = box is None
    box = _sandbox() if box is None else box
    if not box.platform.startswith("linux"):
        return "proposals are evaluated only on Linux, under bubblewrap"
    now = _monotonic()
    if box in _broken:
        reason, since = _broken[box]
        if now - since < REPROBE_AFTER:
            return reason
        del _broken[box]
        problem = _reprobe(box, default)
        if problem is None:  # and the Python it runs, which may be what failed
            box = _sandbox() if default else box
            probe = Path(tempfile.gettempdir()) / f"desman-probe-{os.getpid()}"
            try:
                _probe_runtime(box, sys.executable, probe)
            except Unconfined:
                return _broken[box][0]
        return problem
    problem = box.unavailable()
    if problem is None:
        _failed_at.pop(box, None)
    else:
        failed = _failed_at.setdefault(box, now)
        if now - failed >= REPROBE_AFTER:
            del _failed_at[box]
            problem = _reprobe(box, default)
    return problem


def _reprobe(box: Sandbox, default: bool) -> str | None:
    if default:  # a new one looks for bwrap and prlimit again (installed since)
        _sandbox.cache_clear()
        return _sandbox().unavailable()
    return box.reprobe()


def _unconfined(box: Sandbox, reason: str) -> Unconfined:
    """The sandbox failed a run after passing its probe: no run is offered until it is probed
    again."""
    _broken[box] = (reason, _monotonic())
    return Unconfined(f"the benchmark cannot run confined here: {reason}")


def evaluate(
    conn: sqlite3.Connection,
    proposal: Proposal,
    *,
    mode: bench.Mode,
    tuning: Tuning,
    provider: str | None = None,
    model: str | None = None,
    repo: Path | None = None,
    python: str = sys.executable,
    on_run: Callable[[Side], None] | None = None,
    box: Sandbox | None = None,
) -> Evaluation:
    """Run the benchmark on both versions and record what was found. `tuning`: the person's,
    which the current version runs with and the proposal's settings change. Raises
    `EvaluationError` when the proposal cannot be evaluated at all."""
    if proposal.status is not Status.PROPOSED:
        raise EvaluationError(f"proposal {proposal.id} proposes nothing ({proposal.status})")
    repo = repository() if repo is None else repo
    commit = _git(repo, "rev-parse", "HEAD").strip()
    candidate_tuning = proposed_tuning(tuning, proposal)
    checked = check(
        repo,
        commit,
        changes=proposal.details.get("changes", []),
        base=proposal.details.get("base", {}),
        diff=proposal.diff or "",
        mode=mode,
        base_tuning=tuning,
        candidate_tuning=candidate_tuning,
        python=python,
        on_run=on_run,
        provider=provider,
        model=model,
        box=box,
    )
    if checked.outcome == "refused":
        raise EvaluationError(checked.reason)
    return proposals.record_evaluation(
        conn,
        proposal.id,
        mode=mode,
        commit=commit,
        verdict=checked.verdict or "failed",
        reason=checked.reason,
        base=None if checked.base is None else checked.base.model_dump(mode="json"),
        candidate=None if checked.candidate is None else checked.candidate.model_dump(mode="json"),
    )


@dataclass(frozen=True)
class Checked:
    """A plain-data evaluation, shared with another node's proposal check (G.7b)."""

    outcome: Literal["refused", "failed", "done"]
    reason: str | None = None
    base: bench.Result | None = None
    candidate: bench.Result | None = None
    verdict: Literal["better", "worse", "same"] | None = None


def check(
    repo: Path,
    commit: str,
    *,
    changes: list[dict[str, Any]],
    base: dict[str, str],
    diff: str,
    mode: bench.Mode,
    base_tuning: Tuning,
    candidate_tuning: Tuning,
    timeout: float | None = None,
    python: str = sys.executable,
    on_run: Callable[[Side], None] | None = None,
    provider: str | None = None,
    model: str | None = None,
    box: Sandbox | None = None,
) -> Checked:
    """Validate plain proposal data before running either isolated benchmark (G.7b). Raises
    `Unconfined` when the runs cannot be confined here, before anything is made."""
    problem = isolation(box)  # None: the default sandbox, made again on a reprobe
    box = _sandbox() if box is None else box
    if problem is not None:
        raise Unconfined(f"the benchmark cannot run confined here: {problem}")
    folder = Path(tempfile.mkdtemp(prefix="desman-eval-"))
    trees: list[Path] = []
    try:
        _probe_runtime(box, python, folder / "probe")
        base_tree, candidate_tree = folder / "current", folder / "proposed"
        for tree in (base_tree, candidate_tree):
            trees.append(tree)
            _git(repo, "worktree", "add", "--detach", str(tree), commit)
        try:
            _check_base(base_tree, base, commit)
            if diff or changes:
                _build(candidate_tree, changes, diff)
                _check_changes(base_tree, candidate_tree, changes)
        except (EvaluationError, OSError, ValueError, SyntaxError) as e:
            if isinstance(e, Unconfined) or (
                isinstance(e, EvaluationError) and str(e).startswith("git ")
            ):
                raise
            return Checked("refused", str(e))
        results: dict[Side, bench.Result] = {}
        sides: tuple[tuple[Side, Path, Tuning], ...] = (
            ("base", base_tree, base_tuning),
            ("candidate", candidate_tree, candidate_tuning),
        )
        for side, tree, side_tuning in sides:
            if on_run is not None:
                on_run(side)
            try:
                results[side] = _run(
                    tree,
                    folder,
                    mode,
                    side_tuning,
                    provider=provider,
                    model=model,
                    python=python,
                    timeout=timeout,
                    box=box,
                )
            except RunFailed as e:
                return Checked(
                    "failed",
                    f"the {_side_name(side)} version's run failed: {e}",
                    results.get("base"),
                )
        return Checked(
            "done",
            base=results["base"],
            candidate=results["candidate"],
            verdict=bench.compare(results["base"], results["candidate"]),
        )
    finally:
        for tree in trees:
            with contextlib.suppress(EvaluationError):
                _git(repo, "worktree", "remove", "--force", str(tree))
        shutil.rmtree(folder, ignore_errors=True)
        with contextlib.suppress(EvaluationError):
            _git(repo, "worktree", "prune")


def _side_name(side: Side) -> str:
    return "current" if side == "base" else "proposed"


def repository() -> Path:
    """The git checkout DESMAN runs from (its package is that checkout's `src/desman`)."""
    package = improve.PACKAGE.resolve()
    try:
        root = Path(_git(package, "rev-parse", "--show-toplevel").strip()).resolve()
    except EvaluationError as e:
        raise EvaluationError(f"DESMAN does not run from a git checkout: {e}") from e
    if (root / "src" / "desman").resolve() != package:
        raise EvaluationError(f"DESMAN does not run from the checkout at {root}")
    return root


def proposed_tuning(tuning: Tuning, proposal: Proposal) -> Tuning:
    """`tuning` with the proposal's settings changed."""
    data = tuning.model_dump()
    for item in proposal.details.get("settings", []):
        setting = improve.SETTINGS_BY_ID.get(str(item.get("setting")))
        if setting is None:
            raise EvaluationError(f"unknown setting {item.get('setting')!r}")
        if setting.is_model:
            data["models"][setting.name] = item.get("new")
        else:
            data[setting.name] = item.get("new")
    try:
        return Tuning.model_validate(data)
    except ValidationError as e:
        raise EvaluationError(f"the proposed settings are not valid: {e}") from e


# ── The worktrees ────────────────────────────────────────────────────────────


def _check_base(tree: Path, base: dict[str, str], commit: str) -> None:
    """The proposal was made from the files the commit has."""
    differ = []
    allowed = {improve.SOURCE_PREFIX + target.path for target in improve.TARGETS}
    for name, digest in base.items():
        if name not in allowed:
            raise EvaluationError("the base names a file outside the allowed targets")
        path = tree / name
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        if hashlib.sha256(text.encode()).hexdigest() != digest:
            differ.append(name)
    if differ:
        raise EvaluationError(
            f"the proposal was made from other versions of {', '.join(differ)} than commit "
            f"{commit[:12]} has: ask for a new analysis"
        )


def _build(tree: Path, changes: list[dict[str, Any]], diff: str) -> None:
    """Make the proposed version in `tree` by code, never by applying the diff: each target's
    new text checked (`improve.checked`) and put into the commit's file (`improve.replace`), as
    the diff was built. The diff that makes must be the proposal's, to the byte, so what runs
    is what the person reads; and afterwards git must see only those files changed."""
    texts: dict[str, str] = {}
    package = tree / "src" / "desman"
    for change in changes:
        target = improve.BY_ID.get(str(change.get("target")))
        if target is None:
            raise EvaluationError(f"unknown target {change.get('target')!r}")
        if target.path not in texts:
            texts[target.path] = (package / target.path).read_text(encoding="utf-8")
        try:
            new = improve.checked(target, improve.current(target, package), change.get("new"))
            texts[target.path] = improve.replace(texts[target.path], target, new)
        except improve.Unfit as e:
            raise EvaluationError(str(e)) from e
    built = "".join(
        improve.file_diff(path, (package / path).read_text(encoding="utf-8"), text)
        for path, text in texts.items()
    )
    if built != diff:
        raise EvaluationError("the diff is not the one the proposal's texts make")
    for path, text in texts.items():
        (package / path).write_text(text, encoding="utf-8", newline="")
    status = _git(tree, "status", "--porcelain", "--untracked-files=all", "--ignored")
    expected = {f" M {improve.SOURCE_PREFIX}{path}" for path in texts}
    if set(status.splitlines()) != expected:
        raise EvaluationError("the proposed version differs from the commit in other files")


def _check_changes(base: Path, candidate: Path, changes: list[dict[str, Any]]) -> None:
    """In the proposed version, only the targets' texts differ, and they hold the new values."""
    by_file: dict[str, dict[str, str]] = {}
    for change in changes:
        target = improve.BY_ID.get(str(change.get("target")))
        if target is None:
            raise EvaluationError(f"unknown target {change.get('target')!r}")
        by_file.setdefault(target.path, {})[target.name] = str(change.get("new"))
    for path in {t.path for t in improve.TARGETS}:
        package = improve.SOURCE_PREFIX + path
        # As bytes: read as Python reads them on import (a coding line, a BOM), not as UTF-8.
        old = (base / package).read_bytes()
        new = (candidate / package).read_bytes()
        values = by_file.get(path, {})
        if _without(old, set(values)) != _without(new, set(values)):
            raise EvaluationError(f"the diff changes more than the targets in {package}")
        written = _values(new, set(values))
        if written != values:
            raise EvaluationError(f"the diff does not give {package} the proposed texts")


def _without(text: str | bytes, names: set[str]) -> list[str]:
    """The module, as its syntax, with each assignment to `names` only marked where it stands:
    its value may differ, not how many there are or what comes before and after them."""
    return [
        _marked(n, name) if (name := _assigned(n)) in names else ast.dump(n)
        for n in _parse(text).body
    ]


def _marked(node: ast.stmt, name: str | None) -> str:
    """A target's assignment without its value: its name, and its annotation (code too)."""
    if isinstance(node, ast.AnnAssign):
        return f"<{name}: {ast.dump(node.annotation)}>"
    return f"<{name}>"


def _parse(text: str | bytes) -> ast.Module:
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError) as e:
        raise EvaluationError(f"a changed file is not valid Python: {e}") from e


def _values(text: str | bytes, names: set[str]) -> dict[str, str]:
    found = {}
    for node in _parse(text).body:
        name = _assigned(node)
        if name in names:
            assert isinstance(node, ast.Assign | ast.AnnAssign)
            if node.value is None:
                raise EvaluationError(f"{name} is given no value")
            try:  # a text only: an f-string, a call or a name is code
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as e:
                raise EvaluationError(f"{name} is not a plain value") from e
            if not isinstance(value, str):  # every target is a text
                raise EvaluationError(f"{name} is not a text")
            found[name] = value
    return found


def _assigned(node: ast.stmt) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        return target.id if isinstance(target, ast.Name) else None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _git(where: Path, *args: str, stdin: str | None = None) -> str:
    try:
        done = subprocess.run(
            ["git", "-c", "core.autocrlf=false", "-C", str(where), *args],
            input=None if stdin is None else stdin.encode("utf-8"),
            capture_output=True,
            timeout=GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise EvaluationError(f"git {args[0]}: {e}") from e
    if done.returncode != 0:
        raise EvaluationError(
            f"git {args[0]}: {done.stderr.decode('utf-8').strip() or done.returncode}"
        )
    return done.stdout.decode("utf-8")


# ── A run ────────────────────────────────────────────────────────────────────


class RunFailed(Exception):
    """A run of the benchmark did not report a result."""


def _run(
    tree: Path,
    folder: Path,
    mode: bench.Mode,
    tuning: Tuning,
    *,
    provider: str | None,
    model: str | None,
    python: str,
    timeout: float | None = None,
    box: Sandbox | None = None,
) -> bench.Result:
    """`python -m desman.bench` with the worktree's code, confined (`Sandbox.program`); its
    result, checked to come from it."""
    box = _sandbox() if box is None else box
    # Its one writable folder is in memory, of a bounded size: nothing it writes reaches the
    # disk (its home, its database, its local-model slots). The config beside it, read-only.
    data, config = folder / f"{tree.name}-data", folder / f"{tree.name}-config"
    config.mkdir(mode=0o700)
    read_only = [*_runtime(python), tree, config]
    env = _python_env(data) | {
        "PYTHONPATH": str(tree / "src"),
        "DESMAN_DATA_DIR": str(data),
        "DESMAN_CONFIG_DIR": str(config),
    }
    if mode == "real":  # only what the benchmark may use; the fake needs none
        # A local model's slots are the run's own, never a folder the person's runs trust: an
        # evaluation may ask a local server alongside them.
        with contextlib.suppress(KeyError, OSError):
            env["USER"] = env["LOGNAME"] = getpass.getuser()
        env["XDG_RUNTIME_DIR"] = str(data)
        try:
            given = bench_config(open_config().load(), provider, model, tuning)
            ConfigManager(config / "config.json").save(given)
        except (ConfigError, OSError) as e:  # the person's to fix: no proposal's failure
            raise EvaluationError(f"cannot read the config: {e}") from e
        # How this machine reaches the network: a proxy, its own certificates.
        for name in NETWORK_ENV:
            value = os.environ.get(name) or os.environ.get(name.lower())
            if value:
                env[name] = value
        for name in ("SSL_CERT_FILE", "SSL_CERT_DIR"):  # a folder list: OpenSSL's way
            kept = [
                Path(p)
                for p in env.pop(name, "").split(os.pathsep)
                if p and Path(p).exists() and _harmless(Path(p), folder)
            ]
            if kept and (name == "SSL_CERT_DIR" or len(kept) == 1):
                env[name] = os.pathsep.join(str(p.absolute()) for p in kept)  # from its worktree
                read_only += [q for p in kept for q in (p.absolute(), p.resolve())]
    argv = [python, "-m", "desman.bench", "--mode", mode, "--tuning", tuning.model_dump_json()]
    if provider is not None:
        argv += ["--provider", provider]
    if model is not None:
        argv += ["--model", model]
    seconds = TIMEOUT[mode] if timeout is None else timeout
    try:
        done = box.program(
            argv,
            read_only=read_only,
            writable=[],
            scratch=[data],
            env=env,
            cwd=tree,
            limits=_limits(seconds),
            network=mode == "real",
        )
    except (SandboxUnavailable, OSError) as e:  # the sandbox's failure, not the version's
        raise _unconfined(box, str(e)) from e
    if done.timed_out:
        raise RunFailed(f"it took longer than {int(seconds)} s")
    if done.exit_code != 0:
        raise RunFailed(_last_line(done.stderr)[-500:] or f"exit code {done.exit_code}")
    try:
        result = bench.Result.model_validate(json.loads(_last_line(done.stdout)))
    except (ValueError, ValidationError) as e:
        raise RunFailed(f"it reported no result ({type(e).__name__})") from e
    if Path(result.source).resolve() != (tree / "src" / "desman").resolve():
        raise RunFailed(f"it ran DESMAN from {result.source}, not from its worktree")
    if result.version != bench.VERSION or result.mode != mode:
        raise RunFailed("it ran another benchmark")
    return result


def _harmless(path: Path, folder: Path) -> bool:
    """A folder named for certificates may be shown only if it holds none of the person's: not
    their home, the checkout, the config, DESMAN's data, nor the evaluation's own folder."""
    forms = (path.absolute(), path.resolve())
    if path.is_file():  # a bundle: anything but the config or a file of DESMAN's data
        secret = [default_config_path().resolve(), default_db_path().parent.resolve()]
        return not any(f == s or f.is_relative_to(s) for f in forms for s in secret)
    held = [Path.home(), improve.PACKAGE, default_config_path(), default_db_path(), folder]
    data = default_db_path().parent.resolve()
    return not any(h.resolve().is_relative_to(f) for h in held for f in forms) and not any(
        f.is_relative_to(data) for f in forms
    )


def _python_env(home: Path) -> dict[str, str]:
    env = {
        "PATH": SANDBOX_PATH,
        "LANG": "C.UTF-8",
        "HOME": str(home),
        "TMPDIR": str(home),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if _user_site():  # after the standard library, as outside; found without the real home
        env["PYTHONUSERBASE"] = site.getuserbase()
    else:
        env["PYTHONNOUSERSITE"] = "1"
    return env


PROBE = "import sqlite3, ssl, json, pydantic, httpx, click, cryptography"
"""What the benchmark needs of the Python it runs on: checked confined before any run."""


def _probe_runtime(box: Sandbox, python: str, home: Path) -> None:
    """The Python DESMAN runs on starts confined, with its libraries, before any version runs:
    a Python whose own libraries are missing there (some builds keep them beside it) is this
    machine's problem, never a proposal's failure."""
    try:
        done = box.program(
            [python, "-c", PROBE],
            read_only=_runtime(python),
            writable=[],
            scratch=[home],
            env=_python_env(home),
            cwd=home,
            limits=_limits(60.0),
        )
    except (SandboxUnavailable, OSError) as e:
        raise _unconfined(box, str(e)) from e
    if done.timed_out:
        raise _unconfined(box, "the Python DESMAN runs on took longer than 60 s to start there")
    if done.exit_code != 0:
        detail = _last_line(done.stderr)[-300:] or f"exit code {done.exit_code}"
        raise _unconfined(box, f"the Python DESMAN runs on does not start there: {detail}")


def _limits(seconds: float) -> Limits:
    """A whole DESMAN runs: more memory, processes and output than a generated program gets."""
    return dataclasses.replace(
        Limits(),
        timeout=seconds,
        cpu_seconds=max(1, int(seconds)),
        tmp_bytes=256 * 2**20,
        memory_bytes=4 * 2**30,
        file_bytes=256 * 2**20,
        processes=64,
        stdout_chars=200_000,
        stderr_chars=4_000,
    )


NETWORK_ENV = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)
"""What a run on real models keeps of the environment: how to reach the network, no more."""


def _runtime(python: str | None = None) -> list[Path]:
    """The Python DESMAN runs on, read-only at their own paths (also as given, through a link):
    the interpreter, its standard library, its packages (a venv's, the user's), its shared
    library. Only those: a prefix may hold more of the person's (`--prefix=~/.local` holds
    DESMAN's own data)."""
    paths: list[Path] = [Path(sys.executable), *_user_site()]
    if python is not None and Path(python).is_absolute():
        paths.append(Path(python))
    venv = Path(sys.prefix) / "pyvenv.cfg"
    if venv.is_file():
        paths.append(venv)
    schemes = [
        sysconfig.get_paths(),
        sysconfig.get_paths(vars={"base": sys.base_prefix, "platbase": sys.base_exec_prefix}),
    ]
    for scheme in schemes:
        paths += [Path(scheme[name]) for name in ("stdlib", "platstdlib", "purelib", "platlib")]
    libdir, library = sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("LDLIBRARY")
    if sysconfig.get_config_var("Py_ENABLE_SHARED") and libdir and library:
        paths += sorted(Path(libdir).glob(f"{library}*"))
    # A Python of its own (conda, standalone builds) keeps the libraries its modules load
    # beside it: those files, never the folder.
    for prefix in {Path(sys.base_prefix), Path(sys.base_exec_prefix)}:
        if not prefix.resolve().is_relative_to("/usr"):
            paths += sorted(p for p in (prefix / "lib").glob("*.so*") if p.is_file())
    found: list[Path] = []
    for given in paths:
        if not given.exists():
            continue
        for path in (given.absolute(), given.resolve()):
            if path.is_relative_to("/usr") or any(path.is_relative_to(p) for p in found):
                continue
            found = [p for p in found if not p.is_relative_to(path)] + [path]
    return found


def _user_site() -> list[Path]:
    """The user's site-packages, when DESMAN's packages may be there (`pip install --user`)."""
    folder = site.getusersitepackages() if site.ENABLE_USER_SITE else None
    return [Path(folder)] if folder and folder in sys.path and Path(folder).is_dir() else []


def bench_config(
    config: Config,
    provider: str | None,
    model: str | None = None,
    tuning: Tuning | None = None,
) -> Config:
    """What a confined benchmark on real models is given of the person's config: the keys of
    the providers its run may ask, as the router chooses them (the free ones, the one named by
    `--provider`, those of the models `tuning` puts first for a role, paid or not), the
    settings (hosts of local servers, account ids), the language, the model choices and master
    marks. Never the permissions, desman-global, the profile or another key."""
    names = [provider] if provider else []  # `--model` alone: a model of a free provider
    chosen = (tuning or config.tuning).models.model_dump().values()
    names += [str(label).split("/", 1)[0] for label in chosen if label]
    keep = {p.key_env for p in providers.FREE if p.key_env}
    for name in names:
        with contextlib.suppress(providers.UnknownProvider):
            if env := providers.find(name).key_env:
                keep.add(env)
    return Config(
        configured=config.configured,
        language=config.language,
        api_keys={env: keys for env, keys in config.api_keys.items() if env in keep},
        settings=config.settings,
        tuning=config.tuning,
        master=config.master,
    )


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def metrics_of(evaluation: Evaluation) -> tuple[dict[str, int] | None, dict[str, int] | None]:
    """The compared numbers of both runs (None for a run that did not report)."""

    def of(data: dict[str, Any] | None) -> dict[str, int] | None:
        if data is None:
            return None
        try:
            return bench.metrics(bench.Result.model_validate(data))
        except ValidationError:
            return None

    return of(evaluation.base), of(evaluation.candidate)
