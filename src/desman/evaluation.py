"""Evaluating a proposal (ROADMAP 5.5): the benchmark (`desman.bench`) run on the current
version of DESMAN and on it with the proposal, each in a git worktree of its own, and compared.

Both runs start from the same commit (HEAD of the checkout DESMAN runs from): the current
version is that commit as it is; the proposed one is that commit with the proposal's change
made and its settings changed. The proposal must have been made from the files that commit
has (`details.base`, their SHA-256), or it answers another text and is not evaluated. Each run
is a process of its own (`python -m desman.bench`) with the worktree's code first on its path,
a database of its own in a temporary folder, and the config only in real mode (read, never
written). The worktrees are removed afterwards. G.7b shares validation and runs through `check()`.

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
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from desman import bench, improve
from desman.config import Tuning, default_config_path
from desman.core import proposals
from desman.core.proposals import Evaluation, Proposal, Status

Side = Literal["base", "candidate"]

TIMEOUT: dict[bench.Mode, float] = {"fake": 600.0, "real": 3600.0}
"""Seconds one run of the benchmark may take before it is stopped."""
GIT_TIMEOUT = 120.0


class EvaluationError(Exception):
    """The proposal cannot be evaluated (nothing is recorded): the reason, for the person."""


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
) -> Checked:
    """Validate plain proposal data before running either isolated benchmark (G.7b)."""
    folder = Path(tempfile.mkdtemp(prefix="desman-eval-"))
    trees: list[Path] = []
    try:
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
            if isinstance(e, EvaluationError) and str(e).startswith("git "):
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
        old = (base / package).read_text(encoding="utf-8")
        new = (candidate / package).read_text(encoding="utf-8")
        values = by_file.get(path, {})
        if _without(old, set(values)) != _without(new, set(values)):
            raise EvaluationError(f"the diff changes more than the targets in {package}")
        written = _values(new, set(values))
        if written != values:
            raise EvaluationError(f"the diff does not give {package} the proposed texts")


def _without(text: str, names: set[str]) -> list[str]:
    """The module, as its syntax, without the assignments to `names`."""
    return [ast.dump(n) for n in ast.parse(text).body if _assigned(n) not in names]


def _values(text: str, names: set[str]) -> dict[str, str]:
    found = {}
    for node in ast.parse(text).body:
        name = _assigned(node)
        if name in names:
            assert isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None
            value = ast.literal_eval(node.value)
            found[name] = value if isinstance(value, str) else repr(value)
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
) -> bench.Result:
    """`python -m desman.bench` with the worktree's code; its result, checked to come from it."""
    data = folder / f"{tree.name}-data"
    data.mkdir(mode=0o700)
    env = (
        dict(os.environ)
        if mode == "real"
        else {name: os.environ[name] for name in ("PATH", "LANG", "LC_ALL") if name in os.environ}
    )
    env["PYTHONPATH"] = str(tree / "src")
    env["DESMAN_DATA_DIR"] = str(data)
    if mode == "real":
        env["DESMAN_CONFIG_DIR"] = str(default_config_path().parent)
    else:  # the fake needs no config: an empty one, never the person's
        home = folder / f"{tree.name}-home"
        home.mkdir(mode=0o700)
        env["HOME"] = str(home)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        empty = folder / f"{tree.name}-config"
        empty.mkdir(mode=0o700)
        env["DESMAN_CONFIG_DIR"] = str(empty)
    argv = [python, "-m", "desman.bench", "--mode", mode, "--tuning", tuning.model_dump_json()]
    if provider is not None:
        argv += ["--provider", provider]
    if model is not None:
        argv += ["--model", model]
    try:
        done = subprocess.run(
            argv,
            cwd=tree,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=TIMEOUT[mode] if timeout is None else timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise RunFailed(
            f"it took longer than {int(TIMEOUT[mode] if timeout is None else timeout)} s"
        ) from e
    except OSError as e:
        raise RunFailed(str(e)) from e
    if done.returncode != 0:
        raise RunFailed(_last_line(done.stderr)[-500:] or f"exit code {done.returncode}")
    try:
        result = bench.Result.model_validate(json.loads(_last_line(done.stdout)))
    except (ValueError, ValidationError) as e:
        raise RunFailed(f"it reported no result ({type(e).__name__})") from e
    if Path(result.source).resolve() != (tree / "src" / "desman").resolve():
        raise RunFailed(f"it ran DESMAN from {result.source}, not from its worktree")
    if result.version != bench.VERSION or result.mode != mode:
        raise RunFailed("it ran another benchmark")
    return result


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
