"""Tool registry: everything an executor may do besides answering (ROADMAP 3.1).

An executor never touches files or anything else itself. It returns structured tool calls
(`ToolCall`: a tool's name and its arguments), and the coordinator runs them through the
subtask's `Toolbox`, which

- knows only the tools of its `Registry`; each tool declares a schema for its arguments (a
  Pydantic model, checked before the tool runs) and the permission it needs;
- offers and runs only what its `Access` allows (core/policy.py: the tool policy of the role,
  bounded by the human's permissions) and what can run here (`Tool.ready`: `run_command`
  needs a working sandbox), and records every refusal;
- checks the subtask's lease before running anything, so work whose lease was lost has no more
  effects (StaleStateError propagates, the coordinator discards the work);
- records every call in the audit, with long arguments cut short; arguments a tool declares
  `private` (file contents) are recorded only by their size, since a workspace file may hold a
  secret and the history is shown to anyone who looks at the task.

A call that is the model's mistake (unknown tool, bad arguments, a path outside the workspace,
a missing file) comes back as a failed `ToolResult` the model can read and correct, not as an
exception. Anything else (a full disk, a bug) is recorded as a failed call and propagates,
interrupting the attempt.

Tool outputs are plain JSON, sized for a model's context.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Annotated, Any, Final, Generic, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
)

from desman.core.policy import Access, Permission, executor_access
from desman.core.sandbox import Sandbox
from desman.core.workspace import PathEscapeError, Workspace

__all__ = ["Access", "Permission"]  # re-exported: tools are declared with a Permission

A = TypeVar("A", bound=BaseModel)

MAX_READ_CHARS: Final = 8_000
"""Text returned by `read_file` (≈2000 tokens); the rest is cut with a marker: free models have
small contexts, and every round of tool results stays in the conversation."""
MAX_WRITE_CHARS: Final = 1_000_000
MAX_LISTED_FILES: Final = 500
AUDIT_TEXT: Final = 200
"""Characters of a string argument kept in the audit."""
MAX_ERROR_CHARS: Final = 500
MAX_RUN_SECONDS: Final = 60
"""Longest `run_command`: well inside a lease (the lease is renewed just before the call)."""
MAX_COMMAND_ARGS: Final = 64
MAX_COMMAND_ARG_CHARS: Final = 100_000
_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")


class ToolCall(BaseModel):
    """One call asked for by an executor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    output: Any = None
    """JSON: what the tool returned."""
    error: str | None = None
    """Why the call failed, written for the model to correct it."""


@dataclass(frozen=True)
class Tool(Generic[A]):
    name: str
    description: str
    """For the model: what the tool does and what it returns."""
    args: type[A]
    permission: Permission
    run: Callable[[A, Workspace], Any]
    """Returns plain JSON. Raises PathEscapeError, FileNotFoundError, NotADirectoryError,
    IsADirectoryError or ToolError for a call that is the caller's mistake."""
    private: frozenset[str] = frozenset()
    """Arguments kept out of the audit (only their size is recorded)."""
    ready: Callable[[], str | None] = lambda: None
    """Why the tool cannot run on this machine (e.g. no sandbox), or None. A tool that is not
    ready is neither offered nor run, whatever the policy allows."""

    def spec(self) -> dict[str, Any]:
        """How the tool is shown to a model."""
        return {
            "name": self.name,
            "description": self.description,
            "args": self.args.model_json_schema(),
        }


class ToolError(ValueError):
    """A call the tool cannot carry out because of its arguments (shown to the model)."""


class Registry:
    """The tools that exist. Being registered grants nothing: see `Toolbox`."""

    def __init__(self, tools: Iterable[Tool[Any]] = ()) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool[Any]) -> None:
        if not _NAME.fullmatch(tool.name):
            raise ValueError(f"bad tool name {tool.name!r}: lowercase letters, digits and '_'")
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any] | None:
        return self._tools.get(name)

    def __iter__(self) -> Iterator[Tool[Any]]:
        return iter(self._tools.values())

    @property
    def names(self) -> list[str]:
        return list(self._tools)


Record = Callable[[str, dict[str, Any]], None]
"""Writes an audit entry for the subtask: (action, details)."""


class Toolbox:
    """The tools one subtask may use, bound to its workspace, lease and audit."""

    def __init__(
        self,
        workspace: Workspace,
        registry: Registry | None = None,
        *,
        access: Access | None = None,
        record: Record = lambda action, details: None,
        guard: Callable[[], None] = lambda: None,
    ) -> None:
        self.workspace = workspace
        self.registry = registry if registry is not None else builtin_registry()
        self.access = access if access is not None else executor_access()
        """The shipped policy for the executor, no switches on, by default."""
        self.record = record
        self.guard = guard
        """Raises StaleStateError when the lease is lost: nothing runs after that."""

    def available(self) -> list[Tool[Any]]:
        return [tool for tool in self.registry if self._refusal(tool) is None]

    def _refusal(self, tool: Tool[Any]) -> str | None:
        refusal = self.access.refusal(tool)
        if refusal is None:
            problem = tool.ready()  # only asked for tools the role may use: it may cost a probe
            if problem is not None:
                refusal = f"{tool.name!r} cannot run here: {problem}"
        return refusal

    def specs(self) -> list[dict[str, Any]]:
        """The tools to offer a model: only those it may use."""
        return [tool.spec() for tool in self.available()]

    def call(self, call: ToolCall) -> ToolResult:
        tool = self.registry.get(call.tool)
        private = tool.private if tool is not None else frozenset()
        summary = {"tool": call.tool, "args": summarize(call.args, private)}
        if tool is None:
            return self._failed(summary, f"unknown tool {call.tool!r}; available: {self._names()}")
        if tool.permission is not Permission.WORKSPACE:
            summary["permission"] = tool.permission.value
        refusal = self._refusal(tool)
        if refusal is not None:
            # Never offered to the model, so asking for it is worth a separate entry.
            self.record("tool_refused", {**summary, "reason": refusal})
            return ToolResult(ok=False, error=f"refused: {refusal}")
        try:
            args = tool.args.model_validate(call.args)
        except ValidationError as e:
            return self._failed(summary, f"bad arguments for {call.tool!r}: {_problems(e)}")
        self.guard()
        if tool.permission is Permission.ADMIN:
            summary["elevated"] = True  # admin_rights is on: every such action stands out
        try:
            output = tool.run(args, self.workspace)
        except (
            ToolError,
            PathEscapeError,
            FileNotFoundError,
            NotADirectoryError,
            IsADirectoryError,
        ) as e:
            return self._failed(summary, f"{type(e).__name__}: {e}")
        except Exception as e:
            # Not the model's mistake, but the attempted call must stay in the history.
            self.record("tool_called", {**summary, "ok": False, "crashed": type(e).__name__})
            raise
        self.record("tool_called", {**summary, "ok": True})
        return ToolResult(ok=True, output=output)

    def _failed(self, summary: dict[str, Any], error: str) -> ToolResult:
        error = error[:MAX_ERROR_CHARS]
        self.record("tool_called", {**summary, "ok": False, "error": error})
        return ToolResult(ok=False, error=error)

    def _names(self) -> str:
        return ", ".join(tool.name for tool in self.available()) or "none"


def summarize(value: Any, private: frozenset[str] = frozenset()) -> Any:
    """`value` with long strings cut, for the audit. The `private` keys of a top-level dict are
    replaced by their size."""
    if isinstance(value, dict) and private:
        return {str(k): _size(v) if k in private else summarize(v) for k, v in value.items()}
    if isinstance(value, str):
        return value if len(value) <= AUDIT_TEXT else f"{value[:AUDIT_TEXT]}…[{len(value)} chars]"
    if isinstance(value, dict):
        return {str(k): summarize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [summarize(v) for v in value]
    return value


def _size(value: Any) -> str:
    return f"[{len(value)} chars]" if isinstance(value, str) else "[hidden]"


def _problems(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'args'}: {e['msg']}" for e in error.errors()
    )


# ── Built-in tools: the task's workspace ────────────────────────────────────


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadFileArgs(_Args):
    path: str = Field(description="Relative to the workspace, '/'-separated.")
    offset: int = Field(
        default=0, ge=0, description="First character to return (to read past a cut)."
    )


class WriteFileArgs(_Args):
    path: str = Field(description="Relative to the workspace, '/'-separated.")
    content: str = Field(max_length=MAX_WRITE_CHARS, description="The whole text of the file.")


class ListFilesArgs(_Args):
    offset: int = Field(default=0, ge=0, description="Index of the first path to list.")


def _read_file(args: ReadFileArgs, workspace: Workspace) -> dict[str, Any]:
    try:
        text = workspace.read_text(args.path)
    except UnicodeDecodeError as e:
        raise ToolError(f"{args.path!r} is not a UTF-8 text file") from e
    if args.offset == 0 and len(text) <= MAX_READ_CHARS:
        return {"path": args.path, "content": text}
    end = args.offset + MAX_READ_CHARS
    read: dict[str, Any] = {
        "path": args.path,
        "content": text[args.offset : end],
        "offset": args.offset,
        "chars": len(text),
    }
    if end < len(text):
        read["truncated"] = True
        read["next_offset"] = end
    return read


def _write_file(args: WriteFileArgs, workspace: Workspace) -> dict[str, Any]:
    path = workspace.write(args.path, args.content)
    return {"path": path, "chars": len(args.content)}


def _list_files(args: ListFilesArgs, workspace: Workspace) -> dict[str, Any]:
    files = workspace.files()
    end = args.offset + MAX_LISTED_FILES
    listed: dict[str, Any] = {"files": files[args.offset : end]}
    if len(files) > end:
        listed["more"] = len(files) - end
        listed["next_offset"] = end
    return listed


READ_FILE: Final = Tool(
    name="read_file",
    description="Read a text file of the task's workspace. Returns {path, content}; a long "
    "file is cut (truncated: true, chars: its full length, next_offset: pass it as offset to "
    "read on).",
    args=ReadFileArgs,
    permission=Permission.WORKSPACE,
    run=_read_file,
)
WRITE_FILE: Final = Tool(
    name="write_file",
    description="Create or replace a text file in the task's workspace (parent directories "
    "are made). Files stay there for the later subtasks of the task. Returns {path, chars}.",
    args=WriteFileArgs,
    permission=Permission.WORKSPACE,
    run=_write_file,
    private=frozenset({"content"}),
)
LIST_FILES: Final = Tool(
    name="list_files",
    description="List the files of the task's workspace. Returns {files: [path, ...]}; a "
    "long list is cut (more: how many are left, next_offset: pass it as offset to go on).",
    args=ListFilesArgs,
    permission=Permission.WORKSPACE,
    run=_list_files,
)


# ── Running code: only in the sandbox ───────────────────────────────────────


def _no_nul(value: str) -> str:
    if "\0" in value:
        raise ValueError("NUL characters are not allowed")
    return value


CommandArg = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_COMMAND_ARG_CHARS),
    AfterValidator(_no_nul),
]


class RunCommandArgs(_Args):
    command: list[CommandArg] = Field(
        min_length=1,
        max_length=MAX_COMMAND_ARGS,
        description='Program and its arguments, not a shell line: ["python3", "main.py"]. '
        'For pipes or redirection, run ["sh", "-c", "..."].',
    )
    stdin: Annotated[str, AfterValidator(_no_nul)] = Field(
        default="", max_length=MAX_WRITE_CHARS, description="Text given to the program's input."
    )
    timeout: int = Field(
        default=30, ge=1, le=MAX_RUN_SECONDS, description="Seconds before it is killed."
    )


def run_command_tool(sandbox: Sandbox) -> Tool[RunCommandArgs]:
    def run(args: RunCommandArgs, workspace: Workspace) -> dict[str, Any]:
        outcome = sandbox.run(args.command, workspace.root, stdin=args.stdin, timeout=args.timeout)
        result: dict[str, Any] = {
            "exit_code": outcome.exit_code,
            "stdout": outcome.stdout,
            "stderr": outcome.stderr,
        }
        for flag in ("timed_out", "stdout_cut", "stderr_cut"):
            if getattr(outcome, flag):
                result[flag] = True
        return result

    return Tool(
        name="run_command",
        description="Run a program in an isolated sandbox: the task's workspace is the "
        "working directory (/workspace, writable; files written there stay for later subtasks), "
        "the system's programs (python3, sh, ...) are read-only, there is no network and no "
        "other files. Time, memory, file size and processes are limited. Returns {exit_code, "
        "stdout, stderr}; timed_out: true when it was killed (exit_code null), stdout_cut / "
        "stderr_cut when output was cut (the start of stdout and the end of stderr are kept). "
        "An exit code of 128+N means killed by signal N.",
        args=RunCommandArgs,
        permission=Permission.SHELL,
        run=run,
        private=frozenset({"stdin"}),
        ready=sandbox.unavailable,
    )


def builtin_registry(sandbox: Sandbox | None = None) -> Registry:
    """The tools every DESMAN has. A new registry each time: registering is not global.
    Whether the sandbox works is checked only when `run_command` may be used."""
    return Registry(
        [
            READ_FILE,
            WRITE_FILE,
            LIST_FILES,
            run_command_tool(sandbox if sandbox is not None else Sandbox()),
        ]
    )
