"""Effect journal (ROADMAP 3.4, D6): a rerun does not repeat what already happened.

An execution of a subtask can be interrupted after some of its tool calls ran: the process is
killed, the provider fails, the lease runs out. The next run of the subtask starts the executor
again, and with the response cache (llm/cache.py) the model asks for the same calls in the same
order. Each call therefore goes through the journal (table `effects`):

1. intent: before the call, a row (execution, ordinal, sha256 of tool and arguments) is written
   while the lease is still held;
2. the tool runs;
3. confirmation: its result is stored in the row.

A later run of the same execution walks the journal along with its own calls. As long as call
number N is the same tool with the same arguments as row N, the stored result is returned and
nothing runs: the workspace is already in the state the replayed calls left it in. Reads are
replayed too: run now, they would see the files as the whole interrupted run left them, not as
they were at that point. The first call that differs ends the replay; the rows from there on
belonged to a run that went another way and are replaced, and every call after it runs for real.

An executor that saved a checkpoint resumes from it and skips the calls made before it, so the
replay starts at the number of calls the journal had at that checkpoint (`Store.effect_cursor`).

A row that was started but never confirmed is a call cut off midway: its effects are unknown.
A `repeatable` tool (reading, writing a file) runs it again; any other tool is not rerun, and
the model gets an error saying so, to check the workspace and call it again if it still needs to (that call
is a new one and runs).

An execution ends when the executor hands in something to judge (`Store.executions`): a new
attempt after a rejection starts a new journal, since the model is then meant to act again.
Results are kept in the database because a replay must return them (including what a read
returned); arguments are not (only their hash), so what a call writes is not copied there.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Final

from pydantic import BaseModel

from desman.core.context import (
    CHARS_PER_TOKEN,
    Knowledge,
    evidence_shares,
    fair_shares,
    fit_knowledge,
    given,
    is_document_combine,
    truncate,
)
from desman.core.models import Subtask
from desman.core.store import Store
from desman.core.tools import Replayed, Tool, ToolResult
from desman.core.workspace import PathEscapeError, Workspace, is_work_file, workspace_root_for

INTERRUPTED: Final = (
    "this exact call was started by an earlier run of this subtask and interrupted before it "
    "finished, so it was not run again: its effects on the workspace are unknown. Check the "
    "workspace, then call it again if it is still needed."
)


EVIDENCE_TOKENS: Final = 4000
"""What a reviewer gets of the evidence, at most (shared fairly between its items)."""


def evidence(
    store: Store,
    subtask: Subtask,
    budget_tokens: int = EVIDENCE_TOKENS,
    *,
    knowledge: Knowledge | None = None,
    workspace: Workspace | None = None,
) -> dict[str, Any]:
    """What the result under review can rest on: the subtask's inputs, the accepted results of
    its dependencies, the knowledge tree's nodes it was given (looked up by `knowledge`), and
    what the latest execution returned, plus earlier web lookups of this same subtask
    marked earlier_attempt. A document part's inputs are kept whole. Latest results take
    the remaining budget first; older lookups share what remains, newest execution first.
    Document notes use the same separate allowance as in the executor context.
    In call order within each execution, fitted to
    `budget_tokens`; a cut item says so. Arguments are not kept (only their hash), so a call
    shows its tool and its result. Files successfully written in that execution also show
    their current workspace text (or an unreadable/binary note), sharing the same budget."""
    execution = store.executions(subtask.id) - 1
    rows = store.effects(subtask.id, execution) if execution >= 0 else []
    dependencies = store.dependency_results(subtask.id)
    budget = budget_tokens * CHARS_PER_TOKEN
    out: dict[str, Any] = {"inputs": None, "results_of_dependencies": [], "tool_results": []}
    goal, is_final = store.subtask_goal(subtask)
    if is_final:
        out["goal"] = goal
    elif "document" in subtask.inputs:
        out["task_goal"] = goal
    offered = store.conn.execute(
        "SELECT details_json FROM audit_log WHERE subtask_id = ? "
        "AND action = 'attempt_started' ORDER BY seq DESC LIMIT 1",
        (subtask.id,),
    ).fetchone()
    out["tools_available"] = json.loads(offered[0]).get("tools", []) if offered else []
    used_inputs = 0
    if "document" in subtask.inputs and "text" in subtask.inputs:
        # A document review must be able to check even the last quote against its source.
        out["inputs"] = subtask.inputs
        used_inputs = len(_evidence_text("inputs", subtask.inputs))
        budget = max(0, budget - used_inputs)
    # What the executor was given of the knowledge tree, within at most half the budget, as
    # it got it (`context.fit`): the rest is for what the work itself rests on.
    nodes = fit_knowledge(given(knowledge, subtask.knowledge)[0], budget // 2)
    if nodes:
        out["knowledge"] = [node.model_dump(exclude_defaults=True) for node in nodes]
        budget -= sum(len(node.text) for node in nodes)
    items: list[tuple[str, Any]] = []
    if subtask.inputs and not used_inputs:
        items.append(("inputs", subtask.inputs))
    items += [("dependency", d) for d in dependencies]
    items += [("call", row) for row in rows]
    files = _written_files(rows, workspace, store, subtask)
    if files:
        out["written_files"] = []
        items += [("file", file) for file in files]
    texts = [_evidence_text(kind, item) for kind, item in items]
    digests = [kind == "dependency" and item.digest is not None for kind, item in items]
    combine = is_document_combine(subtask)
    shares, used_evidence = evidence_shares(
        [len(text) for text in texts],
        budget,
        document_notes=[combine and kind == "dependency" for kind, _ in items],
        digests=digests,
    )
    for (kind, item), text, chars, digest in zip(items, texts, shares, digests, strict=True):
        whole = chars >= len(text)
        content: Any = json.loads(text) if whole else truncate(text, chars)
        if kind == "inputs":
            out["inputs"] = content
        elif kind == "dependency":
            out["results_of_dependencies"].append(
                {
                    "subtask": item.description,
                    "result": content,
                    "truncated": not whole,
                    **({"digest": True} if digest else {}),
                }
            )
        elif kind == "file":
            out["written_files"].append(
                {"path": item["path"], "content": content, "truncated": not whole}
            )
        elif item["state"] != "done":  # cut off midway: what it returned is unknown
            out["tool_results"].append({"tool": item["tool"], "result": None, "finished": False})
        else:
            out["tool_results"].append(
                {
                    "tool": item["tool"],
                    "at": timestamp(item["finished_at"]),
                    "result": content,
                    "truncated": not whole,
                }
            )
    used = used_inputs + used_evidence + sum(len(node.text) for node in nodes)
    out["tool_results"].extend(
        earlier_lookups(store, subtask, execution, max(0, budget_tokens * CHARS_PER_TOKEN - used))
    )
    return out


def quote_sources(
    store: Store,
    subtask: Subtask,
    *,
    knowledge: Knowledge | None,
    context_budget_tokens: int,
) -> list[object]:
    """Tool outputs from this execution and the knowledge supplied to its executor."""
    execution = store.executions(subtask.id) - 1
    outputs = [
        output
        for row in store.effects(subtask.id, execution)
        if row["state"] == "done"
        and not _model_written(
            row["tool"], output := ToolResult.model_validate_json(row["result_json"]).output
        )
    ]
    nodes = fit_knowledge(
        given(knowledge, subtask.knowledge)[0], context_budget_tokens * CHARS_PER_TOKEN // 2
    )
    return [*outputs, *(node.text for node in nodes), *_documents_read_in_parts(store, subtask)]


def _model_written(tool: str, output: object) -> bool:
    """A file a model wrote in the workspace is no source for a quote (L.2 live run: a combine
    saved its notes to a file, the next one read it back and its quotes passed). The person's
    files and fetched pages are under input/ and web/."""
    if tool != "read_file" or not isinstance(output, dict):
        return False
    path = output.get("path")
    return not (isinstance(path, str) and path.startswith(("input/", "web/")))


def _documents_read_in_parts(store: Store, subtask: Subtask) -> list[str]:
    """The long documents a planned subtask works from (ADR 0014): a quote must come from them
    even when the notes that carried it are several steps away."""
    paths = subtask.inputs.get("read_in_parts")
    if isinstance(subtask.inputs.get("document"), str) and "text" not in subtask.inputs:
        paths = [subtask.inputs["document"]]  # a combine: its quotes come from the document too
    if not isinstance(paths, list):
        return []
    texts = []
    workspace = Workspace(workspace_root_for(store.conn), subtask.task_id)
    for path in paths:
        if isinstance(path, str):
            try:
                texts.append(workspace.read_text(path))
            except (OSError, UnicodeError, PathEscapeError):
                continue
    return texts


def _written_files(
    rows: list[Any], workspace: Workspace | None, store: Store, subtask: Subtask
) -> list[dict[str, str]]:
    paths: list[str] = []
    for row in rows:
        if row["tool"] != "write_file" or row["state"] != "done":
            continue
        result = ToolResult.model_validate_json(row["result_json"])
        if result.ok and isinstance(result.output, dict):
            path = result.output.get("path")
            if isinstance(path, str) and path not in paths:
                paths.append(path)
    if not paths:
        return []
    files = []
    for path in paths:
        try:
            workspace = workspace or Workspace(workspace_root_for(store.conn), subtask.task_id)
            text = workspace.read_text(path)
            if "\x00" in text:
                text = "Binary file: content unavailable."
        except (OSError, UnicodeError, PathEscapeError):
            text = "File unreadable or not UTF-8: content unavailable."
        files.append({"path": path, "text": text})
    return files


def timestamp(value: str | None) -> str | None:
    """The journal's completion time, in UTC to whole seconds."""
    if value is None:
        return None
    return (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        .astimezone(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def earlier_lookups(
    store: Store, subtask: Subtask, before: int, chars: int
) -> list[dict[str, Any]]:
    """Finished web results before `before`, newest execution first, shared fairly."""
    rows = [
        row
        for execution in range(before - 1, -1, -1)
        for row in store.effects(subtask.id, execution)
        if row["tool"] in ("web_search", "web_fetch") and row["state"] == "done"
    ]
    texts = [str(row["result_json"]) for row in rows]
    shares = fair_shares([len(text) for text in texts], chars)
    return [
        {
            "tool": row["tool"],
            "at": timestamp(row["finished_at"]),
            "earlier_attempt": True,
            "untrusted": True,
            "result": json.loads(text) if share >= len(text) else truncate(text, share),
            "truncated": share < len(text),
        }
        for row, text, share in zip(rows, texts, shares, strict=True)
        if share > 0
    ]


def _evidence_text(kind: str, item: Any) -> str:
    if kind == "inputs":
        return json.dumps(item, ensure_ascii=False, sort_keys=True)
    if kind == "file":
        return json.dumps(item["text"], ensure_ascii=False)
    if kind == "dependency":
        return json.dumps(item.content, ensure_ascii=False, sort_keys=True)
    return str(item["result_json"] or "null")


def effect_key(tool: str, args: BaseModel) -> str:
    """Identifies a call by what it does: the tool and its validated arguments."""
    payload = json.dumps(
        {"tool": tool, "args": args.model_dump(mode="json")},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class Effects:
    """The journal of the current execution of one subtask, for its Toolbox. Used by one
    executor holding the lease, one call at a time."""

    def __init__(self, store: Store, subtask_id: str, lease: str) -> None:
        self.store = store
        self.subtask_id = subtask_id
        self.lease = lease
        self.execution = store.executions(subtask_id)
        self._past = {row["ordinal"]: row for row in store.effects(subtask_id, self.execution)}
        self._next = store.effect_cursor(subtask_id)
        self._replaying = True
        self._running: int | None = None

    @property
    def position(self) -> int:
        """How many calls the execution has made so far (saved with a checkpoint)."""
        return self._next

    def begin(self, tool: Tool[Any], args: BaseModel) -> Replayed | None:
        # A previous call that crashed stays started in the journal: interrupted, for a rerun.
        self._running = None
        ordinal = self._next
        self._next += 1
        key = effect_key(tool.name, args)
        past = self._past.get(ordinal) if self._replaying else None
        if past is None or past["key"] != key:
            self._replaying = False
            self.store.start_effect(
                self.subtask_id, self.lease, self.execution, ordinal, tool.name, key
            )
            self._running = ordinal
            return None
        if past["state"] != "started":
            result = ToolResult.model_validate_json(past["result_json"]).model_copy(
                update={"at": timestamp(past["finished_at"])}
            )
            return Replayed(result, interrupted=past["state"] == "interrupted")
        if tool.repeatable:
            self.store.adopt_effect(self.subtask_id, self.lease, self.execution, ordinal)
            self._running = ordinal
            return None
        result = ToolResult(ok=False, error=INTERRUPTED)
        self.store.finish_effect(
            self.subtask_id,
            self.lease,
            self.execution,
            ordinal,
            _stored(result),
            interrupted=True,
        )
        return Replayed(result.model_copy(update={"at": self._at(ordinal)}), interrupted=True)

    def _at(self, ordinal: int) -> str | None:
        row = self.store.conn.execute(
            "SELECT finished_at FROM effects WHERE subtask_id = ? AND execution = ? AND ordinal = ?",
            (self.subtask_id, self.execution, ordinal),
        ).fetchone()
        return timestamp(row["finished_at"]) if row is not None else None

    def finish(self, result: ToolResult) -> ToolResult:
        assert self._running is not None, "no call was started"
        ordinal, self._running = self._running, None
        self.store.finish_effect(
            self.subtask_id, self.lease, self.execution, ordinal, _stored(result)
        )
        return result.model_copy(update={"at": self._at(ordinal)})


def _stored(result: ToolResult) -> dict[str, Any]:
    stored: dict[str, Any] = result.model_dump(mode="json", exclude={"at"})
    if result.missing is None:
        stored.pop("missing")
    return stored


def workspace_files(
    store: Store, subtask: Subtask, workspace: Workspace, chars: int
) -> list[dict[str, Any]]:
    """Bounded retry inventory; a successful write's journal identifies its attempt."""
    if chars <= 0:
        return []
    writers: dict[str, int] = {}
    starts = store.conn.execute(
        "SELECT details_json FROM audit_log WHERE subtask_id = ? AND action = 'attempt_started' ORDER BY seq",
        (subtask.id,),
    ).fetchall()
    attempts = {
        d["execution"]: d["attempt"] for row in starts if "execution" in (d := json.loads(row[0]))
    }
    for execution in range(store.executions(subtask.id)):
        for row in store.effects(subtask.id, execution):
            if row["tool"] != "write_file" or row["state"] != "done":
                continue
            result = ToolResult.model_validate_json(row["result_json"])
            if (
                result.ok
                and isinstance(result.output, dict)
                and isinstance(result.output.get("path"), str)
                and execution in attempts
            ):
                writers[result.output["path"]] = attempts[execution]
    out: list[dict[str, Any]] = []
    for path in workspace.files():
        if chars <= 0:
            break
        if not is_work_file(path):
            continue
        try:
            item: dict[str, Any] = {"path": path, "size": workspace.size(path)}
        except (OSError, PathEscapeError):
            continue
        if path in writers:
            item["earlier_attempt"] = writers[path]
        cost = len(json.dumps(item, ensure_ascii=False)) + 2
        if cost > chars:
            break
        out.append(item)
        chars -= cost
    return out
