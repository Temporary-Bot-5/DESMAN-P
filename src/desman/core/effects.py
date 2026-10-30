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
from typing import Any, Final

from pydantic import BaseModel

from desman.core.context import CHARS_PER_TOKEN, fair_shares, truncate
from desman.core.models import Subtask
from desman.core.store import Store
from desman.core.tools import Replayed, Tool, ToolResult

INTERRUPTED: Final = (
    "this exact call was started by an earlier run of this subtask and interrupted before it "
    "finished, so it was not run again: its effects on the workspace are unknown. Check the "
    "workspace, then call it again if it is still needed."
)


EVIDENCE_TOKENS: Final = 4000
"""What a reviewer gets of the evidence, at most (shared fairly between its items)."""


def evidence(
    store: Store, subtask: Subtask, budget_tokens: int = EVIDENCE_TOKENS
) -> dict[str, Any]:
    """What the result under review can rest on: the subtask's inputs, the accepted results of
    its dependencies, and what the tools returned to the execution that handed it in (the
    latest one ended: a rejected attempt's searches are not this result's). In call order,
    fitted to `budget_tokens`; a cut item says so. Arguments are not kept (only their hash), so
    a call shows its tool and its result."""
    execution = store.executions(subtask.id) - 1
    rows = store.effects(subtask.id, execution) if execution >= 0 else []
    dependencies = store.dependency_results(subtask.id)
    items: list[tuple[str, Any]] = [("inputs", subtask.inputs)] if subtask.inputs else []
    items += [("dependency", d) for d in dependencies]
    items += [("call", row) for row in rows]
    texts = [_evidence_text(kind, item) for kind, item in items]
    shares = fair_shares([len(text) for text in texts], budget_tokens * CHARS_PER_TOKEN)
    out: dict[str, Any] = {"inputs": None, "results_of_dependencies": [], "tool_results": []}
    for (kind, item), text, chars in zip(items, texts, shares, strict=True):
        whole = chars >= len(text)
        content: Any = json.loads(text) if whole else truncate(text, chars)
        if kind == "inputs":
            out["inputs"] = content
        elif kind == "dependency":
            out["results_of_dependencies"].append(
                {"subtask": item.description, "result": content, "truncated": not whole}
            )
        elif item["state"] != "done":  # cut off midway: what it returned is unknown
            out["tool_results"].append({"tool": item["tool"], "result": None, "finished": False})
        else:
            out["tool_results"].append(
                {"tool": item["tool"], "result": content, "truncated": not whole}
            )
    return out


def _evidence_text(kind: str, item: Any) -> str:
    if kind == "inputs":
        return json.dumps(item, ensure_ascii=False, sort_keys=True)
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
            result = ToolResult.model_validate_json(past["result_json"])
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
        return Replayed(result, interrupted=True)

    def finish(self, result: ToolResult) -> None:
        assert self._running is not None, "no call was started"
        ordinal, self._running = self._running, None
        self.store.finish_effect(
            self.subtask_id, self.lease, self.execution, ordinal, _stored(result)
        )


def _stored(result: ToolResult) -> dict[str, Any]:
    stored: dict[str, Any] = result.model_dump(mode="json")
    return stored
