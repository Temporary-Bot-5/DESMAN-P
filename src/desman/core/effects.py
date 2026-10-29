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

from desman.core.store import Store
from desman.core.tools import Replayed, Tool, ToolResult

INTERRUPTED: Final = (
    "this exact call was started by an earlier run of this subtask and interrupted before it "
    "finished, so it was not run again: its effects on the workspace are unknown. Check the "
    "workspace, then call it again if it is still needed."
)


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
