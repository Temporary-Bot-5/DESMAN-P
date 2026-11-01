"""Which waiting work goes first, across every task (ROADMAP 4.3).

Work waits in two queues: subtasks READY to run and results in REVIEW waiting for a verdict.
Both are ordered by *urgency*: the subtask's priority plus a bonus for the time it has waited
since it joined the queue (`queued_at`, set by every status change into READY or REVIEW). That
is aging: a steady stream of high-priority tasks cannot keep a low-priority one waiting forever,
because the longer it waits the more urgent it becomes.

Aging lifts work at most to just below CRITICAL: critical work never queues behind work that has
merely waited long, and a critical subtask itself does not age (it is first already). Among
equal urgency the work that has waited longer goes first.

The ordering is computed in SQL (`urgency_sql`, so a claim picks atomically) and in Python
(`Aging.urgency`, for a subtask already read); a test keeps the two equal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from desman.core import db
from desman.core.models import Priority


@dataclass(frozen=True)
class Aging:
    """`step` priority points for every `every_seconds` waited, up to just below CRITICAL."""

    step: int = 10
    every_seconds: int = 60

    def __post_init__(self) -> None:
        if self.step < 0 or self.every_seconds <= 0:
            raise ValueError("aging needs step >= 0 and every_seconds > 0")

    @property
    def ceiling(self) -> int:
        """The highest urgency aging alone can reach."""
        return int(Priority.CRITICAL) - 1

    def urgency(self, priority: int, queued_at: str | None, now: datetime) -> int:
        """Priority plus the bonus for waiting since `queued_at` (no bonus without one)."""
        if priority >= Priority.CRITICAL or queued_at is None:
            return priority
        waited_ms = round((now - db.parse_iso(queued_at)).total_seconds() * 1000)
        bonus = max(0, waited_ms // (self.every_seconds * 1000)) * self.step
        return min(priority + bonus, self.ceiling)

    def urgency_sql(self) -> str:
        """The same as `urgency`, as an SQL expression over a `subtasks` row; it takes one
        parameter, the current time. Milliseconds are rounded so that floating point never puts
        a step boundary on the other side of where Python puts it."""
        waited_ms = "CAST(ROUND((julianday(?) - julianday(queued_at)) * 86400000) AS INTEGER)"
        return f"""(CASE WHEN priority >= {int(Priority.CRITICAL)} OR queued_at IS NULL
                         THEN priority
                         ELSE MIN(priority + MAX(0, {waited_ms} / {self.every_seconds * 1000})
                                             * {self.step},
                                  {self.ceiling})
                    END)"""


DEFAULT_AGING = Aging()
"""LOW (0) overtakes fresh HIGH (100) work after about 11 minutes of waiting, NORMAL after 6."""
