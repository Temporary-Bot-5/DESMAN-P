"""Live progress of a task while the chat waits for it (ROADMAP C.4).

A bar of the task's subtasks, the subtasks themselves, and a line per module (provider, model
and key) that worked on it: in which role and on which subtask, its calls and tokens, what is
left of its limits, its state (working, waiting for a limit, quota used up, idle).

Everything is read from what is already written: the subtasks from `Store`, the modules from
the call log (`calls.modules`). Nothing here writes, so a reply served from the response cache
(`llm.cache`), which is no call and costs nothing, shows no module. It is redrawn when the coordinator moves
(`Coordinator.watch`) and after each model call (`calls.watching`).

In a terminal the view is redrawn in place and goes away when the task stops (the chat then
tells the outcome). Without one, a line is printed for each change instead.
"""

from __future__ import annotations

from types import TracebackType

from rich.console import Group, RenderableType
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from desman import providers, ui
from desman.config import Config
from desman.core import calls
from desman.core.report import SubtaskReport, TaskReport, task_report
from desman.core.states import SubtaskStatus
from desman.core.store import Store
from desman.i18n import t, tn
from desman.llm.openai_compat import key_id

S = SubtaskStatus
_WORKING = {S.RUNNING, S.REVIEW}
_STYLE = {S.DONE: "good", S.RUNNING: "key", S.REVIEW: "key", S.PAUSED: "warn"}
_BAD = {S.NEEDS_HUMAN, S.FAILED, S.BLOCKED, S.CANCELLED}


def key_labels(config: Config) -> dict[str, str]:
    """Key fingerprint -> how the person tells the key apart: its label (or its number among the
    provider's keys) and its masked value. Never the key."""
    labels = {}
    for p in providers.CLOUD:
        for i, k in enumerate(config.keys(p.key_env or ""), 1):
            name = providers.label(k.label) if k.label else f"#{i}"
            labels[key_id(k.value)] = f"{name} {providers.mask(k.value)}"
    return labels


class Progress:
    """Shows one task while it runs: `with Progress(...) as progress:` and call `refresh()`."""

    def __init__(
        self, store: Store, task_id: str, config: Config, *, terminal: bool | None = None
    ) -> None:
        self.store = store
        self.task_id = task_id
        self.labels = key_labels(config)
        self.terminal = ui.console().is_terminal if terminal is None else terminal
        self._live: Live | None = None
        self._said: dict[str, str] = {}  # without a terminal: what each line said last

    def __enter__(self) -> Progress:
        if self.terminal:
            self._live = Live(console=ui.console(), auto_refresh=False, transient=True)
            self._live.start()
        self.refresh()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def refresh(self) -> None:
        report = task_report(self.store, self.task_id)
        modules = calls.modules(self.store.conn, self.task_id)
        if self._live is not None:
            self._live.update(self.render(report, modules), refresh=True)
            return
        for name, line in self.lines(report, modules):
            if self._said.get(name) != line:
                self._said[name] = line
                ui.out(Text(line, style="muted"))

    # ── What is shown ────────────────────────────────────────────────────────

    def render(self, report: TaskReport, modules: list[calls.ModuleUse]) -> RenderableType:
        bar = Table.grid(padding=(0, 1))
        bar.add_row(
            ProgressBar(total=max(report.total, 1), completed=report.done, width=30),
            Text(tn(report.total, "{done}/{n} subtask", "{done}/{n} subtasks", done=report.done)),
        )
        subtasks = Table.grid(padding=(0, 1))
        for sub in report.subtasks:
            subtasks.add_row(
                Text(_number(sub), style="muted"),
                Text(sub.status.value, style=_status_style(sub.status)),
                Text(_shorten(sub.description, 70)),
            )
        rows = Table.grid(padding=(0, 2))
        by_id = {s.id: s for s in report.subtasks}
        for module in modules:
            state, style = self._state(module, by_id, modules)
            rows.add_row(
                Text(self._name(module), style="key"),
                Text(self._doing(module, by_id)),
                Text(self._spent(module), style="muted"),
                Text(state, style=style),
            )
        return Group(bar, subtasks, rows)

    def lines(self, report: TaskReport, modules: list[calls.ModuleUse]) -> list[tuple[str, str]]:
        """(what, line) for each subtask and module: printed when its line changes. Calls and
        tokens are left out, or every call would print a line."""
        out = [
            (sub.id, f"{_number(sub)} {sub.status.value}  {_shorten(sub.description, 70)}")
            for sub in report.subtasks
        ]
        by_id = {s.id: s for s in report.subtasks}
        for module in modules:
            name = self._name(module)
            state, _ = self._state(module, by_id, modules)
            out.append((f"module {name}", f"{name}: {self._doing(module, by_id)} · {state}"))
        return out

    def _name(self, module: calls.ModuleUse) -> str:
        if module.key_id is None:
            key = t("local model")
        else:
            key = self.labels.get(module.key_id, t("removed ({id}…)", id=module.key_id[:4]))
        return f"{module.provider} {module.model} ({key})"

    def _doing(self, module: calls.ModuleUse, by_id: dict[str, SubtaskReport]) -> str:
        role = module.role.value if module.role is not None else "?"
        sub = by_id.get(module.subtask_id or "")
        return role if sub is None else f"{role} · {_number(sub)}"

    def _spent(self, module: calls.ModuleUse) -> str:
        spent = [
            tn(module.calls, "{n} call", "{n} calls"),
            tn(
                module.prompt_tokens + module.completion_tokens,
                "{n} token",
                "{n} tokens",
            ),
        ]
        found, _ = calls.limits(module.limits)
        spent += [
            t("{name} left {left}", name=limit.name, left=f"{limit.remaining}/{limit.limit}")
            for limit in found
            if limit.remaining is not None and limit.limit is not None
        ][:2]
        return " · ".join(spent)

    def _state(
        self,
        module: calls.ModuleUse,
        by_id: dict[str, SubtaskReport],
        modules: list[calls.ModuleUse],
    ) -> tuple[str, str]:
        if module.error_type == "QuotaExhausted":
            return t("quota used up"), "bad"
        if module.error_type == "RateLimited":
            return t("waiting for a limit"), "warn"
        sub = by_id.get(module.subtask_id or "")
        # Of the modules on a subtask, only the one called last is at work (a retry may have
        # moved it to another model).
        latest = max(m.last_seq for m in modules if m.subtask_id == module.subtask_id)
        if sub is not None and sub.status in _WORKING and module.last_seq == latest:
            return t("working"), "good"
        return t("idle"), "muted"


def _number(sub: SubtaskReport) -> str:
    """As `desman status` and `inbox` number subtasks, a part after the one it is part of."""
    return f"[{sub.seq}]" if sub.part_of is None else f"[{sub.part_of}→{sub.seq}]"


def _status_style(status: SubtaskStatus) -> str:
    return "bad" if status in _BAD else _STYLE.get(status, "muted")


def _shorten(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"
