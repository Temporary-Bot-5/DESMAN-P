"""What a run of the local operator is made of, shared by `desman run`/`resume` and the chat.

The operator (ADR 0003) is the coordinator with its planner, executor and reviewer. Here it is
put together from the config: the models (a lineup per role, `llm.router`; the executor's per
kind of subtask, each with its other lineup for retries with another model), the executor's
access (the tool policy within the human's permissions) and its tools.
"""

from __future__ import annotations

import functools
import re
import shutil
import sqlite3
import stat
import time
from collections.abc import Callable, Collection, Mapping, Set
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

import click
import httpx

from desman import documents, knowledge, net, providers, ui
from desman.agents import (
    CODE_TOOL_ROUNDS,
    MASTER_TOOL_ROUNDS,
    MAX_TOOL_ROUNDS,
    llm_executor,
    llm_reviewer,
)
from desman.config import Config, ConfigError, MasterChoices, Tuning, open_config
from desman.core import calls, holds, protected
from desman.core import choices as module_choices
from desman.core.context import DEFAULT_BUDGET_TOKENS, Knowledge
from desman.core.coordinator import (
    REVIEW_TEST_TIMEOUT,
    Coordinator,
    Crew,
    Executor,
    Failure,
    Labels,
    Outcome,
    ReviewTests,
    TestRun,
    Work,
    review_test_file,
)
from desman.core.effects import evidence, last_review_tests, quote_sources
from desman.core.models import Kind, Subtask, TaskSpec
from desman.core.owner import current_owner, owner_is_gone
from desman.core.policy import Access, PolicyError, Role, default_policy, switched_on
from desman.core.retry import FailureKind, RetryPolicy, Strategy, strategy_for
from desman.core.sandbox import Sandbox, SandboxUnavailable
from desman.core.search import Tavily, client, open_sources
from desman.core.states import Actor
from desman.core.store import DEFAULT_LEASE_SECONDS, DEFAULT_REVIEW_SECONDS, Store
from desman.core.tools import TOOL_RESULTS_DIR, Registry, builtin_registry
from desman.core.web import SearchEngine, Web, WebCache
from desman.core.workspace import PathEscapeError, Workspace, is_work_file
from desman.i18n import t
from desman.llm import catalog, hub, router
from desman.llm.base import LLM, Completion, RateLimited, Request
from desman.llm.cache import Cached
from desman.llm.limiter import Buckets, Limited, Local
from desman.llm.openai_compat import key_id
from desman.net import jobs, protocol
from desman.net.node import EXTEND_WHEN, capabilities
from desman.net.search_lend import Lent, borrower_identity
from desman.planner import MAX_PARTS, MAX_SUBTASKS, decomposer
from desman.searxng import engine as searxng_engine

PAGE_TUNING_DEFAULTS: Mapping[str, int] = {"workers": 3, "chat_turns": 4}

TUNING_DEFAULTS: Mapping[str, int] = {
    "master_rounds": MASTER_TOOL_ROUNDS,
    "code_rounds": CODE_TOOL_ROUNDS,
    "max_subtasks": MAX_SUBTASKS,
    "max_parts": MAX_PARTS,
    "context_tokens": DEFAULT_BUDGET_TOKENS,
    "max_attempts": RetryPolicy().max_attempts,
    "max_transient": RetryPolicy().max_transient,
}
"""What each number of `config.Tuning` is when the person left it unset."""


def tuned(tuning: Tuning, name: str) -> int:
    """A number of the tuning: the person's, or the default."""
    value: int | None = getattr(tuning, name)
    return {**TUNING_DEFAULTS, **PAGE_TUNING_DEFAULTS}[name] if value is None else value


def tuned_spec(spec: TaskSpec, tuning: Tuning) -> TaskSpec:
    """A planned task with the retry budgets of the tuning (a --spec keeps its own)."""
    retry = RetryPolicy(
        max_attempts=tuned(tuning, "max_attempts"), max_transient=tuned(tuning, "max_transient")
    )
    return spec.model_copy(update={"retry": retry})


@dataclass(frozen=True)
class Master:
    """Who may be the master module (ADR 0015), as found at one moment."""

    models: tuple[LLM, LLM | None] | None
    choice: router.Choice | None = None
    problem: str | None = None
    """Why none may (`router.NoProvider`)."""
    other_choice: router.Choice | None = None
    """The model an "other model" attempt of the master asks first (None: there is none)."""


@dataclass(frozen=True)
class Models:
    llm: LLM
    """The executor's models; every role without its own (`roles`) asks them too."""
    other: LLM | None
    """For "other model" retries (ROADMAP 2.9); None if there is no other model."""
    choice: router.Choice | None = None
    """The executor's first model; None for a fake."""
    roles: Mapping[calls.Role, LLM] = field(default_factory=dict)
    other_choice: router.Choice | None = None
    kinds: Mapping[Kind, tuple[LLM, LLM | None]] = field(default_factory=dict)
    """The executor's main and other models for subtasks of each kind (ROADMAP 4.5); a kind
    not in it (all of them, for a fake) asks `llm` and `other`."""
    step_seconds: int = 0
    """The longest one step of an agent may wait on its models (0: unknown, a fake). A lease
    or a review's reservation shorter than that could run out while nobody can renew it."""
    master: tuple[LLM, LLM | None] | None = None
    """The main and other models of a master subtask (ADR 0015); None: no qualified model is
    ready (a fake serves it with `llm`)."""
    master_choice: router.Choice | None = None
    """The model a master subtask asks first (None for a fake, or without one)."""
    no_master: str | None = None
    """Why no model may be the master now (None: one may, or a fake)."""
    find_master: Callable[[], Master] | None = None
    """The master as the tuning is now (a worker's is refreshed while it runs): a model the
    person names after the run started serves the next attempt. None: `master` is all there is."""

    def for_(self, role: calls.Role) -> LLM:
        return self.roles.get(role, self.llm)

    def executor(self, kind: Kind, master: bool = False) -> tuple[LLM, LLM | None]:
        """The main and other models of the executor of a subtask of `kind` (of a master
        subtask: the master's, or the kind's when no model qualifies; the executor refuses
        such work, `_executor`)."""
        current = self.current_master() if master else None
        if current is not None:
            return current
        return self.kinds.get(kind, (self.llm, self.other))

    def master_status(self) -> Master:
        """The current lineup and the information shown to the person choosing it."""
        if self.find_master is not None:
            return self.find_master()
        return Master(self.master, self.master_choice, self.no_master)

    def current_master(self) -> tuple[LLM, LLM | None] | None:
        """The main and other models of a master subtask now; None: none qualifies."""
        return self.master_status().models

    def labels(self, subtask: Subtask) -> Labels:
        """How the coordinator tells apart the models the executor asks for `subtask`."""
        main, other = self.executor(subtask.kind, subtask.master)
        with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id=subtask.task_id)):
            chain = other.current() if isinstance(other, Scoped) else other
            choices = getattr(chain, "choices", None) if chain is not None else None
            main_chain = main.current() if isinstance(main, Scoped) else main
            main_choices = getattr(main_chain, "choices", None)
            if main_choices:
                alternatives = tuple(c.label for c in router.other_lineup(main_choices))
            else:
                alternatives = tuple(c.label for c in choices) if choices else ()
            return label(main), (alternatives or (label(other),)) if other is not None else None

    def first(self, role: calls.Role) -> router.Choice | None:
        """The model `role` asks first (None for a fake)."""
        choices = getattr(self.for_(role), "choices", None)
        return choices[0] if choices else None


def models(
    config: Config,
    conn: sqlite3.Connection,
    *,
    provider: str | None = None,
    model: str | None = None,
    cache: bool = True,
    client: httpx.Client | None = None,
    clock: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] | None = None,
    only: Collection[str] | None = None,
    timeout_cap: float | None = None,
    wrap: Callable[[LLM, router.Choice], LLM] | None = None,
    read_tuning: Callable[[], Tuning] | None = None,
    read_master: Callable[[], MasterChoices] | None = None,
    master_candidate: str | None = None,
    task_conn: sqlite3.Connection | None = None,
) -> Models:
    """A lineup of models per role (`router.lineup`), each model paced by the rate limiter in
    the database, quick blips retried, calls logged and accepted replies kept there. `provider`
    and `model` narrow the choice like `desman run --provider/--model`; only a pinned `model`
    leaves no other model for retries. `only`: the models of these providers alone (the ones
    lent to desman-global); `wrap` goes around each model inside the cache (a quota). Raises
    `router.NoProvider`. `read_tuning` supplies the worker’s in-memory tuning snapshot,
    and `read_master` its admissions. Both are refreshed at its config checks.
    `master_candidate` pins the benchmark executor to one named model. If `task_conn` is
    supplied, benchmark task attribution and per-run call totals are recorded there."""
    clients: dict[tuple[str, str | None, tuple[str, ...]], LLM] = {}
    timeouts: list[float] = []
    buckets = Buckets(conn, clock)
    recorder = calls.recorder(conn, task_conn=task_conn)
    used: dict[tuple[calls.Role, catalog.Lane | None, tuple[str, str] | None], str] = {}

    def member(
        choice: router.Choice, first: str | None, *, held: Set[tuple[str, str]] = frozenset()
    ) -> LLM:
        # Share key state between roles with the same model and effective key order.
        keys = config.keys(choice.provider.key_env or "")
        applicable = first if any(key_id(k.value) == first for k in keys) else None
        # Identical orders share refusal/limit state, including an explicit default key.
        if keys and applicable == key_id(keys[0].value):
            applicable = None
        keys = [k for k in keys if (key_id(k.value), choice.label) not in held]
        cfg = config.model_copy(
            update={
                "api_keys": {
                    **config.api_keys,
                    **({choice.provider.key_env: keys} if choice.provider.key_env else {}),
                }
            }
        )
        identity = (choice.label, applicable, tuple(key_id(k.value) for k in keys))
        if identity not in clients:
            api = router.connect(
                cfg,
                choice,
                first_key=applicable,
                timeout_cap=timeout_cap,
                on_call=recorder,
                client=client,
            )
            timeouts.append(min(api.timeout, api.max_timeout))
            paced: LLM
            server = None
            if choice.provider.local:  # paced by the machine, not by a rate
                server = f"{choice.provider.id}@{api.base_url}"
                paced = Local(api, buckets, choice.label, server, sleep=sleep)
            else:
                key_count = max(1, len(api.keys))
                paced = Limited(
                    api, buckets, choice.label, choice.spec.rpm * key_count, sleep=sleep
                )
            backed: LLM = router.Backoff(paced, sleep=sleep)
            if wrap is not None:
                backed = wrap(backed, choice)
            clients[identity] = (
                Cached(backed, conn, choice.model, server=server) if cache else backed
            )
        return clients[identity]

    def chain(
        choices: list[router.Choice],
        role: calls.Role = calls.Role.EXECUTOR,
        kind: catalog.Lane | None = None,
        current: Config = config,
        *,
        other: bool = False,
        held: Set[tuple[str, str]] = frozenset(),
    ) -> router.Fallback:
        first = router.preferred_key(current, role, kind)
        automatic = (
            provider is None
            and model is None
            and router.preferred(current, role, kind) is None
            and router.preferred_key(current, role, kind) is None
        )
        capacity = (
            router.capacity_order(
                conn, current, router.spread_tier(choices, role, kind), buckets.clock
            )
            if automatic
            else None
        )

        def order(available: list[router.Choice]) -> list[router.Choice]:
            actual = used.get((role, kind, router.call_scope()))
            if other:
                if actual is None:
                    alternatives = router.other_lineup(available)
                    return alternatives or available
                main = next((c for c in available if c.label == actual), None)
                rest = [c for c in available if c.label != actual]
                if main is not None:
                    rest.sort(key=lambda c: c.provider.id == main.provider.id)
                return rest or available
            return capacity(available) if capacity else available

        def remember(label: str) -> None:
            identity = (role, kind, router.call_scope())
            used.pop(identity, None)
            used[identity] = label
            if len(used) > 256:
                used.pop(next(iter(used)))

        def actual_label() -> str | None:
            if other:
                try:
                    alternatives = order(choices)
                except ModuleWaiting:
                    return None
                return alternatives[0].label if alternatives else None
            return used.get((role, kind, router.call_scope()))

        result = router.Fallback(
            [(member(c, first, held=held), c) for c in choices],
            order=order if other or capacity else None,
            sticky=not other,
            health=(
                router.health_order(
                    conn,
                    buckets.clock,
                    keep=choices[0].label
                    if (router.preferred(current, role, kind) is not None or first is not None)
                    and not other
                    else None,
                )
                if model is None
                else None
            ),
            on_used=None if other else remember,
            label_for=actual_label,
            conn=conn,
        )
        if other:
            alternatives = router.other_lineup(choices)
            if alternatives:
                result.label = alternatives[0].label
        return result

    local_choices = {
        (p.id, p.model(config)): router._choice(config, p, p.model(config))
        for p in providers.ready(config)
        if p.local
    }
    candidate = (
        router.master_candidate(config, master_candidate, local_choices=local_choices)
        if master_candidate is not None
        else None
    )
    if (
        candidate is not None
        and provider is None
        and not any(not p.paid for p in providers.ready(config))
    ):
        provider = candidate.provider.id

    def master_identity(current: Config) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return tuple(current.master.admitted), tuple(current.master.revoked)

    def current_config() -> Config:
        if read_tuning is None and read_master is None:
            return config
        return config.model_copy(
            update={
                "tuning": config.tuning if read_tuning is None else read_tuning(),
                "master": config.master if read_master is None else read_master(),
            }
        )

    lineups: dict[
        tuple[
            calls.Role,
            catalog.Lane | None,
            tuple[tuple[str, ...], tuple[str, ...]] | None,
            str | None,
            str | None,
        ],
        list[router.Choice],
    ] = {}

    def lineup(
        role: calls.Role, kind: catalog.Lane | None = None, current: Config = config
    ) -> list[router.Choice]:
        if kind == catalog.MASTER and candidate is not None:
            return [candidate]
        identity = (
            role,
            kind,
            master_identity(current) if kind == catalog.MASTER else None,
            router.preferred(current, role, kind),
            router.preferred_key(current, role, kind),
        )
        if identity not in lineups:
            lineups[identity] = router.lineup(
                current, role, provider, model, kind=kind, local_choices=local_choices
            )
        choices = lineups[identity]
        if only is not None:
            choices = [c for c in choices if c.provider.id in only]
            if not choices:
                raise router.NoProvider("none of the providers lent has a model for this")
        return choices

    def scoped(
        role: calls.Role,
        kind: catalog.Lane | None = None,
        *,
        other: bool = False,
        base: Config = config,
    ) -> LLM | None:
        own = lineup(role, kind, base)
        selected = router.other_lineup(own) if other else own
        if other and (model is not None or not selected):
            return None
        default = chain(own, role, kind, base, other=other)
        field = router.tuning_field(role, kind)
        chains: dict[tuple[str, ...], router.Fallback] = {}

        def current() -> router.Fallback:
            purpose = calls.purpose()
            scope: holds.Scope = (
                "network"
                if only is not None
                else "session"
                if role == calls.Role.INTERLOCUTOR
                else "task"
            )
            identity = (
                ""
                if only is not None
                else (purpose.session_id if scope == "session" else purpose.task_id) or ""
            )
            cfg = current_config() if only is None else config
            if only is None and identity and field:
                own_choice = module_choices.get(
                    conn, "session" if scope == "session" else "task", identity, field
                )
                cfg = module_choices.effective(cfg, field, own_choice)
            own = lineup(role, kind, cfg)
            current_holds = holds.active(conn, scope, identity)
            removed = {(h.source, h.model) for h in current_holds if h.state == "removed"}
            own = [
                c
                for c in own
                if any((source, c.label) not in removed for source in router.sources(cfg, c))
            ]
            # A stopped interlocutor queues instead of silently answering through a replacement.
            if (
                role == calls.Role.INTERLOCUTOR
                and own
                and any(h.state == "stopped" for h in current_holds)
            ):
                first = own[0]
                preferred = router.preferred_key(cfg, role, kind)
                sources = [
                    source
                    for source in router.sources(cfg, first)
                    if (source, first.label) not in removed
                ]
                source = preferred if preferred in sources else next(iter(sources), "")
                session = conn.execute(
                    "SELECT created_at FROM sessions WHERE id=?", (identity,)
                ).fetchone()
                latest = conn.execute(
                    "SELECT provider, model, key_id FROM model_calls WHERE session_id=? "
                    "AND role='interlocutor' AND error_kind IS NULL ORDER BY seq DESC LIMIT 1",
                    (identity,),
                ).fetchone()
                serving = {(source, first.label)}
                if latest is not None:
                    serving.add(
                        (
                            latest["key_id"] or latest["provider"],
                            f"{latest['provider']}/{latest['model']}",
                        )
                    )
                if any(
                    h.state == "stopped"
                    and (h.source, h.model) in serving
                    and (h.scope == "session" or session is not None and session[0] <= h.at)
                    for h in current_holds
                ):
                    raise ModuleWaiting(field or role.value)
            held = {(h.source, h.model) for h in current_holds}
            own = [
                c
                for c in own
                if any((source, c.label) not in held for source in router.sources(cfg, c))
            ]
            selected = own
            if not selected:
                raise ModuleWaiting(field or role.value)
            signature = (
                router.preferred_key(cfg, role, kind) or "",
                *(
                    c.label
                    + ":"
                    + ",".join(
                        source for source in router.sources(cfg, c) if (source, c.label) not in held
                    )
                    for c in selected
                ),
            )
            if signature not in chains:
                chains[signature] = chain(selected, role, kind, cfg, other=other, held=held)
            return chains[signature]

        return Scoped(current, default)

    main_roles: dict[calls.Role, LLM] = {}
    for role in calls.Role:
        main = scoped(role)
        assert main is not None
        main_roles[role] = main
    kinds: dict[Kind, tuple[LLM, LLM | None]] = {}
    for kind in Kind:
        main = scoped(calls.Role.EXECUTOR, kind)
        assert main is not None
        kinds[kind] = (main, scoped(calls.Role.EXECUTOR, kind, other=True))
    found: dict[tuple[str | None, str | None, tuple[tuple[str, ...], tuple[str, ...]]], Master] = {}

    def find_master() -> Master:
        cfg = current_config()
        # Only the master's own choices decide who it is (`router._master_lineup`).
        identity = (cfg.tuning.models.master, cfg.tuning.keys.master, master_identity(cfg))
        if identity not in found:
            try:
                masters = lineup(calls.Role.EXECUTOR, catalog.MASTER, cfg)
            except router.NoProvider as e:
                found[identity] = Master(None, problem=str(e))
            else:
                main = scoped(calls.Role.EXECUTOR, catalog.MASTER, base=cfg)
                assert main is not None
                other = scoped(calls.Role.EXECUTOR, catalog.MASTER, other=True, base=cfg)
                others = router.other_lineup(masters) if other is not None else []
                found[identity] = Master(
                    (main, other), masters[0], other_choice=others[0] if others else None
                )
        return found[identity]

    # Never for desman-global (ADR 0015, decision 6).
    first_master = find_master() if only is None else Master(None)
    executor = lineup(calls.Role.EXECUTOR)
    others = [] if model is not None else router.other_lineup(executor)
    return Models(
        main_roles[calls.Role.EXECUTOR],
        scoped(calls.Role.EXECUTOR, other=True),
        executor[0],
        main_roles,
        others[0] if others else None,
        kinds,
        step_seconds(max(timeouts, default=0.0)),
        first_master.models,
        first_master.choice,
        first_master.problem,
        find_master if only is None else None,
    )


class ModuleWaiting(RateLimited):
    """A human hold is a LIMIT wait: it never spends a task attempt."""

    automatic_wait = False

    def __init__(self, role: str) -> None:
        super().__init__("modules", retry_after=30)
        self.role = role
        self.args = (f"no module left for {role}",)

    def display(self) -> str:
        return t("no module left for {role}", role=self.role)


class Scoped(router.Fallback):
    """Resolve a task/session preference at every call, retaining acceptance on its chain."""

    def __init__(self, current: Callable[[], router.Fallback], default: router.Fallback) -> None:
        super().__init__(default.members)
        self.current = current
        self.last: router.Fallback | None = None

    @property
    def label(self) -> str:
        return self.current().label if hasattr(self, "current") else self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    @property
    def choices(self) -> list[router.Choice]:
        return self.current().choices

    @property
    def last_call(self) -> router.LastCall:
        return self.last.last_call if self.last is not None else router.LastCall()

    @property
    def has_untried(self) -> bool:
        return self.last is not None and self.last.has_untried

    def complete(self, request: Request) -> Completion:
        self.last = self.current()
        return self.last.complete(request)

    def accept(self, request: Request, reply: Completion) -> None:
        if self.last is not None:
            self.last.accept(request, reply)


CALLS_PER_STEP = 2
"""Model calls one step of an agent may make in a row: its request and one repair request
(`llm.structured.complete_json`)."""
WAITS_PER_CALL = 30.0
"""Seconds a call may spend besides the model's answer: quick repeats (`router.Backoff`) and
waiting its turn (`llm.limiter`)."""


def step_seconds(timeout: float) -> int:
    """How long one step may take when each call may take up to `timeout` seconds."""
    if timeout <= 0:
        return 0
    return int(CALLS_PER_STEP * (timeout + WAITS_PER_CALL) + WAITS_PER_CALL)


def label(llm: LLM) -> str:
    """How the coordinator tells the executor's models apart (`Failure.unfit_provider`)."""
    return str(getattr(llm, "label", llm.name))


def executor_access(
    config: Config, *, stderr: bool = False, sandbox: Sandbox | None = None
) -> Access:
    """What the executor's tools may do: the tool policy within the human's permissions. Warns
    about switches that are on but cannot work, and about admin_rights at every run."""
    permissions = config.permissions
    if permissions.admin_rights:  # at every run, not only when it was turned on
        ui.warn(t("admin_rights is on: agents may act with elevated privileges."), stderr=stderr)
    if permissions.shell_execution:
        sandbox = Sandbox() if sandbox is None else sandbox
        problem = sandbox.unavailable()
        if problem is not None:  # the switch alone would look like commands can run
            ui.warn(
                t(
                    "shell_execution is on, but there is no sandbox to run commands in "
                    "({reason}): run_command stays off.",
                    reason=problem,
                ),
                stderr=stderr,
            )
            if sandbox.image_present is False:
                ui.warn(t("Set up the Docker sandbox with `desman sandbox setup`."), stderr=stderr)
            elif sandbox.backend == "docker" and not sandbox.docker:
                ui.warn(t("Install Docker with `desman docker install`."), stderr=stderr)
    if (
        permissions.network_access
        and not providers.search_engines(config)
        and borrower_identity(config) is None
    ):
        ui.warn(
            t(
                "network_access is on, but no engine for the whole web is set up: web_search "
                "covers only Wikipedia, Stack Overflow, arXiv and GitHub. Add one in `desman "
                "search setup` or use `desman config`."
            ),
            stderr=stderr,
        )
    try:
        policy = default_policy()
    except PolicyError as e:
        raise click.ClickException(t("Cannot use the tool policy: {error}", error=e)) from e
    return Access(policy, Role.EXECUTOR, switched_on(permissions.model_dump()))


def guard(store: Store) -> None:
    """Refuse to run while protected files changed without the person's approval (ROADMAP
    5.3): called before any work starts, and again by every coordinator put together here."""
    try:
        protected.require(store.conn)
    except protected.ProtectedError as e:
        raise click.ClickException(t("Cannot check the protected files: {error}", error=e)) from e
    except protected.Blocked as e:
        raise click.ClickException(blocked(e.changes)) from e


def blocked(changes: list[protected.Change]) -> str:
    return t(
        "Protected files changed without your approval: {files}. Nothing runs until you look "
        "at the change (`desman protected show --diff`) and approve it (`desman protected "
        "approve`).",
        files=", ".join(c.path for c in changes),
    )


def web(
    config: Config,
    store: Store,
    http: httpx.Client | None = None,
    *,
    load: Callable[[], Config] | None = None,
) -> Web:
    """The web tools' view of the config: the open sources, the engines the human set up (in
    `providers.SEARCH` order), the cache. The engines share `http`; a new client without it
    (a one-task process), so a long-lived caller passes its own and closes it."""
    http = client() if http is None else http
    engines: list[SearchEngine] = []
    for p in providers.search_engines(config):
        if p.id == "searxng":
            engines.append(searxng_engine(config, http, store.clock))
        elif p.id == "tavily" and p.key_env is not None:
            engines.append(Tavily([k.value for k in config.keys(p.key_env)], http))
    if not engines and (lent := Lent.from_config(config, store.conn, http, store.clock, load=load)):
        engines.append(lent)
    return Web(engines=engines, sources=open_sources(http), cache=WebCache(store.conn, store.clock))


def document_web(
    goal: str, config: Config, store: Store, access: Access, http: httpx.Client | None = None
) -> Web | None:
    """Prefetch only when the executor could use the registered fetch tool."""
    if not documents.reading_urls(goal):
        return None
    fetch = builtin_registry().get("web_fetch")
    if fetch is None or access.refusal(fetch) is not None:
        return None
    return web(config, store, http)


def prefetch_notices(fetched: documents.Prefetched) -> list[str]:
    chars = {item["url"]: item.get("chars") for item in fetched.details}
    return [
        t(
            "Saved the page {url} ({chars} characters) to read it in parts.",
            url=file.source,
            chars=chars[file.source],
        )
        for file in fetched.given
    ]


def task_tree(conn: sqlite3.Connection, personalize: bool) -> tuple[knowledge.Node, ...]:
    """The knowledge tree tasks choose from (K.3): the release part, and the person's nodes
    (K.4) while they agreed to personalisation."""
    return knowledge.for_tasks(knowledge.personal(conn) if personalize else ())


def _personalize() -> bool:
    """Whether the person agrees to personalisation now (no config that loads: no)."""
    try:
        return open_config().load().personalize
    except ConfigError:
        return False


_REVIEW_COPY_PREFIX = "review-"


def _sweep_review_copies(root: Path) -> None:
    cutoff = time.time() - 3_600
    for path in root.glob(f"{_REVIEW_COPY_PREFIX}*"):
        with suppress(OSError):
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode) and info.st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)


def _run_review_tests(
    sandbox: Sandbox,
    subtask: Subtask,
    workspace: Workspace,
    module: str,
    attempt: int,
    cover: Callable[[float], None] = lambda _: None,
) -> TestRun:
    cover(30)
    file = review_test_file(subtask.id, attempt)
    config = "[pytest]\npythonpath = ..\n"
    skipped = 0
    skipped_names: list[str] = []

    def skip(path: str) -> None:
        nonlocal skipped
        skipped += 1
        if len(skipped_names) < 5:
            skipped_names.append(path)

    def over_quota(entries: int, size: int) -> TestRun:
        limits = sandbox.limits
        message = (
            f"The workspace is too large for the reviewer's test run "
            f"(at least {entries} entries and {size} bytes; sandbox quota: "
            f"{limits.workspace_files} entries and {limits.workspace_bytes} bytes). "
            "Remove files the reviewer's run does not need."
        )
        return TestRun(
            ran=False,
            reason_not_run=message,
            copy_over_quota=message,
            skipped_files=skipped,
            skipped_names=skipped_names,
        )

    try:
        if problem := sandbox.unavailable():
            raise SandboxUnavailable(problem)
        root = workspace.root.parent
        _sweep_review_copies(root)
        paths = []
        estimated = 0
        for path in workspace.files(on_skip=skip):
            if not is_work_file(path) or path.startswith(("web/", TOOL_RESULTS_DIR)):
                continue
            try:
                estimated += workspace.size(path)
            except (FileNotFoundError, PermissionError, PathEscapeError):
                skip(path)
                continue
            paths.append(path)
        limits = sandbox.limits

        def copy_seconds(remaining: int) -> float:
            return 30 + min(estimated, limits.workspace_bytes) / 10_000_000 + 0.05 * remaining

        cover(copy_seconds(len(paths)))
        with TemporaryDirectory(
            prefix=_REVIEW_COPY_PREFIX, dir=root, ignore_cleanup_errors=True
        ) as directory:
            location = Path(directory)
            copy = Workspace(location.parent, location.name)
            directories: set[PurePosixPath] = set()
            entries = size = 0
            for index, path in enumerate(paths):
                if index and index % 500 == 0:
                    cover(copy_seconds(len(paths) - index))
                parents = set(PurePosixPath(path).parents) - {PurePosixPath(".")}
                next_entries = entries + 1 + len(parents - directories)
                try:
                    next_size = size + workspace.size(path)
                    if next_entries > limits.workspace_files or next_size > limits.workspace_bytes:
                        return over_quota(next_entries, next_size)
                    data = workspace.read(path, limit=limits.workspace_bytes - size + 1)
                except (FileNotFoundError, PermissionError, PathEscapeError):
                    skip(path)
                    continue
                next_size = size + len(data)
                if next_size > limits.workspace_bytes:
                    return over_quota(next_entries, next_size)
                copy.write(path, data)
                entries, size = next_entries, next_size
                directories.update(parents)
            copy.write(file, module)
            copy.write(".review/pytest.ini", config)
            review_limits = replace(
                limits,
                workspace_files=limits.workspace_files + len(PurePosixPath(file).parents) + 1,
                workspace_bytes=limits.workspace_bytes + len(module.encode()) + len(config),
            )
            cover(REVIEW_TEST_TIMEOUT)
            # python3 is the interpreter name supplied by every Sandbox backend.
            outcome = sandbox.run(
                [
                    "python3",
                    "-I",
                    "-m",
                    "pytest",
                    "-c",
                    ".review/pytest.ini",
                    "--rootdir",
                    ".review",
                    "--noconftest",
                    "-p",
                    "no:cacheprovider",
                    "-o",
                    "addopts=",
                    "-q",
                    file,
                ],
                copy.root,
                timeout=REVIEW_TEST_TIMEOUT,
                limits=review_limits,
            )
    except SandboxUnavailable as error:
        return TestRun(
            ran=False,
            reason_not_run=str(error)[:1_000],
            sandbox_unavailable=True,
            skipped_files=skipped,
            skipped_names=skipped_names,
        )
    summary = outcome.stdout.splitlines()
    counts: dict[str, int] = {}
    for line in reversed(summary):
        if match := re.fullmatch(
            r"(?:=+\s+)?(\d+ [a-z][a-z ]*(?:, \d+ [a-z][a-z ]*)*) in \d+\.\d+s(?:\s+=+)?",
            line.strip(),
        ):
            counts = {
                status: int(count)
                for count, status in re.findall(
                    r"\b([0-9]{1,9}) (passed|failed|errors?)\b", match[1]
                )
            }
            break
    failing = [
        match.group(1)[:250]
        for line in summary
        if (match := re.match(r"^(?:FAILED|ERROR) (.*?)(?: - |$)", line))
    ]
    tail = "\n".join(
        part
        for part in (outcome.stdout, outcome.stderr, outcome.stopped, outcome.workspace_full)
        if part
    )
    return TestRun(
        ran=True,
        exit_code=outcome.exit_code,
        passed=counts.get("passed", 0),
        failed=counts.get("failed", 0),
        errors=counts.get("errors", counts.get("error", 0)),
        failing=failing[:20],
        tail=tail[-4_000:],
        timed_out=outcome.timed_out,
        workspace_full=outcome.workspace_full,
        stopped=outcome.stopped,
        disk_low=outcome.disk_low,
        skipped_files=skipped,
        skipped_names=skipped_names,
    )


def _review_test_context(store: Store, subtask: Subtask) -> dict[str, Any]:
    goal, is_final = store.subtask_goal(subtask)
    return {
        **({} if is_final else {"task_goal": goal}),
        "previous_review_tests": last_review_tests(store, subtask.id),
    }


def coordinator(
    store: Store,
    models: Models,
    access: Access,
    config: Config,
    *,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    http: httpx.Client | None = None,
    watch: Callable[[], None] | None = None,
    network: Crew | None = None,
    sandbox: Sandbox | None = None,
    document_reader: Web | None = None,
    load: Callable[[], Config] | None = None,
    read_tuning: Callable[[], Tuning] | None = None,
) -> Coordinator:
    http = client() if http is None else http
    # The person's own work only: never a job's (`network`). Their nodes are read fresh, and
    # only while personalisation is on (turned off mid-run: no longer given).
    tree = knowledge.reader(lambda: task_tree(store.conn, _personalize()))
    sandbox = sandbox if sandbox is not None else Sandbox()
    return _coordinator(
        store,
        models,
        access=access,
        tools=builtin_registry(
            sandbox,
            web=document_reader
            if document_reader is not None
            else web(config, store, http, load=load),
            # Without the person's token: see `core.tools.suggest_model_tool`.
            models=lambda ref: hub.inspect(ref, http),
            knowledge=tree,
        ),
        review_tests=functools.partial(_run_review_tests, sandbox),
        knowledge=tree,
        lease_seconds=lease_seconds,
        watch=watch,
        tuning=config.tuning,
        read_tuning=read_tuning,
        network=network,
    )


def network_crew(
    store: Store,
    config: Config,
    load: Callable[[], Config],
    *,
    given: Models | None = None,
    cache: bool = True,
    clock: Callable[[], datetime] | None = None,
) -> Crew | None:
    """Who works jobs from desman-global in this worker (ROADMAP G.6b): the models of the
    providers lent (`net.node.capabilities`), each within its day's quota (`jobs.Quota`, the
    config read at every call), tools by the `network_*` roles within the person's switches.
    None while desman-global is off or nothing usable is lent: the worker takes no jobs then
    (a change later counts from the worker's next start). `given`: these models instead (a
    fake: its scripts count every call)."""
    if net.registry(config) is None:
        return None
    lent = {c.provider for c in capabilities(config) if c.provider != protocol.SEARCH_PROVIDER}
    if not lent:
        return None
    lent_models = given
    if lent_models is None:
        now = clock or (lambda: datetime.now(UTC))
        try:
            lent_models = models(
                config,
                store.conn,
                cache=cache,
                clock=clock,
                only=lent,
                # Workers poll the registry between calls. Leave room for pacing and backoff
                # within the shortest remaining lease that the registry client won't extend.
                timeout_cap=EXTEND_WHEN.total_seconds() - WAITS_PER_CALL,
                wrap=lambda llm, choice: jobs.Quota(llm, store.conn, choice.provider.id, load, now),
            )
        except router.NoProvider:
            return None
    crew_models = lent_models
    try:
        policy = default_policy()
    except PolicyError as e:
        raise click.ClickException(t("Cannot use the tool policy: {error}", error=e)) from e
    reviewer = crew_models.for_(calls.Role.REVIEWER)
    executor = llm_executor(
        crew_models.llm,
        crew_models.other,
        by_kind=crew_models.kinds,
        clock=store.clock,
        format_final=False,
        code_rounds=MAX_TOOL_ROUNDS,  # lent models share their owners' quotas
        tz=UTC,  # another node rechecks: one date for both, wherever they are
    )
    return Crew(
        jobs.executor(store.conn, executor, clock=store.clock),
        jobs.reviewer(
            store.conn,
            llm_reviewer(
                reviewer, evidence=functools.partial(evidence, store), clock=store.clock, tz=UTC
            ),
            llm_reviewer(reviewer, clock=store.clock, tz=UTC),
            clock=store.clock,
        ),
        # The person's switches as they are at each tool call, not at the worker's start: a
        # permission turned off holds from the next call on (a config that fails: none on).
        Access(policy, Role.NETWORK_EXECUTOR, lambda: switched_on(load().permissions.model_dump())),
        strategies=functools.partial(strategy_for, other_model=crew_models.other is not None),
        providers=crew_models.labels,
        planner=crew_models.for_(calls.Role.NETWORK_PLANNER),
        step_seconds=crew_models.step_seconds,
    )


def summarizer(
    store: Store, models: Models, tuning: Tuning | None = None, *, master_check: bool = False
) -> Coordinator:
    """The operator for a summary of a chat session (`desman.compaction`): an ordinary run, but
    its executor has no tools at all, whatever the permissions. What it reads is a conversation,
    and nothing written in it can make the work act. The benchmark (`desman.bench`) runs on it
    too, with the tuning it evaluates."""
    return _coordinator(
        store, models, access=None, tools=Registry(), tuning=tuning, master_check=master_check
    )


NO_MASTER = (
    "no model qualified for a master module is ready: add a key of a provider whose model the "
    "catalog marks or the operator admitted, or name a local model with "
    "`desman tuning --model master=...`; then "
    "retry the subtask (a worker or the page sees a new key once it restarts)"
)
"""Why a master subtask waits for the person when nobody can be its master (audit, English)."""


def _executor(
    models: Models,
    store: Store,
    read_tuning: Callable[[], Tuning] = Tuning,
    *,
    master_check: bool = False,
) -> Executor:
    def execute(work: Work) -> Outcome:
        if work.subtask.master and models.choice is not None and models.current_master() is None:
            # Never worked by a model that was not qualified for it (ADR 0015, decision 2).
            return Failure(kind=FailureKind.PERMANENT, reason=NO_MASTER)
        main, other = models.executor(work.subtask.kind, work.subtask.master)
        if work.context.strategy is Strategy.OTHER_MODEL and other is not None:
            # Holds may remove the alternate after the retry strategy was selected.
            available = getattr(other, "choices", None)
            if available is not None and len(available) <= 1:
                store.note(
                    work.subtask.id,
                    Actor.EXECUTOR,
                    "strategy_fallback",
                    {"from": "other_model", "to": "rephrase", "reason": "no other module"},
                )
                other = None  # The executor already rephrases when no alternate exists.
        tuning = read_tuning()
        return llm_executor(
            main,
            other,
            clock=store.clock,
            master_rounds=tuned(tuning, "master_rounds"),
            code_rounds=tuned(tuning, "code_rounds"),
            allow_delegation=not master_check,
        )(work)

    return execute


def _coordinator(
    store: Store,
    models: Models,
    *,
    access: Access | None,
    tools: Registry,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    watch: Callable[[], None] | None = None,
    tuning: Tuning | None = None,
    network: Crew | None = None,
    knowledge: Knowledge | None = None,
    read_tuning: Callable[[], Tuning] | None = None,
    master_check: bool = False,
    review_tests: ReviewTests | None = None,
) -> Coordinator:
    tuning = tuning or Tuning()
    guard(store)
    # A step may not outlast what holds the subtask while it waits: a slow local model gets a
    # longer lease and review reservation (each only as long as needed, since a longer one
    # delays taking over the work of a run that died elsewhere).
    return Coordinator(
        store,
        _executor(models, store, read_tuning or (lambda: tuning), master_check=master_check),
        llm_reviewer(
            models.for_(calls.Role.REVIEWER),
            evidence=functools.partial(evidence, store, knowledge=knowledge),
            dependencies=store.dependency_results,
            was_decomposed=store.was_decomposed,
            test_context=functools.partial(_review_test_context, store) if review_tests else None,
            quote_evidence=functools.partial(
                quote_sources,
                store,
                knowledge=knowledge,
                context_budget_tokens=tuned(tuning, "context_tokens"),
            ),
            clock=store.clock,
        ),
        owner=current_owner(),
        lease_seconds=max(lease_seconds, models.step_seconds),
        review_seconds=max(DEFAULT_REVIEW_SECONDS, models.step_seconds),
        abandoned=owner_is_gone,  # a run killed on this machine is taken over at once
        # Without another provider, "other model" attempts are rephrased instead (and audited so).
        strategies=functools.partial(strategy_for, other_model=models.other is not None),
        providers=models.labels,
        decomposer=None
        if master_check
        else decomposer(
            models.for_(calls.Role.DECOMPOSER), limit=tuned(tuning, "max_parts"), clock=store.clock
        ),
        context_budget_tokens=tuned(tuning, "context_tokens"),
        access=access,
        tools=tools,
        watch=watch,
        gate=functools.partial(guard, store),  # a running worker stops at a later change too
        network=network,
        knowledge=knowledge,
        review_tests=review_tests,
    )
