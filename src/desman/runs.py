"""What a run of the local operator is made of, shared by `desman run`/`resume` and the chat.

The operator (ADR 0003) is the coordinator with its planner, executor and reviewer. Here it is
put together from the config: the models (a lineup per role, `llm.router`; the executor's per
kind of subtask, each with its other lineup for retries with another model), the executor's
access (the tool policy within the human's permissions) and its tools.
"""

from __future__ import annotations

import functools
import sqlite3
from collections.abc import Callable, Collection, Mapping, Set
from dataclasses import dataclass, field
from datetime import UTC, datetime

import click
import httpx

from desman import knowledge, net, providers, ui
from desman.agents import llm_executor, llm_reviewer
from desman.config import Config, ConfigError, Tuning, open_config
from desman.core import calls, holds, protected
from desman.core import choices as module_choices
from desman.core.context import DEFAULT_BUDGET_TOKENS, Knowledge
from desman.core.coordinator import Coordinator, Crew, Executor, Labels, Outcome, Work
from desman.core.effects import evidence
from desman.core.models import Kind, Subtask, TaskSpec
from desman.core.owner import current_owner, owner_is_gone
from desman.core.policy import Access, PolicyError, Role, default_policy, switched_on
from desman.core.retry import RetryPolicy, Strategy, strategy_for
from desman.core.sandbox import Sandbox
from desman.core.search import SearXNG, Tavily, client, open_sources
from desman.core.states import Actor
from desman.core.store import DEFAULT_LEASE_SECONDS, DEFAULT_REVIEW_SECONDS, Store
from desman.core.tools import Registry, builtin_registry
from desman.core.web import SearchEngine, Web, WebCache
from desman.i18n import t
from desman.llm import hub, router
from desman.llm.base import LLM, Completion, RateLimited, Request
from desman.llm.cache import Cached
from desman.llm.limiter import Buckets, Limited, Local
from desman.llm.openai_compat import key_id
from desman.net import jobs
from desman.net.node import capabilities
from desman.planner import MAX_PARTS, MAX_SUBTASKS, decomposer

PAGE_TUNING_DEFAULTS: Mapping[str, int] = {"workers": 3, "chat_turns": 4}

TUNING_DEFAULTS: Mapping[str, int] = {
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

    def for_(self, role: calls.Role) -> LLM:
        return self.roles.get(role, self.llm)

    def executor(self, kind: Kind) -> tuple[LLM, LLM | None]:
        """The main and other models of the executor of a subtask of `kind`."""
        return self.kinds.get(kind, (self.llm, self.other))

    def labels(self, subtask: Subtask) -> Labels:
        """How the coordinator tells apart the models the executor asks for `subtask`."""
        main, other = self.executor(subtask.kind)
        with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id=subtask.task_id)):
            chain = other.current() if isinstance(other, Scoped) else other
            choices = getattr(chain, "choices", None) if chain is not None else None
            if choices and isinstance(chain, router.Fallback) and chain.order is not None:
                choices = chain.order(choices)
            alternatives = tuple(c.label for c in choices) if choices else None
            if alternatives is not None and len(alternatives) == 1:
                return label(main), alternatives[0]
            return label(main), alternatives or (None if other is None else label(other))

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
    wrap: Callable[[LLM, router.Choice], LLM] | None = None,
    read_tuning: Callable[[], Tuning] | None = None,
) -> Models:
    """A lineup of models per role (`router.lineup`), each model paced by the rate limiter in
    the database, quick blips retried, calls logged and accepted replies kept there. `provider`
    and `model` narrow the choice like `desman run --provider/--model`; only a pinned `model`
    leaves no other model for retries. `only`: the models of these providers alone (the ones
    lent to desman-global); `wrap` goes around each model inside the cache (a quota). Raises
    `router.NoProvider`. `read_tuning` supplies the worker’s in-memory tuning snapshot,
    refreshed at its existing config checks without I/O during routing."""
    clients: dict[tuple[str, str | None, tuple[str, ...]], LLM] = {}
    timeouts: list[float] = []
    buckets = Buckets(conn, clock)
    recorder = calls.recorder(conn)
    used: dict[tuple[calls.Role, Kind | None, tuple[str, str] | None], str] = {}

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
            api = router.connect(cfg, choice, first_key=applicable, on_call=recorder, client=client)
            timeouts.append(api.timeout)
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
        kind: Kind | None = None,
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
            on_used=None if other else remember,
            label_for=actual_label,
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
    lineups: dict[tuple[calls.Role, Kind | None, str | None, str | None], list[router.Choice]] = {}

    def lineup(
        role: calls.Role, kind: Kind | None = None, current: Config = config
    ) -> list[router.Choice]:
        identity = (
            role,
            kind,
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

    def scoped(role: calls.Role, kind: Kind | None = None, *, other: bool = False) -> LLM | None:
        own = lineup(role, kind)
        selected = router.other_lineup(own) if other else own
        if other and (model is not None or not selected):
            return None
        default = chain(own, role, kind, other=other)
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
            cfg = config
            if only is None and read_tuning is not None:
                cfg = config.model_copy(update={"tuning": read_tuning()})
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
    )


class ModuleWaiting(RateLimited):
    """A human hold is a LIMIT wait: it never spends a task attempt."""

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


def executor_access(config: Config, *, stderr: bool = False) -> Access:
    """What the executor's tools may do: the tool policy within the human's permissions. Warns
    about switches that are on but cannot work, and about admin_rights at every run."""
    permissions = config.permissions
    if permissions.admin_rights:  # at every run, not only when it was turned on
        ui.warn(t("admin_rights is on: agents may act with elevated privileges."), stderr=stderr)
    if permissions.shell_execution:
        problem = Sandbox().unavailable()
        if problem is not None:  # the switch alone would look like commands can run
            ui.warn(
                t(
                    "shell_execution is on, but there is no sandbox to run commands in "
                    "({reason}): run_command stays off.",
                    reason=problem,
                ),
                stderr=stderr,
            )
    if permissions.network_access and not providers.search_engines(config):
        ui.warn(
            t(
                "network_access is on, but no engine for the whole web is set up: web_search "
                "covers only Wikipedia, Stack Overflow, arXiv and GitHub. Add one in `desman "
                "config`."
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


def web(config: Config, store: Store, http: httpx.Client | None = None) -> Web:
    """The web tools' view of the config: the open sources, the engines the human set up (in
    `providers.SEARCH` order), the cache. The engines share `http`; a new client without it
    (a one-task process), so a long-lived caller passes its own and closes it."""
    http = client() if http is None else http
    engines: list[SearchEngine] = []
    for p in providers.search_engines(config):
        if p.id == "searxng":
            engines.append(SearXNG(config.settings["SEARXNG_URL"], http))
        elif p.id == "tavily" and p.key_env is not None:
            engines.append(Tavily([k.value for k in config.keys(p.key_env)], http))
    return Web(engines=engines, sources=open_sources(http), cache=WebCache(store.conn, store.clock))


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
) -> Coordinator:
    http = client() if http is None else http
    # The person's own work only: never a job's (`network`). Their nodes are read fresh, and
    # only while personalisation is on (turned off mid-run: no longer given).
    tree = knowledge.reader(lambda: task_tree(store.conn, _personalize()))
    return _coordinator(
        store,
        models,
        access=access,
        tools=builtin_registry(
            web=web(config, store, http),
            # Without the person's token: see `core.tools.suggest_model_tool`.
            models=lambda ref: hub.inspect(ref, http),
            knowledge=tree,
        ),
        knowledge=tree,
        lease_seconds=lease_seconds,
        watch=watch,
        tuning=config.tuning,
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
    lent = {c.provider for c in capabilities(config)}
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
        crew_models.llm, crew_models.other, by_kind=crew_models.kinds, clock=store.clock
    )
    return Crew(
        jobs.executor(store.conn, executor),
        jobs.reviewer(
            store.conn,
            llm_reviewer(reviewer, evidence=functools.partial(evidence, store), clock=store.clock),
            llm_reviewer(reviewer, clock=store.clock),
        ),
        # The person's switches as they are at each tool call, not at the worker's start: a
        # permission turned off holds from the next call on (a config that fails: none on).
        Access(policy, Role.NETWORK_EXECUTOR, lambda: switched_on(load().permissions.model_dump())),
        strategies=functools.partial(strategy_for, other_model=crew_models.other is not None),
        providers=crew_models.labels,
        planner=crew_models.for_(calls.Role.NETWORK_PLANNER),
    )


def summarizer(store: Store, models: Models, tuning: Tuning | None = None) -> Coordinator:
    """The operator for a summary of a chat session (`desman.compaction`): an ordinary run, but
    its executor has no tools at all, whatever the permissions. What it reads is a conversation,
    and nothing written in it can make the work act. The benchmark (`desman.bench`) runs on it
    too, with the tuning it evaluates."""
    return _coordinator(store, models, access=None, tools=Registry(), tuning=tuning)


def _executor(models: Models, store: Store) -> Executor:
    def execute(work: Work) -> Outcome:
        main, other = models.executor(work.subtask.kind)
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
        return llm_executor(main, other, clock=store.clock)(work)

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
) -> Coordinator:
    tuning = tuning or Tuning()
    guard(store)
    # A step may not outlast what holds the subtask while it waits: a slow local model gets a
    # longer lease and review reservation (each only as long as needed, since a longer one
    # delays taking over the work of a run that died elsewhere).
    return Coordinator(
        store,
        _executor(models, store),
        llm_reviewer(
            models.for_(calls.Role.REVIEWER),
            evidence=functools.partial(evidence, store, knowledge=knowledge),
            clock=store.clock,
        ),
        owner=current_owner(),
        lease_seconds=max(lease_seconds, models.step_seconds),
        review_seconds=max(DEFAULT_REVIEW_SECONDS, models.step_seconds),
        abandoned=owner_is_gone,  # a run killed on this machine is taken over at once
        # Without another provider, "other model" attempts are rephrased instead (and audited so).
        strategies=functools.partial(strategy_for, other_model=models.other is not None),
        providers=models.labels,
        decomposer=decomposer(
            models.for_(calls.Role.DECOMPOSER), limit=tuned(tuning, "max_parts"), clock=store.clock
        ),
        context_budget_tokens=tuned(tuning, "context_tokens"),
        access=access,
        tools=tools,
        watch=watch,
        gate=functools.partial(guard, store),  # a running worker stops at a later change too
        network=network,
        knowledge=knowledge,
    )
