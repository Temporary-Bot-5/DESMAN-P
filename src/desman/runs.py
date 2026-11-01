"""What a run of the local operator is made of, shared by `desman run`/`resume` and the chat.

The operator (ADR 0003) is the coordinator with its planner, executor and reviewer. Here it is
put together from the config: the models (a lineup per role, `llm.router`, and for retries with
another model the executor's other lineup), the executor's access (the tool policy within the
human's permissions) and its tools.
"""

from __future__ import annotations

import functools
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime

import click
import httpx

from desman import providers, ui
from desman.agents import llm_executor, llm_reviewer
from desman.config import Config
from desman.core import calls
from desman.core.coordinator import Coordinator
from desman.core.effects import evidence
from desman.core.owner import current_owner, owner_is_gone
from desman.core.policy import Access, PolicyError, Role, default_policy, switched_on
from desman.core.retry import strategy_for
from desman.core.sandbox import Sandbox
from desman.core.search import SearXNG, Tavily, client, open_sources
from desman.core.store import DEFAULT_LEASE_SECONDS, Store
from desman.core.tools import Registry, builtin_registry
from desman.core.web import SearchEngine, Web, WebCache
from desman.i18n import t
from desman.llm import router
from desman.llm.base import LLM
from desman.llm.cache import Cached
from desman.llm.limiter import Buckets, Limited
from desman.planner import decomposer


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

    def for_(self, role: calls.Role) -> LLM:
        return self.roles.get(role, self.llm)

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
) -> Models:
    """A lineup of models per role (`router.lineup`), each model paced by the rate limiter in
    the database, quick blips retried, calls logged and accepted replies kept there. `provider`
    and `model` narrow the choice like `desman run --provider/--model`; only a pinned `model`
    leaves no other model for retries. Raises `router.NoProvider`."""
    clients: dict[str, LLM] = {}
    buckets = Buckets(conn, clock)
    recorder = calls.recorder(conn)

    def member(choice: router.Choice) -> LLM:
        # One client per model, shared by the roles: its keys' state (refused, limited) is one.
        if choice.label not in clients:
            api = router.connect(config, choice, on_call=recorder, client=client)
            keys = max(1, len(api.keys))
            paced = Limited(api, buckets, choice.label, choice.spec.rpm * keys, sleep=sleep)
            backed = router.Backoff(paced, sleep=sleep)
            clients[choice.label] = Cached(backed, conn, choice.model) if cache else backed
        return clients[choice.label]

    def chain(choices: list[router.Choice]) -> router.Fallback:
        return router.Fallback([(member(c), c) for c in choices])

    lineups = {role: router.lineup(config, role, provider, model) for role in calls.Role}
    roles = {role: chain(choices) for role, choices in lineups.items()}
    executor = lineups[calls.Role.EXECUTOR]
    others = [] if model is not None else router.other_lineup(executor)
    return Models(
        roles[calls.Role.EXECUTOR],
        chain(others) if others else None,
        executor[0],
        roles,
        others[0] if others else None,
    )


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


def coordinator(
    store: Store,
    models: Models,
    access: Access,
    config: Config,
    *,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    http: httpx.Client | None = None,
    watch: Callable[[], None] | None = None,
) -> Coordinator:
    return _coordinator(
        store,
        models,
        access=access,
        tools=builtin_registry(web=web(config, store, http)),
        lease_seconds=lease_seconds,
        watch=watch,
    )


def summarizer(store: Store, models: Models) -> Coordinator:
    """The operator for a summary of a chat session (`desman.compaction`): an ordinary run, but
    its executor has no tools at all, whatever the permissions. What it reads is a conversation,
    and nothing written in it can make the work act."""
    return _coordinator(store, models, access=None, tools=Registry())


def _coordinator(
    store: Store,
    models: Models,
    *,
    access: Access | None,
    tools: Registry,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    watch: Callable[[], None] | None = None,
) -> Coordinator:
    return Coordinator(
        store,
        llm_executor(models.llm, models.other),
        llm_reviewer(models.for_(calls.Role.REVIEWER), evidence=functools.partial(evidence, store)),
        owner=current_owner(),
        lease_seconds=lease_seconds,
        abandoned=owner_is_gone,  # a run killed on this machine is taken over at once
        # Without another provider, "other model" attempts are rephrased instead (and audited so).
        strategies=functools.partial(strategy_for, other_model=models.other is not None),
        providers=(label(models.llm), None if models.other is None else label(models.other)),
        decomposer=decomposer(models.for_(calls.Role.DECOMPOSER)),
        access=access,
        tools=tools,
        watch=watch,
    )
