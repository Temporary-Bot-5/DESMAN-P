"""What a run of the local operator is made of, shared by `desman run`/`resume` and the chat.

The operator (ADR 0003) is the coordinator with its planner, executor and reviewer. Here it is
put together from the config: the models (a provider and, for retries with another model,
another free one), the executor's access (the tool policy within the human's permissions) and
its tools.
"""

from __future__ import annotations

import functools
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

import click
import httpx

from desman import providers, ui
from desman.agents import llm_executor, llm_reviewer
from desman.config import Config
from desman.core import calls
from desman.core.coordinator import Coordinator
from desman.core.owner import current_owner, owner_is_gone
from desman.core.policy import Access, PolicyError, Role, default_policy, switched_on
from desman.core.retry import strategy_for
from desman.core.sandbox import Sandbox
from desman.core.search import SearXNG, Tavily, client, open_sources
from desman.core.store import DEFAULT_LEASE_SECONDS, Store
from desman.core.tools import builtin_registry
from desman.core.web import SearchEngine, Web, WebCache
from desman.i18n import t
from desman.llm import router
from desman.llm.base import LLM
from desman.llm.cache import Cached
from desman.llm.openai_compat import OpenAICompatible
from desman.planner import decomposer


@dataclass(frozen=True)
class Models:
    llm: LLM
    other: LLM | None
    """Another free provider for "other model" retries (ROADMAP 2.9); None if there is none."""
    choice: router.Choice | None = None
    """None for a fake."""


def models(config: Config, conn: sqlite3.Connection, *, cache: bool = True) -> Models:
    """The preferred ready provider and another one, as the chat uses them: quick blips retried,
    calls logged and accepted replies kept in the database. Raises `router.NoProvider`."""
    llm, choice = router.build(config)
    alternative = router.alternative(config, choice)
    other = router.Backoff(alternative[0]) if alternative is not None else None
    return Models(
        attach(router.Backoff(llm), conn, cache=cache),
        None if other is None else attach(other, conn, cache=cache),
        choice,
    )


def attach(llm: LLM, conn: sqlite3.Connection, *, cache: bool) -> LLM:
    """A real provider logs every call (`desman usage`) and keeps accepted replies (`llm.cache`)
    in the task database. The fake does neither: its scripts count every call."""
    client = llm.inner if isinstance(llm, router.Backoff) else llm
    if not isinstance(client, OpenAICompatible):
        return llm
    client.on_call = calls.recorder(conn)
    return Cached(llm, conn, client.model) if cache else llm


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
    return Coordinator(
        store,
        llm_executor(models.llm, models.other),
        llm_reviewer(models.llm),
        owner=current_owner(),
        lease_seconds=lease_seconds,
        abandoned=owner_is_gone,  # a run killed on this machine is taken over at once
        # Without another provider, "other model" attempts are rephrased instead (and audited so).
        strategies=functools.partial(strategy_for, other_model=models.other is not None),
        decomposer=decomposer(models.llm),
        access=access,
        tools=builtin_registry(web=web(config, store, http)),
        watch=watch,
    )
