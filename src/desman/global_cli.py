"""`desman global`: whether this machine takes part in desman-global, and what it lends (G.3).

G.7b shares benchmark snapshots with independent checkers.

Off unless the person joins (ADR 0004, decision 4). Joining creates the node's key (`node.key`,
shown by its fingerprint only) and switches desman-global on; leaving takes back everything
lent and switches it off, while the key stays until the person deletes it. A module lent
(`donate`) is a model provider of this machine with a daily quota of calls and tokens: its key
never leaves the machine, a paid provider is never lent, and a free cloud provider only once
its terms were checked (`providers.can_donate`).

Leaving says goodbye to the registry (`farewell`, `node.tell_left`); explicitly confirmed
proposals and reports also contact it here. The worker handles ongoing contact (`net.node`, G.5). Both get the
address from `net.registry()`, which has none while desman-global is off. Every change is written to `network_log`. The commands
are non-interactive; the chat calls the same functions after the person's "yes".
"""

from __future__ import annotations

import base64
import hashlib
import json
import platform
import sqlite3
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, tzinfo
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final

import click
import httpx
from pydantic import BaseModel
from rich.text import Text

from desman import applying, bench, net, providers, searxng, ui
from desman.config import Config, ConfigManager, Donation, Network
from desman.config_cli import json_option, load_config
from desman.core import db, netlog, operator
from desman.core import proposals as stored_proposals
from desman.core import report as task_reports
from desman.core import stats as task_stats
from desman.core.store import AmbiguousIdError, NotFoundError, Store
from desman.i18n import t, tn
from desman.net import cluster, jobs, node, protocol, search_lend, sign
from desman.providers import Provider

DEFAULT_DONATION = Donation(calls_per_day=100, tokens_per_day=100_000)
"""A module's quota when the person names none."""

NO_KEY: Final = "node_key_unusable"
"""`Left.unheard` when the goodbye could not be signed: nothing was sent."""
NO_SUMMARY: Final = "No summary was recorded."
"""A submission's summary when the analyst wrote none."""


def summary_warning() -> str:
    """Shown with every preview: the analyst's texts are written from the person's tasks."""
    return t(
        "The summary and the expected effect are the analyst's words about your own failed "
        "tasks: read them before sending, they may quote those tasks."
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class QuotaUsage:
    calls: int
    tokens: int
    held: int

    def as_json(self, provider: str = "") -> dict[str, int]:
        data = {"calls": self.calls}
        if provider != protocol.SEARCH_PROVIDER:
            data["tokens"] = self.tokens
        return data


def quota_json(quota: Donation, provider: str) -> dict[str, int]:
    data = {"calls_per_day": quota.calls_per_day}
    if provider != protocol.SEARCH_PROVIDER:
        data["tokens_per_day"] = quota.tokens_per_day
    return data


@contextmanager
def _usage_database() -> Iterator[sqlite3.Connection | None]:
    """Read an existing, compatible database without creating or migrating it."""
    path = db.default_db_path()
    conn = None
    try:
        if path.is_file():
            conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            if db.schema_version(conn) != db.SCHEMA_VERSION:
                conn.close()
                conn = None
            else:
                db.verify_schema(conn)
    except (OSError, sqlite3.Error, db.SchemaMismatchError):
        if conn is not None:
            conn.close()
        conn = None
    try:
        yield conn
    finally:
        if conn is not None:
            conn.close()


def _reset_at(now: datetime) -> datetime:
    return jobs.day_start(now) + timedelta(days=1)


@dataclass(frozen=True)
class Status:
    enabled: bool
    registry: str | None
    """Where the node would go (None: no address known yet)."""
    node: str | None
    """The node key's fingerprint, if there is one."""
    donated: dict[str, Donation]
    used_today: dict[str, QuotaUsage] | None
    now: datetime
    lead: bool = True
    borrow_search: bool = False
    cluster: dict[str, Any] | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "lead": self.lead,
            "borrow_search": self.borrow_search,
            "cluster": self.cluster,
            "registry": self.registry,
            "node": self.node,
            "donated": {name: quota_json(d, name) for name, d in self.donated.items()},
            "used_today": (
                None
                if self.used_today is None
                else {name: usage.as_json(name) for name, usage in self.used_today.items()}
            ),
            "resets_at": _reset_at(self.now).isoformat(),
        }


@dataclass(frozen=True)
class QuotaReport:
    provider: str
    quota: Donation
    used_today: QuotaUsage | None
    now: datetime

    def as_json(self, changed: Lent | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "provider": self.provider,
            "quota": quota_json(self.quota, self.provider),
            "used_today": None
            if self.used_today is None
            else self.used_today.as_json(self.provider),
            "resets_at": _reset_at(self.now).isoformat(),
        }
        if changed is not None:
            data["changed"] = changed.changed
        return data


def _quota_usage(conn: sqlite3.Connection, provider: str, now: datetime) -> QuotaUsage:
    if provider == protocol.SEARCH_PROVIDER:
        return QuotaUsage(search_lend.used_today(conn, now), 0, 0)
    return QuotaUsage(*jobs.quota_usage(conn, provider, now))


def show_quota(
    conn: sqlite3.Connection | None,
    provider: str,
    quota: Donation,
    changed: Lent | None = None,
    *,
    as_json: bool = False,
) -> dict[str, Any]:
    now = _utc_now()
    usage = None if conn is None else _quota_usage(conn, provider, now)
    found = QuotaReport(provider, quota, usage, now)
    data = found.as_json(changed)
    if as_json:
        click.echo(json.dumps(data))
    else:
        print_quota(found, changed)
    return data


Entry = tuple[str, dict[str, Any]]
"""An entry of `network_log`: action, details."""


@contextmanager
def journal(conn: sqlite3.Connection | None) -> Iterator[sqlite3.Connection]:
    """The database whose `network_log` records the change: the caller's, or the default one."""
    if conn is not None:
        yield conn
        return
    try:
        opened = db.open_db()
    except (db.SchemaVersionError, db.SchemaMismatchError) as e:
        raise click.ClickException(str(e)) from e
    try:
        yield opened
    finally:
        opened.close()


def commit(
    conn: sqlite3.Connection,
    entries: list[Entry],
    save: Callable[[], None],
    *,
    journal_borrow: bool = False,
) -> None:
    """Save the config and journal its change together: the entries are written in the
    transaction the config is saved in, so a save that fails journals nothing."""
    with db.transaction(conn):
        for action, details in entries:
            netlog.write(conn, action, details)
            if journal_borrow and action == "borrow_search":
                operator.write(conn, action, details)
        save()


@contextmanager
def _change(
    manager: ConfigManager, conn: sqlite3.Connection | None, *, journal_borrow: bool = False
) -> Iterator[tuple[Config, list[Entry]]]:
    """The config to change and the entries that say how, saved and journaled together under
    the config's lock: two commands at once journal in the order they saved. Nothing is saved
    when the block raises or journals nothing."""
    with journal(conn) as db_conn, manager.locked():
        config = manager.load()
        entries: list[Entry] = []
        yield config, entries
        if entries:
            commit(db_conn, entries, lambda: manager.save(config), journal_borrow=journal_borrow)


def node_fingerprint(key_path: Path | None = None) -> str | None:
    """The node's fingerprint, without creating a key; None if there is none (or it is unusable)."""
    path = sign.default_node_key_path() if key_path is None else key_path
    if not path.exists():
        return None
    try:
        return sign.load_node_key(path).fingerprint
    except sign.KeyFileError:
        return None


def usable_node_key(key_path: Path | None = None, *, create: bool = True) -> sign.NodeKey | None:
    """The node's key (created if `create` and there is none), or why it cannot be used."""
    path = sign.default_node_key_path() if key_path is None else key_path
    try:
        if create:
            return sign.node_key(path)
        return sign.load_node_key(path) if path.exists() or path.is_symlink() else None
    except (sign.KeyFileError, OSError) as e:
        raise click.ClickException(t("The node's key cannot be used: {error}", error=e)) from e


def status(
    config: Config, key_path: Path | None = None, *, conn: sqlite3.Connection | None = None
) -> Status:
    network = config.network
    now = _utc_now()
    return Status(
        enabled=network.enabled,
        registry=network.registry or net.DEFAULT_REGISTRY,
        node=node_fingerprint(key_path),
        donated=dict(network.donated),
        used_today=(
            None
            if conn is None
            else {name: _quota_usage(conn, name, now) for name in network.donated}
        ),
        now=now,
        lead=network.lead,
        borrow_search=network.borrow_search,
        cluster=cluster.role(conn, now) if conn is not None and network.enabled else None,
    )


def set_lead(manager: ConfigManager, on: bool, *, conn: sqlite3.Connection | None = None) -> bool:
    """The person's consent to lead, saved with its journal receipt."""
    with _change(manager, conn) as (config, entries):
        if config.network.lead != on:
            config.network.lead = on
            entries.append(("lead", {"on": on}))
    return on


def borrow_search(
    manager: ConfigManager, on: bool | None = None, *, conn: sqlite3.Connection | None = None
) -> bool:
    if on is None:
        return manager.load().network.borrow_search
    with _change(manager, conn, journal_borrow=conn is None) as (config, entries):
        if config.network.borrow_search != on:
            config.network.borrow_search = on
            entries.append(("borrow_search", {"on": on}))
    return on


def search_text(on: bool, *, joined: bool = True) -> str:
    state = t("Borrow web search") + ": " + (t("on") if on else t("off"))
    if on and not joined:
        state += "\n" + t("Borrowing takes effect after `desman global join`.")
    return state


def cluster_text(details: dict[str, Any] | None) -> str:
    if details is None:
        return ""
    if details["role"] == "operator":
        return tn(
            details["members"],
            "Leads cluster {cluster} (task {task}), {members} member",
            "Leads cluster {cluster} (task {task}), {members} members",
            cluster=details["cluster"],
            task=details["task"],
            members=details["members"],
        )
    return t("Member of cluster {cluster}", cluster=details["cluster"])


def proposal_upload(
    conn: sqlite3.Connection,
    config: Config,
    proposal_id: int,
    *,
    key_path: Path | None = None,
) -> protocol.SignedProposalSubmission:
    """Build the exact, bounded payload that would leave this node for G.7."""
    if not config.network.enabled:
        raise click.ClickException(t("desman-global is off: join it first (`desman global join`)."))
    if net.registry(config) is None:
        raise click.ClickException(
            t(
                "No registry address is known yet: give one with `desman global join --registry URL`."
            )
        )
    found = stored_proposals.get(conn, proposal_id)
    if found is None:
        raise click.ClickException(t("No proposal {id}.", id=proposal_id))
    if found.status is not stored_proposals.Status.PROPOSED:
        raise click.ClickException(t("Proposal {id} has nothing to submit.", id=proposal_id))
    evaluations = stored_proposals.evaluations(conn, proposal_id)
    evaluation = evaluations[0] if evaluations else None
    if (
        evaluation is None
        or evaluation.verdict == "failed"
        or evaluation.base is None
        or evaluation.candidate is None
    ):
        raise click.ClickException(
            t("Proposal {id} needs a completed benchmark with both results.", id=proposal_id)
        )
    if found.details.get("settings") or not found.diff:
        # Settings (`Config.tuning`) name this machine's models and limits: they stay here.
        raise click.ClickException(
            t(
                "Proposal {id} changes settings; only proposals that change texts can be sent.",
                id=proposal_id,
            )
        )
    try:
        payload = protocol.ProposalSubmission.model_validate(
            {
                "proposal": found.id,
                "evaluation": evaluation.id,
                "base_commit": evaluation.commit,
                # Signed: the same in every language, so never translated.
                "summary": found.summary or NO_SUMMARY,
                # Not the analyst's `evidence`: it quotes the person's own tasks.
                "details": {
                    "changes": found.details.get("changes", []),
                    "base": found.details.get("base", {}),
                    "expected_effect": found.details.get("expected_effect", ""),
                },
                "diff": found.diff,
                "verdict": evaluation.verdict,
                "mode": evaluation.mode,
                "base": protocol.benchmark_snapshot(bench.Result.model_validate(evaluation.base)),
                "candidate": protocol.benchmark_snapshot(
                    bench.Result.model_validate(evaluation.candidate)
                ),
            }
        )
    except ValueError as e:
        # Validation messages can echo submitted text; keep them and any accidental secrets out
        # of the terminal and logs.
        raise click.ClickException(
            t("This proposal cannot be sent: {reason}", reason="invalid contents")
        ) from e
    _refuse_secrets(config, payload)
    key = usable_node_key(key_path, create=False)
    if key is None:
        raise click.ClickException(t("This node has no usable key; nothing was sent."))
    signature = base64.b64encode(key.sign(sign.canonical(payload.model_dump(mode="json")))).decode(
        "ascii"
    )
    signed = protocol.SignedProposalSubmission(
        submission=payload, signer=key.fingerprint, signature=signature
    )
    if (
        len(protocol.SubmitProposal(signed=signed).model_dump_json().encode("utf-8"))
        > protocol.MAX_PROPOSAL_BYTES
    ):
        raise click.ClickException(t("This proposal is too large to send."))
    return signed


def _refuse_secrets(config: Config, payload: BaseModel) -> None:
    """Do not let a configured API key leave in a proposal, report, or message."""
    keys = [key.value for group in config.api_keys.values() for key in group if key.value]

    def contains_secret(value: object) -> bool:
        if isinstance(value, str):
            return any(key in value for key in keys)
        if isinstance(value, dict):
            return any(contains_secret(k) or contains_secret(v) for k, v in value.items())
        if isinstance(value, list):
            return any(contains_secret(v) for v in value)
        return False

    if contains_secret(payload.model_dump_json()) or contains_secret(
        payload.model_dump(mode="json")
    ):
        if isinstance(payload, protocol.ProposalSubmission):
            raise click.ClickException(t("This proposal contains an API key; nothing was sent."))
        if isinstance(payload, protocol.MessengerBody):
            raise click.ClickException(t("This message contains an API key; nothing was sent."))
        raise click.ClickException(t("This report contains an API key; nothing was sent."))


def proposal_confirmation(registry: str, body: dict[str, object]) -> str:
    """Hash the destination together with the exact wire body."""
    encoded = json.dumps(
        {"registry": registry, "submission": body}, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def send_proposal(
    manager: ConfigManager,
    conn: sqlite3.Connection,
    signed: protocol.SignedProposalSubmission,
    *,
    registry: str,
    key_path: Path | None = None,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> protocol.ProposalReceived:
    """Send a prepared proposal under the config lock; the caller has already shown it."""
    with manager.locked():
        config = manager.load()
        address = net.registry(config)
        if address != registry:
            raise click.ClickException(t("The registry changed after preview; nothing was sent."))
        if not config.network.enabled or address is None:
            raise click.ClickException(t("desman-global is off; nothing was sent."))
        _refuse_secrets(config, signed.submission)
        key = usable_node_key(key_path, create=False)
        if key is None:
            raise click.ClickException(t("This node has no usable key; nothing was sent."))
        if key.fingerprint != signed.signer:
            raise click.ClickException(t("The node key changed after preview; nothing was sent."))
        try:
            received = node.submit_proposal(address, key, signed, transport=transport, clock=clock)
        except node.ProposalSendError as e:
            if str(e) == protocol.PROPOSAL_LIMIT:
                raise click.ClickException(
                    tn(
                        protocol.MAX_PROPOSALS_PER_DAY,
                        "The registry takes at most {n} proposal a day from one node; "
                        "nothing was sent.",
                        "The registry takes at most {n} proposals a day from one node; "
                        "nothing was sent.",
                    )
                ) from e
            raise click.ClickException(
                t("The registry did not accept the proposal: {error}", error=e)
            ) from e
    try:
        netlog.record(
            conn,
            "proposal_submitted",
            {
                "registry": address,
                "node": key.fingerprint,
                "submission": received.id,
                "proposal": received.proposal,
                "evaluation": received.evaluation,
            },
        )
    except sqlite3.OperationalError:
        # The external write is already durable and retries are idempotent; report the missing
        # local journal to the caller without exposing the body or any keys.
        raise click.ClickException(
            t("The proposal was sent, but the local journal is busy.")
        ) from None
    return received


# ── Changes ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Joined:
    changed: bool
    registry: str
    node: str


def join_registry(config: Config, registry: str | None) -> str:
    """The registry joining would use: the one given, the one saved, or the shipped one."""
    found = registry or config.network.registry or net.DEFAULT_REGISTRY
    if found is None:
        raise click.ClickException(
            t(
                "No registry address is known yet: give one with `desman global join --registry URL`."
            )
        )
    try:
        Network(registry=found)
    except ValueError:
        raise click.ClickException(
            t("{url} is not a registry address: https://host[:port][/path].", url=found)
        ) from None
    return found


def joining(config: Config, registry: str | None, key: sign.NodeKey) -> tuple[Joined, Entry | None]:
    """`config` switched on (changed in place), and its journal entry if anything changed."""
    address = join_registry(config, registry)
    before = config.network.model_copy(deep=True)
    config.network.enabled = True
    if registry is not None or config.network.registry is not None:
        config.network.registry = address
    if config.network == before:
        return Joined(False, address, key.fingerprint), None
    return Joined(True, address, key.fingerprint), (
        "joined",
        {"node": key.fingerprint, "registry": address},
    )


@dataclass(frozen=True)
class Left:
    withdrawn: list[str]
    """What was lent and is taken back."""
    registry: str | None = None
    """The registry to say goodbye to: the one desman-global was on with (None: none)."""
    told: bool = False
    """The registry heard the goodbye (or did not list the node anyway)."""
    unheard: str | None = None
    """Why it was not told: `NO_KEY` (nothing sent), no answer, or its refusal."""
    journaled: bool = True
    """How the goodbye went is in `network_log` (False: the database was busy)."""


def leaving(config: Config) -> tuple[Left | None, list[Entry]]:
    """`config` switched off with nothing lent (changed in place): what was taken back and
    from which registry, None if it was off already; and the journal entries. Nothing is sent:
    the goodbye (`farewell`) waits until this config is saved, so a save that fails leaves
    the registry as it was."""
    if not config.network.enabled and not config.network.donated:
        return None, []
    registry = net.registry(config)
    withdrawn = sorted(config.network.donated)
    config.network.enabled = False
    config.network.donated = {}
    return Left(withdrawn, registry), [("left", {"withdrawn": withdrawn})]


def farewell(
    left: Left | None,
    *,
    conn: sqlite3.Connection | None = None,
    transport: httpx.BaseTransport | None = None,
    key_path: Path | None = None,
) -> Left | None:
    """Once `leaving`'s config is saved: tell the registry this node leaves (G.5), sent once,
    and journal how it went. A goodbye that is not heard is not sent again: the registry
    stops listing a node that falls silent. Callers hold the config's lock, so a `join`
    meanwhile waits and is never undone by this goodbye arriving after it."""
    if left is None or left.registry is None:
        return left
    path = sign.default_node_key_path() if key_path is None else key_path
    try:
        key = sign.load_node_key(path)
    except (sign.KeyFileError, OSError):
        details = {"registry": left.registry, "why": NO_KEY}
        return replace(left, unheard=NO_KEY, journaled=_journal(conn, "leave_not_sent", details))
    unheard = node.tell_left(left.registry, key, transport=transport)
    sent: dict[str, Any] = {"registry": left.registry, "node": key.fingerprint}
    sent["heard"] = unheard is None
    if unheard is not None:
        sent["error"] = unheard
    return replace(
        left,
        told=unheard is None,
        unheard=unheard,
        journaled=_journal(conn, "leave_sent", sent),
    )


def _journal(conn: sqlite3.Connection | None, action: str, details: dict[str, Any]) -> bool:
    """An entry written after the config was saved: a busy database must not turn a leave
    that happened into an error (the next `leave` finds it off and could not tell again)."""
    try:
        with journal(conn) as db_conn:
            netlog.record(db_conn, action, details)
    except sqlite3.OperationalError:
        return False
    return True


def join(
    manager: ConfigManager,
    registry: str | None = None,
    *,
    conn: sqlite3.Connection | None = None,
    key_path: Path | None = None,
) -> Joined:
    """Switch desman-global on, with the node's key (created the first time)."""
    with _change(manager, conn) as (config, entries):
        join_registry(config, registry)  # a bad address creates no key
        key = usable_node_key(key_path)
        assert key is not None
        joined, entry = joining(config, registry, key)
        entries += [entry] if entry else []
    return joined


def leave(
    manager: ConfigManager,
    *,
    conn: sqlite3.Connection | None = None,
    transport: httpx.BaseTransport | None = None,
    key_path: Path | None = None,
) -> Left | None:
    """Switch desman-global off and take back every module lent, then tell the registry: what
    was taken back, None if it was off already. The node's key stays."""
    with journal(conn) as db_conn, manager.locked():
        config = manager.load()
        left, entries = leaving(config)
        if entries:
            commit(db_conn, entries, lambda: manager.save(config))
        # Still under the lock: the goodbye cannot land after a `join` that followed.
        return farewell(left, conn=db_conn, transport=transport, key_path=key_path)


def _donation_provider(name: str) -> Provider:
    if name.strip().lower() == protocol.SEARCH_PROVIDER:
        return providers.find_keyed(name)
    return providers.find(name)


def donatable(config: Config, name: str) -> Provider:
    """The provider `name` names, if the person may lend it now; otherwise why not."""
    if not config.network.enabled:
        raise click.ClickException(t("desman-global is off: join it first (`desman global join`)."))
    try:
        provider = _donation_provider(name)
    except providers.UnknownProvider as e:
        raise click.ClickException(str(e)) from e
    if provider.id == protocol.SEARCH_PROVIDER:
        if not searxng.managed_search(config):
            raise click.ClickException(
                t("Set up local SearXNG with `desman search setup` before lending it.")
            )
        return provider
    if provider.paid:
        raise click.ClickException(
            t("{name} is a paid provider: it is never lent.", name=provider.name)
        )
    if not providers.can_donate(provider):
        raise click.ClickException(
            t(
                "{name}'s free tier has not been checked for serving others, so it cannot be lent.",
                name=provider.name,
            )
        )
    if not provider.is_ready(config):
        raise click.ClickException(
            t("{name} is not set up yet (`desman config`).", name=provider.name)
        )
    return provider


def donation(base: Donation, calls: int | None, tokens: int | None, provider: str = "") -> Donation:
    data = base.model_dump()
    if calls is not None:
        data["calls_per_day"] = calls
    if provider == protocol.SEARCH_PROVIDER:
        if tokens is not None:
            raise click.ClickException(t("SearXNG quotas count searches only."))
        data["tokens_per_day"] = 0
    elif tokens is not None:
        data["tokens_per_day"] = tokens
    try:
        if provider != protocol.SEARCH_PROVIDER and data["tokens_per_day"] == 0:
            raise ValueError("model donations need a token quota")
        return Donation.model_validate(data)
    except ValueError:
        raise click.ClickException(
            t(
                "A quota is 1–100000 calls and 1000–100000000 tokens a day.",
            )
        ) from None


@dataclass(frozen=True)
class Lent:
    provider: str
    old: Donation | None
    new: Donation | None

    @property
    def changed(self) -> bool:
        return self.old != self.new


def donate(
    manager: ConfigManager,
    name: str,
    calls: int | None = None,
    tokens: int | None = None,
    *,
    conn: sqlite3.Connection | None = None,
) -> Lent:
    """Lend a provider (or change the quota of one lent already)."""
    with _change(manager, conn) as (config, entries):
        provider = donatable(config, name)
        old = config.network.donated.get(provider.id)
        new = donation(old or DEFAULT_DONATION, calls, tokens, provider.id)
        config.network.donated[provider.id] = new
        if old != new:
            action = "donated" if old is None else "quota"
            entries.append((action, {"provider": provider.id, **new.model_dump()}))
    return Lent(provider.id, old, new)


def lent_provider(config: Config, name: str) -> str:
    """The id of a provider that is lent now, named any way `providers.find` takes."""
    try:
        provider = _donation_provider(name)
    except providers.UnknownProvider as e:
        raise click.ClickException(str(e)) from e
    if provider.id not in config.network.donated:
        raise click.ClickException(t("{name} is not lent.", name=provider.name))
    return provider.id


def set_quota(
    manager: ConfigManager,
    name: str,
    calls: int | None,
    tokens: int | None,
    *,
    conn: sqlite3.Connection | None = None,
) -> Lent:
    """Change the quota of a module lent already."""
    with _change(manager, conn) as (config, entries):
        lent = lent_provider(config, name)
        old = config.network.donated[lent]
        new = donation(old, calls, tokens, lent)
        config.network.donated[lent] = new
        if old != new:
            entries.append(("quota", {"provider": lent, **new.model_dump()}))
    return Lent(lent, old, new)


def undonate(manager: ConfigManager, name: str, *, conn: sqlite3.Connection | None = None) -> Lent:
    with _change(manager, conn) as (config, entries):
        lent = lent_provider(config, name)
        old = config.network.donated.pop(lent)
        entries.append(("undonated", {"provider": lent}))
    return Lent(lent, old, None)


def left_json(left: Left | None) -> dict[str, Any]:
    return {
        "changed": left is not None,
        "withdrawn": [] if left is None else left.withdrawn,
        "registry_told": left is not None and left.told,
    }


# ── Output ───────────────────────────────────────────────────────────────────


def quota_text(donation: Donation, provider: str = "") -> str:
    if provider == protocol.SEARCH_PROVIDER:
        return t("{calls} searches a day", calls=donation.calls_per_day)
    return t(
        "{calls} calls and {tokens} tokens a day",
        calls=donation.calls_per_day,
        tokens=donation.tokens_per_day,
    )


def _usage_text(quota: Donation, used: QuotaUsage, provider: str = "") -> str:
    if provider == protocol.SEARCH_PROVIDER:
        return t(
            "Searches used today: {calls}. Remaining: {calls_left}.",
            calls=used.calls,
            calls_left=max(0, quota.calls_per_day - used.calls),
        )
    return t(
        "Used today (calls/tokens): {calls}/{tokens}. Remaining: {calls_left}/{tokens_left}.",
        calls=used.calls,
        tokens=used.tokens,
        calls_left=max(0, quota.calls_per_day - used.calls - used.held),
        tokens_left=max(0, quota.tokens_per_day - used.tokens),
    )


def _reset_text(now: datetime, tz: tzinfo | None) -> str:
    reset = _reset_at(now).astimezone(tz)
    when = reset.strftime(
        "%H:%M" if reset.date() == now.astimezone(tz).date() else "%Y-%m-%d %H:%M"
    )
    return t("Daily quotas reset at {when} local time.", when=when)


def print_quota(
    found: QuotaReport, changed: Lent | None = None, *, tz: tzinfo | None = None
) -> None:
    if changed is None:
        ui.out(f"{provider_name(found.provider)}: {quota_text(found.quota, found.provider)}")
    else:
        print_lent(changed)
    if found.used_today is not None:
        ui.out(_usage_text(found.quota, found.used_today, found.provider))
    ui.note(_reset_text(found.now, tz))


def provider_name(provider_id: str) -> str:
    try:
        return _donation_provider(provider_id).name
    except providers.UnknownProvider:
        return provider_id


def print_status(found: Status, *, tz: tzinfo | None = None) -> None:
    table = ui.table("", "", title=t("desman-global"))
    table.show_header = False
    table.add_row(Text(t("taking part"), style="muted"), t("yes") if found.enabled else t("no"))
    table.add_row(Text(t("registry"), style="muted"), found.registry or t("not known yet"))
    table.add_row(Text(t("node key"), style="muted"), found.node or t("none yet"))
    table.add_row(t("Lead clusters"), t("yes") if found.lead else t("off"))
    table.add_row(t("Borrow web search"), t("on") if found.borrow_search else t("off"))
    if found.cluster:
        table.add_row(cluster_text(found.cluster), "")
    for provider_id, donation in sorted(found.donated.items()):
        details = quota_text(donation, provider_id)
        if found.used_today is not None:
            details += "\n" + _usage_text(donation, found.used_today[provider_id], provider_id)
        table.add_row(
            Text(t("lent: {name}", name=provider_name(provider_id)), style="muted"), details
        )
    ui.out(table)
    if found.donated:
        ui.note(_reset_text(found.now, tz))
    if not found.enabled:
        ui.note(
            t(
                "Off: nothing leaves this machine for desman-global. `desman global join` "
                "switches it on; lending a module is `desman global donate PROVIDER`."
            )
        )


def print_joined(joined: Joined) -> None:
    if not joined.changed:
        ui.note(t("Already taking part in desman-global."))
    else:
        ui.ok(t("Taking part in desman-global, registry {url}.", url=joined.registry))
    ui.note(t("This node's key: {fingerprint}", fingerprint=joined.node))


def print_left(left: Left | None) -> None:
    if left is None:
        ui.note(t("desman-global is off already."))
        return
    ui.ok(t("desman-global is off; nothing lent any more."))
    for provider_id in left.withdrawn:
        ui.ok(t("Taken back: {name}", name=provider_name(provider_id)))
    if left.unheard == NO_KEY:
        ui.warn(
            t(
                "The registry was not told: the node's key cannot be used, so nothing was "
                "sent. It stops listing this node once it falls silent."
            )
        )
    elif left.unheard:
        ui.warn(
            t(
                "The registry could not be told ({error}); it may list this node a while "
                "longer. Nothing more is sent to it.",
                error=left.unheard,
            )
        )
    if not left.journaled:
        ui.warn(
            t(
                "The database was busy: how the registry was told is not in the network's "
                "journal. desman-global is off all the same."
            )
        )
    ui.note(t("The node's key stays; delete it yourself if you want it gone."))


def print_lent(lent: Lent) -> None:
    name = provider_name(lent.provider)
    if not lent.changed:
        ui.note(t("Nothing changed."))
    elif lent.new is None:
        ui.ok(t("{name} is not lent any more.", name=name))
    elif lent.old is None:
        ui.ok(t("{name} lent: {quota}.", name=name, quota=quota_text(lent.new, lent.provider)))
    else:
        ui.ok(
            t(
                "{name}: {old} → {new}",
                name=name,
                old=quota_text(lent.old, lent.provider),
                new=quota_text(lent.new, lent.provider),
            )
        )


# ── The commands ─────────────────────────────────────────────────────────────


@click.group("global", invoke_without_command=True)
@click.pass_context
def global_group(ctx: click.Context) -> None:
    """desman-global: whether this machine takes part, and what it lends."""
    if ctx.invoked_subcommand is None:
        with _usage_database() as conn:
            print_status(status(load_config().load(), conn=conn))


@global_group.command("status")
@json_option
def status_command(as_json: bool) -> None:
    """Whether this machine takes part in desman-global, its key, what it lends."""
    with _usage_database() as conn:
        found = status(load_config().load(), conn=conn)
    if as_json:
        click.echo(json.dumps(found.as_json(), ensure_ascii=False, indent=2))
    else:
        print_status(found)


@global_group.command("join")
@click.option("--registry", metavar="URL", help="The registry's address (https://...).")
@json_option
def join_command(registry: str | None, as_json: bool) -> None:
    """Take part in desman-global: creates this node's key the first time."""
    joined = join(load_config(), registry)
    if as_json:
        click.echo(
            json.dumps(
                {"changed": joined.changed, "registry": joined.registry, "node": joined.node}
            )
        )
    else:
        print_joined(joined)


@global_group.command("leave")
@json_option
def leave_command(as_json: bool) -> None:
    """Stop taking part in desman-global and take back everything lent. The key stays."""
    left = leave(load_config())
    if as_json:
        click.echo(json.dumps(left_json(left)))
    else:
        print_left(left)


calls_option = click.option("--calls", type=int, metavar="N", help="Calls a day (1–100000).")
tokens_option = click.option(
    "--tokens", type=int, metavar="N", help="Tokens a day (1000–100000000)."
)


def _lent_json(lent: Lent) -> str:
    return json.dumps(
        {
            "provider": lent.provider,
            "changed": lent.changed,
            "quota": None if lent.new is None else quota_json(lent.new, lent.provider),
        }
    )


@global_group.command("lead")
@click.argument("state", type=click.Choice(["on", "off"]))
def lead_command(state: str) -> None:
    """Allow this node to lead clusters with the modules lent."""
    set_lead(load_config(), state == "on")
    ui.ok(t("Lead clusters") + ": " + (t("yes") if state == "on" else t("off")))


@global_group.command("search")
@click.argument("state", type=click.Choice(["on", "off"]), required=False)
def search_command(state: str | None) -> None:
    """Show or change whether this node borrows web search."""
    manager = load_config()
    on = borrow_search(manager, None if state is None else state == "on")
    ui.out(search_text(on, joined=manager.load().network.enabled))


@global_group.command("donate")
@click.argument("provider")
@calls_option
@tokens_option
@json_option
def donate_command(provider: str, calls: int | None, tokens: int | None, as_json: bool) -> None:
    """Lend PROVIDER to desman-global with a daily quota (its key stays on this machine)."""
    lent = donate(load_config(), provider, calls, tokens)
    if as_json:
        click.echo(_lent_json(lent))
    else:
        print_lent(lent)


@global_group.command("undonate")
@click.argument("provider")
@json_option
def undonate_command(provider: str, as_json: bool) -> None:
    """Take PROVIDER back from desman-global."""
    lent = undonate(load_config(), provider)
    if as_json:
        click.echo(_lent_json(lent))
    else:
        print_lent(lent)


@global_group.command("quota")
@click.argument("provider")
@calls_option
@tokens_option
@json_option
def quota_command(provider: str, calls: int | None, tokens: int | None, as_json: bool) -> None:
    """Show or change the daily quota of PROVIDER, lent to desman-global."""
    manager = load_config()
    if calls is None and tokens is None:
        config = manager.load()
        lent = lent_provider(config, provider)
        with _usage_database() as conn:
            show_quota(conn, lent, config.network.donated[lent], as_json=as_json)
        return
    with journal(None) as conn:
        changed = set_quota(manager, provider, calls, tokens, conn=conn)
        assert changed.new is not None
        show_quota(conn, changed.provider, changed.new, changed, as_json=as_json)


@global_group.command("submit-proposal")
@click.argument("proposal", type=click.IntRange(min=1))
@click.option("--confirm", metavar="SHA256", help="Send only if this matches the preview's hash.")
@json_option
def submit_proposal_command(proposal: int, confirm: str | None, as_json: bool) -> None:
    """Preview a proposal upload; --confirm sends only that exact payload."""
    manager = load_config()
    config = manager.load()
    with journal(None) as conn:
        payload = proposal_upload(conn, config, proposal)
        wire = protocol.SubmitProposal(signed=payload)
        address = net.registry(config)
        assert address is not None
        preview = wire.model_dump(mode="json", exclude_none=True)
        digest = proposal_confirmation(address, preview)
        if confirm is None:
            if as_json:
                click.echo(
                    json.dumps(
                        {
                            "sent": False,
                            "registry": address,
                            "confirmation": digest,
                            "submission": preview,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                ui.note(
                    t(
                        "Nothing was sent. Review this payload, then repeat with --confirm {hash}.",
                        hash=digest,
                    )
                )
                ui.out(t("Registry: {address}", address=address))
                ui.out(Text(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True)))
                ui.note(summary_warning())
            return
        if digest != confirm.strip().casefold():
            raise click.ClickException(
                t("The preview changed; nothing was sent. Preview it again.")
            )
        received = send_proposal(manager, conn, payload, registry=address)
    result = {
        "sent": True,
        "id": received.id,
        "proposal": received.proposal,
        "evaluation": received.evaluation,
    }
    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False))
    else:
        ui.ok(
            t(
                "Proposal {proposal} sent to the registry as {id}.",
                proposal=received.proposal,
                id=received.id,
            )
        )


def update_plan(
    conn: sqlite3.Connection,
    action: str,
    release: int | None = None,
) -> applying.UpdatePlan:
    try:
        return (
            applying.plan_update_revert(conn)
            if action == "revert"
            else applying.plan_update(conn, release)
        )
    except applying.ApplyError as e:
        raise click.ClickException(str(e)) from e


def update_lines(plan: applying.UpdatePlan) -> list[str]:
    lines = [
        t("Release {number}: {summary}", number=plan.release.number, summary=plan.release.summary),
        t(
            "Branch: {branch}; commit: {commit}; proposal: {proposal}",
            branch=plan.release.branch,
            commit=plan.release.commit,
            proposal=plan.release.proposal,
        ),
        t(
            "Checkout: {branch}; {head} → {target}",
            branch=plan.branch,
            head=plan.head,
            target=plan.target,
        ),
        t("Commits: {count}", count=plan.commit_count),
        *plan.commits,
        t("Files: {files}", files=", ".join(plan.files)),
    ]
    if plan.commit_count > len(plan.commits) and plan.action == "applied":
        lines.append(t("More commits: {count}", count=plan.commit_count - len(plan.commits)))
    if plan.protected:
        lines.append(
            t(
                "Protected files: {files}. Run `desman protected approve` after the update; tasks are blocked until then.",
                files=", ".join(plan.protected),
            )
        )
    if plan.dependencies:
        lines.append(t("pyproject.toml changes: reinstall dependencies after the update."))
    lines.append(t("Restart running workers and chats after moving the checkout."))
    return lines


def update_data(plan: applying.UpdatePlan) -> dict[str, Any]:
    return {
        "release": plan.release.model_dump(mode="json"),
        "action": plan.action,
        "head": plan.head,
        "branch": plan.branch,
        "target": plan.target,
        "commits": plan.commits,
        "commit_count": plan.commit_count,
        "files": plan.files,
        "protected": plan.protected,
        "dependencies": plan.dependencies,
        "schema": plan.schema,
    }


def install_update(
    conn: sqlite3.Connection,
    plan: applying.UpdatePlan,
    *,
    session_id: str | None = None,
) -> stored_proposals.Update:
    try:
        return applying.carry_out_update(conn, plan, session_id=session_id)
    except applying.ApplyError as e:
        raise click.ClickException(str(e)) from e


def print_update(plan: applying.UpdatePlan, *, done: bool = False) -> None:
    for line in update_lines(plan):
        ui.out(Text(line))
    if done:
        ui.ok(t("The checkout moved to commit {commit}.", commit=plan.target))


@global_group.command("update")
@click.option(
    "--show",
    type=int,
    is_flag=False,
    flag_value=-1,
    default=None,
    help="Show release N, or the newest received release when N is omitted.",
)
@click.option(
    "--apply",
    "apply_release",
    type=click.IntRange(min=1),
    default=None,
    help="Install exactly release N, after inspecting it with --show.",
)
@click.option("--revert", is_flag=True, help="Go back to the checkout before the last update.")
@json_option
def update_command(
    show: int | None, apply_release: int | None, revert: bool, as_json: bool
) -> None:
    """Inspect, install or revert a root-signed release; nothing installs automatically."""
    if sum((show is not None, apply_release is not None, revert)) > 1:
        raise click.UsageError(t("Choose only one of --show, --apply and --revert."))
    if show is not None and show != -1 and show < 1:
        raise click.UsageError(t("The release number must be positive."))
    action = "revert" if revert else "apply" if apply_release is not None else "show"
    number = apply_release if apply_release is not None else None if show in (None, -1) else show
    with journal(None) as conn:
        plan = update_plan(conn, action, number)
        result = update_data(plan)
        if action != "show":
            result["application"] = install_update(conn, plan).model_dump(mode="json")
    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False))
    else:
        print_update(plan, done=action != "show")


def _report_ready(config: Config) -> None:
    if not config.network.enabled:
        raise click.ClickException(t("desman-global is off: join it first (`desman global join`)."))
    if net.registry(config) is None:
        raise click.ClickException(
            t(
                "No registry address is known yet: give one with `desman global join --registry URL`."
            )
        )


def _sign_report(
    config: Config, report: protocol.BugReport | protocol.MemoryShare, key_path: Path | None
) -> protocol.SignedReport:
    _refuse_secrets(config, report)
    key = usable_node_key(key_path, create=False)
    if key is None:
        raise click.ClickException(t("This node has no usable key; nothing was sent."))
    signed = protocol.SignedReport(
        report=report,
        signer=key.fingerprint,
        signature=base64.b64encode(key.sign(protocol.report_content(report))).decode("ascii"),
    )
    if (
        len(protocol.SubmitReport(signed=signed).model_dump_json().encode())
        > protocol.MAX_REPORT_BYTES
    ):
        raise click.ClickException(t("This report is too large to send."))
    return signed


def _task_snapshot(conn: sqlite3.Connection, task_id: str) -> protocol.TaskSnapshot:
    with db.read_snapshot(conn):
        found = task_reports.task_report(Store(conn), task_id)
        parts = []
        for sub in found.subtasks[:50]:
            row = conn.execute(
                "SELECT details_json FROM audit_log WHERE subtask_id = ? "
                "AND json_extract(details_json, '$.failure') IS NOT NULL ORDER BY seq DESC LIMIT 1",
                (sub.id,),
            ).fetchone()
            details = json.loads(row[0]) if row else {}
            cause = details.get("cause") or details.get("reason")
            parts.append(
                protocol.SubtaskSnapshot(
                    kind=sub.kind.value,
                    status=sub.status.value,
                    description=sub.description[:300],
                    attempts=sub.attempts.model_dump(),
                    failure=details.get("failure"),
                    cause=str(cause)[:500] if cause is not None else None,
                )
            )
        return protocol.TaskSnapshot(
            status=found.status.value, goal=found.goal[:500], subtasks=parts
        )


def report_upload(
    conn: sqlite3.Connection,
    config: Config,
    text: str,
    task_id: str | None,
    *,
    key_path: Path | None = None,
) -> protocol.SignedReport:
    _report_ready(config)
    try:
        snapshot = _task_snapshot(conn, task_id) if task_id is not None else None
    except (NotFoundError, AmbiguousIdError) as e:
        raise click.ClickException(t("No task matches {id}.", id=task_id)) from e
    except ValueError as e:
        raise click.ClickException(t("This report cannot be sent: invalid contents.")) from e
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            stdin=subprocess.DEVNULL,
            encoding="utf-8",
        )
        commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    try:
        installed_version = version("desman-p")
    except PackageNotFoundError:
        installed_version = "unknown"
    try:
        report = protocol.BugReport(
            text=text,
            version=installed_version,
            commit=commit,
            python=platform.python_version(),
            system=(platform.system() + " " + platform.release())[:200],
            language=config.language,
            task=snapshot,
        )
    except ValueError as e:
        raise click.ClickException(t("This report cannot be sent: invalid contents.")) from e
    return _sign_report(config, report, key_path)


def memory_upload(
    conn: sqlite3.Connection,
    config: Config,
    *,
    stats: bool,
    failures: bool,
    days: int,
    key_path: Path | None = None,
) -> protocol.SignedReport:
    _report_ready(config)
    if not stats and not failures:
        raise click.ClickException(t("Choose statistics or failures to share."))
    if not 1 <= days <= 365:
        raise click.ClickException(t("This report cannot be sent: invalid contents."))
    since = (_utc_now() - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    start = db.to_iso(since)
    try:
        with db.read_snapshot(conn):
            report = protocol.MemoryShare(
                since=start,
                stats=task_stats.collect(conn, since, skip=stored_proposals.PURPOSE).as_json()
                if stats
                else None,
                failures=stored_proposals.failures(conn, start) if failures else None,
                plan_fallbacks=stored_proposals.plan_fallbacks(conn, start) if failures else None,
            )
    except ValueError as e:
        raise click.ClickException(t("This report cannot be sent: invalid contents.")) from e
    return _sign_report(config, report, key_path)


def send_report(
    manager: ConfigManager,
    conn: sqlite3.Connection,
    signed: protocol.SignedReport,
    *,
    registry: str | None = None,
    key_path: Path | None = None,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> protocol.ReportReceived:
    with manager.locked():
        config = manager.load()
        address = net.registry(config)
        if registry is not None and address != registry:
            raise click.ClickException(t("The registry changed after preview; nothing was sent."))
        if not config.network.enabled or address is None:
            raise click.ClickException(t("desman-global is off; nothing was sent."))
        _refuse_secrets(config, signed.report)
        key = usable_node_key(key_path, create=False)
        if key is None:
            raise click.ClickException(t("This node has no usable key; nothing was sent."))
        if key.fingerprint != signed.signer:
            raise click.ClickException(t("The node key changed after preview; nothing was sent."))
        try:
            received = node.submit_report(address, key, signed, transport=transport, clock=clock)
        except node.ReportSendError as e:
            raise click.ClickException(
                t("The registry did not accept the report: {error}", error=e)
            ) from e
    try:
        netlog.record(
            conn,
            "report_sent",
            {
                "registry": address,
                "node": key.fingerprint,
                "id": received.id,
                "kind": received.kind,
            },
        )
    except sqlite3.OperationalError:
        raise click.ClickException(
            t("The report was sent, but the local journal is busy.")
        ) from None
    return received


def report_warning(report: protocol.BugReport | protocol.MemoryShare) -> str:
    if report.kind == "bug":
        return t("The operator of the registry will read this report.")
    if report.failures is not None:
        return t("Failures quote your own tasks' goals and the reviewers' words.")
    return ""


def _report_command(
    manager: ConfigManager,
    conn: sqlite3.Connection,
    config: Config,
    signed: protocol.SignedReport,
    confirm: str | None,
    as_json: bool,
) -> None:
    address = net.registry(config)
    assert address is not None
    preview = protocol.SubmitReport(signed=signed).model_dump(mode="json", exclude_none=True)
    digest = proposal_confirmation(address, preview)
    if confirm is None:
        if as_json:
            click.echo(
                json.dumps(
                    {
                        "sent": False,
                        "registry": address,
                        "confirmation": digest,
                        "submission": preview,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            ui.note(
                t(
                    "Nothing was sent. Review this payload, then repeat with --confirm {hash}.",
                    hash=digest,
                )
            )
            ui.out(t("Registry: {address}", address=address))
            ui.out(Text(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True)))
            ui.note(report_warning(signed.report))
        return
    if digest != confirm.strip().casefold():
        raise click.ClickException(t("The preview changed; nothing was sent. Preview it again."))
    received = send_report(manager, conn, signed, registry=address)
    if as_json:
        click.echo(json.dumps({"sent": True, **received.model_dump()}))
    else:
        ui.ok(t("Report sent to the registry as {id}.", id=received.id))


@global_group.command("report")
@click.argument("text")
@click.option("--task", "task_id", help="Include this task's snapshot.")
@click.option("--confirm", metavar="SHA256", help="Send only if this matches the preview's hash.")
@json_option
def report_command(text: str, task_id: str | None, confirm: str | None, as_json: bool) -> None:
    """Preview a bug report; --confirm sends the exact payload."""
    manager = load_config()
    config = manager.load()
    with journal(None) as conn:
        signed = report_upload(conn, config, text, task_id)
        _report_command(manager, conn, config, signed, confirm, as_json)


@global_group.command("share")
@click.option("--stats/--no-stats", default=True, help="Include aggregate task statistics.")
@click.option(
    "--failures/--no-failures", default=False, help="Include failures and plan fallbacks."
)
@click.option(
    "--days",
    type=click.IntRange(1, 365),
    default=30,
    help="Share the last N days, from UTC midnight.",
)
@click.option("--confirm", metavar="SHA256", help="Send only if this matches the preview's hash.")
@json_option
def share_command(
    stats: bool, failures: bool, days: int, confirm: str | None, as_json: bool
) -> None:
    """Preview shared memory; --confirm sends this one snapshot."""
    manager = load_config()
    config = manager.load()
    with journal(None) as conn:
        signed = memory_upload(conn, config, stats=stats, failures=failures, days=days)
        _report_command(manager, conn, config, signed, confirm, as_json)
