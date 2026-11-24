"""Operator panel reads. Mutations use the settings action and catalog confirmation."""

from collections.abc import Callable
from typing import Any

import click
from pydantic import BaseModel, ValidationError

from desman import net, operator_cli, providers
from desman.config import ConfigError, open_config
from desman.gui.panels import redact_snapshot
from desman.i18n import t
from desman.net import operator, sign
from desman.net import protocol as p


def snapshot(query: dict[str, list[str]]) -> dict[str, Any]:
    view = query.get("view", ["status"])[0]
    ref = query.get("id", [""])[0]
    status = query.get("status", [""])[0] or None
    data: dict[str, Any] = {"has_key": operator.operator_key_path().exists()}
    if not data["has_key"]:
        return data

    try:
        if view == "status":
            key = operator_cli.current_key()
            if key is None:
                return {"has_key": False}
            data.update(fingerprint=key.fingerprint, public_key=sign.encode_public(key.public))
            address = net.operator_registry(open_config().load())
            data["registry"] = address
            if address is None:
                raise click.ClickException(t("No registry address. Set it with --registry URL."))
            client = operator.OperatorClient(address, key)
            identity = operator_cli.call(
                client, operator.OperatorClient.whoami, show_assignment=False
            )
            data.update(scopes=identity.scopes, until=identity.until.isoformat())
        else:
            read, scope = read_view(view, ref, status)
            client = operator_cli.connect(open_config())
            reply = operator_cli.call(client, read, scope, show_assignment=False)
            data.update(reply.model_dump(mode="json"))
    except (click.ClickException, ConfigError) as exc:
        data["error"] = exc.format_message() if isinstance(exc, click.ClickException) else str(exc)
    try:
        config = open_config().load()
    except (ConfigError, OSError):
        config = None
    return redact_snapshot(data, lambda text: providers.redact_keys(text, config)[0])


def read_view(
    view: str, ref: str, status: str | None
) -> tuple[Callable[[operator.OperatorClient], BaseModel], sign.Scope | None]:
    scope = sign.Scope.MODERATION if view in {"complaints", "complaint", "suspensions"} else None
    try:
        match view:
            case "inbox":
                return operator.OperatorClient.inbox, scope
            case "jobs":
                jobs = p.OperatorJobQuery.model_validate({"status": status})
                return lambda c: c.jobs(jobs.status), scope
            case "clusters":
                clusters = p.OperatorClusterQuery.model_validate({"status": status})
                return lambda c: c.clusters(clusters.status), scope
            case "complaints":
                return lambda c: c.complaints(all=status == "all"), scope
            case "suspensions":
                return operator.OperatorClient.suspensions, scope
            case "nodes":
                return operator.OperatorClient.nodes, scope
            case "reports":
                return operator.OperatorClient.reports, scope
            case "log":
                return operator.OperatorClient.log, scope
            case "job" | "cluster" | "complaint" | "report":
                asked = p.OperatorRef(id=ref)
                return lambda c: detail(c, view, asked.id), scope
    except ValidationError as exc:
        raise ValueError("invalid operator filter or reference") from exc
    raise ValueError("unknown operator view")


def detail(client: operator.OperatorClient, view: str, ref: str) -> BaseModel:
    match view:
        case "job":
            return client.job(ref)
        case "cluster":
            return client.cluster(ref)
        case "complaint":
            return client.complaint(ref)
        case "report":
            return client.report(ref)
    raise ValueError("unknown operator detail")
