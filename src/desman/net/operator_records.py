"""Build signed operator records on the machine that holds the private key."""

from datetime import datetime, timedelta

from desman.net import protocol, sign


class AssignmentEnded(ValueError):
    pass


def _build(
    key: sign.NodeKey,
    kind: sign.RecordKind,
    body: protocol.Job | protocol.ClusterTask | protocol.Moderation,
    *,
    now: datetime,
    lifetime: timedelta,
    until: datetime | None,
) -> sign.Signed:
    if until is not None:
        lifetime = min(lifetime, until - now.replace(microsecond=0))
    if lifetime <= timedelta(0):
        raise AssignmentEnded
    return sign.sign(key, kind, body.model_dump(mode="json"), issued_at=now, lifetime=lifetime)


def job(
    key: sign.NodeKey, body: protocol.Job, *, now: datetime, days: int, until: datetime | None
) -> sign.Signed:
    return _build(
        key, sign.RecordKind.TASK, body, now=now, lifetime=timedelta(days=days), until=until
    )


CLUSTER_DAYS = 30


def cluster_task(
    key: sign.NodeKey,
    body: protocol.ClusterTask,
    *,
    now: datetime,
    days: int,
    until: datetime | None,
) -> sign.Signed:
    if not 1 <= days <= CLUSTER_DAYS:
        raise ValueError("invalid cluster task term")
    return _build(
        key, sign.RecordKind.CLUSTER_TASK, body, now=now, lifetime=timedelta(days=days), until=until
    )


def moderation(
    key: sign.NodeKey,
    body: protocol.Moderation,
    *,
    now: datetime,
    lifetime: timedelta,
    until: datetime | None,
) -> sign.Signed:
    return _build(key, sign.RecordKind.MODERATION, body, now=now, lifetime=lifetime, until=until)
