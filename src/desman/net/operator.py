"""Signed registry calls and the operator's local Ed25519 key (ADR 0016)."""

from __future__ import annotations

import os
import stat
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import httpx
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import BaseModel

from desman.core import db
from desman.net import node, sign
from desman.net import protocol as p


def operator_key_path() -> Path:
    return db.default_db_path().parent / "operator.key"


def load_key() -> sign.NodeKey | None:
    path = operator_key_path()
    try:
        return sign.load_node_key(path)
    except sign.KeyFileError as e:
        if isinstance(e.__cause__, FileNotFoundError):
            return None
        raise


def read_key(src: Path) -> sign.NodeKey:
    """Read and validate an Ed25519 PEM for import."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(src, flags), "rb") as file:
        if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
            raise sign.KeyFileError("expected a regular file")
        data = file.read(16385)
        if len(data) > 16384:
            raise sign.KeyFileError("private key file is too large")
    try:
        private = serialization.load_pem_private_key(data, password=None)
    except (ValueError, TypeError, UnsupportedAlgorithm) as e:
        raise sign.KeyFileError("invalid private key") from e
    if not isinstance(private, Ed25519PrivateKey):
        raise sign.KeyFileError("expected an Ed25519 private key")
    return sign.NodeKey(private)


def import_key(src: Path, *, replace: bool = False) -> str:
    key = read_key(src)
    install_key(key, replace=replace)
    return key.fingerprint


def install_key(key: sign.NodeKey, *, replace: bool = False) -> None:
    path = operator_key_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = key._private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as file:
            if sys.platform != "win32":
                os.fchmod(file.fileno(), 0o600)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            try:
                os.link(tmp, path)
            except FileExistsError:
                existing = load_key()
                if existing is None or existing.fingerprint != key.fingerprint:
                    raise
    finally:
        tmp.unlink(missing_ok=True)


def create_key() -> str:
    """Create a new key and return its public half in the registry's assignment format."""
    key = sign.NodeKey.generate()
    install_key(key)
    return sign.encode_public(key.public)


def remove_key() -> None:
    operator_key_path().unlink(missing_ok=True)


class OperatorError(Exception):
    def __init__(self, reason: str, status: int | None = None) -> None:
        self.reason = reason
        self.status = status
        super().__init__(reason)


M = TypeVar("M", bound=BaseModel)


class OperatorClient:
    def __init__(
        self,
        base_url: str,
        key: sign.NodeKey,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.base_url = base_url
        self.key = key
        self.transport = transport
        self.clock = clock

    def _post(self, route: str, body: BaseModel, response: type[M]) -> M:
        with httpx.Client(
            transport=self.transport, timeout=node.TIMEOUT, follow_redirects=False
        ) as client:
            try:
                return node.signed_call(
                    client,
                    self.key,
                    self.base_url,
                    "POST",
                    route,
                    body,
                    response,
                    self.clock(),
                    time.monotonic() + node.TIMEOUT,
                    include_nulls=True,
                )
            except node.Problem as e:
                raise OperatorError(e.error, e.status) from e
            except node.Unreachable as e:
                raise OperatorError(str(e)) from e

    def whoami(self) -> p.OperatorIdentity:
        identity = self._post(p.OPERATOR_WHOAMI, p.OperatorEmpty(), p.OperatorIdentity)
        if identity.fingerprint != self.key.fingerprint:
            raise OperatorError("bad_reply")
        return identity

    def inbox(self) -> p.OperatorInbox:
        return self._post(p.OPERATOR_INBOX, p.OperatorEmpty(), p.OperatorInbox)

    def jobs(self, status: p.JobStatus | None = None) -> p.OperatorJobs:
        return self._post(p.OPERATOR_JOBS, p.OperatorJobQuery(status=status), p.OperatorJobs)

    def job(self, id: str) -> p.OperatorJobDetail:
        return self._post(p.OPERATOR_JOB, p.OperatorRef(id=id), p.OperatorJobDetail)

    def job_add(self, record: sign.Signed) -> p.OperatorJob:
        return self._post(p.OPERATOR_JOB_ADD, p.OperatorRecord(record=record), p.OperatorJob)

    def job_settle(self, id: str, decision: p.OperatorDecision) -> p.OperatorJob:
        return self._post(
            p.OPERATOR_JOB_SETTLE, p.OperatorSettle(id=id, decision=decision), p.OperatorJob
        )

    def clusters(self, status: p.ClusterStatus | None = None) -> p.OperatorClusters:
        return self._post(
            p.OPERATOR_CLUSTERS, p.OperatorClusterQuery(status=status), p.OperatorClusters
        )

    def cluster(self, id: str) -> p.OperatorClusterDetail:
        return self._post(p.OPERATOR_CLUSTER, p.OperatorRef(id=id), p.OperatorClusterDetail)

    def cluster_add(self, record: sign.Signed) -> p.OperatorCluster:
        return self._post(
            p.OPERATOR_CLUSTER_ADD, p.OperatorRecord(record=record), p.OperatorCluster
        )

    def cluster_settle(
        self, id: str, decision: p.OperatorDecision, note: str | None = None
    ) -> p.OperatorCluster:
        body = p.OperatorClusterSettle(id=id, decision=decision, note=note)
        return self._post(p.OPERATOR_CLUSTER_SETTLE, body, p.OperatorCluster)

    def complaints(self, *, all: bool = False) -> p.OperatorComplaints:
        return self._post(
            p.OPERATOR_COMPLAINTS, p.OperatorComplaintsQuery(all=all), p.OperatorComplaints
        )

    def complaint(self, id: str) -> p.ComplaintDetail:
        return self._post(p.OPERATOR_COMPLAINT, p.OperatorRef(id=id), p.ComplaintDetail)

    def complaint_dismiss(self, id: str) -> p.OperatorRef:
        return self._post(p.OPERATOR_COMPLAINT_DISMISS, p.OperatorRef(id=id), p.OperatorRef)

    def moderation(self, record: sign.Signed) -> p.ModerationRow:
        return self._post(p.OPERATOR_MODERATION, p.OperatorRecord(record=record), p.ModerationRow)

    def suspensions(self) -> p.OperatorSuspensions:
        return self._post(p.OPERATOR_SUSPENSIONS, p.OperatorEmpty(), p.OperatorSuspensions)

    def reports(self, kind: p.ReportKind | None = None) -> p.OperatorReports:
        return self._post(p.OPERATOR_REPORTS, p.OperatorReportQuery(kind=kind), p.OperatorReports)

    def report(self, id: str) -> p.OperatorReportDetail:
        return self._post(p.OPERATOR_REPORT, p.OperatorRef(id=id), p.OperatorReportDetail)

    def nodes(self) -> p.OperatorNodes:
        return self._post(p.OPERATOR_NODES, p.OperatorEmpty(), p.OperatorNodes)

    def owner(self, node: str, owner: str | None) -> p.OperatorNode:
        return self._post(p.OPERATOR_OWNER, p.OperatorOwner(node=node, owner=owner), p.OperatorNode)

    def log(self, limit: int = 100) -> p.OperatorLog:
        return self._post(p.OPERATOR_LOG, p.OperatorLogQuery(limit=limit), p.OperatorLog)
