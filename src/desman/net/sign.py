"""Signatures of desman-global (ADR 0004, decision 3; ROADMAP G.2).

Every key is Ed25519 (`cryptography`). A key is named by its fingerprint: the SHA-256 of its raw
public bytes, in hex. Only fingerprints and public keys ever leave the machine.

- **The node's key** (`node.key` in the data directory, next to the database) is the node's
  identity. It is created once, readable by its owner only (`0600`), and a key file anyone else
  may read is refused. Like an API key it never reaches a model, the chat or a log: `NodeKey`
  shows its fingerprint, never the key.
- **The root key** stays on the owner's server. Its public half is `root.pub` next to this file,
  a protected file (5.3): replacing the root is a change the person has to approve.
- **A signed record** (`Signed`) is a `Record` (`kind`, `id`, `issued_at`, `expires_at`,
  `signer`, `body`) and the signature of its canonical JSON (`canonical()`: sorted keys, no
  spaces, UTF-8).

**Authority** (`Trust`) is a chain of at most one link below the root. The root signs anything:
an *assignment* names an operator's key, its scopes and its term; releases are signed by the
root only. An operator signs tasks (scope `tasks`), moderation decisions and revocations of
nodes (scope `moderation`), only while its assignment runs and is not revoked. A *revocation*
names a key: nothing that key signed, nor anything resting on its assignment, is accepted from
then on. The root itself cannot be revoked; replacing it is a change of `root.pub`. An
operator's revocation of a key the root assigns afterwards gives way to the assignment.

`Trust.receive()` is where a record enters: schema, signer, signature, authority, time, replay
— and only a record that passed all of it is returned. Any failure is `Refused`, with a `Reason`
for the network log. Assignments and revocations are the state the registry hands out again on
every poll, so receiving one twice is harmless; any other record is refused the second time
(`Seen`). A revocation is honoured even after its own `expires_at`: obeying it only restricts.

**A trust list** (G.6c) is signed by the root only, by the person (`desman-registry trust
sign`): it names every assignment and revocation in effect when it was signed. The registry
hands it out with them, and cannot leave one out: a node that holds a list still in its term,
younger than TRUST_LIST_MAX_AGE, and every record it names is `fresh()`; otherwise it takes no
work from any operator. So a
registry that hides a revocation can do so only while the list is young, at most
TRUST_LIST_MAX_AGE (a node keeps the newest list it received).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Final, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from desman.core import db

ROOT_PUB: Final = Path(__file__).resolve().parent / "root.pub"
NODE_KEY: Final = "node.key"
TRUST_LIST_MAX_AGE: Final = timedelta(days=2)
SKEW: Final = timedelta(minutes=5)
"""How far ahead of our clock a record's `issued_at` may be."""

Fingerprint = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
RecordId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
PublicKeyText = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9+/]{43}=$")]
SignatureText = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9+/]{86}==$")]


# ── Keys ─────────────────────────────────────────────────────────────────────


class KeyFileError(ValueError):
    """A key file that cannot be used."""


def public_bytes(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def fingerprint(key: Ed25519PublicKey) -> str:
    return hashlib.sha256(public_bytes(key)).hexdigest()


def encode_public(key: Ed25519PublicKey) -> str:
    """A public key as text (base64 of its 32 raw bytes), as records carry it."""
    return base64.b64encode(public_bytes(key)).decode("ascii")


def decode_public(text: str) -> Ed25519PublicKey:
    try:
        return Ed25519PublicKey.from_public_bytes(base64.b64decode(text, validate=True))
    except (binascii.Error, ValueError) as e:
        raise KeyFileError(f"not an Ed25519 public key: {e}") from e


@dataclass(frozen=True)
class NodeKey:
    """A private key held by this process. Its repr is the fingerprint, never the key."""

    _private: Ed25519PrivateKey = field(repr=False)

    @property
    def public(self) -> Ed25519PublicKey:
        return self._private.public_key()

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.public)

    def sign(self, data: bytes) -> bytes:
        return self._private.sign(data)

    def __repr__(self) -> str:
        return f"NodeKey({self.fingerprint})"

    @classmethod
    def generate(cls) -> NodeKey:
        return cls(Ed25519PrivateKey.generate())


def default_node_key_path() -> Path:
    """Next to the database: $DESMAN_DATA_DIR/node.key or ~/.local/share/desman/node.key."""
    return db.default_db_path().parent / NODE_KEY


def create_node_key(path: Path) -> NodeKey:
    """A new key in a new file only its owner may read; an existing file is never replaced."""
    key = NodeKey.generate()
    pem = key._private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Written whole under another name, then linked into place: a failed write leaves no
    # half-written key behind, and the link fails if the file is there already.
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        try:
            if os.name != "nt":
                os.fchmod(fd, 0o600)
            os.write(fd, pem)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return key


def load_node_key(path: Path) -> NodeKey:
    # The permissions checked are those of the file read: one descriptor, never a link.
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as e:
        raise KeyFileError(f"{path} cannot be opened (it must be a plain file): {e}") from e
    with os.fdopen(fd, "rb") as file:
        if os.name != "nt" and os.fstat(fd).st_mode & 0o077:
            raise KeyFileError(
                f"{path} may be read by others; only its owner may (chmod 600 {path})"
            )
        data = file.read()
    try:
        private = serialization.load_pem_private_key(data, password=None)
    except (ValueError, TypeError) as e:
        raise KeyFileError(f"{path} is not a private key: {e}") from e
    if not isinstance(private, Ed25519PrivateKey):
        raise KeyFileError(f"{path} is not an Ed25519 key")
    return NodeKey(private)


def node_key(path: Path | None = None) -> NodeKey:
    """This node's key, created on first use."""
    path = default_node_key_path() if path is None else path
    try:
        return create_node_key(path)
    except FileExistsError:
        return load_node_key(path)


def public_pem(key: Ed25519PublicKey) -> bytes:
    """A public key as `root.pub` keeps it."""
    return key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


def load_public(path: Path) -> Ed25519PublicKey:
    try:
        public = serialization.load_pem_public_key(path.read_bytes())
    except FileNotFoundError as e:
        raise KeyFileError(f"{path} is missing") from e
    except ValueError as e:
        raise KeyFileError(f"{path} is not a public key: {e}") from e
    if not isinstance(public, Ed25519PublicKey):
        raise KeyFileError(f"{path} is not an Ed25519 key")
    return public


def load_root(path: Path = ROOT_PUB) -> Ed25519PublicKey:
    return load_public(path)


# ── Records ──────────────────────────────────────────────────────────────────


def canonical(value: Any) -> bytes:
    """What is signed: JSON with sorted keys, no spaces, UTF-8; no NaN or infinity."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


class RecordKind(StrEnum):
    ASSIGNMENT = "assignment"
    """The root names an operator's key, its scopes and its term."""
    REVOCATION = "revocation"
    """A key that must not be trusted any more."""
    TASK = "task"
    MODERATION = "moderation"
    RELEASE = "release"
    RESULT = "result"
    """A node's work on a task, signed by the node's own key: checked with `check_signature`
    against the key that sent it, never by `Trust` (a node is no authority)."""
    VERDICT = "verdict"
    """A node's review of another node's result, signed like a result."""
    TRUST_LIST = "trust_list"
    """The root names every assignment and revocation in effect (`TrustList`)."""


STATE: Final = frozenset({RecordKind.ASSIGNMENT, RecordKind.REVOCATION, RecordKind.TRUST_LIST})
"""Kinds handed out again on every poll: receiving one twice is not a replay."""


class Scope(StrEnum):
    TASKS = "tasks"
    MODERATION = "moderation"


OPERATOR_KINDS: Final = {
    RecordKind.TASK: Scope.TASKS,
    RecordKind.MODERATION: Scope.MODERATION,
    RecordKind.REVOCATION: Scope.MODERATION,
}
"""What an operator may sign, by the scope it needs. Anything else is the root's."""


def _utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        raise ValueError("a time without a time zone")
    return moment.astimezone(UTC).replace(microsecond=0)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: RecordKind
    id: RecordId
    issued_at: datetime
    expires_at: datetime
    signer: Fingerprint
    body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("issued_at", "expires_at")
    @classmethod
    def _whole_seconds_utc(cls, moment: datetime) -> datetime:
        return _utc(moment)

    @model_validator(mode="after")
    def _term(self) -> Record:
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at is not after issued_at")
        return self

    def canonical(self) -> bytes:
        return canonical(self.model_dump(mode="json"))


class Signed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record: Record
    signature: SignatureText


class Assignment(BaseModel):
    """The body of an assignment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: PublicKeyText
    scopes: frozenset[Scope] = Field(min_length=1)

    @property
    def fingerprint(self) -> str:
        return fingerprint(decode_public(self.key))


class Revocation(BaseModel):
    """The body of a revocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: Fingerprint
    reason: str = Field(default="", max_length=500)


class Listed(BaseModel):
    """A record a trust list names."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: RecordId
    until: datetime | None
    """Its `expires_at` (an assignment): required only until then; None for a revocation,
    which is honoured past its term and so required as long as the list lasts."""

    @field_validator("until")
    @classmethod
    def _whole_seconds_utc(cls, moment: datetime | None) -> datetime | None:
        return None if moment is None else _utc(moment)


class TrustList(BaseModel):
    """The body of a trust list."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    records: list[Listed] = Field(max_length=100_000)


def sign(
    key: NodeKey,
    kind: RecordKind,
    body: dict[str, Any],
    *,
    issued_at: datetime,
    lifetime: timedelta,
    id: str | None = None,
) -> Signed:
    record = Record(
        kind=kind,
        id=uuid.uuid4().hex if id is None else id,
        issued_at=issued_at,
        expires_at=issued_at + lifetime,
        signer=key.fingerprint,
        body=body,
    )
    signature = base64.b64encode(key.sign(record.canonical())).decode("ascii")
    return Signed(record=record, signature=signature)


def assign(
    root: NodeKey,
    operator: Ed25519PublicKey,
    scopes: set[Scope],
    *,
    issued_at: datetime,
    lifetime: timedelta,
) -> Signed:
    body = Assignment(key=encode_public(operator), scopes=frozenset(scopes))
    return sign(
        root,
        RecordKind.ASSIGNMENT,
        body.model_dump(mode="json") | {"scopes": sorted(body.scopes)},
        issued_at=issued_at,
        lifetime=lifetime,
    )


def revoke(
    signer: NodeKey, key: str, reason: str, *, issued_at: datetime, lifetime: timedelta
) -> Signed:
    body = Revocation(key=key, reason=reason)
    return sign(
        signer, RecordKind.REVOCATION, body.model_dump(), issued_at=issued_at, lifetime=lifetime
    )


def trust_list(
    root: NodeKey, listed: list[Listed], *, issued_at: datetime, lifetime: timedelta
) -> Signed:
    body = TrustList(records=sorted(listed, key=lambda r: r.id))
    return sign(
        root,
        RecordKind.TRUST_LIST,
        body.model_dump(mode="json"),
        issued_at=issued_at,
        lifetime=lifetime,
    )


# ── Checking ─────────────────────────────────────────────────────────────────


class Reason(StrEnum):
    MALFORMED = "malformed"
    FORGED = "forged"
    UNKNOWN_SIGNER = "unknown_signer"
    NOT_ALLOWED = "not_allowed"
    REVOKED = "revoked"
    EXPIRED = "expired"
    NOT_YET = "not_yet"
    REPLAY = "replay"
    CONFLICT = "conflict"
    """The same `id` with another content."""


class Refused(ValueError):
    """A record that is not accepted, and why."""

    def __init__(
        self, reason: Reason, detail: str, *, id: str | None = None, signer: str | None = None
    ) -> None:
        self.reason = reason
        self.id = id
        self.signer = signer
        super().__init__(f"{reason}: {detail}")


def parse(data: object) -> Signed:
    """A signed record as it arrived (JSON text or already parsed), validated."""
    try:
        if isinstance(data, str | bytes):
            return Signed.model_validate_json(data)
        return Signed.model_validate(data)
    except ValidationError as e:
        raise Refused(Reason.MALFORMED, str(e)) from e


def check_signature(signed: Signed, key: Ed25519PublicKey) -> None:
    """The record was signed by `key`; for records whose signer is known otherwise (nodes)."""
    record = signed.record
    if record.signer != fingerprint(key):
        raise Refused(Reason.UNKNOWN_SIGNER, "signer is not this key", id=record.id)
    try:
        key.verify(base64.b64decode(signed.signature, validate=True), record.canonical())
    except (InvalidSignature, binascii.Error) as e:
        raise Refused(Reason.FORGED, "bad signature", id=record.id, signer=record.signer) from e


class Seen(Protocol):
    def add(self, id: str, expires_at: datetime, now: datetime) -> bool:
        """Remember `id`; False if it was already seen."""
        ...


class MemorySeen:
    """Ids seen by this process, forgotten once their record expired (it is refused anyway)."""

    def __init__(self) -> None:
        self._ids: dict[str, datetime] = {}

    def add(self, id: str, expires_at: datetime, now: datetime) -> bool:
        for old in [i for i, until in self._ids.items() if until <= now]:
            del self._ids[old]
        if id in self._ids:
            return False
        self._ids[id] = expires_at
        return True


@dataclass(frozen=True)
class Operator:
    key: Ed25519PublicKey
    scopes: frozenset[Scope]
    issued_at: datetime
    expires_at: datetime
    """The term of its latest assignment."""

    def renewed(self, other: Operator) -> Operator:
        """The later assignment counts; two of the same second, only what both allow."""
        if other.issued_at != self.issued_at:
            return other if other.issued_at > self.issued_at else self
        return Operator(
            self.key,
            self.scopes & other.scopes,
            self.issued_at,
            min(self.expires_at, other.expires_at),
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Trust:
    """What this node believes: the root, the operators it assigned, the keys revoked."""

    def __init__(
        self,
        root: Ed25519PublicKey,
        *,
        clock: Callable[[], datetime] = _utc_now,
        seen: Seen | None = None,
    ) -> None:
        self.root = root
        self.root_fingerprint = fingerprint(root)
        self._clock = clock
        self._seen: Seen = MemorySeen() if seen is None else seen
        self._operators: dict[str, Operator] = {}
        self._revoked: dict[str, set[str]] = {}
        """Revoked keys, with the keys that signed their revocations."""
        self._state: dict[str, bytes] = {}
        """Assignments and revocations accepted, by id: the canonical form of the signed record."""
        self._list: Record | None = None
        """The newest trust list received."""

    def operator(self, key: str) -> Operator | None:
        return self._operators.get(key)

    def is_revoked(self, key: str) -> bool:
        return key in self._revoked

    def fresh(self) -> str | None:
        """None while the newest trust list is young, in term, and every required record
        was received; otherwise why not (for the network log)."""
        found = self._list
        if found is None:
            return "no trust list received"
        now = _utc(self._clock())
        if now >= found.expires_at:
            return f"the trust list expired at {found.expires_at.isoformat()}"
        if now >= found.issued_at + TRUST_LIST_MAX_AGE:
            return (
                f"the trust list was signed at {found.issued_at.isoformat()}, more than 2 days ago"
            )
        for listed in TrustList.model_validate(found.body).records:
            # Allow for skew: the registry stops handing an assignment out at its term.
            required = listed.until is None or listed.until > now + SKEW
            if required and listed.id not in self._state:
                return f"the trust list names a record not received: {listed.id}"
        return None

    def receive(self, data: object) -> Record:
        """A record that passed every check; anything else is `Refused`."""
        signed = parse(data)
        record = signed.record
        now = _utc(self._clock())

        def refuse(reason: Reason, detail: str) -> Refused:
            return Refused(reason, detail, id=record.id, signer=record.signer)

        if record.kind in STATE and record.id in self._state:
            # Its signature was checked when it first came, and its effect is in place; but
            # what happened since (revocations, assignments) applies to it like to any other.
            if self._state[record.id] != canonical(signed.model_dump(mode="json")):
                raise refuse(Reason.CONFLICT, "another record has this id")
            if record.signer in self._revoked:
                raise refuse(Reason.REVOKED, "the signer's key is revoked")
            self._check_body(record)
            return record

        if record.signer in self._revoked:
            raise refuse(Reason.REVOKED, "the signer's key is revoked")
        key = self._signer_key(record)
        check_signature(signed, key)
        self._check_authority(record, now)
        body = self._check_body(record)

        if record.issued_at > now + SKEW:
            raise refuse(Reason.NOT_YET, f"issued at {record.issued_at.isoformat()}")
        if record.kind is not RecordKind.REVOCATION and now >= record.expires_at:
            raise refuse(Reason.EXPIRED, f"expired at {record.expires_at.isoformat()}")

        if isinstance(body, Assignment):
            target = body.fingerprint
            # An operator may not revoke an operator; one that did before the assignment came
            # is overruled by the root's assignment.
            self._revoked.pop(target, None)
            new = Operator(
                decode_public(body.key), body.scopes, record.issued_at, record.expires_at
            )
            current = self._operators.get(target)
            self._operators[target] = new if current is None else current.renewed(new)
        elif isinstance(body, Revocation):
            self._revoked.setdefault(body.key, set()).add(record.signer)
            self._operators.pop(body.key, None)
        elif isinstance(body, TrustList):
            if self._list is None or record.issued_at > self._list.issued_at:
                self._list = record  # an older list handed out again changes nothing
        elif not self._seen.add(record.id, record.expires_at, now):
            raise refuse(Reason.REPLAY, "this record was already received")

        if record.kind in STATE:
            self._state[record.id] = canonical(signed.model_dump(mode="json"))
        return record

    def _signer_key(self, record: Record) -> Ed25519PublicKey:
        if record.signer == self.root_fingerprint:
            return self.root
        found = self._operators.get(record.signer)
        if found is None:
            raise Refused(
                Reason.UNKNOWN_SIGNER,
                "neither the root nor an operator",
                id=record.id,
                signer=record.signer,
            )
        return found.key

    def _check_authority(self, record: Record, now: datetime) -> None:
        if record.signer == self.root_fingerprint:
            return

        def refuse(reason: Reason, detail: str) -> Refused:
            return Refused(reason, detail, id=record.id, signer=record.signer)

        term = found = self._operators[record.signer]
        needed = OPERATOR_KINDS.get(record.kind)
        if needed is None or needed not in found.scopes:
            raise refuse(Reason.NOT_ALLOWED, f"an operator without it may not sign {record.kind}")
        if not term.issued_at <= record.issued_at < term.expires_at:
            raise refuse(Reason.NOT_ALLOWED, "issued outside the operator's assignment")
        if record.kind is not RecordKind.REVOCATION and now >= term.expires_at:
            raise refuse(Reason.EXPIRED, "the operator's assignment has expired")

    def _check_body(self, record: Record) -> Assignment | Revocation | TrustList | None:
        def refuse(reason: Reason, detail: str) -> Refused:
            return Refused(reason, detail, id=record.id, signer=record.signer)

        try:
            if record.kind is RecordKind.ASSIGNMENT:
                assignment = Assignment.model_validate(record.body)
                target = assignment.fingerprint
                if target == self.root_fingerprint:
                    raise refuse(Reason.NOT_ALLOWED, "the root is not an operator")
                if self.root_fingerprint in self._revoked.get(target, ()):
                    raise refuse(Reason.REVOKED, "the assigned key is revoked")
                return assignment
            if record.kind is RecordKind.REVOCATION:
                revocation = Revocation.model_validate(record.body)
                if revocation.key == self.root_fingerprint:
                    raise refuse(Reason.NOT_ALLOWED, "the root cannot be revoked")
                if record.signer != self.root_fingerprint and revocation.key in self._operators:
                    raise refuse(Reason.NOT_ALLOWED, "only the root revokes an operator")
                return revocation
            if record.kind is RecordKind.TRUST_LIST:
                return TrustList.model_validate(record.body)
        except (ValidationError, KeyFileError) as e:
            raise refuse(Reason.MALFORMED, str(e)) from e
        return None
