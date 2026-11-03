"""What a node and the registry say to each other (ADR 0004; ROADMAP G.4, G.6).

Both sides use these models: the node builds its requests from them (G.5) and the registry
validates every body with them where it enters, like any contract crossing a boundary. Every
request is signed (`net.request`); the bodies carry no secret. A lent module is named by its
provider, its model and the fingerprint of the API key (`openai_compat.key_id`), never the key.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from desman.core.models import Kind
from desman.net.sign import Fingerprint, PublicKeyText, RecordId, Signed

REGISTER: Final = "/v1/nodes/register"
HEARTBEAT: Final = "/v1/nodes/heartbeat"
LEAVE: Final = "/v1/nodes/leave"
TRUST: Final = "/v1/trust"
CLAIM: Final = "/v1/work/claim"
EXTEND: Final = "/v1/work/extend"
HAND_IN: Final = "/v1/work/hand-in"

MAX_BODY: Final = 256 * 1024
MAX_RESULT_BYTES: Final = 160 * 1024
"""A result's text and files together, as JSON strings with every non-ASCII character escaped
(`\\uXXXX`, the longest any serializer writes them): what fits a request (`MAX_BODY`) with
room to spare, however the node encodes it."""
MAX_INPUT_BYTES: Final = 32 * 1024
MAX_CAPABILITIES: Final = 32
HEARTBEAT_SECONDS: Final = 300
"""How often the registry asks a node to say it is alive."""

ProviderId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")]
ModelName = Annotated[str, StringConstraints(pattern=r"^[\x21-\x7e]{1,200}$")]
KeyId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{12}$")]


class Capability(BaseModel):
    """A module a node lends, and how much of it a day."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider: ProviderId
    model: ModelName | None = None
    key: KeyId | None = None
    """The API key's fingerprint; None for a local model (the person's own hardware)."""
    calls_per_day: int = Field(ge=0, le=10_000_000)
    tokens_per_day: int = Field(ge=0, le=10_000_000_000)


def _distinct(capabilities: list[Capability]) -> None:
    names = [(c.provider, c.model, c.key) for c in capabilities]
    if len(set(names)) != len(names):
        raise ValueError("the same module twice")


class Register(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    capabilities: list[Capability] = Field(default_factory=list, max_length=MAX_CAPABILITIES)

    @model_validator(mode="after")
    def _once_each(self) -> Register:
        _distinct(self.capabilities)
        return self


class Heartbeat(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    capabilities: list[Capability] | None = Field(default=None, max_length=MAX_CAPABILITIES)
    """None: as before."""

    @model_validator(mode="after")
    def _once_each(self) -> Heartbeat:
        _distinct(self.capabilities or [])
        return self


class NodeState(BaseModel):
    """The registry's answer to register and heartbeat."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node: Fingerprint
    heartbeat_seconds: int


class Left(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node: Fingerprint
    changed: bool


class TrustState(BaseModel):
    """Assignments still in their term and every revocation, oldest first, then the newest
    trust list in its term (G.6c): what a node checks on every poll (`sign.Trust.receive`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    records: list[Signed]


class Problem(BaseModel):
    """Any refusal. `error` is a `sign.Reason` or one of the registry's own words."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    error: str
    detail: str = ""


# ── Work (G.6) ───────────────────────────────────────────────────────────────
#
# A job is one self-contained subtask (no dependencies), the body of a `task` record an
# operator signed. The registry lends it to one node at a time with a lease and its token,
# like `Store.claim()`: first to be done (stage `execute`), then, the result in hand, to a node
# of another owner to be reviewed (stage `recheck`). What a node hands in is a `result` or a
# `verdict` record signed by its own key.

LeaseToken = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
Stage = Literal["execute", "recheck"]
InputName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]
FilePath = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*(/[A-Za-z0-9_-][A-Za-z0-9_.-]*){0,7}$"
    ),
]


def _encoded(*texts: str) -> int:
    """Bytes the texts take as JSON strings at worst (ASCII-escaped)."""
    return sum(len(json.dumps(text, ensure_ascii=True)) for text in texts)


class Job(BaseModel):
    """The body of a `task` record: what a node is asked to do."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str = Field(min_length=1, max_length=4000)
    acceptance_criteria: str = Field(min_length=1, max_length=2000)
    inputs: dict[InputName, str] = Field(default_factory=dict, max_length=16)
    kind: Kind = Kind.TEXT
    runs_code: bool = False

    @model_validator(mode="after")
    def _inputs_fit(self) -> Job:
        if _encoded(*self.inputs.values()) > MAX_INPUT_BYTES:
            raise ValueError(f"the inputs are over {MAX_INPUT_BYTES} bytes")
        return self


class Claim(BaseModel):
    """A node asks for work. Empty for now: the registry knows what it lends."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Work(BaseModel):
    """Work lent to a node until `expires_at`, unless extended (`EXTEND`) with its token."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lease: LeaseToken
    expires_at: datetime
    stage: Stage
    task: Signed
    """The operator's `task` record, as the node checks it (`sign.Trust`)."""
    result: Signed | None = None
    """For `recheck`: the result to review, signed by the node that did the work."""
    result_key: PublicKeyText | None = None
    """For `recheck`: that node's public key, so the reviewer checks the signature itself."""

    @model_validator(mode="after")
    def _recheck_has_result(self) -> Work:
        if (self.stage == "recheck") != (self.result is not None and self.result_key is not None):
            raise ValueError("a recheck, and only a recheck, comes with a result and its key")
        return self


class Offer(BaseModel):
    """The answer to a claim: work, or none for now."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    work: Work | None = None


class Extend(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    lease: LeaseToken


class Lease(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lease: LeaseToken
    expires_at: datetime


class ResultFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: FilePath
    text: str


class Result(BaseModel):
    """The body of a `result` record: a node's work on a job."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    job: RecordId
    lease: LeaseToken
    outcome: Literal["done", "failed"]
    """`failed`: the node could not do it (its models said so, or its retries ran out)."""
    output: str = Field(default="", max_length=MAX_RESULT_BYTES)
    files: list[ResultFile] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def _fits(self) -> Result:
        if len({f.path for f in self.files}) != len(self.files):
            raise ValueError("the same file twice")
        if _encoded(self.output, *(f.text for f in self.files)) > MAX_RESULT_BYTES:
            raise ValueError(f"the result is over {MAX_RESULT_BYTES} bytes")
        return self


class Verdict(BaseModel):
    """The body of a `verdict` record: a node's review of another node's result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    job: RecordId
    lease: LeaseToken
    result: RecordId
    """The `result` record reviewed."""
    accept: bool
    reason: str = Field(default="", max_length=2000)


class HandIn(BaseModel):
    """A signed `result` (stage `execute`) or `verdict` (stage `recheck`) and its lease."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lease: LeaseToken
    record: Signed


class Received(BaseModel):
    """The registry kept what was handed in; `status` is the job's now."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    job: RecordId
    status: str
