"""What a node and the registry say to each other (ADR 0004; ROADMAP G.4).

Both sides use these models: the node builds its requests from them (G.5) and the registry
validates every body with them where it enters, like any contract crossing a boundary. Every
request is signed (`net.request`); the bodies carry no secret. A lent module is named by its
provider, its model and the fingerprint of the API key (`openai_compat.key_id`), never the key.
"""

from __future__ import annotations

from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from desman.net.sign import Fingerprint, Signed

REGISTER: Final = "/v1/nodes/register"
HEARTBEAT: Final = "/v1/nodes/heartbeat"
LEAVE: Final = "/v1/nodes/leave"
TRUST: Final = "/v1/trust"

MAX_BODY: Final = 64 * 1024
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
    """Assignments still in their term and every revocation, oldest first: what a node checks
    on every poll (`sign.Trust.receive`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    records: list[Signed]


class Problem(BaseModel):
    """Any refusal. `error` is a `sign.Reason` or one of the registry's own words."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    error: str
    detail: str = ""
