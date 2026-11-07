"""Config-independent contracts for human settings fields (also imported by registry UI)."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from desman.tuning_limits import TUNING_BOUNDS


class Fields(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)


Provider = Annotated[str, Field(pattern=r"^[a-z0-9_-]+$", min_length=1, max_length=40)]
Fingerprint = Annotated[str, Field(pattern=r"^[0-9a-f]{12}$")]
Role = Literal["planner", "reviewer", "decomposer", "text", "code", "analysis", "interlocutor"]


class General(Fields):
    section: Literal["general"]
    language: Literal["en", "ru", "zh", "es", "pt", "de", "fr", "ja", "ko", "pl", "tr", "uk", "hi"]


class Wizard(Fields):
    section: Literal["wizard"]


class Keys(Fields):
    section: Literal["keys"]
    operation: Literal["add", "remove", "check"]
    provider: Provider
    key: Fingerprint | None = None
    label: str = Field(default="", max_length=60)

    @model_validator(mode="after")
    def removal_key(self) -> "Keys":
        if (self.operation == "remove") != (self.key is not None):
            raise ValueError("only removal requires a fingerprint")
        return self


class ProviderSetting(Fields):
    section: Literal["provider"]
    provider: Provider
    setting: str = Field(pattern=r"^[A-Z0-9_]+$", max_length=80)
    value: str = Field(max_length=1000)


class Roles(Fields):
    section: Literal["roles"]
    role: Role
    model: str | None = Field(pattern=r"^[a-z0-9_-]+/\S+$", max_length=200)
    key: Fingerprint | None


class Permission(Fields):
    section: Literal["permissions"]
    name: Literal["admin_rights", "fs_outside_access", "network_access", "shell_execution"]
    on: bool


class Personalize(Fields):
    section: Literal["personalize"]
    on: bool


class Forget(Fields):
    section: Literal["forget"]
    note: int = Field(ge=1)


class Behaviour(Fields):
    section: Literal["behaviour"]
    length: Literal["auto", "short", "detailed"] = "auto"
    tone: Literal["auto", "formal", "informal"] = "auto"
    style: Literal["auto", "simple", "technical"] = "auto"
    custom: str = Field(default="", max_length=500)
    reset: bool = False


class Limits(Fields):
    section: Literal["limits"]
    name: Literal["max_subtasks", "max_parts", "context_tokens", "max_attempts", "max_transient"]
    value: int | Literal["auto"]

    @model_validator(mode="after")
    def bounds(self) -> "Limits":
        low, high = TUNING_BOUNDS[self.name]
        if self.value != "auto" and not low <= self.value <= high:
            raise ValueError("outside tuning bounds")
        return self


class Models(Fields):
    section: Literal["models"]
    operation: Literal["show", "add", "use"]
    model: str = Field(min_length=3, max_length=200)
    server: Literal["ollama", "llamacpp"] = "ollama"


class Network(Fields):
    section: Literal["network"]
    operation: Literal["join", "leave", "donate", "undonate", "quota"]
    provider: Provider | None = None
    registry: str | None = Field(default=None, max_length=200)
    calls: int | None = Field(default=None, ge=1, le=100000)
    tokens: int | None = Field(default=None, ge=1000, le=100000000)

    @model_validator(mode="after")
    def needs_provider(self) -> "Network":
        if (self.operation in ("donate", "undonate", "quota")) != (self.provider is not None):
            raise ValueError("provider required for lending")
        return self


class Protected(Fields):
    section: Literal["protected"]
    operation: Literal["show", "approve"]


Setting = Annotated[
    General
    | Wizard
    | Keys
    | ProviderSetting
    | Roles
    | Permission
    | Personalize
    | Forget
    | Behaviour
    | Limits
    | Models
    | Network
    | Protected,
    Field(discriminator="section"),
]
