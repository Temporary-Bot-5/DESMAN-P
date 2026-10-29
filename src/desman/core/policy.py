"""Tool policy: which role may use which tool, within what the human allowed (ROADMAP 3.2).

Two layers, both set by people only:

- the human's switches in config.json (`permissions`, set in `desman config`) are the upper
  bound: a tool that needs the network, a shell, files outside the workspace or admin rights is
  refused while its switch is off, whatever the policy says;
- `policy.toml` (shipped in the package, under version control) says which tools each role may
  use. A tool it does not list is refused even when every switch is on.

A tool is usable only when both allow it; the workspace tools need no switch. Agents can change
neither layer: they reach files only through the workspace tools, and neither file is in a
workspace. Every refusal is recorded in the audit (`tool_refused`, with the reason).

`admin_rights` stays available for the owner's test machines: off by default, and while it is
on, every run warns about it and every call of a tool that needs it is audited as elevated.
"""

from __future__ import annotations

import functools
import tomllib
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, ValidationError

if TYPE_CHECKING:
    from desman.core.tools import Tool

POLICY_PATH: Final = Path(__file__).resolve().parent.parent / "policy.toml"


class Permission(StrEnum):
    """What a tool needs. The values are the human's switches in config.json (`permissions`)."""

    WORKSPACE = "workspace"
    """The task's own directory: needs no switch."""
    NETWORK = "network_access"
    SHELL = "shell_execution"
    FS_OUTSIDE = "fs_outside_access"
    ADMIN = "admin_rights"


class Role(StrEnum):
    """Roles that may be given tools."""

    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"


class PolicyError(ValueError):
    """policy.toml cannot be used. Never treated as "no policy": that would refuse everything
    silently or, worse, be mistaken for permission."""


class RolePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tools: list[str] = []


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    roles: dict[Role, RolePolicy] = {}

    def tools_for(self, role: Role) -> frozenset[str]:
        rules = self.roles.get(role)
        return frozenset(rules.tools) if rules else frozenset()


def load_policy(path: Path = POLICY_PATH, known_tools: Iterable[str] | None = None) -> Policy:
    """Read and check a policy; with `known_tools`, a tool name not among them is an error (a
    misspelt name would otherwise refuse that tool without saying why)."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise PolicyError(f"{path}: cannot read it ({e})") from e
    try:
        policy = Policy.model_validate(data)
    except ValidationError as e:
        raise PolicyError(f"{path}: {e}") from e
    if known_tools is not None:
        known = set(known_tools)
        for role, rules in policy.roles.items():
            unknown = sorted(set(rules.tools) - known)
            if unknown:
                raise PolicyError(f"{path}: unknown tools for {role.value}: {', '.join(unknown)}")
    return policy


@functools.cache
def default_policy() -> Policy:
    """The shipped policy, read once (a broken one raises PolicyError every time)."""
    from desman.core.tools import builtin_registry

    return load_policy(known_tools=builtin_registry().names)


def switched_on(permissions: Mapping[str, Any]) -> frozenset[Permission]:
    """The human's switches that are on, e.g. from `Config.permissions.model_dump()`."""
    names = {p.value for p in Permission}
    return frozenset(Permission(name) for name, on in permissions.items() if on and name in names)


class Access:
    """What one role may use: the policy's list, bounded by the human's switches."""

    def __init__(self, policy: Policy, role: Role, switches: Iterable[Permission] = ()) -> None:
        self.policy = policy
        self.role = role
        self.switches = frozenset(switches)

    def refusal(self, tool: Tool[Any]) -> str | None:
        """Why `tool` may not be used, or None if it may."""
        if tool.name not in self.policy.tools_for(self.role):
            return f"the tool policy does not give {tool.name!r} to the {self.role.value}"
        if tool.permission is not Permission.WORKSPACE and tool.permission not in self.switches:
            return (
                f"{tool.name!r} needs the permission {tool.permission.value!r}, "
                "which the human has not turned on"
            )
        return None


def executor_access(switches: Iterable[Permission] = ()) -> Access:
    """The executor's access under the shipped policy."""
    return Access(default_policy(), Role.EXECUTOR, switches)
