"""Tool policy (ROADMAP 3.2): the role's tools, bounded by the human's permissions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result
from pydantic import BaseModel

from desman import cli as cli_module
from desman.cli import EXIT_DONE, cli
from desman.config import ConfigManager, Permissions, default_config_path
from desman.core import db
from desman.core.policy import (
    POLICY_PATH,
    Access,
    Permission,
    Policy,
    PolicyError,
    Role,
    RolePolicy,
    default_policy,
    load_policy,
    switched_on,
)
from desman.core.tools import READ_FILE, Registry, Tool, Toolbox, ToolCall, builtin_registry
from desman.core.workspace import Workspace
from helpers import make_config


class _NoArgs(BaseModel):
    pass


def tool(name: str, permission: Permission, ran: list[str] | None = None) -> Tool[_NoArgs]:
    def run(args: _NoArgs, workspace: Workspace) -> dict[str, Any]:
        if ran is not None:
            ran.append(name)
        return {}

    return Tool(name, f"{name} tool", _NoArgs, permission, run)


SEARCH = tool("search", Permission.NETWORK)
SHELL = tool("shell", Permission.SHELL)
READ = READ_FILE


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# ── The shipped policy ───────────────────────────────────────────────────────


def test_the_shipped_policy_gives_the_file_tools_to_the_executor_only() -> None:
    policy = default_policy()
    assert policy.tools_for(Role.EXECUTOR) == {"read_file", "write_file", "list_files"}
    assert policy.tools_for(Role.PLANNER) == policy.tools_for(Role.REVIEWER) == frozenset()
    assert POLICY_PATH.name == "policy.toml" and POLICY_PATH.parent.name == "desman"


# ── Loading ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("[roles.executor\n", "cannot read it"),
        ('[roles.janitor]\ntools = ["read_file"]\n', "janitor"),
        ('[roles.executor]\ntools = ["read_file"]\nadmin = true\n', "admin"),
        ('[roles.executor]\ntools = "read_file"\n', "tools"),
        ('[roles.executor]\ntools = ["raed_file"]\n', "unknown tools for executor: raed_file"),
        ('everyone = "all tools"\n', "everyone"),
    ],
)
def test_a_broken_policy_is_an_error(tmp_path: Path, text: str, problem: str) -> None:
    path = write(tmp_path / "policy.toml", text)
    with pytest.raises(PolicyError, match=problem):
        load_policy(path, known_tools=builtin_registry().names)


def test_a_missing_policy_is_an_error_not_an_empty_one(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match="cannot read it"):
        load_policy(tmp_path / "nope.toml")


def test_roles_left_out_get_no_tools(tmp_path: Path) -> None:
    policy = load_policy(write(tmp_path / "p.toml", '[roles.executor]\ntools = ["read_file"]\n'))
    assert policy.tools_for(Role.REVIEWER) == frozenset()


# ── Both layers must allow a tool ────────────────────────────────────────────


def test_a_tool_needs_the_policy_and_the_human_switch() -> None:
    policy = Policy(roles={Role.EXECUTOR: RolePolicy(tools=["read_file", "search"])})
    every_switch = list(Permission)

    off = Access(policy, Role.EXECUTOR)
    assert off.refusal(READ) is None  # the workspace needs no switch
    reason = off.refusal(SEARCH)
    assert reason is not None and "'network_access'" in reason and "not turned on" in reason

    on = Access(policy, Role.EXECUTOR, [Permission.NETWORK])
    assert on.refusal(SEARCH) is None
    # Every switch on does not give a tool the policy does not list.
    unlisted = Access(policy, Role.EXECUTOR, every_switch).refusal(SHELL)
    assert unlisted is not None and "does not give 'shell' to the executor" in unlisted
    # Nor a tool listed for another role.
    reviewer = Access(policy, Role.REVIEWER, every_switch).refusal(READ)
    assert reviewer is not None and "to the reviewer" in reviewer


def test_switches_come_from_the_human_config() -> None:
    assert switched_on(Permissions().model_dump()) == frozenset()
    permissions = Permissions(network_access=True, admin_rights=True)
    assert switched_on(permissions.model_dump()) == {Permission.NETWORK, Permission.ADMIN}
    assert {p.value for p in Permission} - {"workspace"} == set(Permissions.model_fields)


def test_refusals_and_elevated_calls_are_audited(tmp_path: Path) -> None:
    ran: list[str] = []
    admin = tool("reboot", Permission.ADMIN, ran)
    registry = Registry([admin, SEARCH])
    policy = Policy(roles={Role.EXECUTOR: RolePolicy(tools=["reboot"])})
    entries: list[tuple[str, dict[str, Any]]] = []

    def box(*switches: Permission) -> Toolbox:
        access = Access(policy, Role.EXECUTOR, switches)
        return Toolbox(
            Workspace(tmp_path, "t"),
            registry,
            access=access,
            record=lambda a, d: entries.append((a, d)),
        )

    assert box().specs() == []  # nothing usable is offered
    assert not box().call(ToolCall(tool="reboot")).ok
    assert not box(*Permission).call(ToolCall(tool="search")).ok  # not in the policy
    assert box(Permission.ADMIN).call(ToolCall(tool="reboot")).ok
    assert ran == ["reboot"]
    assert [(a, d.get("permission"), d.get("elevated")) for a, d in entries] == [
        ("tool_refused", "admin_rights", None),
        ("tool_refused", "network_access", None),
        ("tool_called", "admin_rights", True),
    ]
    assert "not turned on" in entries[0][1]["reason"]
    assert "does not give 'search'" in entries[1][1]["reason"]


def test_agents_cannot_reach_the_policy_or_the_config(tmp_path: Path) -> None:
    box = Toolbox(Workspace(tmp_path, "t"))
    config = default_config_path()
    for target in (str(POLICY_PATH), str(config), f"../../{config.name}", "../policy.toml"):
        result = box.call(ToolCall(tool="write_file", args={"path": target, "content": "{}"}))
        assert not result.ok and "PathEscapeError" in (result.error or "")
    assert not config.exists()


# ── desman run ───────────────────────────────────────────────────────────────


def fake_script(tmp_path: Path) -> Path:
    routes = [
        {"match": "Execute", "steps": ['{"ok": 1}']},
        {"match": "Review", "steps": ['{"verdict": "accept"}'], "repeat_last": True},
    ]
    return write(tmp_path / "fake.json", json.dumps({"routes": routes}))


def desman(*args: str | Path) -> Result:
    return CliRunner().invoke(cli, [str(a) for a in args])


ADMIN_WARNING = "admin_rights is on: agents may act with elevated privileges."


def test_every_run_warns_while_admin_rights_is_on(tmp_path: Path) -> None:
    quiet = desman("run", "g", "--fake-llm", fake_script(tmp_path))
    assert quiet.exit_code == EXIT_DONE, quiet.output
    assert ADMIN_WARNING not in quiet.output

    config = make_config()
    config.permissions.admin_rights = True
    ConfigManager(default_config_path()).save(config)
    loud = desman("run", "g", "--fake-llm", fake_script(tmp_path))
    assert loud.exit_code == EXIT_DONE, loud.output
    assert ADMIN_WARNING in loud.output
    as_json = desman("run", "g", "--fake-llm", fake_script(tmp_path), "--json")
    assert ADMIN_WARNING in as_json.stderr
    json.loads(as_json.stdout)  # the warning does not break machine output
    task_id = db.open_db().execute("SELECT id FROM tasks LIMIT 1").fetchone()[0]
    resumed = desman("resume", task_id, "--fake-llm", fake_script(tmp_path))
    assert ADMIN_WARNING in resumed.output


def test_a_broken_policy_stops_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> Policy:
        raise PolicyError("policy.toml: cannot read it (gone)")

    monkeypatch.setattr(cli_module, "default_policy", broken)
    result = desman("run", "g", "--fake-llm", fake_script(tmp_path))
    assert result.exit_code == 1
    assert "Cannot use the tool policy: policy.toml: cannot read it (gone)" in result.output
    assert not db.default_db_path().exists()  # stopped before creating a task
