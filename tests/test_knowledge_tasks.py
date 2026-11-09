"""Tasks on the knowledge tree (ADR 0009, K.3): the planner names the nodes each subtask needs,
the executor gets their texts within its budget, `read_node` repairs a miss, the choice is
audited and shown by `status`; a job from desman-global never sees the tree."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import FakeClock
from desman import knowledge
from desman.actions import print_report
from desman.agents import KNOWLEDGE_INSTRUCTIONS, llm_executor
from desman.core import db
from desman.core.context import DependencyResult, Knowledge, KnowledgeText, build_context, fit
from desman.core.coordinator import Coordinator, Crew, Review, Success, Verdict, Work
from desman.core.effects import evidence
from desman.core.models import MAX_KNOWLEDGE, Origin, Subtask, SubtaskSpec, TaskSpec
from desman.core.report import task_report
from desman.core.states import TaskStatus
from desman.core.store import Store
from desman.core.tools import Registry, Toolbox, ToolCall, builtin_registry, read_node_tool
from desman.core.workspace import Workspace
from desman.knowledge import Node
from desman.llm.fake import FakeLLM
from desman.planner import KNOWLEDGE_RULES, continuation, plan

TREE = (
    Node("DESMAN/Manual/tasks", "how work is planned", "Tasks are planned by a planner."),
    Node("DESMAN/Docs/ADR/0009/Decision", "the tree", "Nodes have paths.", plan=True),
    Node("DESMAN/Docs/ADR/0009/Context", "why", "Prompts were too large."),
)


def read_tree(path: str) -> KnowledgeText | None:
    return knowledge.read(TREE, path)


def sub(description: str, *context: str, depends_on: tuple[int, ...] = ()) -> dict[str, Any]:
    return {
        "description": description,
        "acceptance_criteria": f"{description} is done",
        "depends_on": list(depends_on),
        "context": list(context),
    }


def reply(*subtasks: dict[str, Any]) -> str:
    return json.dumps({"subtasks": list(subtasks)})


# ── The tree for tasks ───────────────────────────────────────────────────────


def test_tasks_see_the_docs_but_not_the_chat_s_commands() -> None:
    tree = knowledge.task_tree()
    assert tree and not any(n.path.startswith("Commands/") for n in tree)
    assert {n.path for n in tree} < {n.path for n in knowledge.release_tree()}


def test_read_gives_a_node_s_text_a_section_s_contents_or_nothing() -> None:
    node = knowledge.read(TREE, "desman/docs/adr/0009/decision")
    assert node == KnowledgeText(
        path="DESMAN/Docs/ADR/0009/Decision", text="Nodes have paths.", plan=True
    )
    section = knowledge.read(TREE, "DESMAN/Docs/ADR/0009/")
    assert section is not None and section.path == "DESMAN/Docs/ADR/0009/"
    assert "DESMAN/Docs/ADR/0009/Context: why" in section.text
    assert "Prompts were too large" not in section.text  # a table of contents, never texts
    whole = knowledge.read(TREE, "")
    assert whole is not None and whole.path == "/" and "DESMAN/Manual/tasks" in whole.text
    assert knowledge.read(TREE, "tasks") == knowledge.read(TREE, "DESMAN/Manual/tasks")
    assert knowledge.read(TREE, "DESMAN/Nowhere") is None


# ── The planner ──────────────────────────────────────────────────────────────


def test_the_planner_sees_the_table_of_contents_and_names_nodes_per_subtask() -> None:
    llm = FakeLLM(
        reply(
            sub("explain the tree", "desman/docs/adr/0009/decision", "DESMAN/Docs/ADR/0009/"),
            sub("write it up", depends_on=(0,)),
        )
    )
    planned = plan(llm, "explain the knowledge tree", tree=TREE)
    [call] = llm.calls
    system = call.messages[0].content
    assert KNOWLEDGE_RULES in system and "DESMAN/Docs/ADR/0009/Decision [plan]: the tree" in system
    assert "Nodes have paths." not in system  # never the texts
    first, second = planned.spec.subtasks
    assert first.knowledge == ["DESMAN/Docs/ADR/0009/Decision", "DESMAN/Docs/ADR/0009/"]
    assert second.knowledge == []
    assert planned.details() == {"plan": "llm"}


def test_unknown_paths_are_left_out_and_recorded_repeats_and_extras_dropped() -> None:
    many = ["tasks", "DESMAN/Manual/tasks", "DESMAN/Docs/ADR/0009/Context", "made/up", "x/"]
    llm = FakeLLM(reply(sub("one", *many, "DESMAN/Docs/ADR/0009/Decision")))
    planned = plan(llm, "goal", tree=TREE)
    [only] = planned.spec.subtasks
    assert len(only.knowledge) <= MAX_KNOWLEDGE
    assert only.knowledge == [
        "DESMAN/Manual/tasks",
        "DESMAN/Docs/ADR/0009/Context",
        "DESMAN/Docs/ADR/0009/Decision",
    ]
    assert planned.details() == {"plan": "llm", "context_unknown": ["made/up", "x/"]}


@pytest.mark.parametrize(
    ("written", "kept"),
    [
        (None, []),
        ("tasks", ["DESMAN/Manual/tasks"]),
        ([{"path": "tasks"}, "tasks"], ["DESMAN/Manual/tasks"]),
        (7, []),
    ],
)
def test_a_badly_written_context_never_costs_the_plan(written: Any, kept: list[str]) -> None:
    llm = FakeLLM(reply({**sub("one"), "context": written}))
    planned = plan(llm, "goal", tree=TREE)
    assert len(llm.calls) == 1 and planned.source == "llm"
    assert planned.spec.subtasks[0].knowledge == kept


def test_without_a_tree_the_planner_is_not_told_and_names_nothing() -> None:
    llm = FakeLLM(reply(sub("one", "DESMAN/Manual/tasks")))
    planned = plan(llm, "goal")
    assert KNOWLEDGE_RULES not in llm.calls[0].messages[0].content
    assert planned.spec.subtasks[0].knowledge == []
    assert "context_unknown" not in planned.details()


def test_a_continuation_names_nodes_too() -> None:
    llm = FakeLLM(reply(sub("add a section", "DESMAN/Manual/tasks")))
    planned = continuation(llm, "goal", [(0, "draft", "done")], "add more", tree=TREE)
    assert KNOWLEDGE_RULES in llm.calls[0].messages[0].content
    assert planned.spec.subtasks[0].knowledge == ["DESMAN/Manual/tasks"]


# ── The store ────────────────────────────────────────────────────────────────


def spec(*knowledge_per_subtask: list[str]) -> TaskSpec:
    return TaskSpec(
        goal="g",
        subtasks=[
            SubtaskSpec(description=f"s{i}", acceptance_criteria="ok", knowledge=paths)
            for i, paths in enumerate(knowledge_per_subtask)
        ],
    )


def chosen(store: Store, task_id: str) -> list[tuple[str, list[str]]]:
    rows = store.conn.execute(
        "SELECT subtask_id, details_json FROM audit_log"
        " WHERE task_id = ? AND action = 'context_chosen' ORDER BY seq",
        (task_id,),
    ).fetchall()
    return [(r[0], json.loads(r[1])["paths"]) for r in rows]


def test_the_choice_is_kept_with_the_subtask_and_audited(store: Store) -> None:
    task_id = store.create_task(spec(["DESMAN/Manual/tasks"], []))
    first, second = store.list_subtasks(task_id)
    assert first.knowledge == ["DESMAN/Manual/tasks"] and second.knowledge == []
    assert chosen(store, task_id) == [(first.id, ["DESMAN/Manual/tasks"])]


def test_a_continuation_keeps_and_audits_its_choice(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(spec([]))
    Coordinator(store, ok, accept, sleep=clock.sleep).run(task_id)
    [base] = store.list_subtasks(task_id)
    [new] = store.extend(task_id, spec(["DESMAN/Docs/ADR/0009/"]).subtasks, [base.id])
    assert store.get_subtask(new).knowledge == ["DESMAN/Docs/ADR/0009/"]
    assert chosen(store, task_id) == [(new, ["DESMAN/Docs/ADR/0009/"])]


def test_the_parts_of_a_split_subtask_are_given_and_audited_as_it_was(store: Store) -> None:
    task_id = store.create_task(spec(["DESMAN/Manual/tasks"]))
    claimed = store.claim("w")
    assert claimed is not None and claimed.lease_token is not None
    parts = [SubtaskSpec(description=f"p{i}", acceptance_criteria="ok") for i in range(2)]
    made = store.decompose(claimed.id, claimed.lease_token, parts)
    assert [p.knowledge for p in made] == [["DESMAN/Manual/tasks"]] * 2
    assert chosen(store, task_id)[1:] == [(p.id, ["DESMAN/Manual/tasks"]) for p in made]


def test_a_job_from_desman_global_is_never_given_the_tree(store: Store) -> None:
    with pytest.raises(ValueError, match="never given the knowledge tree"):
        store.create_task(spec(["DESMAN/Manual/tasks"]), origin=Origin.NETWORK)


def test_a_database_before_k3_upgrades_with_no_knowledge(tmp_path: Path) -> None:
    path = tmp_path / "v44.db"
    old = db.connect(path)
    for v, script in enumerate(db.MIGRATIONS[:44], start=1):
        with db.transaction(old):
            for statement in db._split_sql(script):
                old.execute(statement)
            old.execute(f"PRAGMA user_version = {v}")
    old.execute(
        "INSERT INTO tasks (id, goal, priority, created_at) VALUES ('t', 'g', 50, '2026-11-09')"
    )
    old.execute(
        "INSERT INTO subtasks (id, task_id, seq, description, acceptance_criteria, inputs_json,"
        " priority, created_at, updated_at)"
        " VALUES ('s', 't', 0, 'old', 'ok', '{}', 50, '2026-11-09', '2026-11-09')"
    )
    old.close()
    conn = db.open_db(path)
    try:
        assert Store(conn).get_subtask("s").knowledge == []
    finally:
        conn.close()


# ── The executor's context ───────────────────────────────────────────────────


def ok(work: Work) -> Success:
    return Success(content={"done": work.subtask.description})


def accept(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
    return Review(verdict=Verdict.ACCEPT)


def test_knowledge_and_dependency_results_share_one_budget() -> None:
    long = KnowledgeText(path="a", text="x" * 4000)
    short = KnowledgeText(path="b", text="short")
    result = DependencyResult(subtask_id="d", description="dep", content={"r": "y" * 4000})
    deps, nodes = fit([result], [long, short], budget_tokens=500)
    assert nodes[1] == short  # small ones whole
    assert nodes[0].truncated and len(nodes[0].text) < 4000
    assert len(nodes[0].text) + len(nodes[1].text) <= 500 * 4 // 2  # material: half at most
    assert deps[0].truncated
    total = len(nodes[0].text) + len(nodes[1].text) + len(str(deps[0].content))
    assert total <= 500 * 4


def test_knowledge_never_crowds_out_the_results_it_leaves_room_for() -> None:
    result = DependencyResult(subtask_id="d", description="dep", content={"r": "y" * 100})
    deps, nodes = fit([result], [KnowledgeText(path="a", text="x" * 10_000)], budget_tokens=500)
    assert not deps[0].truncated and nodes[0].truncated


def test_the_executor_gets_the_texts_it_was_given(store: Store, clock: FakeClock) -> None:
    seen: list[list[KnowledgeText]] = []

    def executor(work: Work) -> Success:
        seen.append(work.context.knowledge)
        return ok(work)

    paths = ["DESMAN/Docs/ADR/0009/Decision", "gone/node", "DESMAN/Docs/ADR/0009/"]
    task_id = store.create_task(spec(paths))
    result = Coordinator(store, executor, accept, sleep=clock.sleep, knowledge=read_tree).run(
        task_id
    )
    assert result.status is TaskStatus.DONE
    [given] = seen
    assert [n.path for n in given] == ["DESMAN/Docs/ADR/0009/Decision", "DESMAN/Docs/ADR/0009/"]
    assert given[0].text == "Nodes have paths." and given[0].plan
    [missing] = store.conn.execute(
        "SELECT details_json FROM audit_log WHERE action = 'context_missing'"
    ).fetchall()
    assert json.loads(missing[0]) == {"paths": ["gone/node"]}  # a miss stays visible


def test_without_a_reader_nobody_is_given_anything(store: Store, clock: FakeClock) -> None:
    seen: list[list[KnowledgeText]] = []

    def executor(work: Work) -> Success:
        seen.append(work.context.knowledge)
        return ok(work)

    task_id = store.create_task(spec(["DESMAN/Manual/tasks"]))
    Coordinator(store, executor, accept, sleep=clock.sleep).run(task_id)
    assert seen == [[]]


def test_a_job_s_crew_is_never_given_the_tree(store: Store, clock: FakeClock) -> None:
    """Even a job stored with paths (it cannot be, but the coordinator does not rely on it)."""
    seen: list[list[KnowledgeText]] = []

    def executor(work: Work) -> Success:
        seen.append(work.context.knowledge)
        return ok(work)

    task_id = store.create_task(spec([]), origin=Origin.NETWORK)
    [job] = store.list_subtasks(task_id)
    store.conn.execute(
        "UPDATE subtasks SET knowledge_json = ? WHERE id = ?",
        (json.dumps(["DESMAN/Manual/tasks"]), job.id),
    )
    crew = Crew(executor, accept, None)
    Coordinator(store, ok, accept, sleep=clock.sleep, knowledge=read_tree, network=crew).run(
        task_id
    )
    assert seen == [[]]


def test_the_model_reads_the_nodes_with_their_marks(store: Store, tmp_path: Path) -> None:
    llm = FakeLLM('{"answer": 42}')
    task_id = store.create_task(spec(["DESMAN/Docs/ADR/0009/Decision"]))
    [subtask] = store.list_subtasks(task_id)
    node = read_tree("DESMAN/Docs/ADR/0009/Decision")
    assert node is not None
    context = build_context(subtask, [], None, knowledge=[node])
    no_tools = Toolbox(Workspace(tmp_path, task_id), Registry())
    work = Work(subtask=subtask, context=context, heartbeat=lambda _: None, tools=no_tools)
    assert llm_executor(llm)(work) == Success(content={"answer": 42})
    [call] = llm.calls
    assert KNOWLEDGE_INSTRUCTIONS in call.messages[0].content
    payload = json.loads(call.prompt.split("\n", 1)[1])
    assert payload["knowledge"] == [
        {"path": "DESMAN/Docs/ADR/0009/Decision", "text": "Nodes have paths.", "plan": True}
    ]


def test_the_reviewer_s_evidence_holds_the_nodes(store: Store) -> None:
    task_id = store.create_task(spec(["DESMAN/Manual/tasks", "gone/node"]))
    [subtask] = store.list_subtasks(task_id)
    shown = evidence(store, subtask, knowledge=read_tree)
    assert shown["knowledge"] == [
        {"path": "DESMAN/Manual/tasks", "text": "Tasks are planned by a planner."}
    ]
    assert "knowledge" not in evidence(store, subtask)


# ── read_node ────────────────────────────────────────────────────────────────


def call(tool_knowledge: Knowledge, path: str, tmp_path: Path) -> dict[str, Any]:
    tool = read_node_tool(tool_knowledge)
    result = tool.run(tool.args(path=path), Workspace(tmp_path, "t"))
    assert isinstance(result, dict)
    return result


def test_read_node_reads_nodes_and_sections_and_marks_plans(tmp_path: Path) -> None:
    assert call(read_tree, "DESMAN/Docs/ADR/0009/Decision", tmp_path) == {
        "path": "DESMAN/Docs/ADR/0009/Decision",
        "text": "Nodes have paths.",
        "plan": True,
    }
    whole = call(read_tree, "", tmp_path)
    assert whole["path"] == "/" and "DESMAN/Manual/tasks" in whole["text"]
    untrusted = call(lambda p: KnowledgeText(path=p, text="t", untrusted=True), "x", tmp_path)
    assert untrusted["untrusted"]


def test_read_node_tells_a_wrong_path_how_to_find_the_right_one(tmp_path: Path) -> None:
    registry = builtin_registry(knowledge=read_tree)
    tool = registry.get("read_node")
    assert tool is not None and tool.repeatable
    box = Toolbox(Workspace(tmp_path, "t"), registry)
    result = box.call(ToolCall(tool="read_node", args={"path": "nowhere"}))
    assert not result.ok and "table of contents" in (result.error or "")
    assert box.call(ToolCall(tool="read_node", args={"path": "tasks"})).ok


# ── status ───────────────────────────────────────────────────────────────────


def test_status_shows_what_each_subtask_was_given(
    store: Store, capsys: pytest.CaptureFixture[str]
) -> None:
    task_id = store.create_task(spec(["DESMAN/Manual/tasks"], []))
    report = task_report(store, task_id)
    assert [s.knowledge for s in report.subtasks] == [["DESMAN/Manual/tasks"], []]
    print_report(report)
    out = capsys.readouterr().out
    assert out.count("knowledge: DESMAN/Manual/tasks") == 1


def test_without_a_tree_read_node_finds_nothing(tmp_path: Path) -> None:
    box = Toolbox(Workspace(tmp_path, "t"), builtin_registry())
    assert not box.call(ToolCall(tool="read_node", args={"path": "tasks"})).ok


def test_a_spec_file_names_only_nodes_of_the_tree(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from desman.cli import cli

    def run(paths: list[str]) -> Any:
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(
            json.dumps(
                {
                    "goal": "g",
                    "subtasks": [
                        {"description": "d", "acceptance_criteria": "ok", "knowledge": paths}
                    ],
                }
            )
        )
        return CliRunner().invoke(
            cli, ["run", "--spec", str(spec_path), "--db", str(tmp_path / "d.db"), "--json"]
        )

    for wrong in (["nowhere"], [""], ["/"]):
        result = run(wrong)
        assert result.exit_code != 0 and "no such node of the knowledge tree" in result.output
