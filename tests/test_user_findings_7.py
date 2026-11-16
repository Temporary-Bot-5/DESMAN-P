"""Seventh round of user findings: prose, review, planning and limits."""

from datetime import timedelta

import pytest

from conftest import FakeClock
from desman import actions, agents, planner
from desman.config_cli import _usage_reset
from desman.core import calls
from desman.core.audit import history
from desman.core.coordinator import Coordinator, Review, Verdict
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.store import Store
from desman.llm.fake import FakeLLM, Timeout
from desman.llm.router import Fallback
from test_planner import reply, sub
from test_routing import backed, members


@pytest.mark.parametrize("key", ["text", "summary", "reply"])
def test_result_drops_only_repeated_extras(key: str) -> None:
    prose = "The paragraph is here.\n```python\nprint('hi')\n```\n" * 3
    result = {
        key: prose,
        "paragraph": "The paragraph is here.",
        "script": "print('hi')",
        "files_written": ["main.py"],
        "written_files": ["a"],
        "files": ["b"],
        "type": "text",
        "price": 20,
        "source": "https://example.org",
    }
    assert actions.result_text(result) == prose + "\nprice: 20\nsource: https://example.org"
    assert "files_written" in result


@pytest.mark.parametrize(
    "source", [r"$0^\circ$C", r"$0^{\circ}$C", r"$0\degree$C", r"$0\textdegree$C", r"\(0^\circ\)C"]
)
def test_degrees(source: str) -> None:
    assert actions.result_text({"text": "около " + source}) == "около 0°C"
    assert actions.result_text("0 °°C") == "0 °C"
    assert actions.result_text("$5 and $10") == "$5 and $10"


@pytest.mark.parametrize("continuation", [False, True])
def test_planning_slow_model_moves_to_untried(clock: FakeClock, continuation: bool) -> None:
    slow = backed(FakeLLM(Timeout(after=60), name="slow", sleep=clock.sleep), clock)
    second = FakeLLM(reply(sub("answer")), name="next")
    llm = Fallback(members(slow, second))
    result = (
        planner.continuation(llm, "goal", [], "change")
        if continuation
        else planner.plan(llm, "goal", provider_fallback=False)
    )
    assert result.source == "llm"
    assert len(second.calls) == 1


@pytest.mark.parametrize("continuation", [False, True])
def test_planning_slow_retry_is_bounded(clock: FakeClock, continuation: bool) -> None:
    llm = backed(FakeLLM(*[Timeout(after=60)] * 4, sleep=clock.sleep), clock)
    result = (
        planner.continuation(llm, "goal", [], "change")
        if continuation
        else planner.plan(llm, "goal")
    )
    assert result.source == "template"
    assert isinstance(llm.inner, FakeLLM)
    assert len(llm.inner.calls) == 3


@pytest.mark.parametrize("text", ["Однопarentalные", "maternalным", "abcЖ", "Жabc", "éЖ"])
def test_mixed_script_retry_is_audited_without_model_call(store: Store, text: str) -> None:
    task = store.create_task(
        TaskSpec(
            goal="Ответ", subtasks=[SubtaskSpec(description="Ответ", acceptance_criteria="ok")]
        )
    )
    executor = FakeLLM('{"text": "' + text + '"}', '{"text":"Corrected"}')
    reviewer = FakeLLM()
    Coordinator(store, agents.llm_executor(executor), agents.llm_reviewer(reviewer)).tick(task, [])
    Coordinator(store, agents.llm_executor(executor), agents.llm_reviewer(reviewer)).tick(task, [])
    assert reviewer.calls == []
    entries = history(store.conn, task)
    assert any(
        e.details.get("verdict") == "retry" and text in e.details.get("feedback", "")
        for e in entries
    )


def test_mixed_script_exclusions_and_limit(store: Store) -> None:
    task = store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="ok")])
    )
    subtask = store.list_subtasks(task)[0]
    llm = FakeLLM('{"verdict":"accept"}')
    artifact = {
        "text": "iPhone-ы Wi-Fi A1Б www.example.org/abcЖ https://example.org/abcЖ `abcЖ`\n```python\nabcЖ\n```"
    }
    review = agents.llm_reviewer(llm)(subtask, artifact)
    assert isinstance(review, Review) and review.verdict is Verdict.ACCEPT
    assert len(llm.calls) == 1
    assert agents.mixed_script_words("aБ bВ cГ dД eЕ fЖ") == ["aБ", "bВ", "cГ", "dД", "eЕ"]


def test_unnamed_openrouter_limits(clock: FakeClock) -> None:
    reset = clock() + timedelta(hours=7, minutes=32)
    headers = {
        "x-ratelimit-limit": "50",
        "x-ratelimit-remaining": "0",
        "x-ratelimit-reset": str(int(reset.timestamp() * 1000)),
    }
    limits, other = calls.limits(headers)
    assert other == {}
    assert limits == [calls.Limit("requests", "0", "50", headers["x-ratelimit-reset"])]
    assert not limits[0].exhausted_for_good
    assert calls.limits({"x-ratelimit-limit": "0"})[0][0].exhausted_for_good
    assert _usage_reset(limits[0].reset, clock()) == "7h32m"
    assert _usage_reset(str(int(reset.timestamp())), clock()) == "7h32m"
    assert _usage_reset("2m", clock()) == "2m"


def test_prompt_rules() -> None:
    for prompt in (planner.PLANNER_SYSTEM, planner.SPLITTER_SYSTEM):
        assert "URL patterns or a count of sources" in prompt
        assert "unless the person’s goal names them" in prompt
    assert "equivalent reliable source" in agents.REVIEWER_SYSTEM
    assert "newest dated data" in agents.REVIEWER_EVIDENCE
    assert "older figure" in agents.REVIEWER_EVIDENCE
    assert "language of the goal" in agents.TOOLS_INSTRUCTIONS
    assert "retrieval time" not in agents.TOOLS_INSTRUCTIONS


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("decision", ["retry", "accept"])
def test_decision_continues_only_foreground(
    store: Store, background: bool, decision: str, capsys: pytest.CaptureFixture[str]
) -> None:
    from desman import commands
    from desman.config import open_config
    from desman.core.retry import FailureKind
    from desman.core.states import Actor, SubtaskStatus, TaskStatus

    task = store.create_task(
        TaskSpec(
            goal="goal",
            subtasks=[
                SubtaskSpec(description="first", acceptance_criteria="ok"),
                SubtaskSpec(description="next", acceptance_criteria="ok", depends_on=[0]),
            ],
        )
    )
    claimed = store.claim("worker", task_id=task)
    assert claimed is not None
    store.record_failure(
        claimed.id,
        SubtaskStatus.RUNNING,
        Actor.EXECUTOR,
        FailureKind.PERMANENT,
        lease=claimed.lease_token,
        reason="Which dates?",
    )
    resumed: list[str] = []
    ctx = commands.Context(
        store, open_config(), lambda *_: None, resumed.append, background_tasks=background
    )
    entry = commands.find(decision)
    assert entry is not None
    proposal = entry.handler(
        ctx, commands.DecisionArgs(subtask=claimed.id, note="Tomorrow", if_version=None)
    )
    assert isinstance(proposal, commands.Proposal)
    assert resumed == []
    proposal.apply()
    assert resumed == ([] if background else [task])
    assert store.task_status(task) is TaskStatus.ACTIVE
    assert "Continue with: desman resume" not in capsys.readouterr().out
