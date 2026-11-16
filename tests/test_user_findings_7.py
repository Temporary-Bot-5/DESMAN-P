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
    assert (
        actions.result_text(result)
        == prose + "\nscript: print('hi')\nprice: 20\nsource: https://example.org"
    )
    assert "files_written" in result


@pytest.mark.parametrize(
    "source", [r"$0^\circ$C", r"$0^{\circ}$C", r"$0\degree$C", r"$0\textdegree$C", r"\(0^\circ\)C"]
)
def test_degrees(source: str) -> None:
    assert actions.result_text({"text": "около " + source}) == "около 0°C"
    assert actions.result_text("0 °°C") == "0 °C"
    assert actions.result_text("$5 and $10") == "$5 and $10"


@pytest.mark.parametrize("slow_count", [2, 3])
@pytest.mark.parametrize("continuation", [False, True])
def test_planning_slow_model_moves_to_untried(
    clock: FakeClock, continuation: bool, slow_count: int
) -> None:
    slow = [
        backed(FakeLLM(Timeout(after=60), name=f"slow-{i}", sleep=clock.sleep), clock)
        for i in range(slow_count)
    ]
    second = FakeLLM(reply(sub("answer")), name="next")
    llm = Fallback(members(*slow, second))
    result = (
        planner.continuation(llm, "goal", [], "change")
        if continuation
        else planner.plan(llm, "goal", provider_fallback=False)
    )
    assert result.source == "llm"
    assert len(second.calls) == 1


def test_planning_through_scoped_moves_to_untried(clock: FakeClock) -> None:
    """runs.Scoped (the planner's model in a real run) delegates to a Fallback: the planner
    must see that Fallback's untried models."""
    from desman.runs import Scoped

    slow = backed(FakeLLM(Timeout(after=60), name="slow", sleep=clock.sleep), clock)
    second = FakeLLM(reply(sub("answer")), name="next")
    inner = Fallback(members(slow, second))
    result = planner.plan(Scoped(lambda: inner, inner), "goal", provider_fallback=False)
    assert result.source == "llm"
    assert len(second.calls) == 1


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("continuation", [False, True])
def test_planning_slow_retry_is_bounded(
    clock: FakeClock, continuation: bool, fallback: bool
) -> None:
    llm = backed(FakeLLM(*[Timeout(after=60)] * 4, sleep=clock.sleep), clock)
    provider = Fallback(members(llm)) if fallback else llm
    result = (
        planner.continuation(provider, "goal", [], "change")
        if continuation
        else planner.plan(provider, "goal")
    )
    assert result.source == "template"
    assert isinstance(llm.inner, FakeLLM)
    assert len(llm.inner.calls) == 1


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


@pytest.mark.parametrize("can_resume", [False, True])
@pytest.mark.parametrize("decision", ["retry", "accept"])
def test_decision_hint_depends_on_resume_result(
    store: Store, can_resume: bool, decision: str, capsys: pytest.CaptureFixture[str]
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

    def resume(task_id: str) -> bool:
        resumed.append(task_id)
        return can_resume

    ctx = commands.Context(store, open_config(), lambda *_: None, resume)
    entry = commands.find(decision)
    assert entry is not None
    proposal = entry.handler(
        ctx, commands.DecisionArgs(subtask=claimed.id, note="Tomorrow", if_version=None)
    )
    assert isinstance(proposal, commands.Proposal)
    assert resumed == []
    proposal.apply()
    assert resumed == [task]
    assert store.task_status(task) is TaskStatus.ACTIVE
    assert ("Continue with: desman resume" in capsys.readouterr().out) is (not can_resume)


@pytest.mark.parametrize(
    "source", ["description", "acceptance_criteria", "inputs", "goal", "evidence"]
)
@pytest.mark.parametrize("invented", [False, True])
def test_review_preserves_supplied_mixed_words(store: Store, source: str, invented: bool) -> None:
    name = "Яndex"
    spec = SubtaskSpec(description="Answer", acceptance_criteria="ok")
    if source in {"description", "acceptance_criteria"}:
        spec = spec.model_copy(update={source: name})
    elif source == "inputs":
        spec = spec.model_copy(update={"inputs": {"brand": name}})
    task = store.create_task(TaskSpec(goal="g", subtasks=[spec]))
    subtask = store.list_subtasks(task)[0]
    supplied = (
        {"goal": name} if source == "goal" else {"quote": name} if source == "evidence" else {}
    )
    llm = FakeLLM('{"verdict":"accept"}')
    seen: list[str] = []

    def evidence(subtask: object) -> dict[str, str]:
        seen.append("read")
        return supplied

    review = agents.llm_reviewer(llm, evidence)(
        subtask, {"text": name + (" inventedЖ" if invented else "")}
    )
    assert seen == ["read"]
    assert isinstance(review, Review)
    assert review.verdict is (Verdict.RETRY if invented else Verdict.ACCEPT)
    assert len(llm.calls) == (0 if invented else 1)


@pytest.mark.parametrize("key", ["files", "files_written", "written_files"])
@pytest.mark.parametrize("value", [{"answer.txt": "contents"}, [{"path": "a", "content": "b"}]])
def test_file_contents_remain_visible(key: str, value: object) -> None:
    assert actions._result_extras({"text": "Answer", key: value}, "text") == {key: value}


@pytest.mark.parametrize("value", ["USD", "ok", "", "nineteen characters", "twenty characters!!!"])
def test_repeated_extra_length(value: str) -> None:
    extras = actions._result_extras({"text": "Answer " + value, "detail": value}, "text")
    assert extras == ({} if len(value) >= 20 else {"detail": value})


def test_named_request_limits_keep_unsuffixed_headers() -> None:
    headers = {
        "x-ratelimit-limit": "50",
        "x-ratelimit-reset": "123",
        "X-RateLimit-Remaining-Requests": "7",
    }
    found, other = calls.limits(headers)
    assert found == [calls.Limit("requests", "7", None, None)]
    assert other == {"x-ratelimit-limit": "50", "x-ratelimit-reset": "123"}


def test_usage_json_preserves_raw_reset(clock: FakeClock) -> None:
    from desman.config import Config
    from desman.config_cli import usage_json, usage_view

    reset = str(int((clock() + timedelta(hours=7)).timestamp() * 1000))
    headers = {"x-ratelimit-reset": reset}
    row = calls.KeyUsage("openrouter", None, 1, 0, 0, 1, 1, None, None, headers)
    assert usage_json(Config(), [row])[0]["limits"] == headers
    assert usage_view(Config(), [row], None)["limits"][0]["reset"] == reset
    assert _usage_reset(reset, clock()) == "7h"
