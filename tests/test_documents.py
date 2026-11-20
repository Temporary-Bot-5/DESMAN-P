"""L.2: long text reaches workers and reviewers as exact, bounded document slices."""

from __future__ import annotations

import json
import random
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman import documents, runs, task_files
from desman.cli import cli
from desman.config import Config, open_config
from desman.core import db
from desman.core.audit import history
from desman.core.models import Kind, SubtaskSpec, TaskSpec, Tools
from desman.core.policy import Access, Permission, Policy, Role, RolePolicy, executor_access
from desman.core.store import Store
from desman.core.tools import ToolCall
from desman.core.web import Web, WebCache
from desman.core.workspace import Workspace, workspace_root_for
from desman.llm.base import Request
from desman.llm.fake import FakeLLM
from desman.planner import PLAN
from helpers import make_config
from test_cli_tasks import desman, only_task, script
from test_given_files import install_chat_models
from test_tools import tool_calls
from test_web import Site, fetcher, toolbox

QUOTE = "На берегу стоял дом."
LONG_URL = "https://example.org/long"
SHORT_URL = "https://example.org/short"
BAD_URL = "https://example.org/missing"


def chapters(size: int = documents.PART_CHARS) -> str:
    sections = []
    for name in ("I", "II", "III", "IV", "V"):
        start = f"{name}\n\n{QUOTE} "
        sections.append(start + "я" * (size - len(start) - 2) + "\n\n")
    return "".join(sections)


def test_split_invariants() -> None:
    rng = random.Random(914)
    chunks = ["", "текст", "\n\n", "!", "… ", "\r\n", "\n# Heading\n", "\nГлава III\n", "x" * 90]
    corpus = ["", "   ", "n" * 1000, "Раз. Два! Три?", chapters(), chapters().replace("\n", "\r\n")]
    corpus.extend("".join(rng.choices(chunks, k=rng.randrange(100))) for _ in range(200))
    for text in corpus:
        for target in (1, 13, 51, 150, documents.PART_CHARS):
            parts = documents.split(text, target)
            assert "".join(text[part.start : part.end] for part in parts) == text
            ends = [0, *(part.end for part in parts)]
            assert [part.start for part in parts] == ends[:-1]
            assert all(0 < part.end - part.start <= target for part in parts)
            if 0 < len(text) <= target:
                assert len(parts) == 1


@pytest.mark.parametrize(
    "heading",
    ["# Глава 3", "Глава 3", "ГЛАВА III", "Chapter 12: The End", "Part II", "Часть 2", "РАЗДЕЛ"],
)
def test_section_title_carries_through_paragraph_sentence_and_hard_cuts(heading: str) -> None:
    first = "До главы идёт предисловие.\r\n\r\n"
    section = heading + "\r\n\r\nАбзац.\r\n\r\n" + "Длинное предложение! " + "ы" * 36
    text = first + section
    parts = documents.split(text, 33)
    assert parts[0] == documents.Part(0, len(first), None)
    assert parts[1].start == len(first)
    assert all(part.title == heading.removeprefix("# ") for part in parts[1:])
    assert text[parts[1].start : parts[1].end].endswith("Абзац.\r\n\r\n")
    assert text[parts[2].start : parts[2].end].endswith("Длинное предложение!")


def test_small_russian_chapters_begin_at_their_heading() -> None:
    sections = [f"{title}\r\n\r\nСнег тает.\r\n\r\n" for title in ("I", "II", "III.")]
    parts = documents.split("".join(sections), 35)
    assert parts == [
        documents.Part(0, len(sections[0]), "I"),
        documents.Part(len(sections[0]), len(sections[0] + sections[1]), "II"),
        documents.Part(len(sections[0] + sections[1]), sum(map(len, sections)), "III."),
    ]


def plan_spec() -> TaskSpec:
    return TaskSpec(
        goal="Summarise the setting",
        priority=40,
        tools=Tools.NONE,
        subtasks=[
            SubtaskSpec(description="Organise", acceptance_criteria="Complete"),
            SubtaskSpec(description="Answer", acceptance_criteria="Clear", depends_on=[0]),
        ],
    )


@pytest.mark.parametrize("size", [300_000, 2_000_000])
def test_large_documents_use_at_most_the_part_limit(size: int) -> None:
    line = "Глава 3\n\n" + "Текст. " * 600 + "\n\n"
    text = (line * (size // len(line) + 1))[:size]
    spec = documents.in_parts(plan_spec(), [documents.Document("input/big.txt", text)])
    parts = [sub.inputs["text"] for sub in spec.subtasks if "text" in sub.inputs]
    assert "".join(parts) == text
    assert len(parts) <= documents.MAX_PARTS
    if size == 300_000:
        assert max(map(len, parts)) <= documents.PART_CHARS
    else:
        assert len(parts) == documents.MAX_PARTS
        assert max(map(len, parts)) == 31_250


@pytest.mark.parametrize(("part_count", "combines"), [(3, 2), (9, 3), (11, 4)])
def test_document_digests_precede_and_feed_the_planners_graph(
    part_count: int, combines: int
) -> None:
    base = plan_spec()
    book = documents.Document("input/book.txt", "а" * (part_count * documents.PART_CHARS))
    extra = documents.Document("input/extra.txt", "b" * 12_001)
    short = documents.Document("input/brief.txt", "b" * 12_000)
    assert documents.in_parts(base, [short]) is base
    assert documents.goal_note([short]) == ""
    spec = documents.in_parts(base, [book, short, extra])
    assert TaskSpec.model_validate(spec.model_dump()) == spec
    assert (spec.goal, spec.tools, spec.retry, spec.priority) == (
        base.goal,
        base.tools,
        base.retry,
        base.priority,
    )
    indices = spec.subtasks[-2].depends_on
    assert len(indices) == 2
    assert spec.subtasks[-1].depends_on == [len(spec.subtasks) - 2]
    assert spec.subtasks[-1].model_dump(exclude={"depends_on", "inputs"}) == base.subtasks[
        -1
    ].model_dump(exclude={"depends_on", "inputs"})
    assert spec.subtasks[-1].inputs == {
        **base.subtasks[-1].inputs,
        documents.READ_IN_PARTS: ["input/book.txt", "input/extra.txt"],
    }

    def covered(index: int) -> list[SubtaskSpec]:
        sub = spec.subtasks[index]
        assert sub.kind is Kind.ANALYSIS
        if not sub.depends_on:
            assert "digest" not in sub.inputs
            return [sub]
        if index in indices:
            assert sub.inputs["digest"] is True
            assert "no more than about 12000 characters" in sub.description
        else:
            assert "digest" not in sub.inputs
            assert "no more than about 8000 characters" in sub.description
        assert 2 <= len(sub.depends_on) <= documents.GROUP
        assert sub.depends_on == sorted(sub.depends_on)
        return [part for dep in sub.depends_on for part in covered(dep)]

    for doc, index, count in zip((book, extra), indices, (part_count, 3), strict=True):
        parts = covered(index)
        assert len(parts) == count
        assert all("no more than about 2500 characters" in sub.description for sub in parts)
        assert "".join(sub.inputs["text"] for sub in parts) == doc.text
        assert [sub.inputs["part"] for sub in parts] == list(range(1, count + 1))
        assert all(
            sub.inputs["document"] == doc.path and sub.inputs["of"] == count for sub in parts
        )
    assert sum("Combine" in sub.description for sub in spec.subtasks) == combines


def test_urls_keep_order_strip_punctuation_and_limit_duplicates() -> None:
    assert documents.urls(
        f'Use ({LONG_URL}), «{SHORT_URL}» and "{LONG_URL}". Try {BAD_URL}; http://other.org'
    ) == [LONG_URL, SHORT_URL, BAD_URL]


def text_page(text: str) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/plain; charset=utf-8"}, text=text)


def test_prefetch_records_outcomes_and_respects_permission() -> None:
    text = chapters()
    goal = f"Read {LONG_URL}, {SHORT_URL}, and {BAD_URL}."
    site = Site({LONG_URL: text_page(text), SHORT_URL: text_page("Short.")})
    web = Web(fetcher=fetcher(site))
    assert documents.prefetch(goal, web, allowed=False) == documents.Prefetched([], [])
    assert not site.requests
    result = documents.prefetch(goal, web, allowed=True)
    assert result.details == [
        {"url": LONG_URL, "chars": len(text)},
        {"url": SHORT_URL, "chars": 6},
        {"url": BAD_URL, "error": "HTTP 404"},
    ]
    [file] = result.given
    assert file.name == "input/web-example.org.txt"
    assert file.text == f"Source: {LONG_URL}\n\n{text}"
    assert len(site.requests) == 3


def test_prefetch_checks_the_role_policy_as_well_as_the_switch(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*args: object) -> None:
        pytest.fail("created a reader without executor access")

    monkeypatch.setattr(runs, "web", unexpected)
    denied = executor_access()
    no_fetch_policy = Policy(roles={Role.EXECUTOR: RolePolicy(tools=["web_search"])})
    only_search = Access(no_fetch_policy, Role.EXECUTOR, [Permission.NETWORK])
    for access in (denied, only_search):
        assert runs.document_web(f"Read {LONG_URL}", Config(), store, access) is None


def test_prefetched_file_has_its_own_byte_limit_and_does_not_count_as_an_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(task_files, "MAX_GIVEN_BYTES", 29_000)
    site = Site({LONG_URL: text_page("ю" * 28_000)})
    fetched = documents.prefetch(f"Read {LONG_URL}", Web(fetcher=fetcher(site)), allowed=True)
    [file] = fetched.given
    assert len(file.text.encode()) <= task_files.MAX_GIVEN_BYTES
    assert not file.text.endswith("�")
    given = [task_files.given_upload("web-example.org.txt", b"x")] * task_files.MAX_GIVEN_FILES
    expected_paths = [doc.path for doc in documents.attached(given + fetched.given)]
    paths = task_files.place(tmp_path, "t", given, prefetched=fetched.given)
    assert paths == expected_paths
    assert paths[-1] == "input/web-example.org-11.txt"
    assert Workspace(tmp_path, "t").read_text(paths[-1]) == file.text


def payload(request: Request) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(request.prompt.split("\n", 1)[1])
    return data


def test_cli_reads_a_long_file_in_parts_and_reviews_quotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = tmp_path / "setting.txt"
    text = chapters()
    file.write_bytes(text.encode())
    seen_parts: dict[int, str] = {}
    reviewed_parts: set[int] = set()
    combine_attempts = 0
    invented = "Вдали стоял высокий замок."
    brief = tmp_path / "brief.txt"
    brief_quote = "На тропинке лежал камень."
    brief.write_bytes(brief_quote.encode())
    goal = "Describe the setting"
    digest = "Notes. " * 2271 + f'End: "{QUOTE}"'
    assert len(digest) > 15_900

    def execute(request: Request) -> str:
        nonlocal combine_attempts
        data = payload(request)
        inputs = data["inputs"]
        if "text" in inputs:
            assert goal in data["task_goal"]
            start, end = inputs["chars"]
            assert inputs["text"] == text[start:end] and QUOTE in inputs["text"]
            seen_parts[inputs["part"]] = inputs["text"]
            # A part is offered no tools: its text is in the inputs (L.2 live run).
            assert "tools" not in data
            return json.dumps({"text": f'Part {inputs["part"]}: "{QUOTE}"'})
        if data["subtask"].startswith("Combine"):
            assert goal in data["task_goal"]
            assert len(data["results_of_dependencies"]) == 5
            combine_attempts += 1
            if combine_attempts == 1:
                return json.dumps({"text": f'Invented: "{invented}"'})
            return json.dumps({"text": digest})
        [dep] = data["results_of_dependencies"]
        assert dep["result"] == {"text": digest}
        assert dep["digest"] is True and not dep["truncated"]
        assert "input/setting.txt were read in parts by code" in data["documents_read"]
        return tool_calls(("read_file", {"path": "input/brief.txt"}))

    def review(request: Request) -> str:
        data = payload(request)
        inputs = data["evidence"]["inputs"] or {}
        rule = "found word for word there means retry"
        if "text" in inputs:
            assert rule in request.messages[0].content
            quote = data["result"]["text"].split('"')[1]
            assert quote == QUOTE and quote in inputs["text"]
            assert inputs["text"] == seen_parts[inputs["part"]]
            assert goal in data["evidence"]["task_goal"]
            reviewed_parts.add(inputs["part"])
        else:
            assert rule not in request.messages[0].content
        assert invented not in data["result"]["text"]
        if data["subtask"] == "Write the answer":
            [dep] = data["evidence"]["results_of_dependencies"]
            assert dep["result"] == {"text": digest} and dep["digest"] is True
            assert "against the relevant dependency notes" in request.messages[0].content
            assert brief_quote in data["result"]["text"]
        return '{"verdict":"accept"}'

    planned = {
        "subtasks": [
            {"description": "Write the answer", "acceptance_criteria": "Complete"},
        ]
    }
    fake = FakeLLM().on(PLAN, json.dumps(planned))

    def after_read(request: Request) -> str:
        if brief_quote in request.prompt:
            return json.dumps({"text": f'The riverbank: "{QUOTE}" The path: "{brief_quote}"'})
        return json.dumps({"text": f'Part 1: "{QUOTE}"'})

    fake.on("Execute subtask (tool results)", after_read, repeat_last=True)
    fake.on("Execute", execute, repeat_last=True).on("Review", review, repeat_last=True)
    monkeypatch.setattr(FakeLLM, "from_file", lambda *args, **kwargs: fake)
    result = desman(
        "run",
        goal,
        "--file",
        file,
        "--file",
        brief,
        "--plan",
        "llm",
        "--fake-llm",
        script(tmp_path, []),
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert combine_attempts == 2
    assert reviewed_parts == set(seen_parts) == set(range(1, 6))
    assert "".join(seen_parts[i] for i in range(1, 6)) == text
    with closing(db.open_db()) as conn:
        entries = history(conn, only_task())
        [created] = [entry for entry in entries if entry.action == "task_created"]
        [rejected] = [
            entry
            for entry in entries
            if entry.action == "verdict" and entry.details["verdict"] == "retry"
        ]
        assert f"«{invented}»" in rejected.details["feedback"]
        assert created.details["documents"] == [
            {"path": "input/setting.txt", "chars": len(text), "parts": 5}
        ]
    [request] = [call for call in fake.calls if call.prompt.startswith(PLAN)]
    assert "is read in parts by code" in request.prompt and QUOTE not in request.prompt
    assert "never a subtask to read the file" in request.prompt


def test_chat_task_splits_its_long_attachment(
    tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_chat_models(clock, monkeypatch)
    path = tmp_path / "setting.txt"
    text = chapters()
    path.write_bytes(text.encode())
    result = CliRunner().invoke(cli, [], input=f'/file "{path}"\nRead it\n/exit\n')
    assert result.exit_code == 0, (result.output, result.exception)
    with closing(db.open_db()) as conn:
        task = only_task()
        subtasks = Store(conn).list_subtasks(task)
        assert "".join(sub.inputs["text"] for sub in subtasks if "text" in sub.inputs) == text
        assert subtasks[-1].description == "Read the attachment"
        assert Workspace(workspace_root_for(conn), task).read_text("input/setting.txt") == text


def test_cli_prefetch_saves_only_the_long_page_and_audits_all_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = Site({LONG_URL: text_page(chapters()), SHORT_URL: text_page("Short.")})
    config = make_config(language="en", permissions={"network_access": True})
    open_config().save(config)

    def web(config: Config, store: Store, http: httpx.Client | None = None) -> Web:
        return Web(fetcher=fetcher(site), cache=WebCache(store.conn, store.clock))

    monkeypatch.setattr(runs, "web", web)
    fake = script(tmp_path, ['{"text":"Notes."}'] * 16)
    result = desman("run", f"Read {LONG_URL} {SHORT_URL} {BAD_URL}", "--fake-llm", fake)
    assert result.exit_code == 0, (result.output, result.exception)
    assert result.output.count("Saved the page ") == 1 and LONG_URL in result.output
    with closing(db.open_db()) as conn:
        task = only_task()
        subtasks = Store(conn).list_subtasks(task)
        workspace = Workspace(workspace_root_for(conn), task)
        saved = workspace.read_text("input/web-example.org.txt")
        assert saved == f"Source: {LONG_URL}\n\n{chapters()}"
        assert "".join(sub.inputs["text"] for sub in subtasks if "text" in sub.inputs) == saved
        [created] = [entry for entry in history(conn, task) if entry.action == "task_created"]
        assert created.details["prefetch"] == [
            {"url": LONG_URL, "chars": len(chapters())},
            {"url": SHORT_URL, "chars": 6},
            {"url": BAD_URL, "error": "HTTP 404"},
        ]
        assert created.details["documents"][0]["path"] == "input/web-example.org.txt"
        assert list(workspace.files()) == ["input/web-example.org.txt"]
        assert not task_files.listing(workspace_root_for(conn), task)
    assert len(site.requests) == 3


def test_web_fetch_saves_a_long_page_once_and_keeps_short_results_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = chapters()
    site = Site({LONG_URL: text_page(text), SHORT_URL: text_page("Short.")})
    box = toolbox(tmp_path, Web(fetcher=fetcher(site)), Permission.NETWORK)
    call = ToolCall(tool="web_fetch", args={"url": LONG_URL})
    first = box.call(call)
    assert first.ok and "read_file" in first.output["hint"]
    saved = first.output["saved"]
    assert saved.startswith("web/example.org-")
    assert box.workspace.read_text(saved) == text

    def fail(*args: object) -> str:
        pytest.fail("rewrote an identical saved page")

    monkeypatch.setattr(Workspace, "write", fail)
    assert box.call(call).output["saved"] == saved
    short = box.call(ToolCall(tool="web_fetch", args={"url": SHORT_URL}))
    assert short.ok and "saved" not in short.output


def test_web_fetch_still_returns_text_when_the_workspace_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = chapters()
    box = toolbox(
        tmp_path, Web(fetcher=fetcher(Site({LONG_URL: text_page(text)}))), Permission.NETWORK
    )

    def fail(*args: object) -> str:
        raise OSError("cannot save in this workspace")

    monkeypatch.setattr(Workspace, "write", fail)
    result = box.call(ToolCall(tool="web_fetch", args={"url": LONG_URL}))
    assert result.ok and text.startswith(result.output["content"])
    assert "saved" not in result.output


def test_parts_are_never_a_bare_heading_when_paragraphs_are_single_line_breaks() -> None:
    # Found live (Chekhov, lib.ru): paragraphs are "\n   -- …" with no blank line, so the only
    # blank line of a chapter follows its heading; parts of three characters were made.
    body = "".join(f"   -- Реплика номер {n}, и ещё немного слов.\n" for n in range(60))
    text = "".join(f"{title}\n\n{body}\n" for title in ("I", "II", "III"))
    parts = documents.split(text, 1000)
    assert all(part.end - part.start >= 250 for part in parts[:-1])
    assert all(text[part.end - 1] == "\n" for part in parts[:-1])
    assert [p.start for p in parts if text[p.start : p.start + 3] in ("I\n\n", "II\n", "III")]


def test_a_short_shouted_reply_is_not_a_heading() -> None:
    text = "Начало.\n\n-- А!\n\nПродолжение.\n"
    assert documents._headings(text) == ([], [])


def test_the_planner_is_not_told_to_read_a_long_document_with_read_file() -> None:
    # Found live (L.2): "Read them with read_file" next to "never a subtask to read the file";
    # the planner planned "Read the input file …" and nothing that answered the goal.
    book = task_files.Given("book.txt", "а" * 40_000, 40_000, "book.txt")
    brief = task_files.Given("brief.txt", "b" * 100, 100, "brief.txt")
    note = documents.files_note([book, brief], documents.attached([book, brief]))
    assert "read in parts by code" in note
    assert "brief.txt" in note.split("read in parts")[0]
    assert "book.txt" not in note.split("read in parts")[0].split("Read them with")[0]
    only_long = documents.files_note([book], documents.attached([book]))
    assert "read_file" not in only_long


def test_a_part_names_the_headings_that_start_inside_it() -> None:
    # Found live (L.2): notes keyed "Part 3" became "Chapter 3" in the answer; chapter VI, which
    # began inside a part, was never named. Code tells each part which headings it holds.
    chapter = "Абзац текста главы, и ещё немного слов.\n" * 40
    text = "".join(f"{title}\n\n{chapter}\n" for title in ("I", "II", "III", "IV"))
    spec = documents.in_parts(plan_spec(), [documents.Document("input/a.txt", text * 6)])
    parts = [sub for sub in spec.subtasks if "part" in sub.inputs]
    assert all("Never number parts as chapters" in sub.description for sub in parts)
    named = [
        h for sub in parts for h in [sub.inputs.get("section"), *sub.inputs.get("headings", [])]
    ]
    assert {"I", "II", "III", "IV"} <= set(named)
    for sub in parts:
        start, end = sub.inputs["chars"]
        for heading in sub.inputs.get("headings", []):
            assert f"\n{heading}\n" in (text * 6)[start:end]


@pytest.mark.parametrize(
    "line",
    [
        "Part in the story he left.",
        "Chapter discusses many things",
        "Part IIIrd",
        "Chapter 12th",
        "-- СТОЙТЕ",
        "- СТОЙТЕ",
        "— СТОЙТЕ",
        "СТОЙТЕ!",
        "КТО ТАМ?",
        "КАК БЫ…",
    ],
)
def test_prose_and_shouted_dialogue_are_not_headings(line: str) -> None:
    assert documents._headings(f"Beginning.\n\n{line}\n\nEnd.") == ([], [])


def test_urls_keep_balanced_parentheses() -> None:
    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    assert documents.urls(f"Read ({url}).") == [url]
    assert documents.urls(f"Read {url}.") == [url]


@pytest.mark.parametrize(
    "verb",
    [
        "Please summarize",
        "Проанализируйте",
        "阅读",
        "要約",
        "요약",
        "पढ़ें",
        "Résume",
        "Analiza",
        "Resuma",
        "Übersetze",
        "Streść",
        "Özetle",
        "Підсумуй",
    ],
)
def test_reading_intent_recognizes_direct_requests(verb: str) -> None:
    assert documents.reading_urls(f"{verb} {LONG_URL}") == [LONG_URL]


def test_prefetch_ignores_urls_mentioned_without_reading_intent() -> None:
    site = Site({LONG_URL: text_page(chapters())})
    web = Web(fetcher=fetcher(site))
    for goal in (f"Save this link: {LONG_URL}", f"Do not read {LONG_URL}", LONG_URL):
        assert documents.prefetch(goal, web, allowed=True) == documents.Prefetched([], [])
    assert not site.requests


def test_saved_pages_are_omitted_from_task_outputs_and_handover(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, "source")
    workspace.write("input/web-example.org.txt", "prefetched source")
    workspace.write("web/example.org.txt", "fetched source")
    workspace.write("answer.md", "answer")
    assert [file.path for file in task_files.listing(tmp_path, "source")] == ["answer.md"]
    assert [item["path"] for item in task_files.for_retelling(tmp_path, "source")] == ["answer.md"]
    handover = task_files.handover(tmp_path, "source", "abcdef")
    assert handover.paths() == ["from-abcdef/answer.md"]
    assert task_files.hand_over(tmp_path, "next", handover) == handover.paths()
    assert list(Workspace(tmp_path, "next").files()) == handover.paths()


def test_every_planned_subtask_names_the_documents_it_works_from() -> None:
    book = documents.Document("input/book.txt", "а" * (3 * documents.PART_CHARS))
    short = documents.Document("input/brief.txt", "b" * 100)
    spec = documents.in_parts(plan_spec(), [book, short])
    planned = [sub for sub in spec.subtasks if "document" not in sub.inputs]
    assert len(planned) == len(plan_spec().subtasks)
    assert all(sub.inputs[documents.READ_IN_PARTS] == ["input/book.txt"] for sub in planned)
    assert all(
        documents.READ_IN_PARTS not in sub.inputs for sub in spec.subtasks if sub not in planned
    )
