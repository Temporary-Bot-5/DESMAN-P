"""Human role preferences: safe key selectors, independent orders and wizard/chat reuse."""

from __future__ import annotations

import json
from typing import Any

import click
import httpx
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from conftest import FakeClock
from desman import commands, improve, providers, runs, tuning_cli, wizard
from desman.cli import cli
from desman.config import ApiKey, Config, KeyChoices, ModelChoices, Tuning, open_config
from desman.core import db, operator
from desman.core.calls import Role
from desman.core.models import Kind
from desman.llm import router
from desman.llm.base import Message, Request
from desman.llm.openai_compat import key_id
from helpers import make_config

FIRST = "first-secret-value-111111"
SECOND = "second-secret-value-222222"
REQUEST = Request(messages=[Message(role="user", content="hello")])


def config() -> Config:
    result = make_config(language="en", configured=True)
    result.api_keys["GROQ_API_KEY"] = [
        ApiKey(value=FIRST, label="main"),
        ApiKey(value=SECOND, label="backup"),
    ]
    return Config.model_validate(result.model_dump())


def response(text: str = "ok") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_v5_load_save_and_fingerprint_contract() -> None:
    manager = open_config()
    manager.config_path.parent.mkdir(parents=True, exist_ok=True)
    manager.config_path.write_text(json.dumps({"schema_version": 5, "language": "en"}))
    loaded = manager.load()
    assert loaded.schema_version == 7 and loaded.tuning == Tuning()
    manager.save(loaded)
    assert json.loads(manager.config_path.read_text())["schema_version"] == 7
    assert KeyChoices(planner=key_id(FIRST)).planner == key_id(FIRST)
    assert Tuning(keys=KeyChoices(planner=key_id(FIRST))).is_set()
    for value in (FIRST, "a" * 11, "A" * 12, "g" * 12, "a" * 13):
        with pytest.raises(ValidationError):
            KeyChoices(planner=value)
    with pytest.raises(ValidationError):
        KeyChoices.model_validate({"other": key_id(FIRST)})


@pytest.mark.parametrize("status", [401, 403, 429])
def test_chosen_key_first_then_other_key(status: int) -> None:
    cfg = config()
    seen = []

    def send(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return (
            httpx.Response(status, json={"error": {"message": "limited"}})
            if len(seen) == 1
            else response()
        )

    choice = router.lineup(cfg, Role.PLANNER)[0]
    api = router.connect(
        cfg,
        choice,
        first_key=key_id(SECOND),
        client=httpx.Client(transport=httpx.MockTransport(send)),
    )
    assert api.complete(REQUEST).text == "ok"
    assert seen == [f"Bearer {SECOND}", f"Bearer {FIRST}"]


def test_missing_or_other_provider_key_keeps_order() -> None:
    cfg = config()
    cfg.api_keys["MISTRAL_API_KEY"] = config().api_keys["GROQ_API_KEY"][:1]
    groq = router.lineup(cfg, Role.PLANNER, provider="groq")[0]
    mistral = router.lineup(cfg, Role.PLANNER, provider="mistral")[0]
    assert router.connect(cfg, groq, first_key="0" * 12).keys == [FIRST, SECOND]
    assert router.connect(cfg, mistral, first_key=key_id(SECOND)).keys == [FIRST]
    cfg.tuning.keys.planner = "0" * 12
    assert tuning_cli.role_rows(cfg)[0]["key_label"] == "missing"


def test_roles_on_same_model_use_their_own_order(clock: FakeClock) -> None:
    cfg = config()
    cfg.tuning.keys.planner = key_id(SECOND)
    cfg.tuning.keys.reviewer = key_id(FIRST)
    cfg.tuning.keys.code = key_id(SECOND)
    seen = []

    def send(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return response()

    conn = db.open_db(":memory:")
    crew = runs.models(
        cfg,
        conn,
        provider="groq",
        model=providers.find("groq").default_model,
        cache=False,
        client=httpx.Client(transport=httpx.MockTransport(send)),
        clock=clock,
        sleep=clock.sleep,
    )
    crew.for_(Role.PLANNER).complete(REQUEST)
    crew.for_(Role.REVIEWER).complete(REQUEST)
    crew.executor(Kind.CODE)[0].complete(REQUEST)
    assert seen == [f"Bearer {SECOND}", f"Bearer {FIRST}", f"Bearer {SECOND}"]
    # All role clients use the same provider/model rate bucket.
    assert conn.execute("SELECT count(*) FROM rate_buckets").fetchone()[0] == 1


def test_interlocutor_preference_reaches_actual_chat(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    cfg = config()
    chosen = router.candidates(cfg, "interlocutor")[-1]
    cfg.tuning.models.interlocutor = chosen
    cfg.tuning.keys.interlocutor = key_id(SECOND)
    open_config().save(cfg)
    seen = []

    def send(request: httpx.Request) -> httpx.Response:
        seen.append((json.loads(request.content)["model"], request.headers["authorization"]))
        return response(json.dumps({"type": "reply", "text": "Hello from chosen model."}))

    original = runs.models

    def models(config: Config, conn: Any, **kwargs: Any) -> runs.Models:
        kwargs["cache"] = False
        return original(
            config,
            conn,
            **kwargs,
            client=httpx.Client(transport=httpx.MockTransport(send)),
            clock=clock,
            sleep=clock.sleep,
        )

    monkeypatch.setattr(runs, "models", models)
    result = CliRunner().invoke(cli, [], input="hello\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "Hello from chosen model." in result.output
    assert seen[0] == (chosen.split("/", 1)[1], f"Bearer {SECOND}")


def test_cli_key_set_show_json_reset_errors_and_warning() -> None:
    cfg = config()
    open_config().save(cfg)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["tuning", "--key", "planner=groq:backup", "--model", "interlocutor=groq/test"]
    )
    assert result.exit_code == 0, result.output
    assert "backup" in result.output and FIRST not in result.output and SECOND not in result.output
    assert open_config().load().tuning.keys.planner == key_id(SECOND)
    shown = runner.invoke(cli, ["tuning"])
    assert "key: planner" in shown.output and "backup" in shown.output
    data = json.loads(runner.invoke(cli, ["tuning", "--json"]).output)
    assert data["keys"]["planner"] == key_id(SECOND)
    assert data["roles"][0]["key_label"] == f"backup {providers.mask(SECOND)}"
    assert FIRST not in json.dumps(data) and SECOND not in json.dumps(data)
    for selector in ("groq:2", "groq:#2"):
        assert runner.invoke(cli, ["tuning", "--key", f"planner={selector}"]).exit_code == 0
    warning = runner.invoke(cli, ["tuning", "--model", "planner=mistral/test"])
    assert "only applies to models from groq" in warning.output
    for selector in ("unknown:#1", "groq:absent", "groq:", "groq:²", f"groq:{SECOND}"):
        error = runner.invoke(cli, ["tuning", "--key", f"planner={selector}"])
        assert error.exit_code == 1 and "Available:" in error.output
        assert "backup" in error.output and SECOND not in error.output
    assert runner.invoke(cli, ["tuning", "--key", "wrong=auto"]).exit_code == 2
    runner.invoke(cli, ["tuning", "--key", "planner=auto"])
    assert open_config().load().tuning.keys.planner is None
    runner.invoke(cli, ["tuning", "--key", "code=groq:#2"])
    runner.invoke(cli, ["tuning", "--reset"])
    assert open_config().load().tuning == Tuning()


def test_removed_key_is_kept_but_shown_missing() -> None:
    cfg = config()
    cfg.tuning.keys.planner = key_id(SECOND)
    cfg.api_keys["GROQ_API_KEY"] = cfg.api_keys["GROQ_API_KEY"][:1]
    open_config().save(cfg)
    result = CliRunner().invoke(cli, ["tuning"])
    assert result.exit_code == 0 and "missing" in result.output
    assert open_config().load().tuning.keys.planner == key_id(SECOND)


def test_wizard_numbered_section_and_cancellation() -> None:
    open_config().save(config())
    runner = CliRunner()

    @click.command()
    def section() -> None:
        wizard.run_section(open_config(), "roles")

    result = runner.invoke(section, [], input="1\n2\n3\n8\n")
    assert result.exit_code == 0, result.output
    tuned = open_config().load().tuning
    assert tuned.models.planner is not None and tuned.keys.planner == key_id(SECOND)
    assert "key: planner" in result.output and "backup" in result.output
    assert FIRST not in result.output and SECOND not in result.output
    # The whole wizard exposes the section after Done, preserving existing provider numbers.
    role_number = len(providers.PROVIDERS + providers.SERVICES) + 2
    result = runner.invoke(cli, ["config"], input=f"1\n{role_number}\n8\n")
    assert "Models and keys for roles" in result.output
    assert open_config().load().tuning == tuned


def test_chat_key_window_and_operator_journal() -> None:
    open_config().save(config())
    result = CliRunner().invoke(cli, [], input="!tuning --key reviewer=groq:backup\n/exit\n")
    assert result.exit_code == 0, result.output
    assert open_config().load().tuning.keys.reviewer == key_id(SECOND)
    entries = operator.entries(db.open_db())
    assert any(
        e.action == "operator_tuning" and e.details == {"changed": ["keys.reviewer"], "typed": True}
        for e in entries
    )
    assert SECOND not in result.output
    args = commands.TuningArgs(models=None, keys={"reviewer": "groq:backup"})
    assert args.change().keys == {"reviewer": "groq:backup"}


@pytest.mark.parametrize(
    "target", ["tuning.model.interlocutor", "tuning.keys.planner", "tuning.key.planner"]
)
def test_analyst_never_offers_or_accepts_key_or_interlocutor(target: str) -> None:
    assert target not in improve.SETTINGS_BY_ID
    change = improve.Change(target=target, value=key_id(FIRST))
    with pytest.raises(improve.Unfit, match="not a target or a setting"):
        improve.render([change])


def test_shared_orders_share_client_state(clock: FakeClock) -> None:
    cfg = config()
    cfg.tuning.keys.reviewer = key_id(FIRST)  # Same effective order as auto.
    crew = runs.models(
        cfg,
        db.open_db(":memory:"),
        provider="groq",
        model=providers.find("groq").default_model,
        cache=False,
        client=httpx.Client(transport=httpx.MockTransport(lambda _: response())),
        clock=clock,
        sleep=clock.sleep,
    )
    planner = crew.for_(Role.PLANNER)
    reviewer = crew.for_(Role.REVIEWER)
    assert isinstance(planner, router.Fallback) and isinstance(reviewer, router.Fallback)
    assert planner.members[0][0] is reviewer.members[0][0]


def test_preview_is_pure_and_role_data_contains_no_secrets() -> None:
    cfg = config()
    old, new, preview = tuning_cli.changed_tuning(
        cfg,
        tuning_cli.TuningChange(
            models={"interlocutor": "groq/test"}, keys={"interlocutor": "groq:backup"}
        ),
    )
    assert old == cfg.tuning == Tuning()
    assert new == preview.tuning and new.keys.interlocutor == key_id(SECOND)
    assert new.models.interlocutor == "groq/test"
    assert set(KeyChoices.model_fields) == set(ModelChoices.model_fields)
    rows = tuning_cli.role_rows(preview)
    assert len(rows) == 7 and rows[-1]["key_fingerprint"] == key_id(SECOND)
    assert FIRST not in json.dumps(rows) and SECOND not in json.dumps(rows)


def test_natural_chat_key_change_requires_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    from desman.interlocutor import CHAT
    from desman.llm.fake import FakeLLM

    cfg = config()
    open_config().save(cfg)
    ask = json.dumps(
        {"type": "command", "name": "tuning", "args": {"keys": {"reviewer": "groq:backup"}}}
    )
    fake = FakeLLM().on(CHAT, ask, ask)
    monkeypatch.setattr(runs, "models", lambda config, conn, **_: runs.Models(fake, None))
    result = CliRunner().invoke(cli, [], input="choose backup\nn\n/exit\n")
    assert result.exit_code == 0, result.output
    assert open_config().load().tuning.keys.reviewer is None
    result = CliRunner().invoke(cli, [], input="choose backup\ny\n/exit\n")
    assert result.exit_code == 0, result.output
    assert open_config().load().tuning.keys.reviewer == key_id(SECOND)
    assert any(
        e.action == "operator_tuning" and e.details == {"changed": ["keys.reviewer"]}
        for e in operator.entries(db.open_db())
    )


def test_chat_wizard_role_section_journals_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    from desman.interlocutor import CHAT
    from desman.llm.fake import FakeLLM

    open_config().save(config())
    ask = json.dumps({"type": "command", "name": "config", "args": {"section": "roles"}})
    fake = FakeLLM().on(CHAT, ask)
    monkeypatch.setattr(runs, "models", lambda config, conn, **_: runs.Models(fake, None))
    result = CliRunner().invoke(cli, [], input="configure roles\ny\n1\n2\n3\n8\n/exit\n")
    assert result.exit_code == 0, result.output
    assert open_config().load().tuning.keys.planner == key_id(SECOND)
    assert any(
        e.action == "operator_tuning"
        and e.details == {"changed": ["models.planner", "keys.planner"]}
        for e in operator.entries(db.open_db())
    )


@pytest.mark.parametrize("missing", ["model", "key"])
@pytest.mark.parametrize("reset", [False, True])
def test_wizard_keeps_unavailable_defaults_unless_auto_selected(missing: str, reset: bool) -> None:
    cfg = config()
    if missing == "model":
        cfg.tuning.models.planner = "mistral/saved-model"
        cfg.api_keys.pop("MISTRAL_API_KEY", None)
    else:
        cfg.tuning.keys.planner = key_id(SECOND)
        cfg.api_keys["GROQ_API_KEY"] = cfg.api_keys["GROQ_API_KEY"][:1]
    saved = cfg.tuning.model_copy(deep=True)
    open_config().save(cfg)

    @click.command()
    def section() -> None:
        wizard.run_section(open_config(), "roles")

    # Empty answers in the numbered fallback retain both menu defaults.
    answers = "1\n1\n1\n8\n" if reset else "1\n\n\n8\n"
    result = CliRunner().invoke(section, [], input=answers)
    assert result.exit_code == 0, result.output
    assert "not available now" in result.output
    if missing == "key":
        assert "missing" in result.output
    else:
        assert "mistral/saved-model" in result.output
    assert open_config().load().tuning == (Tuning() if reset else saved)
    assert FIRST not in result.output and SECOND not in result.output
