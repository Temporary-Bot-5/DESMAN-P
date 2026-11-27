"""Provider terms and explicit consent for lending a new cloud provider."""

import json
from dataclasses import asdict
from datetime import date
from urllib.parse import urlsplit

from desman import providers
from desman.config import Network
from desman.interlocutor import CHAT
from desman.llm.fake import FakeLLM
from test_commands import asks, chat
from test_global import OLLAMA, REGISTRY, configured, network_log, ok, refused, saved
from test_global import fake as fake

GROQ = providers.find("groq")


def test_catalog_terms() -> None:
    for provider in providers.CLOUD:
        terms = provider.terms
        assert terms is not None
        url = urlsplit(terms.url)
        assert url.scheme == "https" and url.hostname
        assert date.fromisoformat(terms.checked).isoformat() == terms.checked
        assert not provider.donatable or terms.lending == "allowed"
    assert all(p.terms is None for p in providers.LOCAL + providers.SERVICES)
    assert not any(p.donatable for p in providers.PROVIDERS + providers.SERVICES)


def test_terms_table_and_json() -> None:
    found = ok("global", "terms").output
    assert "Key sharing" in found and "Lending" in found and "Checked" in found
    assert "DESMAN never sends a key to anyone" in found
    rows = json.loads(ok("global", "terms", "--json").stdout)
    assert [row["id"] for row in rows] == [p.id for p in providers.CLOUD]
    assert all(
        set(row) == {"id", "url", "checked", "key_sharing", "lending", "clauses"} for row in rows
    )

    one = ok("global", "terms", "GROQ_API_KEY").output
    assert "GROQ" in one and "NVIDIA" not in one and "2026-11-27" in one
    assert "unclear" in one and "forbidden" in one
    assert json.loads(ok("global", "terms", "GROQ", "--json").stdout) == [
        row for row in rows if row["id"] == "groq"
    ]
    assert "unknown provider" in refused("global", "terms", "nobody")
    assert "runs on your own machine" in ok("global", "terms", "ollama").output
    assert json.loads(ok("global", "terms", "ollama", "--json").stdout) == []


def setup_lending() -> None:
    configured(
        api_keys={"GROQ_API_KEY": "fake-groq-key"},
        settings=OLLAMA,
        network=Network(enabled=True, registry=REGISTRY, lend_cloud=True),
    )


def test_donation_warns_and_requires_yes_only_for_new_cloud_provider() -> None:
    setup_lending()
    before = saved()
    preview = ok("global", "donate", "groq")
    assert "GROQ's terms forbid" in preview.stdout
    assert "Nothing lent. Run it again with --yes to lend GROQ." in preview.stdout

    preview_json = ok("global", "donate", "groq", "--json")
    terms = GROQ.terms
    assert terms is not None
    assert json.loads(preview_json.stdout) == {
        "lent": False,
        "provider": "groq",
        "terms": asdict(terms),
    }
    assert "GROQ's terms forbid" in preview_json.stderr
    assert saved() == before
    assert network_log() == []

    accepted = ok("global", "donate", "groq", "--yes", "--json")
    assert json.loads(accepted.stdout)["provider"] == "groq"
    assert "GROQ's terms forbid" in accepted.stderr
    assert "groq" in saved().network.donated
    assert [action for action, _ in network_log()] == ["donated"]

    quota = ok("global", "donate", "groq", "--calls", "20")
    assert "terms forbid" not in quota.output
    assert saved().network.donated["groq"].calls_per_day == 20
    assert "terms forbid" not in ok("global", "donate", "ollama").output
    assert "ollama" in saved().network.donated


def test_chat_donation_warns_before_confirmation(fake: FakeLLM) -> None:
    setup_lending()
    fake.on(CHAT, asks("global donate", provider="groq"), asks("global donate", provider="groq"))
    declined = chat("lend groq", "n").output
    assert "Others' tasks will use it" in declined
    assert declined.index("GROQ's terms forbid") < declined.index("Apply this change?")
    assert "groq" not in saved().network.donated and network_log() == []
    chat("lend groq", "y")
    assert "groq" in saved().network.donated


def test_cloud_on_lists_free_verdicts() -> None:
    output = ok("global", "cloud", "on").output
    assert "Most free tiers' terms" in output
    assert "`desman global terms`" in output
    for provider in providers.FREE:
        terms = provider.terms
        assert terms is not None
        assert f"{provider.name}: lending {providers.verdict_text(terms.lending)}" in output
    for provider in providers.PAID:
        assert f"{provider.name}: lending" not in output
