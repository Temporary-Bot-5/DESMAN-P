"""Signed registry discovery across CLI, setup, and the local Settings route."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from desman import global_cli, net
from desman.cli import cli
from desman.config import Config, Network, open_config
from desman.core import web
from desman.gui.front import WebFront
from desman.gui.server import Server
from desman.net import catalog, sign
from desman.registry.cli import cli as registry_cli
from test_wizard import DONE, KEEP_PERMISSIONS, wizard

NOW = datetime(2026, 11, 28, tzinfo=UTC)
ROOT = sign.NodeKey(Ed25519PrivateKey.from_private_bytes(bytes(range(32))))
ONE = "https://one.example"
TWO = "https://two.example/desman"
ENTRIES = [{"name": "First", "url": ONE}, {"name": "Second", "url": TWO}]
FIXTURE = Path(__file__).parent / "fixtures" / "registry-catalog.json"


@pytest.fixture(autouse=True)
def catalog_root(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(global_cli, "_utc_now", lambda: NOW)
    monkeypatch.setattr(web, "resolve", lambda host, port: ["93.184.215.14"])
    with monkeypatch.context() as patch:
        patch.setattr(sign, "load_root", lambda *args: ROOT.public)
        yield


def signed_catalog(
    entries: list[dict[str, str]],
    *,
    root: sign.NodeKey = ROOT,
    kind: sign.RecordKind = sign.RecordKind.REGISTRY_CATALOG,
    issued_at: datetime = NOW,
) -> sign.Signed:
    return sign.sign(
        root, kind, {"registries": entries}, issued_at=issued_at, lifetime=timedelta(days=365)
    )


def fetch_bytes(data: bytes) -> catalog.RegistryCatalog:
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=data))
    ) as client:
        return catalog.fetch(client, NOW)


def test_fetch_accepts_signed_fixture() -> None:
    assert [entry.url for entry in fetch_bytes(FIXTURE.read_bytes()).registries] == [ONE, TWO]


@pytest.mark.parametrize(
    "signed",
    [
        signed_catalog(ENTRIES).model_copy(update={"signature": "A" * 88}),
        signed_catalog(ENTRIES, root=sign.NodeKey.generate()),
        signed_catalog(ENTRIES, kind=sign.RecordKind.RELEASE),
        signed_catalog([{"name": "Bad", "url": "http://localhost"}]),
        signed_catalog([{"name": "Bad", "url": "https://example.org:99999"}]),
        signed_catalog([{"name": "Bad", "url": "https://example.org/api?q=1"}]),
        signed_catalog([*ENTRIES, {"name": "Alias", "url": ONE + "/"}]),
        signed_catalog([{"name": str(i), "url": f"https://host{i}.example"} for i in range(51)]),
    ],
)
def test_bad_signed_catalog_is_refused(signed: sign.Signed) -> None:
    with pytest.raises(catalog.Refused, match="could not be verified"):
        fetch_bytes(signed.model_dump_json().encode())


def test_expired_unsigned_and_oversized_catalogs_are_refused() -> None:
    expired = signed_catalog(ENTRIES, issued_at=NOW - timedelta(days=365))
    with pytest.raises(catalog.Expired, match="expired"):
        fetch_bytes(expired.model_dump_json().encode())
    with pytest.raises(catalog.Refused, match="could not be verified"):
        fetch_bytes(expired.record.model_dump_json().encode())
    with pytest.raises(catalog.Refused, match="too large"):
        fetch_bytes(b" " * (catalog.MAX_BYTES + 1))


def test_catalog_is_not_received_into_trust() -> None:
    trust = sign.Trust(ROOT.public, clock=lambda: NOW)
    with pytest.raises(sign.Refused, match="not trust state"):
        trust.receive(FIXTURE.read_bytes())


@pytest.fixture
def setup_network(monkeypatch: pytest.MonkeyPatch) -> tuple[list[httpx.Request], dict[str, str]]:
    requests: list[httpx.Request] = []
    outcomes: dict[str, str] = {}

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "GET"
        original = httpx.URL(net.CATALOG_URL)
        if request.headers["host"] == original.host:
            # The catalog is fetched pinned to a checked public address.
            assert request.url.host == "93.184.215.14"
            assert request.url.path == original.path
            assert request.extensions["sni_hostname"] == original.host
            assert request.extensions["timeout"]["read"] == 10
            if outcomes.get("catalog") == "down":
                return httpx.Response(503)
            return httpx.Response(200, content=FIXTURE.read_bytes())
        assert request.url.path.endswith("/healthz")
        assert request.extensions["timeout"]["read"] == 5
        host = request.url.host  # a registry is reached by its name, as a node reaches it
        if host == "two.example":
            assert request.url.path == "/desman/healthz"
        outcome = outcomes.get(host)
        if outcome == "timeout":
            raise httpx.ReadTimeout("test", request=request)
        if outcome == "garbage":
            return httpx.Response(200, text="not JSON")
        if outcome == "ok":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(503)

    monkeypatch.setattr(
        web, "fetch_client", lambda: httpx.Client(transport=httpx.MockTransport(transport))
    )
    return requests, outcomes


@pytest.mark.parametrize("failure", ["timeout", "garbage", "503"])
def test_probes_measure_only_answering_registries(
    setup_network: tuple[list[httpx.Request], dict[str, str]], failure: str
) -> None:
    requests, outcomes = setup_network
    outcomes.update({"one.example": "ok", "two.example": failure})
    first, second = catalog.discover(NOW)
    assert first.latency_ms is not None and first.latency_ms >= 0 and first.answers
    assert second.latency_ms is None and not second.answers
    assert len(requests) == 3


def test_probe_reaches_a_registry_on_a_private_address(
    setup_network: tuple[list[httpx.Request], dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root-signed entry may resolve to the owner's tailnet (100.64.0.0/10) on their machines;
    the catalog itself is still fetched only from a public address."""
    requests, outcomes = setup_network
    outcomes.update({"one.example": "ok"})

    def resolve(host: str, port: int) -> list[str]:
        if host == httpx.URL(net.CATALOG_URL).host:
            return ["93.184.215.14"]
        return ["93.184.215.14", "127.0.0.1"]

    monkeypatch.setattr(web, "resolve", resolve)
    assert [probe.answers for probe in catalog.discover(NOW)] == [True, False]
    assert len(requests) == 3


@pytest.mark.parametrize("count", [0, 1, 2])
def test_join_selects_only_a_single_answering_registry(
    setup_network: tuple[list[httpx.Request], dict[str, str]], count: int
) -> None:
    _, outcomes = setup_network
    outcomes.update({host: "ok" for host in ["one.example", "two.example"][:count]})
    open_config().save(Config(configured=True))
    result = CliRunner().invoke(cli, ["global", "join"])
    net_config = open_config().load().network
    if count == 1:
        assert result.exit_code == 0, result.output
        assert ONE in result.output and net_config.enabled and net_config.registry == ONE
        return
    assert result.exit_code == 1, result.output
    assert not net_config.enabled and not sign.default_node_key_path().exists()
    if count == 2:
        assert all(part in result.output for part in (ONE, TWO, "--registry URL"))
    else:
        assert "No registry address is known yet" in result.output
        assert "No catalog registries are answering." in result.output


def test_status_and_explicit_join_skip_discovery(
    setup_network: tuple[list[httpx.Request], dict[str, str]],
) -> None:
    requests, _ = setup_network
    manager = open_config()
    manager.save(Config(configured=True))
    runner = CliRunner()
    assert runner.invoke(cli, ["global", "status"]).exit_code == 0
    manager.save(Config(configured=True, network=Network(registry=ONE)))
    assert runner.invoke(cli, ["global", "join"]).exit_code == 0
    assert runner.invoke(cli, ["global", "join", "--registry", TWO]).exit_code == 0
    assert requests == []


@pytest.mark.parametrize(
    ("answers", "catalog_down", "selected"),
    [
        (("n",), False, None),
        (("y", "1"), False, ONE),
        (("y", "2", TWO), False, TWO),
        (("y", "3"), False, None),
        (("y", TWO), True, TWO),
    ],
)
def test_wizard_catalog_and_manual_address(
    setup_network: tuple[list[httpx.Request], dict[str, str]],
    answers: tuple[str, ...],
    catalog_down: bool,
    selected: str | None,
) -> None:
    requests, outcomes = setup_network
    outcomes["one.example"] = "ok"
    if catalog_down:
        outcomes["catalog"] = "down"
    result = wizard("1", DONE, *KEEP_PERMISSIONS, "", "", *answers, "")
    assert result.exit_code == 0, result.output
    config = open_config().load().network
    assert config.enabled == (selected is not None) and config.registry == selected
    if answers == ("n",):
        assert requests == []
    elif catalog_down:
        assert "catalog is unreachable" in result.output
    else:
        assert "Choose a registry" in result.output


def test_registries_cli_json_and_authenticated_gui_route(
    setup_network: tuple[list[httpx.Request], dict[str, str]],
) -> None:
    requests, outcomes = setup_network
    outcomes["one.example"] = "ok"
    manager = open_config()
    manager.save(Config(configured=True, network=Network(enabled=True, registry=ONE)))
    result = CliRunner().invoke(cli, ["global", "registries", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    first, second = payload["registries"]
    assert first["in_use"] and first["answers"] and first["latency_ms"] is not None
    assert not second["in_use"] and not second["answers"] and second["latency_ms"] is None

    app = Server(WebFront(), 12345)
    headers = {"Host": "localhost:12345", "Cookie": f"desman={app.cookie}"}
    requests.clear()
    assert app.handle("GET", "/registries", {"Host": "localhost:12345"}).status == 403
    assert app.handle("GET", "/registries", headers).status == 200
    assert requests == []
    manager.save(Config(configured=True))
    response = app.handle("GET", "/registries", headers)
    assert response.status == 200, response.body
    assert not json.loads(response.body)["registries"][0]["in_use"]
    assert len(requests) == 3


def test_registry_cli_signature_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root_key = tmp_path / "root.key"
    root = sign.create_node_key(root_key)
    output = tmp_path / "registries.json"
    with monkeypatch.context() as patch:
        patch.setattr(sign, "load_root", lambda *args: root.public)
        runner = CliRunner()
        result = runner.invoke(
            registry_cli,
            [
                "--dir",
                str(tmp_path),
                "catalog",
                "sign",
                "--out",
                str(output),
                "--entry",
                ONE,
                "First",
                "--entry",
                TWO,
                "Second",
                "--note",
                "Second note",
                "--entry",
                "https://three.example",
                "Third",
            ],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        signed = sign.parse(output.read_bytes())
        assert signed.record.expires_at - signed.record.issued_at == timedelta(days=180)
        result = runner.invoke(registry_cli, ["catalog", "show", str(output)])
        assert result.exit_code == 0, (result.output, result.exception)
        assert "Second note" in result.output
        found = catalog.verify(output.read_bytes(), signed.record.issued_at)
        assert [item.note for item in found.registries] == [None, "Second note", None]
