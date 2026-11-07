"""A model from Hugging Face for a local server (ROADMAP 4.5b): the bodies of `desman models`,
shared by the CLI, the wizard and the chat's catalog.

`look_up` says what a model is (size, files, gated) and `describe` words it for the person
with the machine's free memory, before anything is downloaded. `fetch` has the server download
it (Ollama pulls it; llama.cpp is told how to start with it) and `use` makes it the server's
model in the config. Problems are `click.ClickException`, worded for the person.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import suppress

import click
import httpx

from desman import providers, ui
from desman.config import Config
from desman.i18n import t
from desman.llm import hub
from desman.providers import Provider

TARGETS = ("ollama", "llamacpp")
"""The local servers a model can be added to."""
OLLAMA_KEYS = "https://huggingface.co/settings/keys"


def token(config: Config) -> str | None:
    """The Hugging Face token, if the person added one."""
    keys = config.keys("HF_TOKEN")
    return keys[0].value if keys else None


def target(name: str) -> Provider:
    if name not in TARGETS:
        raise click.BadParameter(
            t("unknown local server {name!r} (known: {known})", name=name, known=", ".join(TARGETS))
        )
    return providers.find(name)


def parse(text: str) -> hub.Ref:
    try:
        return hub.parse(text)
    except ValueError:
        raise click.BadParameter(
            t(
                "{text!r} is not a Hugging Face model: write author/model, or "
                "author/model:quantization (e.g. Q4_K_M)",
                text=text,
            ),
            param_hint="MODEL",
        ) from None


def look_up(config: Config, ref: hub.Ref, client: httpx.Client) -> hub.Gguf:
    """What `ref` is on Hugging Face, and whether this token may download it."""
    secret = token(config)
    try:
        model = hub.inspect(ref, client, secret)
        hub.check_access(model, client, secret)
    except hub.HubError as e:
        raise click.ClickException(problem(e)) from None
    return model


def problem(e: hub.HubError) -> str:
    name, url = e.ref.spec, e.ref.page
    quants = ", ".join(e.quants)
    match e.problem:
        case hub.Problem.NOT_FOUND:
            return t(
                "{model} is not on Hugging Face, or it is private: check the name, or add a "
                "token with `desman keys add huggingface`.",
                model=name,
            )
        case hub.Problem.NO_ACCESS:
            return t("Your Hugging Face token may not read {model}.", model=name)
        case hub.Problem.NOT_GGUF:
            return t(
                "{model} has no GGUF files, and only GGUF models can be used: look for a "
                "repository with GGUF in its name.",
                model=name,
            )
        case hub.Problem.NO_QUANT:
            return t(
                "{model} has no {quant} files. It has: {quants}",
                model=e.ref.repo,
                quant=e.ref.quant,
                quants=quants,
            )
        case hub.Problem.CHOOSE_QUANT:
            return t(
                "{model} comes in several quantizations: add one to the name, e.g. "
                "{model}:{example}. It has: {quants}",
                model=name,
                example=e.quants[0] if e.quants else hub.DEFAULT_QUANT,
                quants=quants or "-",
            )
        case hub.Problem.LICENSE:
            return t(
                "{model} is gated: accept its license at {url} with the account of your token.",
                model=name,
                url=url,
            )
        case hub.Problem.TOKEN:
            return t(
                "{model} is gated: accept its license at {url}, then add a Hugging Face token "
                "with `desman keys add huggingface`.",
                model=name,
                url=url,
            )
        case hub.Problem.UNEXPECTED:
            return t(
                "Hugging Face answered with something unexpected about {model}: try again later.",
                model=name,
            )
    return t("Hugging Face cannot be reached right now.")


def describe(model: hub.Gguf, machine: hub.Memory) -> tuple[list[str], list[str]]:
    """What the person is shown before a download: (lines, warnings)."""
    lines = [
        t(
            "{model}: {size} to download, about {memory} of memory to run",
            model=model.ref.spec,
            size=hub.size_text(model.size),
            memory=hub.size_text(model.memory),
        )
    ]
    if machine.ram is None:
        lines.append(t("How much memory is free could not be found out."))
    elif machine.video:
        lines.append(
            t(
                "Free memory: {ram} RAM, {video} video",
                ram=hub.size_text(machine.ram),
                video=hub.size_text(machine.video),
            )
        )
    else:
        lines.append(t("Free memory: {ram} RAM", ram=hub.size_text(machine.ram)))
    warnings = []
    if hub.fits(model, machine) is False:
        warnings.append(
            t("It does not fit in the free memory: it would run very slowly, or not at all.")
        )
    return lines, warnings


def plan(config: Config, model: hub.Gguf, server: Provider) -> tuple[list[str], list[str]]:
    """What adding `model` to `server` does: (lines, warnings)."""
    if server.id == "ollama":
        host = server.setting(config, "OLLAMA_HOST")
        lines = [
            t("Ollama at {host} downloads it, then DESMAN uses it as the Ollama model.", host=host)
        ]
        warnings = []
        if model.gated:
            warnings.append(
                t(
                    "Ollama downloads a gated model with its own key: add Ollama's key to your "
                    "Hugging Face account first ({url}).",
                    url=OLLAMA_KEYS,
                )
            )
        if not providers.on_this_machine(host) and not host.lower().startswith("https://"):
            warnings.append(
                t(
                    "Ollama at {host} is not on this machine and is reached without encryption: "
                    "on the way, the request can be read or changed (another model downloaded).",
                    host=host,
                )
            )
        return lines, warnings
    lines = [
        t(
            "DESMAN uses it as the llama.cpp model. llama.cpp downloads it when its server is "
            "started with: {command}",
            command=hub.llama_server_command(model.ref, _port(config, server)),
        )
    ]
    warnings = (
        [t("For a gated model, start it with your Hugging Face token in the HF_TOKEN variable.")]
        if model.gated
        else []
    )
    return lines, warnings


def _port(config: Config, server: Provider) -> int:
    from urllib.parse import urlsplit

    try:
        return urlsplit(server.endpoint(config)).port or 8080
    except ValueError:
        return 8080


def fetch(
    config: Config,
    model: hub.Gguf,
    server: Provider,
    client: httpx.Client,
    *,
    show: Callable[[str], None] = ui.note,
) -> None:
    """Has `server` download the model (Ollama); llama.cpp downloads it on its next start."""
    if server.id != "ollama":
        return
    host = server.setting(config, "OLLAMA_HOST")
    if "://" not in host:
        host = "http://" + host
    reported: set[str] = set()

    def clean(text: str) -> str:
        """Ollama's own words, with anything key-shaped cut out (it is never sent a key, but
        what it prints is not ours to vouch for)."""
        return providers.redact_keys(text, config)[0]

    def step(s: hub.Step) -> None:
        if s.total and s.completed is not None:
            tenth = 10 * s.completed // s.total
            mark = f"{s.status}:{tenth}"
            if tenth and mark not in reported:
                reported.add(mark)
                show(
                    t(
                        "Downloaded {percent}% of {size}",
                        percent=10 * tenth,
                        size=hub.size_text(s.total),
                    )
                )
        elif s.status and s.status not in reported:
            reported.add(s.status)
            show(clean(s.status))  # Ollama's own words

    try:
        hub.pull(host, model.ref, client, step)
    except hub.PullFailed as e:
        raise click.ClickException(
            t(
                "Ollama could not download {model}: {error}",
                model=model.ref.spec,
                error=clean(str(e)),
            )
        ) from None


def use(config: Config, model: hub.Gguf, server: Provider) -> str:
    """Makes the model the server's in `config`; returns the name it is asked by."""
    name = model.ref.ollama if server.id == "ollama" else model.ref.spec
    return use_name(config, name, server)


# Small, bounded cache shared by page polls and role validation. No credentials are cached.
_tags_cache: dict[str, tuple[float, list[hub.InstalledModel], bool]] = {}


def cached_installed(config: Config) -> tuple[list[hub.InstalledModel], bool]:
    """Read picker metadata without discovery or network access."""
    host = providers.find("ollama").setting(config, "OLLAMA_HOST")
    cached = _tags_cache.get(host)
    return (cached[1], cached[2]) if cached else ([], False)


def installed(
    config: Config, *, refresh: bool = False, cached_only: bool = False
) -> tuple[list[hub.InstalledModel], bool]:
    """Installed Ollama models and reachability; a short failure cache avoids poll storms."""
    import time

    host = providers.find("ollama").setting(config, "OLLAMA_HOST")
    cached = _tags_cache.get(host)
    if cached_only:
        return cached_installed(config)
    if not refresh and cached is not None and time.monotonic() - cached[0] < 15:
        return cached[1], cached[2]
    try:
        with httpx.Client() as client:
            models = hub.ollama_tags(host, client)
        ready = True
    except (httpx.HTTPError, httpx.InvalidURL, ValueError):
        models, ready = [], False
    if len(_tags_cache) >= 16:
        _tags_cache.clear()
    _tags_cache[host] = (time.monotonic(), models, ready)
    return models, ready


def select_installed(config: Config, name: str) -> str:
    """Use an installed tag, validated against the server; shares the config body with use()."""
    tags, _ = installed(config, refresh=True)
    if name not in {m.name for m in tags}:
        raise click.ClickException(t("Invalid module choice."))
    return use_name(config, name, target("ollama"))


def use_name(config: Config, name: str, server: Provider) -> str:
    assert server.model_env is not None
    config.settings[server.model_env] = name
    if server.id == "ollama":
        _tags_cache.pop(server.setting(config, "OLLAMA_HOST"), None)
    return name


def ollama_name(name: str) -> str:
    """Ollama interprets an omitted tag as :latest, including names with namespaces."""
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def removal_model(config: Config, name: str, client: httpx.Client) -> hub.InstalledModel:
    """Resolve an exact installed name at the configured Ollama host."""
    host = providers.find("ollama").setting(config, "OLLAMA_HOST")
    try:
        models = hub.ollama_tags(host, client)
    except (httpx.HTTPError, httpx.InvalidURL, ValueError) as e:
        raise click.ClickException(t("Ollama cannot be reached right now.")) from e
    found = next((model for model in models if ollama_name(model.name) == ollama_name(name)), None)
    if found is None:
        raise click.ClickException(t("Ollama model not installed: {model}", model=name))
    return found


def missing_choices(config: Config, name: str, conn: sqlite3.Connection) -> list[str]:
    """Saved references are intentionally retained when their model is removed."""
    label = "ollama/" + ollama_name(name)
    references = [
        "role/" + role
        for role, model in config.tuning.models.model_dump().items()
        if model and model.startswith("ollama/") and "ollama/" + ollama_name(model[7:]) == label
    ]
    if ollama_name(providers.find("ollama").model(config)) == ollama_name(name):
        references.insert(0, "OLLAMA_MODEL")
    references.extend(
        f"{row['scope']}/{row['scope_id']}/{row['role']}"
        for row in conn.execute(
            "SELECT scope, scope_id, role, model FROM module_choices WHERE model LIKE 'ollama/%' ORDER BY scope, scope_id, role",
        )
        if "ollama/" + ollama_name(row["model"][7:]) == label
    )
    return references


def remove(config: Config, name: str, client: httpx.Client, conn: sqlite3.Connection) -> list[str]:
    """Delete only on the configured host, without following redirects or changing choices."""
    host = providers.find("ollama").setting(config, "OLLAMA_HOST")
    if "://" not in host:
        host = "http://" + host
    try:
        response = client.request(
            "DELETE",
            host.rstrip("/") + "/api/delete",
            json={"model": name},
            timeout=30.0,
            follow_redirects=False,
        )
        response.raise_for_status()
    except (httpx.HTTPError, httpx.InvalidURL) as e:
        raise click.ClickException(t("Could not remove Ollama model {model}.", model=name)) from e
    _tags_cache.pop(providers.find("ollama").setting(config, "OLLAMA_HOST"), None)
    references = missing_choices(config, name, conn)
    ui.ok(t("Removed Ollama model {model}.", model=name))
    if references:
        ui.note(
            t(
                "Saved choices now pointing at a missing model: {choices}",
                choices=", ".join(references),
            )
        )
    return references


def ollama_context(config: Config, name: str) -> tuple[int, bool]:
    """Context and memory risk, using cached metadata only (routing never discovers models)."""
    provider = providers.find("ollama")
    if "OLLAMA_CONTEXT" in config.settings:
        return provider.context(config) or 16384, False
    tags, _ = cached_installed(config)
    model = next((m for m in tags if ollama_name(m.name) == ollama_name(name)), None)
    if model is None:
        return 16384, False
    details = model.details
    maximum = details.context_length if details else None
    cap = min(16384, maximum or 16384)
    floor = min(8192, cap)
    # fp16 K and V per layer; GQA reduces the cached embedding width.
    layers = details.block_count if details and details.block_count else 32
    embedding = details.embedding_length if details and details.embedding_length else 4096
    ratio = (
        details.head_count_kv / details.head_count
        if details and details.head_count_kv and details.head_count
        else 1.0
    )
    per_token = max(1, int(4 * layers * embedding * ratio))
    available = hub.memory().total
    if available is None or not model.size:
        return cap, False
    # Leave room for Ollama's compute buffers as well as the weights.
    room = max(0, available - model.size - 512 * 1024 * 1024)
    fit = int(room // per_token)
    return min(cap, max(floor, (fit // 1024) * 1024)), fit < floor


def prepare_ollama_context(config: Config, name: str, client: httpx.Client) -> tuple[int, bool]:
    """Resolve metadata only when actually calling Ollama, never while building a lineup."""
    import time

    if "OLLAMA_CONTEXT" in config.settings:
        return ollama_context(config, name)
    host = providers.find("ollama").setting(config, "OLLAMA_HOST")
    tags, ready = cached_installed(config)
    cached = _tags_cache.get(host)
    if not ready and (cached is None or time.monotonic() - cached[0] >= 15):
        if len(_tags_cache) >= 16:
            _tags_cache.clear()
        _tags_cache[host] = (time.monotonic(), [], False)
        with suppress(httpx.HTTPError, httpx.InvalidURL, ValueError):
            tags = hub.ollama_tags(host, client)
            _tags_cache[host] = (time.monotonic(), tags, True)
    for index, tag in enumerate(tags):
        if ollama_name(tag.name) == ollama_name(name) and (
            tag.details is None or tag.details.block_count is None
        ):
            with suppress(httpx.HTTPError, httpx.InvalidURL, ValueError):
                tags[index] = hub.ollama_details(host, tag, client)
            break
    return ollama_context(config, name)
