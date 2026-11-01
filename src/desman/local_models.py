"""A model from Hugging Face for a local server (ROADMAP 4.5b): the bodies of `desman models`,
shared by the CLI, the wizard and the chat's catalog.

`look_up` says what a model is (size, files, gated) and `describe` words it for the person
with the machine's free memory, before anything is downloaded. `fetch` has the server download
it (Ollama pulls it; llama.cpp is told how to start with it) and `use` makes it the server's
model in the config. Problems are `click.ClickException`, worded for the person.
"""

from __future__ import annotations

from collections.abc import Callable

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
    assert server.model_env is not None
    config.settings[server.model_env] = name
    return name
