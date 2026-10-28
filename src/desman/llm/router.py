"""Which provider and model a run uses, built from the config (ROADMAP 2.2).

One provider per run for now: the one asked for, or the first ready one in
`providers.PREFERENCE` (free tiers only: a paid provider is used only when asked for). All its keys are used (the client moves to the next key when one is
refused or limited). Fallback across providers and per-role models are ROADMAP 4.5.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx

from desman import providers
from desman.config import Config
from desman.i18n import t
from desman.llm.openai_compat import CallRecord, OpenAICompatible
from desman.providers import Provider


class NoProvider(LookupError):
    pass


def choose(config: Config, wanted: str | None = None) -> Provider:
    """The provider to use: `wanted` (id, name or key variable) or the preferred ready one."""
    if wanted is not None:
        try:
            provider = providers.find(wanted)
        except providers.UnknownProvider as e:
            raise NoProvider(str(e)) from e
        if provider.local:
            raise NoProvider(
                t("{name} is local: local models are not supported yet", name=provider.name)
            )
        if not provider.is_ready(config):
            if provider.settings:
                raise NoProvider(
                    t(
                        "{name} is not configured: add a key with `desman keys add {id}` and set "
                        "its settings in `desman config`",
                        name=provider.name,
                        id=provider.id,
                    )
                )
            raise NoProvider(
                t(
                    "{name} is not configured: add a key with `desman keys add {id}`",
                    name=provider.name,
                    id=provider.id,
                )
            )
        return provider
    ready = {p.id: p for p in providers.ready(config)}
    for pid in providers.PREFERENCE:
        if pid in ready:
            return ready[pid]
    paid = [p for p in providers.ready(config) if p.paid]
    if paid:
        raise NoProvider(
            t(
                "only paid providers are configured ({names}), and DESMAN never picks a paid one "
                "by itself: choose it with --provider, e.g. --provider {id}",
                names=", ".join(p.name for p in paid),
                id=paid[0].id,
            )
        )
    raise NoProvider(t("no provider is configured: add a key with `desman keys add PROVIDER`"))


@dataclass(frozen=True)
class Choice:
    provider: Provider
    model: str


def build(
    config: Config,
    provider: str | None = None,
    model: str | None = None,
    *,
    on_call: Callable[[CallRecord], None] | None = None,
    client: httpx.Client | None = None,
) -> tuple[OpenAICompatible, Choice]:
    chosen = choose(config, provider)
    assert chosen.key_env is not None
    use = model or chosen.default_model
    llm = OpenAICompatible(
        chosen.id,
        chosen.endpoint(config),
        [k.value for k in config.keys(chosen.key_env)],
        use,
        client=client,
        on_call=on_call,
        auth=chosen.auth,
        send_temperature=chosen.send_temperature,
        max_tokens=chosen.max_tokens,
        paid=chosen.paid,
    )
    return llm, Choice(chosen, use)
