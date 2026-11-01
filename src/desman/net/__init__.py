"""desman-global, the node's side (ADR 0004). Nothing here runs while desman-global is off."""

from __future__ import annotations

from typing import Final

from desman.config import Config

DEFAULT_REGISTRY: Final[str | None] = None
"""The owner's registry, once it runs (ROADMAP G.4); until then the person names one."""


def registry(config: Config) -> str | None:
    """Where to reach the registry, or None while desman-global is off.

    The one way to the registry's address: everything that talks to it asks here first, so
    with desman-global off there is nowhere to send anything."""
    if not config.network.enabled:
        return None
    return config.network.registry or DEFAULT_REGISTRY
