"""desman-global, the node's side (ADR 0004). Nothing here runs while desman-global is off.

The registry (`desman.registry`) uses `net.sign`, `net.request` and `net.protocol` too, and it
must not import anything that reads `config.json` (ADR 0004, decision 1): `Config` is only a
type here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
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


def operator_registry(config: Config) -> str | None:
    """Registry address for operator mode. Call only when the person installed operator.key.

    The operator key permits operator calls independently of desman-global participation.
    Ordinary node calls still obtain their address from `registry`.
    """
    return config.network.registry or DEFAULT_REGISTRY
