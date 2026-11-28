"""desman-global, the node's side (ADR 0004).

Node traffic requires participation. The person can explicitly request discovery before joining.

The registry (`desman.registry`) uses `net.sign`, `net.request` and `net.protocol` too, and it
must not import anything that reads `config.json` (ADR 0004, decision 1): `Config` is only a
type here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Final

from pydantic import StringConstraints

if TYPE_CHECKING:
    from desman.config import Config

RegistryUrl = Annotated[
    str,
    StringConstraints(
        pattern=r"^(https://[A-Za-z0-9.-]+(:\d+)?|http://(localhost|127\.0\.0\.1)(:\d+)?)(/[^\s]*)?$",
        max_length=200,
    ),
]
"""The registry of desman-global: HTTPS, or plain HTTP to this machine (a registry under test)."""

CATALOG_URL: Final = "https://temporary-bot-5.github.io/DESMAN-P/registries.json"

DEFAULT_REGISTRY: Final[str | None] = None
"""The built-in registry address. Discovery and manual selection use the saved address instead."""


def registry(config: Config) -> str | None:
    """Where to reach the registry, or None while desman-global is off.

    All participation traffic gets its registry address here. Discovery probes catalog entries
    at the person's request, independently of participation."""
    if not config.network.enabled:
        return None
    return config.network.registry or DEFAULT_REGISTRY


def operator_registry(config: Config) -> str | None:
    """Registry address for operator mode. Call only when the person installed operator.key.

    The operator key permits operator calls independently of desman-global participation.
    Ordinary node calls still obtain their address from `registry`.
    """
    return config.network.registry or DEFAULT_REGISTRY
