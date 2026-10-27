"""Helpers shared by tests (import as `from helpers import ...`)."""

from __future__ import annotations

from desman.config import Config, upgrade


def make_config(**data: object) -> Config:
    """A Config from schema-1 style data (`api_keys: {ENV: "key"}`), upgraded like a file."""
    return Config.model_validate(upgrade({"schema_version": 1, **data}))


def flat_keys(config: Config) -> dict[str, str | list[str]]:
    """api_keys as {ENV: "key"}, or {ENV: [keys]} when a provider has several."""
    return {
        env: keys[0].value if len(keys) == 1 else [k.value for k in keys]
        for env, keys in config.api_keys.items()
    }
