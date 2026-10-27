"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from desman.config import CONFIG_DIR_ENV


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point DESMAN at a throwaway config dir so tests never touch ~/.config/desman."""
    config_dir = tmp_path / "config"
    monkeypatch.setenv(CONFIG_DIR_ENV, str(config_dir))
    return config_dir
