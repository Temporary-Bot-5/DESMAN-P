"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from desman.config import CONFIG_DIR_ENV
from desman.core.db import DATA_DIR_ENV


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point DESMAN at throwaway config and data dirs so tests never touch the real ones."""
    config_dir = tmp_path / "config"
    monkeypatch.setenv(CONFIG_DIR_ENV, str(config_dir))
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "data"))
    return config_dir
