"""config.json: where it lives, its schema and how it is read and written.

The interactive wizard is in `wizard.py`; the known providers in `providers.py`.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

import click
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, ValidationError

CONFIG_DIR_ENV = "DESMAN_CONFIG_DIR"


def project_root() -> Path | None:
    """The source checkout DESMAN runs from (src/desman/config.py -> repo), or None when it is
    installed as a package and there is no project folder to keep the config in."""
    root = Path(__file__).resolve().parents[2]
    return (
        root if (root / "pyproject.toml").is_file() and (root / "src" / "desman").is_dir() else None
    )


def default_config_path() -> Path:
    """Config file location, first that applies:

    1. $DESMAN_CONFIG_DIR/config.json;
    2. config.json in the project folder (a source checkout); `.gitignore` keeps it out of git;
    3. ~/.config/desman/config.json for an installed package.
    """
    config_dir = os.environ.get(CONFIG_DIR_ENV)
    if config_dir:
        return Path(config_dir) / "config.json"
    root = project_root()
    base = root if root is not None else home_config_dir()
    return base / "config.json"


def home_config_dir() -> Path:
    return Path.home() / ".config" / "desman"


def open_config() -> ConfigManager:
    """The ConfigManager the CLI uses. A checkout that used to keep its config in the home
    directory moves it into the project folder once, so saved keys are not lost."""
    path = default_config_path()
    old = home_config_dir() / "config.json"
    if path != old and not path.exists() and old.is_file() and CONFIG_DIR_ENV not in os.environ:
        adopt_config(old, path)
    return ConfigManager(path)


def adopt_config(old: Path, new: Path) -> None:
    """Move `old` to `new` (0600). A config already at `new` wins and `old` is left alone."""
    data = old.read_bytes()
    try:
        fd = os.open(new, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:  # created meanwhile, e.g. by another `desman`
        return
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        new.unlink(missing_ok=True)
        raise
    old.unlink()
    click.echo(f"Moved the configuration from {old} to {new}.", err=True)


SCHEMA_VERSION = 3

# Renamed to the provider's own variable name (schema 0 -> 1).
_RENAMED_KEYS = {"VERCEL_AI_GATEWAY_KEY": "AI_GATEWAY_API_KEY"}


def looks_like_secret(text: str) -> bool:
    """A long word without spaces: how keys look, and how labels ("work", "second account")
    do not. Labels are shown unmasked, so such a label is refused, and masked if found."""
    return len(text) >= 16 and not any(c.isspace() for c in text)


class ConfigError(Exception):
    """config.json exists but cannot be used. Never treated as "not configured": the wizard
    would then write over the keys in it."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path


class Permissions(BaseModel):
    """Set by the human in `desman config` only; the upper bound of what agents may do."""

    model_config = ConfigDict(extra="forbid")

    admin_rights: StrictBool = False
    fs_outside_access: StrictBool = False
    network_access: StrictBool = False
    shell_execution: StrictBool = False


BEHAVIOUR_CHARS = 500
"""The longest text the person may add about how the interlocutor talks."""


class Behaviour(BaseModel):
    """How the interlocutor talks, as the person set it (ROADMAP C.7): it holds whether or not
    the person lets it keep notes. None: as the interlocutor sees fit."""

    model_config = ConfigDict(extra="forbid")

    length: Literal["short", "detailed"] | None = None
    tone: Literal["formal", "informal"] | None = None
    style: Literal["simple", "technical"] | None = None
    custom: StrictStr = Field(default="", max_length=BEHAVIOUR_CHARS)
    """Anything else, in the person's words."""

    def is_set(self) -> bool:
        return self != Behaviour()


class ApiKey(BaseModel):
    """One key of a provider. A provider may have several (e.g. two accounts' free tiers)."""

    model_config = ConfigDict(extra="forbid")

    value: StrictStr = Field(min_length=1)
    label: StrictStr = ""


class Config(BaseModel):
    """The whole of config.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[3] = 3
    configured: StrictBool = False
    language: Literal[
        "en", "ru", "zh", "es", "pt", "de", "fr", "ja", "ko", "pl", "tr", "uk", "hi"
    ] = "en"
    permissions: Permissions = Field(default_factory=Permissions)
    # Secrets, by the provider's variable name; in the order they are tried.
    api_keys: dict[str, list[ApiKey]] = Field(default_factory=dict)
    settings: dict[str, StrictStr] = Field(default_factory=dict)  # not secret: hosts, account ids
    # The person lets the interlocutor study the conversation and keep notes about them
    # (ROADMAP C.7); off unless they said yes: it neither sees nor keeps any.
    personalize: StrictBool = False
    behaviour: Behaviour = Field(default_factory=Behaviour)

    def keys(self, env: str) -> list[ApiKey]:
        return list(self.api_keys.get(env, []))

    def add_key(self, env: str, value: str, label: str = "") -> bool:
        """Append a key; False (and nothing changes) if this provider already has it."""
        value = value.strip()
        label = label.strip()
        if not value:
            raise ValueError("an API key cannot be empty")
        if looks_like_secret(label) or (label and label in value):
            raise ValueError("the label looks like a key; a label is a name, like 'work'")
        current = self.api_keys.setdefault(env, [])
        if any(k.value == value for k in current):
            return False
        current.append(ApiKey(value=value, label=label))
        return True

    def remove_key(self, env: str, index: int) -> ApiKey:
        """Remove the key at `index` (0-based) of `env`."""
        current = self.api_keys.get(env, [])
        if not 0 <= index < len(current):
            raise IndexError(f"{env} has no key #{index + 1}")
        removed = current.pop(index)
        if not current:
            del self.api_keys[env]
        return removed


def upgrade(data: dict[str, Any]) -> dict[str, Any]:
    """Bring a parsed config.json up to SCHEMA_VERSION, one step at a time.

    0: written before there was a schema_version (`version: "0.1.0"`, `local_settings`).
    1: one key per provider, `api_keys: {ENV: "key"}`.
    2: a list of keys per provider, `api_keys: {ENV: [{"value": ..., "label": ...}]}`.
    3: `personalize` (off unless it says otherwise) and `behaviour`.
    """
    version = data.get("schema_version", 0)
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError(f"schema_version must be a whole number, not {version!r}")
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {version} is newer than this DESMAN understands ({SCHEMA_VERSION})"
        )
    data = dict(data)
    if version == 0:
        data.pop("version", None)
        data["settings"] = data.pop("local_settings", {})
        keys = data.get("api_keys")
        if isinstance(keys, dict):
            keys = dict(keys)
            for old, new in _RENAMED_KEYS.items():
                value = keys.pop(old, None)
                if value is not None and new not in keys:  # a key under the new name wins
                    keys[new] = value
            data["api_keys"] = keys
        version = 1
    if version == 1:
        keys = data.get("api_keys")
        if isinstance(keys, dict):
            # An empty string meant "no key"; anything else that is not a string is left for
            # validation to reject.
            data["api_keys"] = {
                env: [{"value": v}] if isinstance(v, str) else v
                for env, v in keys.items()
                if v != ""
            }
        version = 2
    if version == 2:
        version = 3  # `personalize` and `behaviour` have defaults
    if data.get("profile") in (True, False):
        # Written by a 3 before its release (62d084b): `profile`, on unless it said otherwise,
        # was never the person's consent, so `personalize` stays as the file says, off if unsaid.
        del data["profile"]
    data["schema_version"] = version
    return data


class ConfigManager:
    """Reads and writes config.json."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        # Only a directory we create gets 0700; an existing one (maybe chosen by the user via
        # DESMAN_CONFIG_DIR) is left alone. The file itself is always 0600 (save).
        self.config_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    def load(self) -> Config:
        """The saved config; an unconfigured one when there is no file. Raises ConfigError
        when the file cannot be read, is not JSON or does not fit the schema."""
        try:
            text = self.config_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return Config()
        except (OSError, UnicodeDecodeError) as e:
            raise ConfigError(self.config_path, f"cannot read it ({e})") from e
        try:
            data = json.loads(text)
        except ValueError as e:
            raise ConfigError(self.config_path, f"not valid JSON ({e})") from e
        if not isinstance(data, dict):
            raise ConfigError(self.config_path, "not a JSON object")
        try:
            return Config.model_validate(upgrade(data))
        except ValidationError as e:
            # Pydantic's own message quotes the rejected value, which may be a key.
            problems = "; ".join(
                f"{'.'.join(map(str, err['loc'])) or 'file'}: {err['msg']}"
                for err in e.errors(include_input=False, include_url=False)
            )
            raise ConfigError(self.config_path, problems) from e
        except ValueError as e:  # from upgrade(): about schema_version only
            raise ConfigError(self.config_path, str(e)) from e

    def is_configured(self) -> bool:
        return self.load().configured

    def save(self, config: Config) -> None:
        """Save configuration atomically, readable by the owner only (it holds API keys).

        Written to a private temporary file and renamed over the old one: a crash never
        leaves a half-written or world-readable config behind. A symlinked config (e.g. into
        a dotfiles directory) is updated at its target, so the link stays. The old file is
        made private before it is replaced, so another hard link to it does not keep the
        previous keys readable by others.
        """
        text = config.model_dump_json(indent=2)
        target = self.config_path.resolve() if self.config_path.is_symlink() else self.config_path
        tmp = target.with_name(f".{target.name}.{secrets.token_hex(6)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            if target.exists():
                os.chmod(target, 0o600)  # the old inode, shared by any other hard link
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def delete(self) -> bool:
        """Remove the config (keys included) and any temp file a crashed save left behind.
        False if there was nothing to remove. A symlinked config loses only the link: its
        target may be a shared dotfile, and is the user's to delete."""
        with self.locked():
            removed = self.config_path.is_symlink() or self.config_path.exists()
            if removed:
                self.config_path.unlink()
            for stray in self.config_path.parent.glob(f".{self.config_path.name}.*.tmp"):
                stray.unlink(missing_ok=True)
            return removed

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the config's lock: other DESMAN processes that change it wait meanwhile.
        Advisory (flock on a lock file next to it); a no-op where flock does not exist."""
        lock_path = self.config_path.parent / f".{self.config_path.name}.lock"
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)  # releases the lock

    @contextmanager
    def edit(self) -> Iterator[Config]:
        """Load, let the caller change, save: all under the lock, so two commands that add
        keys at the same time both keep theirs. Nothing is saved if the block raises."""
        with self.locked():
            config = self.load()
            yield config
            self.save(config)

    def snapshot(self) -> str | None:
        """The file's text as it is now (None if there is no file), to detect later changes."""
        try:
            return self.config_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def save_if_unchanged(self, config: Config, seen: str | None) -> None:
        """Save, unless the file changed since `seen` was taken (another command wrote it
        while, say, the wizard was open). Raises ConfigError then; nothing is written."""
        with self.locked():
            if self.snapshot() != seen:
                raise ConfigError(
                    self.config_path, "it changed while this was open; nothing was saved"
                )
            self.save(config)
