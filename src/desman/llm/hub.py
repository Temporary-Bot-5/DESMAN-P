"""Models from Hugging Face for the local servers (ROADMAP 4.5b).

The person names a model (`author/model:quant`, `hf.co/...`, or its page's address); `inspect`
asks the Hugging Face API what it is: its GGUF files for that quantization, their size, and
whether the model is gated. Only GGUF: a local server loads it as data, nothing in the
repository runs, and neither `transformers` nor `torch` is needed. `fits` compares the size
with the machine's free memory (RAM and video memory).

The download is the server's own: Ollama pulls `hf.co/author/model:quant` (`pull`, its API);
llama.cpp downloads it when started with `-hf author/model:quant` (`llama_server_command`).
Nothing here downloads a model by itself, and nothing here is called unless the person said
"yes" (the CLI command they typed, the wizard, or the window of the chat).

A Hugging Face token (`HF_TOKEN`, kept like a provider's key) is sent only to huggingface.co,
for a private or gated model; never to a local server, never to a model.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx

HUB = "https://huggingface.co"
DEFAULT_QUANT = "Q4_K_M"
"""What Ollama picks when a name has no quantization, and a good size/quality balance."""
OVERHEAD = 1.2
"""Memory a model takes for its size on disk: the weights, plus the context and the server."""
TIMEOUT = 20.0

_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_QUANT = re.compile(r"^[A-Za-z0-9][\w.-]*$")
# A quantization in a file name: Q4_K_M, IQ3_XS, Q8_0, F16, BF16...
_QUANT_IN_NAME = re.compile(
    r"(?<![A-Za-z0-9])(I?Q\d[A-Za-z0-9_]*|BF16|F16|F32)(?![A-Za-z0-9])", re.I
)
_SPLIT = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.I)


@dataclass(frozen=True)
class Ref:
    """A model on Hugging Face and the quantization wanted (None: the default)."""

    repo: str
    quant: str | None = None

    @property
    def spec(self) -> str:
        """`author/model:quant`, as llama.cpp's `-hf` takes it."""
        return self.repo if self.quant is None else f"{self.repo}:{self.quant}"

    @property
    def ollama(self) -> str:
        """The name Ollama pulls and then serves it under."""
        return f"hf.co/{self.spec}"

    @property
    def page(self) -> str:
        return f"{HUB}/{self.repo}"


def parse(text: str) -> Ref:
    """`author/model[:quant]`, with or without `hf.co/` or `https://huggingface.co/` in front.
    Raises ValueError for anything else."""
    text = text.strip()
    if "://" in text:
        parts = urlsplit(text)
        if parts.hostname not in ("huggingface.co", "hf.co", "www.huggingface.co"):
            raise ValueError("not a Hugging Face address")
        text = parts.path.strip("/")
        # A file's or a tree's address: the repository is the first two parts.
        text = "/".join(text.split("/")[:2])
    for prefix in ("hf.co/", "huggingface.co/"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
    repo, _, quant = text.partition(":")
    dots = repo.rsplit("/", 1)[-1] in (".", "..")
    if not _REPO.match(repo) or dots or (quant and not _QUANT.match(quant)):
        raise ValueError("expected author/model or author/model:quantization")
    return Ref(repo, quant or None)


class Problem(StrEnum):
    """Why a model cannot be used: the CLI words each for the person."""

    NOT_FOUND = "not_found"  # no such model, or a private one without a token for it
    NO_ACCESS = "no_access"  # the token may not read it
    NOT_GGUF = "not_gguf"
    NO_QUANT = "no_quant"  # the quantization asked for is not there
    CHOOSE_QUANT = "choose_quant"  # no default one: the person has to pick
    LICENSE = "license"  # gated: the license is not accepted for this token yet
    TOKEN = "token"  # gated: a token is needed
    UNREACHABLE = "unreachable"


class HubError(Exception):
    def __init__(self, problem: Problem, ref: Ref, quants: tuple[str, ...] = ()) -> None:
        super().__init__(f"{ref.spec}: {problem.value}")
        self.problem = problem
        self.ref = ref
        self.quants = quants
        """The quantizations the model has (for NO_QUANT and CHOOSE_QUANT)."""


@dataclass(frozen=True)
class Gguf:
    """A model's GGUF files for one quantization."""

    ref: Ref
    """With the quantization found (never None)."""
    files: tuple[str, ...]
    size: int
    """Bytes, all parts of a split file together."""
    gated: bool
    quants: tuple[str, ...]
    """Every quantization the model has."""

    @property
    def memory(self) -> int:
        """Bytes of memory it needs to run, roughly."""
        return int(self.size * OVERHEAD)


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def inspect(ref: Ref, client: httpx.Client, token: str | None = None) -> Gguf:
    """What `ref` is on Hugging Face; raises `HubError`. Downloads nothing but the listing."""
    url = f"{HUB}/api/models/{quote(ref.repo)}"
    try:
        response = client.get(
            url, params={"blobs": "true"}, headers=_headers(token), timeout=TIMEOUT
        )
    except httpx.HTTPError:
        raise HubError(Problem.UNREACHABLE, ref) from None
    if response.status_code in (401, 404):
        raise HubError(Problem.NOT_FOUND, ref)
    if response.status_code == 403:
        raise HubError(Problem.NO_ACCESS, ref)
    if response.status_code != 200:
        raise HubError(Problem.UNREACHABLE, ref)
    try:
        data = response.json()
    except ValueError:
        raise HubError(Problem.UNREACHABLE, ref) from None
    siblings = data.get("siblings") if isinstance(data, dict) else None
    files = {
        str(s["rfilename"]): int(s.get("size") or 0)
        for s in siblings or []
        if isinstance(s, dict)
        and str(s.get("rfilename", "")).lower().endswith(".gguf")
        # A vision model's projector is not the model.
        and "mmproj" not in str(s.get("rfilename", "")).lower()
    }
    if not files:
        raise HubError(Problem.NOT_GGUF, ref)
    by_quant: dict[str, list[str]] = {}
    for name in files:
        found = _QUANT_IN_NAME.search(_SPLIT.sub(".gguf", name.rsplit("/", 1)[-1]))
        if found:
            by_quant.setdefault(found[1].upper(), []).append(name)
    quants = tuple(sorted(by_quant))
    if ref.quant is not None:
        wanted = ref.quant.upper()
        if wanted not in by_quant:
            raise HubError(Problem.NO_QUANT, ref, quants)
    elif DEFAULT_QUANT in by_quant:
        wanted = DEFAULT_QUANT
    elif len(by_quant) == 1:
        wanted = quants[0]
    else:  # several, or none named (the servers find a file by its quantization's name)
        raise HubError(Problem.CHOOSE_QUANT, ref, quants)
    chosen = tuple(sorted(by_quant[wanted]))
    return Gguf(
        Ref(ref.repo, wanted),
        chosen,
        sum(files[f] for f in chosen),
        _gated(data),
        quants,
    )


def _gated(data: object) -> bool:
    gated = data.get("gated") if isinstance(data, dict) else None
    return gated not in (None, False, "false")


def check_access(model: Gguf, client: httpx.Client, token: str | None) -> None:
    """For a gated model: may this token download it? Asks for the first file's headers only
    (no redirect followed, nothing downloaded). Raises `HubError` (TOKEN, LICENSE)."""
    if not model.gated:
        return
    if not token:
        raise HubError(Problem.TOKEN, model.ref)
    url = f"{HUB}/{quote(model.ref.repo)}/resolve/main/{quote(model.files[0])}"
    try:
        response = client.head(url, headers=_headers(token), timeout=TIMEOUT)
    except httpx.HTTPError:
        raise HubError(Problem.UNREACHABLE, model.ref) from None
    if response.status_code == 401:
        raise HubError(Problem.TOKEN, model.ref)
    if response.status_code == 403:
        raise HubError(Problem.LICENSE, model.ref)


# ── The machine's memory ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Memory:
    ram: int | None
    """Bytes of RAM available now (None: unknown)."""
    video: int | None
    """Bytes of video memory free (None: no GPU found, or unknown)."""

    @property
    def total(self) -> int | None:
        if self.ram is None:
            return None
        return self.ram + (self.video or 0)


def memory() -> Memory:
    return Memory(_ram(), _video())


def _ram() -> int | None:
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        text = ""
    found = re.search(r"^MemAvailable:\s+(\d+) kB", text, re.M)
    if found:
        return int(found[1]) * 1024
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def _video() -> int | None:
    """Free video memory: NVIDIA's own tool (a fixed command DESMAN runs, nothing generated),
    or the AMD driver's counters."""
    tool = shutil.which("nvidia-smi")
    if tool is not None:
        try:
            done = subprocess.run(
                [tool, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return sum(int(line) for line in done.stdout.split()) * 1024 * 1024
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    free = 0
    for card in Path("/sys/class/drm").glob("card*/device"):
        try:
            total = int((card / "mem_info_vram_total").read_text())
            used = int((card / "mem_info_vram_used").read_text())
        except (OSError, ValueError):
            continue
        free += max(0, total - used)
    return free or None


def fits(model: Gguf, machine: Memory) -> bool | None:
    """Whether the model has room to run (None: the memory is unknown)."""
    if machine.total is None:
        return None
    return model.memory <= machine.total


# ── The download, by the local server ───────────────────────────────────────


@dataclass(frozen=True)
class Step:
    """How a pull goes: Ollama's status line and, while a file comes, its bytes."""

    status: str
    completed: int | None = None
    total: int | None = None


class PullFailed(Exception):
    """Ollama refused or broke off the pull; the message is its own (English)."""


PULL_TIMEOUT = httpx.Timeout(TIMEOUT, read=600.0)
"""A pull reports progress as it goes; ten minutes without a line is a stalled one."""


def pull(
    host: str, ref: Ref, client: httpx.Client, on_step: Callable[[Step], None] | None = None
) -> None:
    """Has Ollama at `host` (its address, without /v1) download `ref`. Raises `PullFailed`."""
    body = {"model": ref.ollama, "stream": True}
    try:
        with client.stream(
            "POST", f"{host.rstrip('/')}/api/pull", json=body, timeout=PULL_TIMEOUT
        ) as response:
            if response.status_code != 200:
                response.read()
                raise PullFailed(_ollama_error(response.text) or f"HTTP {response.status_code}")
            last = ""
            for step in _steps(response.iter_lines()):
                last = step.status
                if on_step is not None:
                    on_step(step)
            if last != "success":
                raise PullFailed(last or "the pull ended without success")
    except httpx.HTTPError as e:
        raise PullFailed(type(e).__name__) from None


def _steps(lines: Iterator[str]) -> Iterator[Step]:
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("error"):
            raise PullFailed(str(data["error"])[:300])
        completed, total = data.get("completed"), data.get("total")
        yield Step(
            str(data.get("status", "")),
            completed if isinstance(completed, int) else None,
            total if isinstance(total, int) else None,
        )


def _ollama_error(text: str) -> str:
    try:
        data = json.loads(text)
    except ValueError:
        return text[:300].strip()
    return str(data.get("error", ""))[:300] if isinstance(data, dict) else ""


def llama_server_command(ref: Ref, port: int = 8080) -> str:
    """How to start llama.cpp's server with `ref`: it downloads the model on its first start
    (a gated one with the token in the HF_TOKEN variable) and serves it under that name."""
    return f"llama-server -hf {ref.spec} --alias {ref.spec} --port {port}"


def size_text(size: int) -> str:
    """Bytes for people: "4.7 GB", "820 MB" (decimal units, as Hugging Face shows them)."""
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.1f} GB"
    return f"{max(1, round(size / 1_000_000))} MB"
