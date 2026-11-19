# ADR 0013: Working without a preinstalled Docker

Status: accepted (owner, 2026-11-19: "keep all the options"; "offer to install Docker during
`config`: after Y DESMAN installs it itself with the command for the OS")

## Context

Two features need Docker today: the code sandbox on Windows and macOS (ADR 0012) and the local
SearXNG search engine (ADR 0011). Docker cannot be embedded in DESMAN: containers need a Linux
kernel, which on Windows and macOS comes from a virtual machine that Docker Desktop (or WSL2)
installs, and Docker Desktop's license does not allow redistributing it. For a person on Windows
Docker Desktop is heavy (WSL2, administrator rights, sometimes a reboot, gigabytes on disk) and
paid in larger companies, and both features are unproven on a real Windows machine.

Probes on 2026-11-19:
- **SearXNG without Docker on Windows 11 Home** (desman-test, Python 3.12, source archive of
  SearXNG master, its `requirements.txt` in a venv): it fails on `import pwd`; with a three-line
  `pwd` stand-in first on `PYTHONPATH` it serves JSON searches (49 results for one query). Two
  engines fail to load (ahmia, torch: Tor), harmless. The archive has a file whose name contains
  `:` (a uWSGI socket config), which cannot be extracted on Windows and is not needed.
- **CPython built for WASI under `wasmtime`** (pip package `wasmtime`, ~30 MB; CPython 3.15.2
  WASI build from `brettcannon/cpython-wasi-build`, ~16 MB zipped, 44 MB unpacked): no network
  (no `getaddrinfo`), no files but the directories handed in, no processes (`subprocess` raises).
  pytest (pure-Python wheels, 7.5 MB) passes and fails as it should with `-p no:cacheprovider
  --capture=sys -p no:faulthandler -p no:logging` and `os.chdir` into the workspace first;
  about 6 s per run, mostly start-up. Packages with C code (numpy …) cannot be used.
- Windows 11 Home has `winget`; WSL is not enabled by default (needs administrator rights).

## Decision

DESMAN keeps every option. Each feature gets an order of backends; the first one available is
used, and `status` commands say which one and why.

1. **Installing Docker is offered, never required (T.6u).** `desman docker [status]` and
   `desman docker install [--yes]` (module `docker_cli.py`, bodies shared with the wizard and the
   chat's catalog). Without `--yes` it only shows the plan: the exact commands, that they need
   administrator rights, the download size, a possible reboot, and Docker Desktop's license
   terms (free for personal use, education and small companies). With `--yes` it runs them
   with fixed argument lists, the person's terminal attached (sudo's password, Windows' UAC
   window and package managers' questions reach the person directly); the first failure stops
   it with the command's own message. Plans:
   - Windows: `winget install --exact --id Docker.DockerDesktop --source winget
     --accept-package-agreements --accept-source-agreements`; then it starts Docker Desktop and
     says that its agreement must be accepted in its window and that Windows may ask for a
     reboot. No winget → the download link.
   - macOS: `brew install --cask docker-desktop`, then `open -a Docker`. No Homebrew → the link.
   - Linux, by `/etc/os-release`: the distribution's package (`docker.io` with apt, `moby-engine`
     with dnf, `docker` with pacman and zypper) plus `bubblewrap` when `bwrap` is missing (it is
     the Linux sandbox), `systemctl enable --now docker`, and adding the person to the `docker`
     group (the plan says plainly that this group is equal to administrator rights and that a new
     login is needed). An unknown distribution → the link. Never a piped install script.
   - Already installed and answering → nothing to do; installed but stopped → only the start.
   Afterwards it checks `docker info` and names the next steps: `desman sandbox setup --yes`
   (not on Linux) and `desman search setup --yes`. The wizard gets a step **Docker** (only when
   `docker` is missing or does not answer) that explains what Docker gives (code checked by
   running it on Windows/macOS, the whole web through local SearXNG), shows the plan and asks
   y/N; on "yes" it runs the plan, then offers the sandbox image and SearXNG. In the chat the
   command is a change shown in the window; it runs only where a terminal is attached, elsewhere
   the person gets the command to run.
2. **SearXNG without Docker (T.6v).** When Docker is not there, `desman search setup` installs
   SearXNG as a Python program: a source archive pinned by commit and SHA-256, a venv of its own
   in DESMAN's data directory, the `pwd` stand-in on Windows, bound to `127.0.0.1`. DESMAN starts
   it when a search needs it and it is not running, and `search start|stop|status` manage it.
   Docker stays first where it answers. Officially SearXNG supports only Linux, so a Windows
   problem falls back to the other engines with a plain warning, never a failed task.
3. **Python sandbox in WebAssembly (T.6w).** A third `Sandbox` backend, `wasi`, used on Windows
   and macOS when Docker is not available: `wasmtime` running a pinned CPython WASI build
   (fetched by `desman sandbox setup` with the person's yes, checked by SHA-256), the workspace
   as the only writable directory, the standard library and pytest read-only, a time limit
   (epoch interruption), a memory limit (store limiter), output limits. It runs only `python …`
   and `python -m pytest …`; anything else, or an import of a package it cannot have, is an
   ordinary failed run with a plain reason the executor sees. Docker stays first where it
   answers, because it runs any command and packages with C code. `wasmtime` becomes a
   dependency on Windows and macOS only (an environment marker), justified by this ADR.
4. **WSL backend (T.6x).** On Windows with WSL2 enabled (or after `desman sandbox setup --wsl`,
   which asks for administrator rights): DESMAN imports a small pinned Linux root file system as
   its own WSL distribution and runs code there under bubblewrap, exactly as on Linux; SearXNG
   can run there too. Order on Windows: Docker, WSL, WASI.
5. **Search lent through desman-global (G.13).** A node that has a working SearXNG may lend
   searches like it lends models (quota per day, journaled, never the person's own queries
   revealed to the operator beyond what a lent job needs). A node without any whole-web engine
   may use it when the person turned this on.
6. **Windows' own isolation (AppContainer, T.6y) is research only:** a written assessment and a
   prototype outside the product; it becomes a backend only after its own ADR and an
   independent review, because an error there would run code with the person's rights.

Never any fallback to running generated code unconfined, whatever is missing.

## Consequences

- A person on Windows without anything installed gets checked Python code (WASI) and the whole
  web (native SearXNG) after `desman sandbox setup` / `desman search setup`; Docker or WSL adds
  any command and C packages.
- More backends to test: each gets tests with fakes and a real run in CI where it exists (WASI
  and native SearXNG run on every platform; Docker on Linux CI; WSL only on desman-test).
- Order of work: T.6u, T.6v, T.6w, T.6x, G.13, T.6y.
