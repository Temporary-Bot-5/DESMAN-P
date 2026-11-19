# ADR 0012: Generated code in a Docker container where bubblewrap does not exist

Status: accepted (owner's choice, 2026-11-19: "Docker")

## Context

`core/sandbox.py` runs generated code only inside bubblewrap (ROADMAP 3.3), and bubblewrap
exists only on Linux. On Windows (and macOS) the sandbox is *unavailable*: `run_command` is
never offered, a subtask marked `runs_code` waits for the person at once, and "write a script and
check it" tasks cannot be done there at all. The pilot (T.6b, T.7) has Windows participants.

Options considered: WSL2 + bubblewrap (the same isolation, but WSL setup, a distribution and
path/encoding translation between Windows and WSL); a Windows-native jail (AppContainer, job
objects: no simple way to take the network away, a large amount of Win32 code to get right);
Docker. DESMAN already uses Docker for SearXNG (ADR 0011), Docker Desktop runs on Windows Home
and macOS, and a Linux container gives the same kind of isolation bubblewrap gives.

## Decision

1. **Two backends, one contract.** `Sandbox` keeps its interface (`unavailable()`, `run()`,
   `Outcome`, `Limits`). On Linux it is bubblewrap, as now; there is no fallback to Docker on
   Linux (a missing bwrap stays "unavailable" with its reason). On every other platform it is
   Docker. Never any fallback to running code unconfined.
2. **The image is pinned:** an official `python:3.12-slim` image by its `sha256` digest, written
   in `sandbox.py` (a protected file; a new digest is a reviewed commit). The container never
   pulls (`--pull never`). The image is fetched only by the person: `desman sandbox setup`
   (shows the size, downloads with `--yes`; the wizard can offer a confirmation when the
   image is missing). `desman sandbox status` says which backend is used and why it is or is not
   available. Both have catalog entries for the chat (setup: a window).
3. **What the container gets:** `--network none`, `--read-only` root file system, `--cap-drop
   ALL`, `--security-opt no-new-privileges`, an unprivileged user, `--pids-limit`, `--memory`
   (= swap, no extra), `--cpus 1`, `--ulimit` for CPU time, file size, open files and core dumps,
   a size-limited `/tmp` tmpfs, the task's workspace bound at `/workspace` (the working directory
   and `HOME`) and nothing else of the machine; only the fixed environment the bwrap backend sets
   (`-e` per variable; the docker CLI's own environment never reaches the container). pytest and
   its pure-Python runtime packages are bound read-only under `/opt/desman/python` as in the bwrap
   backend. Every container has a random name, the label `desman.sandbox`, and an init process
   to reap child processes.
4. **Timeout and leftovers:** killing the `docker` client does not stop a container, so a timeout
   (or Ctrl+C) runs `docker kill NAME`. After a normal client exit, container inspection reports
   whether it started and gives the program exit code. Cleanup always attempts `docker rm -f NAME`.
   The probe uses the daemon clock to remove leftover containers after the larger of the default
   and configured timeouts plus ten minutes.
   Output is read by threads (pipes cannot be `select`ed on Windows), with the same head/tail
   limits.
5. **Availability is checked, not assumed:** `docker` on PATH, the daemon answers, the server runs
   *Linux* containers (Docker Desktop in Windows-containers mode → unavailable, with that reason),
   the pinned image is present, and an empty container with all the flags above starts and exits 0.
6. **Tests:** the argument list and the availability reasons are unit-tested with a stub `docker`
   on every platform; real runs (no network, read-only root, limits, timeout kill, workspace
   writes) are tested where a Linux Docker daemon exists (Linux CI) and skipped elsewhere. Windows
   CI runners have no Linux containers, so the Windows path is checked by hand on the test box
   (`docs/sandbox-docker.md`).

## Consequences

- Windows/macOS participants can run code once they install Docker Desktop and run `desman
  sandbox setup --yes`; without it nothing changes from today (code does not run, they are told why
  and how to enable it).
- The isolation boundary on those platforms is Docker's (a Linux VM under Docker Desktop), not
  bubblewrap's. The person's files outside the workspace and the configuration are not mounted.
- A new protected digest to keep current; updating it is a person's commit.
