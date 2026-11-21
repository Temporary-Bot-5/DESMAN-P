# AppContainer as a sandbox for generated code (T.6y, research)

ADR 0013, decision 6: Windows' own isolation is studied, not shipped. This is the written
assessment. The prototype is in `research/appcontainer/` and is not part of DESMAN. It becomes a
backend only after an ADR of its own and an independent review, because a mistake in it runs
generated code with the person's own rights.

## Question

Can DESMAN confine generated code on Windows with AppContainer and Job objects alone, without
Docker, WSL or WebAssembly, as well as bubblewrap does on Linux?

## Method

- `confine.py` (ctypes, standard library only) does the following:
  - creates the AppContainer profile `desman.research.sandbox` with **no capabilities**;
  - grants the container's SID access to two DESMAN-owned folders only: the workspace
    (read/write) and a *private copy* of Python (read/execute). No ACL on the person's own
    folders is changed, and `--cleanup` removes exactly the entries it added and the profile;
  - starts the command suspended, inside a Job object, then resumes it. The Job sets
    kill-on-close, at most 32 processes, a memory limit, and UI limits (desktop, clipboard,
    global atoms);
  - gives the command a minimal environment and pipes its output.
- `probe.py` runs inside the container and reports, check by check, what it can do.
- The test ran on desman-test, Windows 11 build 26200, with CPython 3.12.16 (a uv build). It ran
  twice, with the UI limits on and off, in normal AppContainer and LPAC (Less Privileged
  AppContainer) modes. The cleanup was verified: no container entries left on the workspace and
  no ACL journal.

## Results (normal AppContainer, no capabilities)

| Check | Result |
| --- | --- |
| List `%USERPROFILE%\Documents`, `%USERPROFILE%\.ssh` | denied |
| Write in `%USERPROFILE%`, the user's `%TEMP%`, `C:\Users\Public` | denied |
| List `C:\` | denied |
| Read `System32\drivers\etc\hosts` | **allowed** (system files readable by design) |
| Enumerate `HKCU\Software` | **allowed**: the same 15 subkeys as outside |
| Outbound TCP 1.1.1.1:80 | denied (WinError 10013) |
| DNS lookup | denied |
| Connect to a localhost listener of the person | denied (no `privateNetworkClientServer`, no loopback exemption) |
| Open the person's named pipe | denied |
| Start `cmd.exe /c echo` | allowed: the child inherits the container and the Job |
| Start PowerShell | starts, fails at once (0xC0000142, DLL initialisation) |
| Write in the workspace | allowed (as intended) |
| `import sqlite3, lzma, decimal, _multiprocessing` | work |
| `import ctypes`, `import ssl` | **fail**: "DLL initialization routine failed", with or without UI limits |
| LPAC mode | **the process does not start** (0xC0000022, access denied) |

Two practical findings from building it:

- `CreateProcessW` for an AppContainer fails with ERROR_ENVVAR_NOT_FOUND (203) unless the
  environment block contains `LOCALAPPDATA`. Windows derives the container's own folder from it.
- The interpreter must be readable by the container. Copying it into a DESMAN-owned folder
  avoids touching the person's ACLs, but costs ~100 MB and a copy per Python update.

## Assessment

What AppContainer gives, it gives well. The person's files, the network (including localhost and
DNS), other sessions' pipes and the rest of the disk are closed by the kernel's access checks,
and child processes stay inside. This is comparable to bubblewrap for the filesystem and the
network.

The gaps:

1. **Broad read access.** Anything granted to `ALL APPLICATION PACKAGES` stays readable: system
   files and much of `HKCU`. Some programs keep tokens or paths there. LPAC removes this
   group's grants, and this is exactly the mode the prototype could not start. LPAC needs
   grants to `ALL RESTRICTED APPLICATION PACKAGES` (or the container) on every file the runtime
   loads, and possibly extra capabilities. It is not finished.
2. **Python is crippled.** Inside the container `ctypes` and `ssl` fail to load (the cause is not
   found yet: a DLL's initialisation refuses, independent of the Job's UI limits). Many packages
   need one of them. PowerShell does not start.
3. **No equivalent of `--unshare-all`.** There is no PID or IPC namespace. Kernel objects in the
   session namespace are protected only by their ACLs. The Job limits processes and memory but
   has no file-size limit (bubblewrap's prlimit has one).
4. **Risk of the backend itself.** The confinement is built by DESMAN's own Win32 code (profile,
   SIDs, ACLs, attribute lists). A bug there, for example a forgotten security-capabilities
   attribute, silently gives the code the person's full rights. Docker, WSL and WASI fail
   closed instead: without the runtime nothing runs.

## Recommendation

Do not make AppContainer a backend now. Keep the Windows order **Docker → WSL → WASI** (T.6r,
T.6x, T.6w). They already cover the cases AppContainer would serve. WASI needs nothing from the
system for Python code, and WSL runs any command under bubblewrap.

Reconsider only if a real group of participants (T.7) has neither WSL2 (no virtualisation) nor
enough for WASI. Then the next steps would be:

1. Get LPAC running. Find the minimal grants, so that `ALL APPLICATION PACKAGES` reads (HKCU)
   are gone.
2. Find why `_ctypes`/`_ssl` fail. A test with the python.org build as well as the uv build
   would tell whether it is the build or AppContainer.
3. Make the self-check fail closed and complete. The prototype already reads the suspended
   child's `TokenAppContainerSid` and compares it before resuming. A backend would also check
   that the child has zero capabilities and that it is in the Job, and refuse otherwise.
4. Write an ADR and get an independent review.
