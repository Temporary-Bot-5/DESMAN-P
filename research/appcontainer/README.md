# AppContainer research prototype

This experiment accompanies ADR 0013, decision 6. Use an ordinary Windows test account or a disposable VM and run the supplied probe. Any proposal to use AppContainer for generated code needs its own ADR and independent review.

## Files

- `confine.py` launches a Windows executable in the `desman.research.sandbox` profile with zero capabilities. It grants the profile SID read/write/delete and directory traversal in the workspace, and read/execute in the private runtime. It verifies the AppContainer SID before resuming the process and places it in a Job with a 32-process limit, per-process and aggregate memory limits, kill-on-close, and selected UI restrictions. It retains up to 64 KiB per stream and prints a single JSON object. The default limits are 30 seconds and 256 MiB.
- `probe.py` emits one JSON line per check. The runner passes in host paths and fixture names as arguments because the child receives a minimal environment. The pytest check runs a tiny workspace test when pytest is available in the copied interpreter.
- `run_probe.ps1` copies the running interpreter's `sys.base_prefix` into `%LOCALAPPDATA%\desman-research\python`, stages the probe in `%LOCALAPPDATA%\desman-research\ws`, and provides a localhost listener and a host named pipe. It prints both the launcher JSON and a table of checks for each mode, then invokes cleanup.

## Run on Windows

Use a regular installed CPython 3.11+ and a recent Windows version supporting LPAC. From the repository root:

```powershell
.\research\appcontainer\run_probe.ps1
# Or select a particular interpreter.
.\research\appcontainer\run_probe.ps1 -Python 'C:\Python312\python.exe'
```

The runner leaves both private directories for inspection. Move them aside after cleanup before another run. Keep the launcher directory outside both ACL trees. Use dedicated local NTFS directories with no reparse points or hard-linked files. Keep them free of concurrent changes by other host processes during the experiment. Reserve the profile name and run a single copy of the experiment per Windows user account.

To exercise the launcher directly with the staged runtime and workspace:

```powershell
$root = Join-Path $env:LOCALAPPDATA 'desman-research'
python .\research\appcontainer\confine.py --workspace "$root\ws" --runtime "$root\python" --timeout 10 --memory 256 --lpac -- python -B -c "print('hello')"
python .\research\appcontainer\confine.py --cleanup
```

Bare command names resolve inside the runtime. You can also pass an absolute executable path. The launcher supplies `SystemRoot`, `PATH` pointing to the runtime, and `TEMP` and `TMP` pointing to the workspace. Child exit status is reported inside the result JSON. A launcher failure exits nonzero and prints its diagnostic to stderr.

Cleanup removes the recorded explicit ACEs from the two roots and lets Windows update inherited entries on descendants. It deletes the reserved profile, including a profile reused by this experiment. If the launcher is interrupted, keep the original directories and `acl-journal.json` in place and rerun `--cleanup` with the same host account. The journal records intent before applying an ACL change and refuses to remove an ambiguous duplicate or edit a replacement directory.

## Interpret a run

Look for a complete set of 17 rows and a successful workspace write in both modes. A negative row may reflect a missing path, a missing package, an unavailable service, or a denial. Compare the details with the host's setup. Registry reads may refer to the container's view. Process enumeration and opening a process are separate observations, so inspect both counts. Starting a child is recorded separately from its exit code.

Windows also decides access via the existing ACLs of system resources, the AppContainer profile, and network policy. Normal AppContainers may access resources granted to All Application Packages. LPAC opts out of that group. [Microsoft AppContainer and LPAC launch documentation](https://learn.microsoft.com/en-us/windows/win32/secauthz/implementing-an-appcontainer)

## Check on Linux

```sh
python3 -m py_compile research/appcontainer/confine.py research/appcontainer/probe.py
python3 research/appcontainer/confine.py --help
```

## Results

The assessment, with the results from Windows 11 (desman-test, 2026-11-21), is in
`docs/research/appcontainer.md`. The `DESMAN_RESEARCH_ENV` (extra variable names) and
`DESMAN_RESEARCH_NO_UI` (Job UI limits off) environment toggles of `confine.py` exist for that
investigation.
