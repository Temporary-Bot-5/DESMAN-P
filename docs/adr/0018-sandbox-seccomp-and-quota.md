# ADR 0018: A system-call filter and a workspace quota for the sandbox

Status: accepted (owner, 2026-11-25: wants options for bwrap without seccomp; ROADMAP S.2)

## Context

Generated code runs under bubblewrap (ADR 0001, ROADMAP 3.3): new namespaces, no network, no
capabilities, a read-only `/usr`, the workspace at `/workspace`, rlimits and a timeout. What it
can still reach is the kernel's whole system-call interface. The local privilege escalations of
recent years mostly go through interfaces an ordinary program never calls: BPF, userfaultfd,
perf events, io_uring, keyrings, ptrace, the new mount API. Docker's default profile refuses
them; our bwrap sandbox (Linux and WSL) did not. The second gap is the disk: each file written is
capped (`RLIMIT_FSIZE`), their number is not, so a run could fill the person's disk.

Options looked at for the filter:

1. **libseccomp** (Python bindings, or `scmp_bpf_export` at install time): the usual way, but a
   native dependency that is not on PyPI as a wheel for every platform, and a second thing that
   must be installed for the sandbox to work.
2. **A filter written here** as a classic BPF program (`struct sock_filter`): a deny list is a
   few dozen instructions; bwrap loads it from a descriptor (`--seccomp FD`, in bwrap since 0.1).
3. **Landlock** or an allow list: an allow list breaks unknown programs (pytest plugins, a
   compiler); Landlock restricts files, which the namespaces already do.

For the quota: a size-limited mount (tmpfs, loop device, overlay with a quota) needs privileges
or keeps the workspace in memory; disk quotas need the file system's support and root.

## Decision

1. **`core/seccomp.py` writes the filter** (option 2), no new dependency. It checks the
   architecture first: a call of another ABI (32-bit x86 through `int 0x80`, ARM compat) kills
   the process, x32 numbers on x86-64 are refused; then the denied calls return `EPERM` (as
   Docker does, so a program that probes gets an error, not a crash); everything else is
   allowed. Numbers are listed for x86-64 and aarch64; a test compares them with the kernel's
   headers when they are installed, and a small BPF interpreter in the tests runs the program.
   On another machine, or under a 32-bit userland on a 64-bit kernel (the kernel's name is not
   the programs' ABI), there is no filter and the sandbox runs as before (it is defence in
   depth, not the boundary).
2. **The denied calls** (`seccomp.DENIED`): bpf, userfaultfd, perf_event_open, io_uring_*,
   keyctl/add_key/request_key, ptrace, process_vm_*, kcmp, pidfd_getfd, process_madvise,
   migrate_pages, move_pages, every
   mount call and the new mount API, chroot, setns, unshare, handles (name_to_handle_at,
   open_by_handle_at), fanotify_init, kexec_*, module loading, reboot, swap, acct, quotactl,
   syslog, vhangup, the clock and host-name setters, and on x86 iopl, ioperm, modify_ldt,
   uselib, _sysctl. `personality` and the NUMA policy calls stay allowed (numeric libraries use
   them), and so do `adjtimex` and `clock_adjtime` (`ntp_gettime` reads the clock with them;
   changing it needs a capability the sandbox does not have); `clone` stays allowed (`--disable-userns` already stops new user namespaces).
3. **Where it applies.** bwrap on Linux gets it through a temporary file's descriptor passed to
   the child (`pass_fds`). The availability probe runs with the filter too, so a kernel that
   refuses it makes the sandbox unavailable instead of silently unfiltered. In WSL no
   descriptor can be passed through `wsl.exe`: a fixed `busybox sh` line writes the filter,
   given as octal escapes in one argument, to a temporary file in the distribution, opens it as
   fd 3 and `exec`s bwrap with `--seccomp 3`. Docker keeps its own default profile; WebAssembly
   has no system calls.
4. **A workspace quota** (`Limits.workspace_files` = 10 000 entries, `workspace_bytes` =
   512 MiB), measured by code on the host after every run on every backend (links never
   followed, counting stops past the quota; a folder that cannot be listed counts as over,
   since it could hide anything). Over it, the run's result says so
   (`workspace_full`, with the counts) and a subtask that must run code is not done on such a
   run. Under bwrap the run is also watched about once a second and killed once the workspace
   grows past the quota, or past what it already held, so a full workspace can still be
   cleaned with `rm`; the watch goes on after the program closes its output. A count misses
   what a folder made unreadable holds and files deleted while kept open, so the watch also
   stops a run once the file system under the workspace has lost more free space or inodes
   than twice the quota since it began (`statvfs`). The count walks folders by descriptor
   (`O_NOFOLLOW`, relative to the parent), so code running at the same time cannot swap a
   folder for a link and lead it outside; on Windows junctions are skipped by their
   attribute. The reason a run was stopped goes back with its result. Nothing is deleted by
   DESMAN.

## Consequences

- A program that needs a denied call (a debugger, `strace`, `perf`, io_uring-based servers)
  gets `EPERM` in the sandbox. None is needed by the tasks DESMAN runs.
- The quota is measured, not enforced by the file system: a run can overshoot it by what it
  writes in about a second; other backends than bwrap are only measured after the run.
- The disk's free space also falls with what other programs write (another task's sandbox,
  a download): a run may be stopped for writes it did not make; its result says what was
  seen. Subtasks of one task share the workspace, so one may find it over the quota because
  of another; it is then asked to clean up like any other run.
- `core/seccomp.py` is protected, like the sandbox.
