"""A system-call filter for the bubblewrap sandbox (ROADMAP S.2, ADR 0018).

Namespaces, a read-only `/usr` and dropped capabilities already keep generated code away from
the person's files and the network. The kernel interfaces it can still reach are what a bug in
the kernel would be exploited through: BPF, userfaultfd, perf events, io_uring, keyrings,
tracing other processes, mounting, loading modules and the like. Ordinary programs and pytest
never need them, so the filter refuses them (`EPERM`, as Docker's default profile does) and
lets everything else through.

The filter is a classic BPF program written here, with no library: bwrap loads it from a file
descriptor (`--seccomp FD`) just before it runs the command, and it holds for every process
the command starts. A program built for another ABI on the same machine (32-bit x86 through
`int 0x80`, x32) would use other system-call numbers, so a call of any architecture but the
machine's own is refused too.

`program()` is None on an architecture whose numbers are not listed here: the sandbox then
runs as before, without the filter.
"""

from __future__ import annotations

import contextlib
import errno
import platform
import struct
import tempfile
from collections.abc import Iterator
from typing import Final

# Classic BPF (linux/filter.h, linux/bpf_common.h) and seccomp (linux/seccomp.h).
_LD_W_ABS: Final = 0x20  # BPF_LD | BPF_W | BPF_ABS
_JEQ_K: Final = 0x15  # BPF_JMP | BPF_JEQ | BPF_K
_JGE_K: Final = 0x35  # BPF_JMP | BPF_JGE | BPF_K
_RET_K: Final = 0x06  # BPF_RET | BPF_K
_NR: Final = 0  # offsetof(struct seccomp_data, nr)
_ARCH: Final = 4  # offsetof(struct seccomp_data, arch)
RET_KILL_PROCESS: Final = 0x80000000
RET_ERRNO: Final = 0x00050000
RET_ALLOW: Final = 0x7FFF0000
_X32_BIT: Final = 0x40000000

AUDIT_ARCH_X86_64: Final = 0xC000003E
AUDIT_ARCH_AARCH64: Final = 0xC00000B7

DENIED: Final = (
    # Kernel interfaces a sandbox escape goes through.
    "bpf",
    "userfaultfd",
    "perf_event_open",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
    "keyctl",
    "add_key",
    "request_key",
    # Reading or steering other processes.
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "kcmp",
    "migrate_pages",
    "move_pages",
    # Mounts and namespaces (bwrap has set them up already).
    "mount",
    "umount2",
    "pivot_root",
    "chroot",
    "move_mount",
    "open_tree",
    "fsopen",
    "fsconfig",
    "fsmount",
    "fspick",
    "mount_setattr",
    "setns",
    "unshare",
    "name_to_handle_at",
    "open_by_handle_at",
    "fanotify_init",
    # The machine itself.
    "kexec_load",
    "kexec_file_load",
    "init_module",
    "finit_module",
    "delete_module",
    "reboot",
    "swapon",
    "swapoff",
    "acct",
    "quotactl",
    "quotactl_fd",
    "syslog",
    "vhangup",
    "lookup_dcookie",
    "nfsservctl",
    "settimeofday",
    "clock_settime",
    "clock_adjtime",
    "adjtimex",
    "sethostname",
    "setdomainname",
    # x86 only: port I/O, segment tables, a.out libraries, the old sysctl.
    "iopl",
    "ioperm",
    "modify_ldt",
    "uselib",
    "_sysctl",
)
"""What generated code may not call. A name missing from an architecture's table does not
exist there."""

NUMBERS: Final[dict[str, tuple[int, dict[str, int]]]] = {
    "x86_64": (
        AUDIT_ARCH_X86_64,
        {
            "bpf": 321,
            "userfaultfd": 323,
            "perf_event_open": 298,
            "io_uring_setup": 425,
            "io_uring_enter": 426,
            "io_uring_register": 427,
            "keyctl": 250,
            "add_key": 248,
            "request_key": 249,
            "ptrace": 101,
            "process_vm_readv": 310,
            "process_vm_writev": 311,
            "kcmp": 312,
            "migrate_pages": 256,
            "move_pages": 279,
            "mount": 165,
            "umount2": 166,
            "pivot_root": 155,
            "chroot": 161,
            "move_mount": 429,
            "open_tree": 428,
            "fsopen": 430,
            "fsconfig": 431,
            "fsmount": 432,
            "fspick": 433,
            "mount_setattr": 442,
            "setns": 308,
            "unshare": 272,
            "name_to_handle_at": 303,
            "open_by_handle_at": 304,
            "fanotify_init": 300,
            "kexec_load": 246,
            "kexec_file_load": 320,
            "init_module": 175,
            "finit_module": 313,
            "delete_module": 176,
            "reboot": 169,
            "swapon": 167,
            "swapoff": 168,
            "acct": 163,
            "quotactl": 179,
            "quotactl_fd": 443,
            "syslog": 103,
            "vhangup": 153,
            "lookup_dcookie": 212,
            "nfsservctl": 180,
            "settimeofday": 164,
            "clock_settime": 227,
            "clock_adjtime": 305,
            "adjtimex": 159,
            "sethostname": 170,
            "setdomainname": 171,
            "iopl": 172,
            "ioperm": 173,
            "modify_ldt": 154,
            "uselib": 134,
            "_sysctl": 156,
        },
    ),
    "aarch64": (
        AUDIT_ARCH_AARCH64,
        {
            "bpf": 280,
            "userfaultfd": 282,
            "perf_event_open": 241,
            "io_uring_setup": 425,
            "io_uring_enter": 426,
            "io_uring_register": 427,
            "keyctl": 219,
            "add_key": 217,
            "request_key": 218,
            "ptrace": 117,
            "process_vm_readv": 270,
            "process_vm_writev": 271,
            "kcmp": 272,
            "migrate_pages": 238,
            "move_pages": 239,
            "mount": 40,
            "umount2": 39,
            "pivot_root": 41,
            "chroot": 51,
            "move_mount": 429,
            "open_tree": 428,
            "fsopen": 430,
            "fsconfig": 431,
            "fsmount": 432,
            "fspick": 433,
            "mount_setattr": 442,
            "setns": 268,
            "unshare": 97,
            "name_to_handle_at": 264,
            "open_by_handle_at": 265,
            "fanotify_init": 262,
            "kexec_load": 104,
            "kexec_file_load": 294,
            "init_module": 105,
            "finit_module": 273,
            "delete_module": 106,
            "reboot": 142,
            "swapon": 224,
            "swapoff": 225,
            "acct": 89,
            "quotactl": 60,
            "quotactl_fd": 443,
            "syslog": 116,
            "vhangup": 58,
            "lookup_dcookie": 18,
            "nfsservctl": 42,
            "settimeofday": 170,
            "clock_settime": 112,
            "clock_adjtime": 266,
            "adjtimex": 171,
            "sethostname": 161,
            "setdomainname": 162,
        },
    ),
}
"""System-call numbers per machine (`platform.machine()`): the audit architecture and the
numbers of the denied calls (from the kernel's unistd headers)."""

_MACHINES: Final = {"amd64": "x86_64", "arm64": "aarch64"}


def _insn(code: int, k: int, jt: int = 0, jf: int = 0) -> bytes:
    return struct.pack("=HBBI", code, jt, jf, k)


def program(machine: str | None = None) -> bytes | None:
    """The filter for `machine` (this one by default) as a `struct sock_filter` array, or None
    when its system-call numbers are not known here."""
    name = (machine or platform.machine()).lower()
    entry = NUMBERS.get(_MACHINES.get(name, name))
    if entry is None:
        return None
    arch, numbers = entry
    denied = sorted({numbers[call] for call in DENIED if call in numbers})
    deny = RET_ERRNO | errno.EPERM
    head = [
        _insn(_LD_W_ABS, _ARCH),
        _insn(_JEQ_K, arch, jt=1),
        _insn(_RET_K, RET_KILL_PROCESS),  # another ABI: its numbers mean other calls
        _insn(_LD_W_ABS, _NR),
    ]
    if arch == AUDIT_ARCH_X86_64:
        # x32 shares the architecture and marks its numbers with this bit.
        head.append(_insn(_JGE_K, _X32_BIT, jt=len(denied) + 1))
    checks = [_insn(_JEQ_K, nr, jt=len(denied) - i) for i, nr in enumerate(denied)]
    tail = [_insn(_RET_K, RET_ALLOW), _insn(_RET_K, deny)]
    return b"".join(head + checks + tail)


@contextlib.contextmanager
def program_fd(machine: str | None = None) -> Iterator[int | None]:
    """A file descriptor holding the filter, open for bwrap's `--seccomp` (None: no filter)."""
    code = program(machine)
    if code is None:
        yield None
        return
    with tempfile.TemporaryFile() as file:
        file.write(code)
        file.flush()
        file.seek(0)
        yield file.fileno()
