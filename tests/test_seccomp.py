"""The sandbox's system-call filter (ROADMAP S.2, ADR 0018), checked without a kernel: the
program is run here by a small classic-BPF interpreter, and the numbers are compared with the
kernel's headers when they are installed."""

from __future__ import annotations

import errno
import re
import struct
from pathlib import Path

import pytest

from desman.core import seccomp

DENY = seccomp.RET_ERRNO | errno.EPERM


def evaluate(code: bytes, arch: int, nr: int) -> int:
    """What the kernel would return for a call `nr` of `arch`."""
    data = struct.pack("=iI", nr, arch) + bytes(56)  # struct seccomp_data, args zeroed
    program = [struct.unpack("=HBBI", code[i : i + 8]) for i in range(0, len(code), 8)]
    assert len(program) <= 4096  # BPF_MAXINSNS
    acc = pc = 0
    while True:
        op, jt, jf, k = program[pc]
        pc += 1
        if op == 0x20:
            (acc,) = struct.unpack_from("=I", data, k)
        elif op == 0x15:
            pc += jt if acc == k else jf
        elif op == 0x35:
            pc += jt if acc >= k else jf
        elif op == 0x06:
            return int(k)
        else:
            raise AssertionError(f"unexpected instruction {op:#x}")


@pytest.mark.parametrize("machine", sorted(seccomp.NUMBERS))
def test_denied_calls_are_refused_and_every_other_is_allowed(machine: str) -> None:
    arch, numbers = seccomp.NUMBERS[machine]
    code = seccomp.program(machine)
    assert code is not None
    denied = set(numbers.values())
    assert len(denied) == len(numbers)  # no two names share a number
    for nr in range(512):
        expected = DENY if nr in denied else seccomp.RET_ALLOW
        assert evaluate(code, arch, nr) == expected, (machine, nr)


@pytest.mark.parametrize("machine", sorted(seccomp.NUMBERS))
def test_a_call_of_another_architecture_kills_the_process(machine: str) -> None:
    code = seccomp.program(machine)
    assert code is not None
    i386, arm = 0x40000003, 0x40000028
    assert evaluate(code, i386, 39) == seccomp.RET_KILL_PROCESS
    assert evaluate(code, arm, 39) == seccomp.RET_KILL_PROCESS


def test_x32_calls_are_refused() -> None:
    code = seccomp.program("x86_64")
    assert code is not None
    for nr in (0, 39, 101 | 0x40000000, 0x40000000 + 512):
        expected = seccomp.RET_ALLOW if nr < 0x40000000 else DENY
        assert evaluate(code, seccomp.AUDIT_ARCH_X86_64, nr) == expected


def test_every_name_is_denied_and_known() -> None:
    assert len(set(seccomp.DENIED)) == len(seccomp.DENIED)
    for _arch, numbers in seccomp.NUMBERS.values():
        assert set(numbers) <= set(seccomp.DENIED)
    x86 = seccomp.NUMBERS["x86_64"][1]
    assert set(x86) == set(seccomp.DENIED)  # every name exists on x86-64


def test_an_unknown_machine_has_no_filter() -> None:
    assert seccomp.program("riscv64") is None
    assert seccomp.program("AMD64") == seccomp.program("x86_64")
    assert seccomp.program("arm64") == seccomp.program("aarch64")
    with seccomp.program_fd("riscv64") as fd:
        assert fd is None


def test_a_32_bit_userland_on_a_64_bit_kernel_gets_no_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    assert seccomp.program() == seccomp.program("aarch64")
    monkeypatch.setattr("struct.calcsize", lambda fmt: 4)
    assert seccomp.program() is None


def test_the_descriptor_holds_the_program() -> None:
    import os

    with seccomp.program_fd("x86_64") as fd:
        assert fd is not None
        assert os.read(fd, 1 << 16) == seccomp.program("x86_64")


HEADERS = {
    "x86_64": Path("/usr/include/x86_64-linux-gnu/asm/unistd_64.h"),
    "aarch64": Path("/usr/include/asm-generic/unistd.h"),
}


@pytest.mark.parametrize("machine", sorted(HEADERS))
def test_the_numbers_are_the_kernels(machine: str) -> None:
    header = HEADERS[machine]
    if not header.exists():
        alternative = Path("/usr/include/asm/unistd_64.h")
        if machine == "x86_64" and alternative.exists():
            header = alternative
        else:
            pytest.skip(f"{header} is not installed")
    known: dict[str, int] = {}
    for match in re.finditer(r"#define __NR(?:3264)?_(\w+)\s+(\d+)", header.read_text()):
        known.setdefault(match.group(1), int(match.group(2)))
    numbers = seccomp.NUMBERS[machine][1]
    newer = {name for name in numbers if name not in known}  # newer than these headers
    assert newer <= {"open_tree_attr"}
    assert {name: known[name] for name in numbers if name in known} == {
        name: nr for name, nr in numbers.items() if name not in newer
    }
    for name in seccomp.DENIED:
        if name not in numbers:
            assert name not in known  # missing only where the call does not exist
