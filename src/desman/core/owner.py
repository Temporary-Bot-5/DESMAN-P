"""Who holds a lease, and whether that holder is certainly gone.

A lease owner is "<host>:<pid>". After a crash (`kill -9`, power loss, a closed laptop lid) the
dead process's leases would block its subtasks until they expire; `resume` would sit idle for
up to the lease length. When the owner ran on this machine and its process no longer exists,
the work can be taken back at once.

`owner_is_gone` answers True only when that is certain. Another machine, an unparsable owner,
our own process, a live process (even one belonging to another user) or a platform without a
reliable check all answer False, and the lease simply runs out as before. A recycled pid makes
a dead owner look alive, which is the safe direction.

One wrong answer is possible: two containers with the same hostname but separate pid namespaces
sharing one database. A live run there can look dead. Nothing is duplicated even then (its lease
is taken by token, so its late result is refused and recorded as discarded work), but the work
is wasted: give such containers distinct hostnames.
"""

from __future__ import annotations

import os
import socket


def current_owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def owner_is_gone(owner: str) -> bool:
    host, sep, pid_text = owner.rpartition(":")
    if not sep or host != socket.gethostname() or not pid_text.isdigit():
        return False
    pid = int(pid_text)
    if pid <= 0 or pid == os.getpid() or os.name != "posix":
        return False
    try:
        os.kill(pid, 0)  # signal 0: existence check only
    except ProcessLookupError:
        return True
    except OSError:  # PermissionError: it exists, under another user
        return False
    return False
