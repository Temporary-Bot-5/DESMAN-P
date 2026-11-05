"""Exclusive process locks on an open descriptor; unlock before closing it."""

from __future__ import annotations

import errno
import os
import sys
import time

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


def lock(fd: int, *, blocking: bool = True) -> None:
    """Lock byte zero on Windows, or the whole file on POSIX.

    Windows' LK_LOCK stops retrying after ten seconds, so use LK_NBLCK and retry
    indefinitely for blocking callers. All users must open the file for writing.
    """
    if sys.platform == "win32":
        while True:
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if not blocking:
                    raise BlockingIOError(exc.errno, exc.strerror) from exc
                time.sleep(0.05)
    else:
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))


def unlock(fd: int) -> None:
    if sys.platform == "win32":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
