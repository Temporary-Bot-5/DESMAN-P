"""UTF-8 text streams for Windows commands and piped keys."""

from __future__ import annotations

import codecs
import io
import sys


def utf8_stream(stream: object) -> None:
    """Reconfigure real text streams, leaving captured/replaced streams alone."""
    if isinstance(stream, io.TextIOWrapper) and codecs.lookup(stream.encoding).name != "utf-8":
        stream.reconfigure(encoding="utf-8", errors="replace")


def configure_output() -> None:
    """Windows pipes and older consoles otherwise use an ANSI code page."""
    if sys.platform == "win32":
        utf8_stream(sys.stdout)
        utf8_stream(sys.stderr)
