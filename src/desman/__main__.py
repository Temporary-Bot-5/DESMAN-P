"""Entry point for DESMAN(P) CLI."""

import signal
import sys

from desman.cli import cli

if __name__ == "__main__":
    # Page workers have their own Windows process group: CTRL_BREAK is their Ctrl+C.
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    cli()
