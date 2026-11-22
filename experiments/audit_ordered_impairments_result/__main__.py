"""Command-line entry point for audit_head."""

from .file import *  # noqa: F401,F403
from .file import main

if __name__ == "__main__":
    raise SystemExit(main())
