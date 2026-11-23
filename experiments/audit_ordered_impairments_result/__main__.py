"""Command-line entry point for the ordered-impairment result audit."""

from .file import *  # noqa: F401,F403
from .file import main

if __name__ == "__main__":
    raise SystemExit(main())
