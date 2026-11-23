"""Entry point for the channel-split lineage audit."""

from .channel_audit import *  # noqa: F401,F403
from .channel_audit import main

if __name__ == "__main__":
    main()
