"""Stable identities for resolved experiment inputs and generated artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def content_identity(value: Any, *, prefix: str = "sha256") -> str:
    """Return a deterministic identity for JSON-compatible content."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"
