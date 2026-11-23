"""Hashing for ordered impairments receiver."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical byte representation used for result hashes."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
