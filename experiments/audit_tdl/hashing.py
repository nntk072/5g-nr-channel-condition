"""SHA-256 helpers shared by the TDL audit modules."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256(raw.encode())
