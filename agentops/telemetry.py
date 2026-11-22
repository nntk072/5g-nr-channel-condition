"""Bounded engineering telemetry that never infers scientific qualification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json_bytes
from .provider import redact


def normalize_provider_usage(value: dict[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {"prompt_tokens": "unknown", "output_tokens": "unknown", "estimated_cost": "unknown"}
    result: dict[str, Any] = {}
    for key in ("prompt_tokens", "output_tokens", "estimated_cost"):
        item = value.get(key, "unknown")
        if item != "unknown" and not isinstance(item, (int, float)):
            raise ContractError(f"provider usage field {key} must be numeric or unknown")
        result[key] = item
    return result


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def append_event(path: Path, event: dict[str, Any], *, maximum_bytes: int = 1_000_000) -> str:
    cleaned = _redact_value(event)
    raw = canonical_json_bytes(cleaned)
    if len(raw) > maximum_bytes:
        raise ContractError("telemetry event exceeds the configured byte limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(raw + b"\n")
        handle.flush()
    return hashlib.sha256(raw).hexdigest()
