"""Canonical JSON, timestamp and strict-object helpers for acquisition packages."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, cast


def _canonical_sha256(value: Any) -> str:
    encoded = _canonical_json(value)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strata_key_set(value: Any) -> set[tuple[str, str, str]] | None:
    if not isinstance(value, list):
        return None
    result: set[tuple[str, str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            return None
        key = (
            item.get("material_id"),
            item.get("polarization"),
            item.get("incidence_stratum_id"),
        )
        if not all(isinstance(part, str) for part in key):
            return None
        normalized_key = cast(tuple[str, str, str], key)
        if normalized_key in result:
            return None
        result.add(normalized_key)
    return result


def _json_bytes(data: bytes, description: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON: {error}") from error
