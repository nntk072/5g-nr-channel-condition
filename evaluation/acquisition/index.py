"""Index and pointer helpers for acquisition-package inventories."""

from __future__ import annotations

from typing import Any


def _unique_index(
    records: Any, *, key: str, label: str, errors: list[str]
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if not isinstance(records, list):
        errors.append(f"{label} must be an array")
        return index
    for item in records:
        if not isinstance(item, dict):
            errors.append(f"{label} contains a non-object record")
            continue
        identity = item.get(key)
        if not isinstance(identity, str) or not identity:
            errors.append(f"{label} record lacks a nonempty {key}")
            continue
        if identity in index:
            errors.append(f"duplicate {label} {key}: {identity}")
        else:
            index[identity] = item
    return index


def _pointer_item(document: Any, pointer: Any, expected_prefix: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith(expected_prefix):
        raise ValueError(f"pointer must start with {expected_prefix}")
    suffix = pointer[len(expected_prefix) :]
    if not suffix.isdecimal():
        raise ValueError("pointer index must be a nonnegative decimal integer")
    records = document
    for part in expected_prefix.strip("/").split("/"):
        if not isinstance(records, dict) or part not in records:
            raise ValueError(f"pointer container {part!r} is missing")
        records = records[part]
    if not isinstance(records, list):
        raise ValueError("pointer target container is not an array")
    index = int(suffix)
    if index >= len(records):
        raise ValueError("pointer index is outside its array")
    return records[index]
