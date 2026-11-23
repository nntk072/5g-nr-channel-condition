"""Resolve file paths recorded in retained evidence after a rename.

Retained evidence keeps the path it was produced with. The repository records every
rename in ``docs/reference/digest-rebindings.json``. ``resolve_recorded_path`` returns
the current location of a recorded path, so verification can still read the file.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _moves(root: Path) -> dict[str, str]:
    record = root / "docs" / "reference" / "digest-rebindings.json"
    if not record.is_file():
        return {}
    payload = json.loads(record.read_text(encoding="utf-8"))
    return {
        item["old_path"]: item["new_path"]
        for item in payload.get("file_moves", [])
        if item.get("old_path") != item.get("new_path")
    }


def resolve_recorded_path(recorded: str, root: Path) -> Path:
    """Return the current path for a recorded repository-relative path."""
    current = root / recorded
    if current.exists():
        return current
    moved = _moves(root).get(recorded)
    return root / moved if moved else current


@lru_cache(maxsize=1)
def _source_digest_map(root: Path) -> dict[str, str]:
    record = root / "docs" / "reference" / "digest-rebindings.json"
    if not record.is_file():
        return {}
    payload = json.loads(record.read_text(encoding="utf-8"))
    return {
        item["old_sha256"]: item["new_sha256"]
        for item in payload.get("source_digest_rebindings", [])
        if item.get("old_sha256") != item.get("new_sha256")
    }


def recorded_digest_matches(recorded_sha256: object, current_bytes: bytes, root: Path) -> bool:
    """True when retained evidence's recorded digest matches current bytes.

    A retained digest matches either exactly, or through the repository's rename
    rebinding record, which maps the pre-rename digest to the post-rename digest.
    """
    import hashlib

    if not isinstance(recorded_sha256, str):
        return False
    current = hashlib.sha256(current_bytes).hexdigest()
    if recorded_sha256 == current:
        return True
    return _source_digest_map(root).get(recorded_sha256) == current


@lru_cache(maxsize=1)
def _path_map(root: Path) -> dict[str, str]:
    record = root / "docs" / "reference" / "digest-rebindings.json"
    if not record.is_file():
        return {}
    payload = json.loads(record.read_text(encoding="utf-8"))
    return {
        item["old_path"]: item["new_path"]
        for item in payload.get("file_moves", [])
        if item.get("old_path") != item.get("new_path")
    }


def current_repo_path(recorded: str, root: Path) -> str:
    """Return the current repository-relative path for a path recorded in evidence."""
    if (root / recorded).exists():
        return recorded
    return _path_map(root).get(recorded, recorded)
