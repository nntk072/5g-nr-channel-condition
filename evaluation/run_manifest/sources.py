"""Evaluator source bundle identity and rooted reads of source files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from evaluation.acquisition_package import (
    MAX_FILE_BYTES,
    _read_inventory_file,
    _safe_relative_path,
)
from evaluation.run_manifest.common import (
    _canonical_json_bytes,
)
from evaluation.run_manifest.constants import (
    _MAX_SOURCE_BUNDLE_BYTES,
    EVALUATOR_SOURCE_PATHS,
    V5_EVALUATOR_SOURCE_PATHS,
)


def _source_bundle_sha256(sources: list[dict[str, Any]]) -> str:
    records = [
        {"path": item["path"], "size_bytes": item["size_bytes"], "sha256": item["sha256"]}
        for item in sources
    ]
    return hashlib.sha256(_canonical_json_bytes(records)).hexdigest()


def _rooted_file_identity(
    root_fd: int, root: Path, file_path: str | Path, *, limit: int, capture: bool
) -> tuple[str, bytes | None, int, str]:
    path = Path(file_path)
    try:
        relative = path.relative_to(root) if path.is_absolute() else path
    except ValueError as error:
        raise ValueError("manifest file must be located below its declared root") from error
    relative_path = relative.as_posix()
    parts = _safe_relative_path(relative_path)
    data, size, digest = _read_inventory_file(root_fd, parts, limit, capture=capture)
    return relative_path, data, size, digest


def _read_rooted_source_file(root: Path, relative_path: str, *, limit: int) -> bytes:
    root_fd = os.open(
        root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    )
    try:
        data, _, _ = _read_inventory_file(
            root_fd, _safe_relative_path(relative_path), limit, capture=True
        )
        if data is None:
            raise ValueError(f"source file was not captured: {relative_path}")
        return data
    finally:
        os.close(root_fd)


def evaluator_source_bundle_identity(
    *,
    source_root: str | Path | None = None,
    include_runtime_inventory: bool = False,
) -> dict[str, Any]:
    """Return the fixed source roster and its digest before a run is launched."""
    root = (
        Path(source_root).resolve(strict=True)
        if source_root is not None
        else Path(__file__).resolve().parents[2]
    )
    if not root.is_dir():
        raise ValueError("source_root must be a directory")
    root_fd = os.open(
        root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    )
    sources: list[dict[str, Any]] = []
    total = 0
    try:
        if not isinstance(include_runtime_inventory, bool):
            raise TypeError("include_runtime_inventory must be a boolean")
        source_roster = (
            V5_EVALUATOR_SOURCE_PATHS if include_runtime_inventory else EVALUATOR_SOURCE_PATHS
        )
        for relative_path in source_roster:
            remaining = _MAX_SOURCE_BUNDLE_BYTES - total
            if remaining <= 0:
                raise ValueError("evaluator source roster exceeds its total byte limit")
            _, size, digest = _read_inventory_file(
                root_fd,
                _safe_relative_path(relative_path),
                min(MAX_FILE_BYTES, remaining),
                capture=False,
            )
            total += size
            sources.append({"path": relative_path, "size_bytes": size, "sha256": digest})
    finally:
        os.close(root_fd)
    return {
        "evaluator_sources": sources,
        "evaluator_source_bundle_sha256": _source_bundle_sha256(sources),
    }
