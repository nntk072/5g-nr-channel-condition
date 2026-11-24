"""Run-manifest source identity: rooted reads, capture, roster and byte-budget rules."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from evaluation.run_manifest import sources
from evaluation.run_manifest.sources import (
    _read_rooted_source_file,
    _rooted_file_identity,
    evaluator_source_bundle_identity,
)


def test_rooted_identity_rejects_paths_outside_the_root(tmp_path: Path) -> None:
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError, match="must be located below its declared root"):
            _rooted_file_identity(
                root_fd, tmp_path, tmp_path.parent / "outside.txt", limit=10, capture=False
            )
    finally:
        os.close(root_fd)


def test_rooted_reads_return_captured_bytes_only_when_requested(tmp_path: Path) -> None:
    (tmp_path / "data.txt").write_bytes(b"abc")
    assert _read_rooted_source_file(tmp_path, "data.txt", limit=10) == b"abc"
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        relative, data, size, digest = _rooted_file_identity(
            root_fd, tmp_path, tmp_path / "data.txt", limit=10, capture=True
        )
    finally:
        os.close(root_fd)
    assert relative == "data.txt" and data == b"abc" and size == 3 and len(digest) == 64


def test_source_root_must_be_a_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="source_root must be a directory"):
        evaluator_source_bundle_identity(source_root=file_path)


def test_bundle_identity_requires_a_boolean_runtime_flag(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="include_runtime_inventory must be a boolean"):
        evaluator_source_bundle_identity(include_runtime_inventory="yes")


def test_bundle_identity_enforces_its_total_byte_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(sources.__file__).resolve().parents[2]
    first = root / sources.EVALUATOR_SOURCE_PATHS[0]
    # Exactly the first source's bytes: the second source finds no remaining budget.
    monkeypatch.setattr(sources, "_MAX_SOURCE_BUNDLE_BYTES", first.stat().st_size)
    with pytest.raises(ValueError, match="total byte limit"):
        evaluator_source_bundle_identity()
