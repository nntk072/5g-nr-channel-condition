"""NR record store tamper rules: content hashes bind manifests and archives; write-side metadata checks."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from datasets.nr_store import read_nr_record, write_nr_record
from tests.test_datasets.test_nr_store_pointer_rules import _record, _written


def _manifest_file(root: Path) -> Path:
    return next(path for path in root.rglob("*.json"))


def test_non_finite_metadata_is_refused_before_any_write(tmp_path: Path) -> None:
    record = dataclasses.replace(
        _record(), provenance={"scene_hash": "sha256:abc", "bad": float("nan")}
    )
    with pytest.raises(TypeError, match="finite JSON-compatible values"):
        write_nr_record(tmp_path, record)
    assert not any(tmp_path.rglob("*.json"))


def test_object_dtype_targets_are_refused_before_any_write(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must use a numeric or boolean dtype"):
        dataclasses.replace(_record(), targets={"geometric_los": np.asarray([True], dtype=object)})


def test_edited_manifest_fails_its_content_hash(tmp_path: Path) -> None:
    stored, _ = _written(tmp_path)
    manifest_path = _manifest_file(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["sample_id"] = "forged-sample"
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError):
        read_nr_record(tmp_path, manifest_path)


def test_replaced_manifest_bytes_fail_before_parsing(tmp_path: Path) -> None:
    stored, _ = _written(tmp_path)
    manifest_path = _manifest_file(tmp_path)
    manifest_path.write_bytes(b"\xff\xfe not json")
    with pytest.raises(ValueError):
        read_nr_record(tmp_path, manifest_path)


def test_corrupted_archive_fails_its_content_hash(tmp_path: Path) -> None:
    stored, _ = _written(tmp_path)
    archive = (
        tmp_path / stored.archive_relative_path
        if hasattr(stored, "archive_relative_path")
        else None
    )
    if archive is None:
        archive = next(path for path in tmp_path.rglob("*.npz"))
    archive.write_bytes(b"corrupted archive bytes")
    with pytest.raises(ValueError, match="content hash does not match|NPZ|hash"):
        read_nr_record(tmp_path, stored)


def test_round_trip_is_unchanged_after_reading_back(tmp_path: Path) -> None:
    stored, _ = _written(tmp_path)
    restored = read_nr_record(tmp_path, stored)
    assert restored.targets["geometric_los"] == np.asarray(True)
