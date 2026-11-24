"""Stored standardized runs: manifest, pointer and record-archive integrity checks.

Each case edits the on-disk store (or the pointer used to read it) and checks the reader's
named rejection. Manifest edits recompute the manifest digest so that only the rule under
test fails.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from datasets.standardized import (
    StandardizedFamily,
    _canonical_json,
    read_standardized_run,
    standardized_run_from_channel,
    write_standardized_run,
)
from tests.test_datasets.test_standardized import _channel


def _write(tmp_path: Path):
    run = standardized_run_from_channel(
        _channel(), family=StandardizedFamily.TDL, run_id="store-rules"
    )
    stored = write_standardized_run(tmp_path, run)
    return stored, tmp_path / stored.relative_path


def _manifest_digest(raw: dict[str, Any]) -> str:
    core = {key: raw[key] for key in ("schema", "schema_version", "run_id", "records")}
    return hashlib.sha256(_canonical_json(core, name="run manifest")).hexdigest()


def _rewrite_manifest(tmp_path: Path, manifest: Path, stored, mutate) -> Any:
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    mutate(raw)
    digest = _manifest_digest(raw)
    raw["content_sha256"] = digest
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    return dataclasses.replace(stored, content_sha256=digest)


def _resave_record(tmp_path: Path, manifest: Path, index: int, change) -> None:
    """Rewrite one record archive under its new content address and update the manifest."""
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    reference = raw["records"][index]
    old_path = tmp_path / reference["relative_path"]
    with np.load(old_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(arrays["metadata_json"].tobytes().decode("utf-8"))
    change(metadata, arrays)
    arrays["metadata_json"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    new_path = old_path.with_name(f"{digest}.npz")
    new_path.write_bytes(payload)
    reference["relative_path"] = new_path.relative_to(tmp_path).as_posix()
    reference["content_sha256"] = digest
    manifest.write_text(json.dumps(raw), encoding="utf-8")


def _reread_with_manifest(tmp_path: Path, manifest: Path, stored) -> Any:
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    digest = _manifest_digest(raw)
    raw["content_sha256"] = digest
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    return dataclasses.replace(stored, content_sha256=digest)


def test_store_reads_from_a_relative_manifest_path(tmp_path: Path) -> None:
    stored, manifest = _write(tmp_path)

    restored = read_standardized_run(tmp_path, stored.relative_path)

    assert restored.run_id == "store-rules"
    assert [record.sample_id for record in restored.records] == list(stored.sample_ids)


@pytest.mark.parametrize(
    ("pointer_change", "message"),
    [
        pytest.param(
            {"relative_path": "../escape.json"},
            "manifest path must remain inside the store",
            id="manifest-escape",
        ),
        pytest.param(
            {"relative_path": "/absolute.json"},
            "manifest path must remain inside the store",
            id="manifest-absolute",
        ),
        pytest.param(
            {"schema_version": 2},
            "unsupported standardized run schema version: 2",
            id="schema-version",
        ),
        pytest.param(
            {"run_id": "other-run"}, "does not match its manifest pointer", id="run-id-pointer"
        ),
        pytest.param({"content_sha256": "0" * 64}, "manifest hash is invalid", id="pointer-hash"),
    ],
)
def test_store_pointer_must_match_its_manifest(
    tmp_path: Path, pointer_change, message: str
) -> None:
    stored, _ = _write(tmp_path)
    pointer = dataclasses.replace(stored, **pointer_change)

    with pytest.raises(ValueError, match=message):
        read_standardized_run(tmp_path, pointer)


def test_store_rejects_manifest_with_another_schema(tmp_path: Path) -> None:
    stored, manifest = _write(tmp_path)
    updated = _rewrite_manifest(tmp_path, manifest, stored, lambda raw: raw.update(schema="x.v9"))

    with pytest.raises(ValueError, match="unsupported standardized run manifest"):
        read_standardized_run(tmp_path, updated)


@pytest.mark.parametrize(
    ("record_path", "message"),
    [
        pytest.param(
            "../outside.npz", "record path must remain inside the store", id="record-escape"
        ),
    ],
)
def test_store_rejects_record_paths_outside_the_store(
    tmp_path: Path, record_path: str, message: str
) -> None:
    stored, manifest = _write(tmp_path)
    updated = _rewrite_manifest(
        tmp_path,
        manifest,
        stored,
        lambda raw: raw["records"][0].update(relative_path=record_path),
    )

    with pytest.raises(ValueError, match=message):
        read_standardized_run(tmp_path, updated)


def test_store_rejects_record_archive_with_another_schema(tmp_path: Path) -> None:
    stored, manifest = _write(tmp_path)

    def change(metadata: dict[str, Any], arrays: dict[str, Any]) -> None:
        metadata["schema"] = "standardized_channel_record.v9"

    _resave_record(tmp_path, manifest, 0, change)
    updated = _reread_with_manifest(tmp_path, manifest, stored)

    with pytest.raises(ValueError, match="unsupported standardized channel record"):
        read_standardized_run(tmp_path, updated)


def test_store_rejects_record_identity_that_disagrees_with_manifest(tmp_path: Path) -> None:
    stored, manifest = _write(tmp_path)

    def change(metadata: dict[str, Any], arrays: dict[str, Any]) -> None:
        metadata["sample_id"] = "someone-else"

    _resave_record(tmp_path, manifest, 0, change)
    updated = _reread_with_manifest(tmp_path, manifest, stored)

    with pytest.raises(ValueError, match="record identity disagrees with run manifest"):
        read_standardized_run(tmp_path, updated)


def test_store_rejects_reordered_samples_against_run_pointer(tmp_path: Path) -> None:
    run = standardized_run_from_channel(
        _channel(), family=StandardizedFamily.TDL, run_id="store-order"
    )
    stored = write_standardized_run(tmp_path, run)
    manifest = tmp_path / stored.relative_path
    pointer = dataclasses.replace(stored, sample_ids=tuple(reversed(stored.sample_ids)))

    with pytest.raises(ValueError, match="sample order disagrees with run pointer"):
        read_standardized_run(tmp_path, pointer)
    assert manifest.exists()


def test_store_accepts_an_absolute_manifest_path_and_checks_record_run_identity(
    tmp_path: Path,
) -> None:
    stored, manifest = _write(tmp_path)
    restored = read_standardized_run(tmp_path, str(manifest.resolve()))
    assert restored.run_id == "store-rules"

    def change(metadata: dict[str, Any], arrays: dict[str, Any]) -> None:
        metadata["run_id"] = "another-run"

    _resave_record(tmp_path, manifest, 0, change)
    updated = _reread_with_manifest(tmp_path, manifest, stored)
    with pytest.raises(ValueError, match="sample identity disagrees with run manifest"):
        read_standardized_run(tmp_path, updated)
