from __future__ import annotations

import hashlib
import json
import tracemalloc
import zipfile
from collections.abc import Generator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from datasets.nr_store import NRStoredRecord, _nr_record_memory_estimate


def _write_synthetic_store(root: Path, manifest_count: int) -> Path:
    """Write a bounded, scan-only manifest inventory and a compressed NPZ fixture."""
    object_path = root / "objects" / "synthetic.npz"
    object_path.parent.mkdir(parents=True)
    np.savez_compressed(object_path, channel=np.zeros((64, 64), dtype=np.complex64))
    manifest_dir = root / "manifests"
    manifest_dir.mkdir()
    for index in range(manifest_count):
        sample_id = f"sample-{index:04d}"
        manifest = {
            "sample_id": sample_id,
            "content_sha256": "0" * 64,
            "relative_path": "objects/synthetic.npz",
            "schema_version": 1,
        }
        key = hashlib.sha256(sample_id.encode()).hexdigest()
        (manifest_dir / f"{key}.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
    return object_path


def _payload_byte_accounting(object_path: Path) -> dict[str, int]:
    compressed_bytes = object_path.stat().st_size
    with zipfile.ZipFile(object_path) as archive:
        uncompressed_member_bytes = sum(member.file_size for member in archive.infolist())
    return {
        "compressed_archive_bytes": compressed_bytes,
        "uncompressed_member_bytes_including_npy_headers": uncompressed_member_bytes,
        "modeled_two_compressed_buffers_bytes": 2 * compressed_bytes,
        "modeled_two_uncompressed_member_equivalents_bytes": 2 * uncompressed_member_bytes,
    }


def _inventory_scan_peak(root: Path) -> int:
    """Trace the bounded manifest scan with payload estimation/decoding stubbed."""
    import datasets.nr_store as nr_store

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nr_store, "_nr_record_memory_estimate", lambda *_args, **_kwargs: 2)
        patch.setattr(nr_store, "read_nr_record", lambda *_args, **_kwargs: None)
        tracemalloc.start()
        try:
            iterator = cast(
                Generator[object, None, None],
                nr_store.iter_nr_record_shards(root, max_shard_bytes=3),
            )
            try:
                next(iterator, None)
                _, peak_bytes = tracemalloc.get_traced_memory()
            finally:
                iterator.close()
        finally:
            tracemalloc.stop()
    return peak_bytes


def test_bounded_synthetic_profile_separates_payload_and_inventory_memory(tmp_path):
    payload_root = tmp_path / "payload"
    payload_root.mkdir()
    payload_path = _write_synthetic_store(payload_root, 0)
    payload = _payload_byte_accounting(payload_path)
    assert payload["compressed_archive_bytes"] > 0
    assert (
        payload["uncompressed_member_bytes_including_npy_headers"]
        > payload["compressed_archive_bytes"]
    )
    assert (
        payload["modeled_two_compressed_buffers_bytes"] == 2 * payload["compressed_archive_bytes"]
    )
    assert payload["modeled_two_uncompressed_member_equivalents_bytes"] == (
        2 * payload["uncompressed_member_bytes_including_npy_headers"]
    )
    estimated_bytes = _nr_record_memory_estimate(
        payload_root,
        NRStoredRecord(
            sample_id="synthetic-payload",
            content_sha256="0" * 64,
            relative_path="objects/synthetic.npz",
        ),
        max_shard_bytes=1_000_000,
    )
    assert estimated_bytes == (
        payload["modeled_two_compressed_buffers_bytes"]
        + payload["modeled_two_uncompressed_member_equivalents_bytes"]
    )

    inventory_peaks: dict[int, int] = {}
    for count in (0, 8, 128, 8192):
        root = tmp_path / f"inventory-{count}"
        root.mkdir()
        _write_synthetic_store(root, count)
        inventory_peaks[count] = _inventory_scan_peak(root)

    traced_growth = inventory_peaks[8192] - inventory_peaks[8]
    assert traced_growth < 1024 * 1024
    report = {
        "label": "synthetic non-production NR store memory profile",
        "manifest_counts": list(inventory_peaks),
        "manifest_scan_peak_traced_python_bytes": inventory_peaks,
        "traced_python_growth_8_to_8192_manifests_bytes": traced_growth,
        **payload,
        "scope": (
            "Payload values are byte accounting and conservative modeled equivalents; "
            "manifest peaks are a bounded tracemalloc observation with payload estimation "
            "and decoding stubbed. "
            "This is not a corpus RSS cap."
        ),
    }
    print(json.dumps(report, sort_keys=True))


def test_sqlite_inventory_order_matches_python_sample_id_order(tmp_path, monkeypatch):
    import datasets.nr_store as nr_store

    sample_ids = ("z", "A", "a", "a2", "ä", "Å", "a10", "sample-0001")
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    for sample_id in sample_ids:
        payload = {
            "sample_id": sample_id,
            "content_sha256": "0" * 64,
            "relative_path": "unused.npz",
            "schema_version": 1,
        }
        key = hashlib.sha256(sample_id.encode()).hexdigest()
        (manifest_dir / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(nr_store, "_nr_record_memory_estimate", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(nr_store, "read_nr_record", lambda _root, manifest: manifest.sample_id)
    shard_ids = tuple(
        sample_id
        for shard in nr_store.iter_nr_record_shards(tmp_path, max_shard_bytes=3)
        for sample_id in shard
    )

    assert shard_ids == tuple(sorted(sample_ids))
