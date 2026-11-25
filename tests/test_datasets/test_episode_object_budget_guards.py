"""Episode object decoding enforces content-addressing and conservative budgets."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import datasets.episode_store.storage as storage
import numpy as np
import pytest


def _write_object(tmp_path: Path, payload: bytes) -> dict[str, str]:
    digest = hashlib.sha256(payload).hexdigest()
    object_path = tmp_path / f"{digest}.npz"
    object_path.write_bytes(payload)
    return {"path": object_path.name, "sha256": digest}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not-an-npz", "not a readable NPZ archive"),
        (b"", "not a readable NPZ archive"),
    ],
)
def test_object_size_estimate_rejects_non_npz_bytes(tmp_path: Path, payload: bytes, message: str):
    reference = _write_object(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        storage._episode_object_size_estimate(tmp_path, reference, max_shard_bytes=1024)


def test_object_size_estimate_rejects_non_npy_archive_members(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("metadata.txt", b"not-an-array")
    reference = _write_object(tmp_path, buffer.getvalue())

    with pytest.raises(ValueError, match="members must be NPY arrays"):
        storage._episode_object_size_estimate(tmp_path, reference, max_shard_bytes=1024)


def test_object_size_estimate_rejects_duplicate_npz_members(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("array.npy", b"first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            # The test archive intentionally contains a repeated member name.
            archive.writestr("array.npy", b"second")
    reference = _write_object(tmp_path, buffer.getvalue())

    with pytest.raises(ValueError, match="unique nonempty members"):
        storage._episode_object_size_estimate(tmp_path, reference, max_shard_bytes=1024)


def test_object_size_estimate_enforces_decoded_array_memory_budget(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, array=np.zeros(2048, dtype=np.float64))
    payload = buffer.getvalue()
    reference = _write_object(tmp_path, payload)

    with pytest.raises(ValueError, match="estimate.*exceeds max_shard_bytes"):
        storage._episode_object_size_estimate(tmp_path, reference, max_shard_bytes=len(payload) + 1)


def test_object_size_estimate_rejects_hash_mismatch(tmp_path: Path) -> None:
    reference = _write_object(tmp_path, b"content")
    object_path = tmp_path / reference["path"]
    object_path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="content hash does not match"):
        storage._episode_object_size_estimate(tmp_path, reference, max_shard_bytes=1024)
