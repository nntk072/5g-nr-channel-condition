"""CAEZ archive reader: local-header, NPY-version and payload-size rules on crafted members."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from datasets import caez
from tests.test_datasets.test_caez_npz_audit import _archive


def _rewrite_member(path: Path, name: str, payload: bytes) -> None:
    """Replace one stored member's bytes, keeping the ZIP directory consistent."""
    with zipfile.ZipFile(path) as source:
        members = {info.filename: source.read(info.filename) for info in source.infolist()}
    members[name] = payload
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as target:
        for member, data in members.items():
            target.writestr(member, data)


def _npy(array: np.ndarray, version=(1, 0)) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, version=version)
    return buffer.getvalue()


def test_truncated_local_header_is_reported(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "a.npz")
    with zipfile.ZipFile(archive) as zf:
        offset = zf.getinfo("H.npy").header_offset
    raw = bytearray(archive.read_bytes())
    # Truncate the archive immediately after the H member's offset to starve the header read.
    truncated = tmp_path / "truncated.npz"
    truncated.write_bytes(bytes(raw[: offset + 10]))
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        caez._numeric_member(truncated, "H")


def test_corrupted_local_signature_is_reported(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "a.npz")
    with zipfile.ZipFile(archive) as zf:
        offset = zf.getinfo("H.npy").header_offset
    raw = bytearray(archive.read_bytes())
    raw[offset : offset + 4] = b"XXXX"
    corrupted = tmp_path / "bad-signature.npz"
    corrupted.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="invalid ZIP local header for H.npy"):
        caez._numeric_member(corrupted, "H")


def test_unsupported_npy_version_is_reported(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "a.npz")
    payload = bytearray(_npy(np.ones(4, dtype=np.float64)))
    payload[6], payload[7] = 3, 0
    _rewrite_member(archive, "noise_var.npy", bytes(payload))
    with pytest.raises(ValueError, match="unsupported NPY format version"):
        caez._numeric_member(archive, "noise_var")


def test_version_two_headers_are_accepted(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "a.npz")
    _rewrite_member(archive, "noise_var.npy", _npy(np.full(12, 0.01), version=(2, 0)))
    memmap, info = caez._numeric_member(archive, "noise_var")
    assert memmap.shape == (12,)
