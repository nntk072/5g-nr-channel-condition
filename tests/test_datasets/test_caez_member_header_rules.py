"""CAEZ NPZ audit: members must be stored numeric arrays whose payload matches the ZIP member exactly."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from datasets import caez


def _npz(path: Path, **arrays) -> Path:
    np.savez(path, **arrays)
    return path


def test_stored_numeric_member_is_mapped_without_pickle(tmp_path: Path) -> None:
    path = _npz(tmp_path / "caez.npz", values=np.arange(6, dtype=np.float32).reshape(2, 3))
    member, info = caez._numeric_member(path, "values")
    assert member.shape == (2, 3)
    assert info.filename == "values.npy"


def test_missing_member_is_named_in_the_error(tmp_path: Path) -> None:
    path = _npz(tmp_path / "caez.npz", values=np.zeros(2))
    with pytest.raises(ValueError, match="required CAEZ array is missing: absent"):
        caez._numeric_member(path, "absent")


def test_compressed_members_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "compressed.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("values.npy", b"\x93NUMPY")
    with pytest.raises(ValueError, match="must be ZIP_STORED"):
        caez._numeric_member(path, "values")


def test_member_header_beyond_end_of_file_is_truncated(tmp_path: Path, monkeypatch) -> None:
    path = _npz(tmp_path / "caez.npz", values=np.zeros(2))
    real_getinfo = zipfile.ZipFile.getinfo

    def beyond_end(self, name):
        info = real_getinfo(self, name)
        info.header_offset = path.stat().st_size + 1
        return info

    monkeypatch.setattr(zipfile.ZipFile, "getinfo", beyond_end)
    with pytest.raises(ValueError, match="truncated ZIP local header"):
        caez._numeric_member(path, "values")


def test_payload_must_match_the_zip_member_size(tmp_path: Path, monkeypatch) -> None:
    path = _npz(tmp_path / "caez.npz", values=np.zeros(4, dtype=np.float64))
    real_header = np.lib.format.read_array_header_1_0

    def wrong_shape(stream):
        _shape, fortran, dtype = real_header(stream)
        return (1,), fortran, dtype

    monkeypatch.setattr(np.lib.format, "read_array_header_1_0", wrong_shape)
    with pytest.raises(ValueError, match="payload size does not match"):
        caez._numeric_member(path, "values")
