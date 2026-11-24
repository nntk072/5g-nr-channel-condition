"""Loaded native-library inventory: unavailable, oversized, unsafe and changing mappings."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from integrations import runtime_inventory


def _maps(tmp_path: Path, lines: list[str]) -> Path:
    maps = tmp_path / "maps"
    maps.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return maps


def _mapping(path: Path) -> str:
    return f"7f000000-7f001000 r-xp 00000000 08:01 123 {path}"


def test_missing_process_maps_is_reported_unavailable_not_complete(tmp_path: Path) -> None:
    report = runtime_inventory._loaded_native_library_identity(tmp_path / "absent-maps")
    assert report["status"] == "UNAVAILABLE"
    assert report["reason"] == "process_maps_unavailable"
    assert report["libraries"] == []
    assert report["total_hashed_bytes"] == 0


def test_oversized_process_maps_are_reported_incomplete(tmp_path: Path, monkeypatch) -> None:
    maps = _maps(tmp_path, [_mapping(tmp_path / "x.so")])
    monkeypatch.setattr(runtime_inventory, "_MAX_PROC_MAPS_BYTES", 8)
    report = runtime_inventory._loaded_native_library_identity(maps)
    assert report["status"] == "INCOMPLETE"
    assert report["reason"] == "process_maps_exceed_byte_limit"


def test_only_shared_object_mappings_are_hashed(tmp_path: Path) -> None:
    library = tmp_path / "libkernel.so.2"
    library.write_bytes(b"kernel")
    text_file = tmp_path / "notes.txt"
    text_file.write_bytes(b"not native")
    maps = _maps(
        tmp_path,
        [
            "short line without six fields",
            f"7f000000-7f001000 r-xp 00000000 08:01 123 {text_file}",
            "7f000000-7f001000 r-xp 00000000 08:01 123 [heap]",
            _mapping(library),
        ],
    )
    report = runtime_inventory._loaded_native_library_identity(maps)
    assert [row["path"] for row in report["libraries"]] == [str(library)]
    assert report["status"] == "COMPLETE"


def test_deleted_mapping_suffix_is_stripped_before_hashing(tmp_path: Path) -> None:
    library = tmp_path / "libdeleted.so"
    library.write_bytes(b"deleted-but-mapped")
    maps = _maps(tmp_path, [_mapping(library) + " (deleted)"])
    report = runtime_inventory._loaded_native_library_identity(maps)
    assert [row["path"] for row in report["libraries"]] == [str(library)]


def test_vanished_and_non_regular_mappings_are_unhashed_with_reasons(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe.so"
    os.mkfifo(fifo)
    vanished = tmp_path / "gone.so"
    maps = _maps(tmp_path, [_mapping(vanished), _mapping(fifo)])
    report = runtime_inventory._loaded_native_library_identity(maps)
    reasons = {row["path"]: row["reason"] for row in report["unhashed"]}
    assert reasons == {
        str(vanished): "mapped_file_unavailable",
        str(fifo): "mapped_file_not_regular",
    }
    assert report["status"] == "INCOMPLETE"
    assert report["reason"] == "one_or_more_mapped_libraries_unhashed"


def test_total_byte_budget_stops_hashing_later_libraries(tmp_path: Path, monkeypatch) -> None:
    first = tmp_path / "a.so"
    second = tmp_path / "b.so"
    first.write_bytes(b"x" * 10)
    second.write_bytes(b"y" * 10)
    monkeypatch.setattr(runtime_inventory, "_MAX_NATIVE_LIBRARY_TOTAL_BYTES", 15)
    maps = _maps(tmp_path, [_mapping(first), _mapping(second)])
    report = runtime_inventory._loaded_native_library_identity(maps)
    assert [row["path"] for row in report["libraries"]] == [str(first)]
    assert report["unhashed"] == [
        {"path": str(second), "reason": "total_byte_limit_exceeded", "size_bytes": 10}
    ]
    assert report["total_hashed_bytes"] == 10


def test_library_that_changes_while_hashed_is_refused(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "moving.so"
    library.write_bytes(b"before")
    maps = _maps(tmp_path, [_mapping(library)])
    real_lstat = os.lstat
    calls = {"count": 0}

    def drifting_lstat(path, *args, **kwargs):
        result = real_lstat(path, *args, **kwargs)
        if str(path) == str(library):
            calls["count"] += 1
            if calls["count"] >= 2:
                # Simulate a rewrite between the identity check and the after-hash check.
                class Drifted:
                    def __getattr__(self, name):
                        return getattr(result, name)

                drifted = Drifted()
                object.__setattr__(drifted, "st_mtime_ns", result.st_mtime_ns + 1)
                return drifted
        return result

    monkeypatch.setattr(runtime_inventory.os, "lstat", drifting_lstat)
    report = runtime_inventory._loaded_native_library_identity(maps)
    assert report["libraries"] == []
    assert report["unhashed"][0]["reason"] == "mapped_file_changed_or_unhashable"


def test_identity_hash_refuses_files_that_exceed_their_bound(tmp_path: Path) -> None:
    payload = tmp_path / "identity.bin"
    payload.write_bytes(b"0123456789")
    with pytest.raises(ValueError, match="exceeds the bounded size limit"):
        runtime_inventory._sha256(payload, max_bytes=4)
    assert (
        runtime_inventory._sha256(payload, max_bytes=10)
        == hashlib.sha256(b"0123456789").hexdigest()
    )
