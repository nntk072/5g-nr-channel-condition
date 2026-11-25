"""Stable episode-store file reads reject links, bounds, and observed races."""

from __future__ import annotations

import os
from pathlib import Path

import datasets.episode_store.storage as storage
import numpy as np
import pytest


def test_stable_file_reader_rejects_hard_links(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(b"record")
    try:
        os.link(path, tmp_path / "alias.json")
    except OSError as error:
        pytest.skip(f"hard-link creation is unavailable: {error}")

    with pytest.raises(ValueError, match="single-link regular file"):
        storage._stable_file_bytes(path, label="fixture")


def test_stable_file_reader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    path = tmp_path / "record.fifo"
    try:
        os.mkfifo(path)
    except OSError as error:
        pytest.skip(f"FIFO creation is unavailable: {error}")

    with pytest.raises(ValueError, match="single-link regular file"):
        storage._stable_file_bytes(path, label="fixture")


def test_stable_file_reader_checks_initial_size_bound(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(b"record")

    with pytest.raises(ValueError, match="configured byte limit"):
        storage._stable_file_bytes(path, label="fixture", max_bytes=2)


def test_stable_file_reader_detects_replacement_while_opening(tmp_path, monkeypatch) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(b"original")
    real_open = storage.os.open
    replaced = False

    def replace_before_open(target, flags, *args, **kwargs):
        nonlocal replaced
        if target == path and not replaced:
            replaced = True
            replacement = tmp_path / "replacement.json"
            replacement.write_bytes(b"replacement")
            replacement.replace(path)
        return real_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(storage.os, "open", replace_before_open)

    with pytest.raises(ValueError, match="changed while opening"):
        storage._stable_file_bytes(path, label="fixture")


def test_stable_file_reader_detects_content_change_during_read(tmp_path, monkeypatch) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(b"record")
    real_read = storage.os.read
    changed = False

    def append_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        data = real_read(descriptor, size)
        if data and not changed:
            changed = True
            with path.open("ab") as stream:
                stream.write(b" changed")
        return data

    monkeypatch.setattr(storage.os, "read", append_after_first_read)

    with pytest.raises(ValueError, match="changed while being read"):
        storage._stable_file_bytes(path, label="fixture")


def test_stable_file_reader_enforces_bound_when_file_grows_during_read(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(b"record")
    real_read = storage.os.read
    changed = False

    def append_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        data = real_read(descriptor, size)
        if data and not changed:
            changed = True
            with path.open("ab") as stream:
                stream.write(b" extra")
        return data

    monkeypatch.setattr(storage.os, "read", append_after_first_read)

    with pytest.raises(ValueError, match="configured byte limit"):
        storage._stable_file_bytes(path, label="fixture", max_bytes=6)


@pytest.mark.parametrize("value", [{"not_json": {1, 2}}, {"not_finite": float("nan")}])
def test_episode_json_serializer_rejects_unsupported_metadata(value) -> None:
    with pytest.raises(TypeError, match="finite JSON-compatible"):
        storage._json_bytes(value)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xff", "strict UTF-8 JSON"),
        (b'{"key":1,"key":2}', "duplicate JSON object key"),
        (b'{"value":NaN}', "non-finite JSON constant is forbidden"),
    ],
)
def test_episode_json_parser_rejects_ambiguous_or_nonstandard_bytes(payload, message) -> None:
    with pytest.raises(ValueError, match=message):
        storage._parse_json_bytes(payload, label="fixture")


@pytest.mark.parametrize(
    "array",
    [np.asarray([object()], dtype=object), np.asarray(["text"], dtype="U4")],
)
def test_episode_npz_packer_rejects_non_numeric_arrays(array) -> None:
    with pytest.raises(TypeError, match="non-object numeric dtype"):
        storage._pack_npz({}, invalid=array)


def test_immutable_episode_object_cannot_be_replaced_with_different_bytes(tmp_path: Path) -> None:
    path = tmp_path / "immutable.npz"
    path.write_bytes(b"original")

    with pytest.raises(FileExistsError, match="different bytes"):
        storage._publish_immutable(path, b"replacement")


def test_immutable_episode_writer_propagates_tempfile_creation_failure(
    tmp_path, monkeypatch
) -> None:
    error = OSError("temporary episode file unavailable")

    def fail_temporary_file(*args, **kwargs):
        raise error

    monkeypatch.setattr(storage.tempfile, "NamedTemporaryFile", fail_temporary_file)

    with pytest.raises(OSError, match="temporary episode file unavailable") as raised:
        storage._publish_immutable(tmp_path / "unpublished.npz", b"payload")

    assert raised.value is error
    assert not (tmp_path / "unpublished.npz").exists()
