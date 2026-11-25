import hashlib
import os

import pytest
from evaluation.acquisition import files
from evaluation.acquisition.files import (
    _read_bounded_regular_file,
    _read_inventory_file,
    _safe_relative_path,
)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "nonempty POSIX relative path"),
        ("", "nonempty POSIX relative path"),
        ("a\\b", "nonempty POSIX relative path"),
        ("a\x00b", "nonempty POSIX relative path"),
        ("/absolute/file", "absolute or contain empty/dot segments"),
        ("a//b", "absolute or contain empty/dot segments"),
        ("a/./b", "absolute or contain empty/dot segments"),
        ("a/../b", "absolute or contain empty/dot segments"),
        ("C:/drive/file", "drive or stream syntax"),
    ],
)
def test_safe_relative_path_rejects_noncanonical_paths(value, message):
    with pytest.raises(ValueError, match=message):
        _safe_relative_path(value)


def test_safe_relative_path_accepts_nested_posix_path():
    assert _safe_relative_path("receipts/episode-01.json") == ("receipts", "episode-01.json")


def test_bounded_reader_accepts_regular_file_and_rejects_other_file_types(tmp_path):
    source = tmp_path / "source.json"
    source.write_bytes(b"{}")
    assert _read_bounded_regular_file(source, limit=2, description="artifact") == b"{}"

    with pytest.raises(ValueError, match="exceeds 1 bytes"):
        _read_bounded_regular_file(source, limit=1, description="artifact")

    with pytest.raises(ValueError, match="regular non-symlink file"):
        _read_bounded_regular_file(tmp_path, limit=8, description="artifact")

    symlink = tmp_path / "link.json"
    symlink.symlink_to(source)
    with pytest.raises(OSError):
        _read_bounded_regular_file(symlink, limit=8, description="artifact")

    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular non-symlink file"):
        _read_bounded_regular_file(fifo, limit=8, description="artifact")


def test_bounded_reader_detects_file_mutation_during_read(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_bytes(b"{}")
    actual_read = os.read
    changed = False

    def mutate_after_first_chunk(descriptor, count):
        nonlocal changed
        chunk = actual_read(descriptor, count)
        if chunk and not changed:
            changed = True
            source.write_bytes(b'{"changed": true}')
        return chunk

    monkeypatch.setattr(files.os, "read", mutate_after_first_chunk)
    with pytest.raises(ValueError, match="changed while being read"):
        _read_bounded_regular_file(source, limit=64, description="artifact")


def test_bounded_reader_rejects_concurrent_growth_past_limit(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_bytes(b"x")
    actual_read = os.read

    def grow_before_read(descriptor, count):
        source.write_bytes(b"expanded")
        monkeypatch.setattr(files.os, "read", actual_read)
        return actual_read(descriptor, count)

    monkeypatch.setattr(files.os, "read", grow_before_read)
    with pytest.raises(ValueError, match="exceeds 1 bytes"):
        _read_bounded_regular_file(source, limit=1, description="artifact")


@pytest.mark.parametrize("capture", [False, True])
def test_inventory_reader_returns_exact_size_hash_and_optional_content(tmp_path, capture):
    root = tmp_path / "package"
    nested = root / "receipts"
    nested.mkdir(parents=True)
    payload = b'{"receipt": 1}'
    (nested / "one.json").write_bytes(payload)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        content, size, digest = _read_inventory_file(
            root_fd, ("receipts", "one.json"), 64, capture=capture
        )
    finally:
        os.close(root_fd)

    assert content == (payload if capture else None)
    assert size == len(payload)
    assert digest == hashlib.sha256(payload).hexdigest()


def test_inventory_reader_rejects_symlink_components_nonfiles_and_oversize(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    (root / "data.json").write_bytes(b"{}")
    (root / "directory").mkdir()
    (root / "alias.json").symlink_to(root / "data.json")
    (root / "alias-directory").symlink_to(root / "directory", target_is_directory=True)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(OSError):
            _read_inventory_file(root_fd, ("alias.json",), 16, capture=False)
        with pytest.raises(OSError):
            _read_inventory_file(root_fd, ("alias-directory", "missing"), 16, capture=False)
        with pytest.raises(ValueError, match="regular file"):
            _read_inventory_file(root_fd, ("directory",), 16, capture=False)
        with pytest.raises(ValueError, match="bounded read limit"):
            _read_inventory_file(root_fd, ("data.json",), 1, capture=False)
    finally:
        os.close(root_fd)


def test_inventory_reader_enforces_capture_budget(tmp_path, monkeypatch):
    root = tmp_path / "package"
    root.mkdir()
    (root / "receipt.json").write_bytes(b"{}")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    monkeypatch.setattr(files, "MAX_JSON_ARTIFACT_BYTES", 1)
    try:
        with pytest.raises(ValueError, match="captured inventory artifact exceeds 1 bytes"):
            _read_inventory_file(root_fd, ("receipt.json",), 8, capture=True)
        content, size, _digest = _read_inventory_file(root_fd, ("receipt.json",), 8, capture=False)
    finally:
        os.close(root_fd)
    assert content is None
    assert size == 2


def test_inventory_reader_rejects_concurrent_growth_past_limit(tmp_path, monkeypatch):
    root = tmp_path / "package"
    root.mkdir()
    source = root / "receipt.json"
    source.write_bytes(b"x")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    actual_read = os.read

    def grow_before_read(descriptor, count):
        source.write_bytes(b"expanded")
        monkeypatch.setattr(files.os, "read", actual_read)
        return actual_read(descriptor, count)

    monkeypatch.setattr(files.os, "read", grow_before_read)
    try:
        with pytest.raises(ValueError, match=r"bounded read limit \(1 bytes\)"):
            _read_inventory_file(root_fd, ("receipt.json",), 1, capture=False)
    finally:
        os.close(root_fd)


def test_inventory_reader_detects_mutation_during_capture(tmp_path, monkeypatch):
    root = tmp_path / "package"
    root.mkdir()
    source = root / "receipt.json"
    source.write_bytes(b"{}")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    actual_read = os.read
    changed = False

    def mutate_after_first_chunk(descriptor, count):
        nonlocal changed
        chunk = actual_read(descriptor, count)
        if chunk and not changed:
            changed = True
            source.write_bytes(b'{"changed": true}')
        return chunk

    monkeypatch.setattr(files.os, "read", mutate_after_first_chunk)
    try:
        with pytest.raises(ValueError, match="changed while being read"):
            _read_inventory_file(root_fd, ("receipt.json",), 64, capture=True)
    finally:
        os.close(root_fd)


def test_inventory_reader_rejects_platform_without_dir_fd_open_support(tmp_path, monkeypatch):
    root = tmp_path / "package"
    root.mkdir()
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    monkeypatch.setattr(files.os, "open", lambda *_args, **_kwargs: -1)
    try:
        with pytest.raises(ValueError, match="platform lacks safe directory-relative"):
            _read_inventory_file(root_fd, ("receipt.json",), 64, capture=False)
    finally:
        os.close(root_fd)
