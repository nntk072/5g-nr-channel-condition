"""Bounded and root-relative file reads for acquisition packages."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from evaluation.acquisition.constants import (
    CHUNK_BYTES,
    MAX_JSON_ARTIFACT_BYTES,
)


def _read_bounded_regular_file(path: Path, *, limit: int, description: str) -> bytes:
    """Read a bounded regular file without following a final symlink or blocking on a FIFO."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("platform lacks safe no-symlink file opening")
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{description} must be a regular non-symlink file")
        if info.st_size > limit:
            raise ValueError(f"{description} exceeds {limit} bytes")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(fd, min(CHUNK_BYTES, limit + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise ValueError(f"{description} exceeds {limit} bytes")
            chunks.append(chunk)
        final_info = os.fstat(fd)
        if (
            size != info.st_size
            or final_info.st_size != info.st_size
            or final_info.st_mtime_ns != info.st_mtime_ns
        ):
            raise ValueError(f"{description} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _safe_relative_path(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("inventory path must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    parts = value.split("/")
    if path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("inventory path must not be absolute or contain empty/dot segments")
    if any(":" in part for part in parts):
        raise ValueError("inventory path must not contain drive or stream syntax")
    return tuple(parts)


def _read_inventory_file(
    root_fd: int,
    parts: tuple[str, ...],
    limit: int,
    *,
    capture: bool,
) -> tuple[bytes | None, int, str]:
    """Open each path component without following symlinks, then read a bounded regular file."""
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or os.open not in os.supports_dir_fd
    ):
        raise ValueError("platform lacks safe directory-relative no-symlink file opening")
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    opened: list[int] = []
    try:
        parent_fd = root_fd
        for component in parts[:-1]:
            child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            opened.append(child_fd)
            parent_fd = child_fd
        file_fd = os.open(parts[-1], flags, dir_fd=parent_fd)
        opened.append(file_fd)
        info = os.fstat(file_fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("inventory path does not resolve to a regular file")
        if info.st_size > limit:
            raise ValueError(f"inventory file exceeds bounded read limit ({limit} bytes)")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(file_fd, min(CHUNK_BYTES, limit + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise ValueError(f"inventory file exceeds bounded read limit ({limit} bytes)")
            digest.update(chunk)
            if capture:
                if size > MAX_JSON_ARTIFACT_BYTES:
                    raise ValueError(
                        f"captured inventory artifact exceeds {MAX_JSON_ARTIFACT_BYTES} bytes"
                    )
                chunks.append(chunk)
        final_info = os.fstat(file_fd)
        if (
            size != info.st_size
            or final_info.st_size != info.st_size
            or final_info.st_mtime_ns != info.st_mtime_ns
        ):
            raise ValueError("inventory file changed while being read")
        content = b"".join(chunks) if capture else None
        return content, size, digest.hexdigest()
    finally:
        for fd in reversed(opened):
            os.close(fd)
