"""Bounded reads and preflight checks for evidence archives and estimator sources."""

from __future__ import annotations

import io
import json
import math
import os
import stat
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    _HEX,
    _MAX_WAVEFORM_EVIDENCE_BYTES,
    _MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES,
)
from .hashing import (
    _sha256,
)


def _curve_report_schema_status(
    report: dict[str, Any], schema_path: Path
) -> tuple[str | None, bool]:
    """Return the exact schema digest and validate the curve against those bytes."""
    try:
        schema_bytes = schema_path.read_bytes()
    except OSError:
        return None, False
    digest = _sha256(schema_bytes)
    try:
        import jsonschema
    except ImportError:
        return digest, False
    try:
        schema_document = json.loads(schema_bytes.decode("utf-8"))
        jsonschema.Draft202012Validator(schema_document).validate(report)
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        jsonschema.exceptions.SchemaError,
        jsonschema.exceptions.ValidationError,
    ):
        return digest, False
    return digest, True


def _estimator_source_checks(
    estimator: dict[str, Any],
    *,
    source_bytes: bytes | None,
    committed_bytes: bytes | None,
) -> dict[str, bool | None]:
    """Verify the fixed estimator identity against disk and an optional Git blob."""
    if estimator.get("name") == "sionna_default_ls":
        valid = estimator == {"name": "sionna_default_ls"}
        return {
            "estimator_source_worktree_digest": valid,
            "estimator_committed_source_digest": True if valid else False,
        }
    if estimator.get("name") != "finite_delay_dft":
        raise ValueError("unsupported channel estimator identity")

    digest = estimator.get("implementation_sha256")
    valid = (
        estimator.get("implementation_path") == "experiments/cp_bounded_dft_estimator.py"
        and estimator.get("min_delay_samples") == 0
        and estimator.get("max_delay_samples") == 7
        and isinstance(digest, str)
        and len(digest) == 64
        and set(digest) <= _HEX
    )
    return {
        "estimator_source_worktree_digest": (
            valid and source_bytes is not None and digest == _sha256(source_bytes)
        ),
        "estimator_committed_source_digest": (
            None if committed_bytes is None else valid and digest == _sha256(committed_bytes)
        ),
    }


def _read_bounded_evidence_file(path: Path, *, maximum_bytes: int) -> bytes:
    """Read one immutable, single-link regular file without following its final symlink."""
    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes < 0:
        raise ValueError("maximum_bytes must be a nonnegative integer")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError("platform lacks no-follow file opening required for waveform evidence")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("waveform evidence must be a single-link regular file")
        if before.st_size > maximum_bytes:
            raise ValueError("waveform evidence exceeds the configured size limit")

        contents = bytearray()
        while len(contents) <= maximum_bytes:
            remaining = maximum_bytes + 1 - len(contents)
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            contents.extend(block)
        if len(contents) > maximum_bytes:
            raise ValueError("waveform evidence exceeds the configured size limit")

        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        identity_at_path = (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_mode,
            path_after.st_nlink,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        if (
            identity_before != identity_after
            or identity_after != identity_at_path
            or not stat.S_ISREG(path_after.st_mode)
            or path_after.st_nlink != 1
            or len(contents) != before.st_size
        ):
            raise ValueError("waveform evidence changed while being read")
        return bytes(contents)
    finally:
        os.close(descriptor)


def _preflight_waveform_archive(
    evidence_bytes: bytes, *, required_arrays: set[str]
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """Check ZIP and NPY metadata before NumPy materializes decompressed arrays."""
    if len(evidence_bytes) > _MAX_WAVEFORM_EVIDENCE_BYTES:
        raise ValueError("waveform evidence exceeds the configured size limit")
    expected_members = {f"{name}.npy" for name in required_arrays}
    with zipfile.ZipFile(io.BytesIO(evidence_bytes)) as compressed:
        members = compressed.infolist()
        if (
            len(members) != len(expected_members)
            or {member.filename for member in members} != expected_members
            or sum(member.file_size for member in members)
            > _MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES
        ):
            raise ValueError("waveform evidence archive exceeds its member or size contract")
        metadata: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
        for info in members:
            with compressed.open(info) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                        member, max_header_size=64 * 1024
                    )
                elif version == (2, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                        member, max_header_size=64 * 1024
                    )
                else:
                    raise ValueError("waveform evidence uses an unsupported NPY format version")
                if (
                    fortran_order
                    or not shape
                    or len(shape) > 8
                    or any(
                        isinstance(size, bool) or not isinstance(size, int) or size < 1
                        for size in shape
                    )
                ):
                    raise ValueError("waveform evidence NPY shape or memory order is invalid")
                if dtype.hasobject or dtype.kind not in "iufc":
                    raise ValueError("waveform evidence contains an unsupported NPY dtype")
                payload_bytes = math.prod(shape) * dtype.itemsize
                if payload_bytes != info.file_size - member.tell():
                    raise ValueError("waveform evidence NPY shape disagrees with its member size")
                metadata[Path(info.filename).stem] = (tuple(shape), dtype)
    return metadata
