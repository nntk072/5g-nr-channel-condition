"""Safely audit official CAEZ numeric NPZ fields without unpickling metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

_LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_FILE_SIGNATURE = 0x04034B50
_REQUIRED_FIELDS = ("H", "noise_var", "timestamps", "pos")


def _numeric_member(path: Path, name: str) -> tuple[np.memmap, zipfile.ZipInfo]:
    """Map a stored numeric NPY member directly from an NPZ, without pickle."""
    member_name = f"{name}.npy"
    with zipfile.ZipFile(path) as archive:
        try:
            info = archive.getinfo(member_name)
        except KeyError as exc:
            raise ValueError(f"required CAEZ array is missing: {name}") from exc
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{member_name} must be ZIP_STORED for bounded-memory audit")

    with path.open("rb") as stream:
        stream.seek(info.header_offset)
        raw = stream.read(_LOCAL_FILE_HEADER.size)
        if len(raw) != _LOCAL_FILE_HEADER.size:
            raise ValueError(f"truncated ZIP local header for {member_name}")
        fields = _LOCAL_FILE_HEADER.unpack(raw)
        if fields[0] != _LOCAL_FILE_SIGNATURE:
            raise ValueError(f"invalid ZIP local header for {member_name}")
        filename_length, extra_length = fields[-2:]
        stream.seek(filename_length + extra_length, 1)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY format version {version} for {member_name}")
        payload_offset = stream.tell()

    dtype = np.dtype(dtype)
    if dtype.hasobject or dtype.kind not in "biufc":
        raise ValueError(f"{member_name} must contain a numeric NPY array, got {dtype}")
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    payload_end = payload_offset + expected_bytes
    member_end = (
        info.header_offset
        + _LOCAL_FILE_HEADER.size
        + filename_length
        + extra_length
        + info.compress_size
    )
    if expected_bytes < 0 or payload_end != member_end:
        raise ValueError(f"NPY payload size does not match ZIP member for {member_name}")
    array = np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=payload_offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )
    return array, info


def _array_summary(array: np.ndarray, *, chunk_rows: int) -> dict[str, Any]:
    if array.ndim == 0 or array.shape[0] == 0:
        raise ValueError("CAEZ arrays must have a non-empty sample axis")
    finite = 0
    total = int(array.size)
    for start in range(0, array.shape[0], chunk_rows):
        block = array[start : start + chunk_rows]
        finite += int(np.isfinite(block).sum())
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_values": finite,
        "total_values": total,
        "all_finite": finite == total,
    }


def audit(path: Path, *, chunk_rows: int = 2048) -> dict[str, Any]:
    """Return aggregate schema, integrity, and time-order metadata only."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    path = path.resolve(strict=True)
    if not zipfile.is_zipfile(path):
        raise ValueError("input is not a valid NPZ/ZIP archive")

    with zipfile.ZipFile(path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"ZIP CRC check failed for {corrupt_member}")
        members = {item.filename for item in archive.infolist()}
    arrays: dict[str, np.memmap] = {}
    member_metadata: dict[str, dict[str, Any]] = {}
    for name in _REQUIRED_FIELDS:
        array, info = _numeric_member(path, name)
        arrays[name] = array
        member_metadata[name] = {
            "member": info.filename,
            "uncompressed_bytes": info.file_size,
        }

    sample_counts = {int(array.shape[0]) for array in arrays.values()}
    if len(sample_counts) != 1:
        raise ValueError("CAEZ numeric fields do not share a sample count")
    summaries = {
        name: _array_summary(array, chunk_rows=chunk_rows) for name, array in arrays.items()
    }
    h_is_complex = bool(np.iscomplexobj(arrays["H"]))
    channel_observation = {
        "stored_dtype": str(arrays["H"].dtype),
        "complex_coefficients_present": h_is_complex,
        "phase_information_available_in_archive": h_is_complex,
        "g3_complex_csi_disposition": (
            "not_established_phase_reference_unverified"
            if h_is_complex
            else "ineligible_no_explicit_complex_phase"
        ),
        "scope_note": (
            "Complex dtype alone does not establish calibration or an absolute phase reference."
            if h_is_complex
            else "The archived H field is real-valued; complex phase cannot be reconstructed."
        ),
    }

    times = arrays["timestamps"]
    if times.ndim != 1 or times.dtype.kind not in "fiu":
        raise ValueError("timestamps must be a one-dimensional real numeric array")
    positions = arrays["pos"]
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("pos must be a two-dimensional array with at least two coordinates")
    timestamp_diffs = np.diff(times)
    strictly_increasing = bool(np.all(timestamp_diffs > 0))
    finite_timestamp_diffs = timestamp_diffs[np.isfinite(timestamp_diffs)]

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)

    return {
        "schema": "caez-preprocessed-npz-audit.v1",
        "source_file": path.name,
        "archive_bytes": path.stat().st_size,
        "archive_sha256": digest.hexdigest(),
        "zip_crc_check": "passed",
        "npz_members": sorted(members),
        "sample_count": sample_counts.pop(),
        "fields": summaries,
        "channel_observation": channel_observation,
        "member_sizes": member_metadata,
        "timestamps": {
            "strictly_increasing": strictly_increasing,
            "first": float(times[0]),
            "last": float(times[-1]),
            "min_positive_step": (
                float(finite_timestamp_diffs[finite_timestamp_diffs > 0].min())
                if np.any(finite_timestamp_diffs > 0)
                else None
            ),
            "max_step": (
                float(finite_timestamp_diffs.max()) if finite_timestamp_diffs.size else None
            ),
        },
        "sample_split_policy": "chronological contiguous blocks; never random per-slot split",
        "label_scope": "CAEZ supplies UE positions; this archive does not establish LOS/NLOS labels",
        "license": "CAEZ Dataset License v1.0; attribution required; no redistribution",
        "source": "https://iip.ethz.ch/datasets/caez.html",
        "required_citation": (
            "R. Wiesmayr, F. Zumegen, S. Taner, C. Dick, and C. Studer, "
            "CSI-Based User Positioning, Channel Charting, and Device Classification "
            "with an NVIDIA 5G Testbed, Asilomar Conference, 2025."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="local official preprocessed CAEZ .npz file")
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--output", type=Path, help="optional aggregate JSON report path")
    args = parser.parse_args()
    report = audit(args.archive, chunk_rows=args.chunk_rows)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
