"""Waveform evidence for ordered impairments receiver."""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    _MAX_WAVEFORM_EVIDENCE_BYTES,
    _MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES,
    _MAX_WAVEFORM_NPZ_MEMBER_NAME_BYTES,
    _MAX_WAVEFORM_NPZ_MEMBERS,
)
from .hashing import (
    _sha256,
)


class _BoundedBytesIO(io.BytesIO):
    """BytesIO that rejects writes whose high-water mark exceeds a fixed cap."""

    def __init__(self, maximum_bytes: int) -> None:
        super().__init__()
        self._maximum_bytes = maximum_bytes
        self._high_water_mark = 0

    def write(self, content: Any) -> int:
        high_water_mark = max(self._high_water_mark, self.tell() + len(content))
        if high_water_mark > self._maximum_bytes:
            raise ValueError("compressed waveform evidence exceeds the configured size limit")
        written = super().write(content)
        self._high_water_mark = max(self._high_water_mark, self.tell())
        return written


def _write_bounded_npz(
    destination: _BoundedBytesIO,
    *,
    channel_count: int,
    channel_arrays: dict[str, list[np.ndarray]],
    scalar_arrays: dict[str, np.ndarray],
) -> None:
    """Stream channel-major arrays into compressed NPY members without stacking."""
    if isinstance(channel_count, bool) or not isinstance(channel_count, int) or channel_count < 1:
        raise ValueError("channel_count must be a positive integer")
    entries: dict[str, tuple[list[np.ndarray], bool]] = {
        **{name: (chunks, True) for name, chunks in channel_arrays.items()},
        **{name: ([array], False) for name, array in scalar_arrays.items()},
    }
    if set(channel_arrays).intersection(scalar_arrays):
        raise ValueError("NPZ member names must be unique")
    if not entries:
        raise ValueError("at least one NPZ array is required")
    if len(entries) > _MAX_WAVEFORM_NPZ_MEMBERS:
        raise ValueError("waveform evidence has too many NPZ members")

    prepared: dict[str, tuple[list[np.ndarray], tuple[int, ...], np.dtype[Any]]] = {}
    uncompressed_bytes = 0
    for name, (chunks, channel_major) in entries.items():
        if (
            not isinstance(name, str)
            or not name
            or not name.isascii()
            or len(name) > _MAX_WAVEFORM_NPZ_MEMBER_NAME_BYTES
            or not name[0].isalnum()
            or any(not (character.isalnum() or character in "_.-") for character in name)
            or not chunks
        ):
            raise ValueError("NPZ array names must be short safe identifiers with nonempty chunks")
        if channel_major and len(chunks) != channel_count:
            raise ValueError(f"NPZ channel count disagrees for member {name}")
        first = np.asarray(chunks[0])
        if first.dtype.hasobject or first.dtype.kind not in "iufc":
            raise ValueError(f"NPZ member {name} has an unsupported dtype")
        for chunk in chunks:
            value = np.asarray(chunk)
            if value.shape != first.shape or value.dtype != first.dtype:
                raise ValueError(f"NPZ member {name} has inconsistent channel shape or dtype")
            uncompressed_bytes += value.nbytes
        shape = (channel_count, *first.shape) if channel_major else first.shape
        prepared[name] = (chunks, shape, first.dtype)
    if uncompressed_bytes + 64 * 1024 > _MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES:
        raise ValueError("uncompressed waveform evidence exceeds the configured size limit")

    with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, (chunks, shape, dtype) in sorted(prepared.items()):
            with archive.open(f"{name}.npy", mode="w", force_zip64=True) as member:
                np.lib.format.write_array_header_2_0(
                    member,
                    {
                        "descr": np.lib.format.dtype_to_descr(dtype),
                        "fortran_order": False,
                        "shape": shape,
                    },
                )
                for chunk in chunks:
                    contiguous = np.ascontiguousarray(chunk)
                    member.write(contiguous.tobytes(order="C"))


def _combine_snr_curve_waveform_evidence(
    point_paths: tuple[Path, ...],
    *,
    output_path: Path,
    snr_values: tuple[float, ...],
) -> dict[str, Any]:
    """Combine per-point evidence while storing shared raw draws only once."""
    if len(point_paths) != len(snr_values) or not point_paths:
        raise ValueError("SNR waveform evidence paths must match the nonempty SNR grid")
    raw_names = (
        "channel_seeds",
        "tx_waveform",
        "tx_phase_noise_rad",
        "rx_phase_noise_rad",
        "actual_capture",
        "awgn_standard_normal_real",
        "awgn_standard_normal_imag",
    )
    condition_names = {
        "clean": "clean_receiver_input",
        "uncorrected": "uncorrected_receiver_input",
        "corrected": "corrected_receiver_input",
    }
    expected_point_names = {*raw_names, *condition_names.values()}
    planned_uncompressed_bytes = 0
    staging_compressed_bytes = 0
    for point_index, point_path in enumerate(point_paths):
        staging_compressed_bytes += point_path.stat().st_size
        with zipfile.ZipFile(point_path) as point_archive:
            members = point_archive.infolist()
            if len(members) != len(expected_point_names) or {
                member.filename for member in members
            } != {f"{name}.npy" for name in expected_point_names}:
                raise ValueError("per-point waveform evidence has an unexpected array roster")
            member_bytes = {Path(member.filename).stem: member.file_size for member in members}
            planned_uncompressed_bytes += sum(
                member_bytes[name] for name in (raw_names if point_index == 0 else ())
            )
            planned_uncompressed_bytes += sum(
                member_bytes[name] for name in condition_names.values()
            )
    if planned_uncompressed_bytes + 64 * 1024 > _MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES:
        raise ValueError(
            "multi-SNR waveform evidence exceeds the configured uncompressed size limit"
        )
    if staging_compressed_bytes > 2 * _MAX_WAVEFORM_EVIDENCE_BYTES:
        raise ValueError("multi-SNR waveform evidence staging exceeds its disk size limit")
    raw_arrays: dict[str, np.ndarray] | None = None
    condition_chunks: dict[str, list[np.ndarray]] = {
        f"{name}_receiver_input_by_snr": [] for name in condition_names
    }
    uncompressed_bytes = 0
    for point_index, point_path in enumerate(point_paths):
        with np.load(point_path, allow_pickle=False) as archive:
            if set(archive.files) != expected_point_names:
                raise ValueError("per-point waveform evidence has an unexpected array roster")
            if point_index == 0:
                raw_arrays = {name: archive[name] for name in raw_names}
                uncompressed_bytes += sum(array.nbytes for array in raw_arrays.values())
            else:
                if raw_arrays is None:
                    raise RuntimeError("shared raw waveform arrays were not initialized")
                for name in raw_names:
                    if not np.array_equal(raw_arrays[name], archive[name]):
                        raise ValueError("raw waveform draws differ across paired SNR points")
            for condition, source_name in condition_names.items():
                values = archive[source_name]
                if not np.iscomplexobj(values) or not np.isfinite(values).all():
                    raise ValueError(f"{condition} waveform evidence is invalid")
                uncompressed_bytes += values.nbytes
                if uncompressed_bytes + 64 * 1024 > _MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES:
                    raise ValueError(
                        "multi-SNR waveform evidence exceeds the configured uncompressed size limit"
                    )
                condition_chunks[f"{condition}_receiver_input_by_snr"].append(values)
    if raw_arrays is None:
        raise ValueError("SNR waveform evidence did not contain shared raw draws")
    evidence_buffer = _BoundedBytesIO(_MAX_WAVEFORM_EVIDENCE_BYTES)
    _write_bounded_npz(
        evidence_buffer,
        channel_count=len(snr_values),
        channel_arrays=condition_chunks,
        scalar_arrays=raw_arrays,
    )
    evidence_bytes = evidence_buffer.getvalue()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(evidence_bytes)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.link(temporary_name, output_path)
        Path(temporary_name).unlink()
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return {
        "schema": "umcd.g5_ordered_impairments_waveform_evidence.v4",
        "file_name": output_path.name,
        "sha256": _sha256(evidence_bytes),
        "byte_count": len(evidence_bytes),
        "channel_count": int(raw_arrays["channel_seeds"].shape[0]),
        "snr_point_count": len(snr_values),
        "snr_db": list(snr_values),
        "arrays": sorted([*raw_names, *condition_chunks]),
    }
