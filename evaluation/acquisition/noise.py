"""Noise RMS summary for acquisition-package verification."""

from __future__ import annotations

import io
import math


def _noise_rms_summary(data: bytes) -> tuple[float, float, int, tuple[int, ...]]:
    """Summarize one NPY capture of noise-only channel estimates."""
    import numpy as np

    stream = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY format version {version!r}; expected 1.0 or 2.0")
    except (ValueError, OSError, EOFError, OverflowError) as error:
        raise ValueError(f"noise measurement has an invalid NPY header: {error}") from error
    if dtype.kind != "c" or dtype.itemsize not in (8, 16):
        raise ValueError("noise-only CFR estimate must contain complex64 or complex128 values")
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError("noise-only CFR estimate must contain at least one complex value")
    sample_count = math.prod(shape)
    data_offset = stream.tell()
    expected_data_bytes = sample_count * dtype.itemsize
    if expected_data_bytes != len(data) - data_offset:
        raise ValueError("noise measurement NPY payload is truncated or has trailing bytes")
    values = np.frombuffer(data, dtype=dtype, count=sample_count, offset=data_offset)
    real = values.real.astype(np.float64, copy=False).reshape(-1)
    imag = values.imag.astype(np.float64, copy=False).reshape(-1)
    if not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError("noise-only CFR estimate values must be finite")
    scale = max(float(np.max(np.abs(real))), float(np.max(np.abs(imag))))
    if scale == 0.0:
        return 0.0, 0.0, sample_count, tuple(shape)
    normalized_magnitude = np.hypot(real / scale, imag / scale)
    normalized_sum_squares = float(np.dot(normalized_magnitude, normalized_magnitude))
    if not math.isfinite(normalized_sum_squares):
        raise ValueError("noise measurement RMS accumulation is not finite")
    return scale, normalized_sum_squares, sample_count, tuple(shape)
