"""Scoring numeric helpers and analytical sample admission (remaining branches)."""

from __future__ import annotations

import numpy as np
import pytest
from evaluation.acquisition.noise import _noise_rms_summary
from evaluation.scoring.numeric import finite_real_array, finite_real_number


def test_numeric_scalars_reject_nonfinite_numpy_values() -> None:
    with pytest.raises(ValueError, match="must be a finite real number"):
        finite_real_number(np.float64("nan"), "x")


def test_array_parsing_reports_element_names_and_dimensions() -> None:
    with pytest.raises(ValueError, match="must be a 2-dimensional finite real array"):
        finite_real_array([1.0, 2.0], "x", ndim=2)
    with pytest.raises(ValueError, match="must be a finite real array"):
        finite_real_array("text", "x")


def test_array_conversion_errors_are_reported_as_value_errors() -> None:
    class InvalidArray:
        def __array__(self, dtype=None, copy=None):
            raise TypeError("cannot expose array data")

    with pytest.raises(ValueError, match="must be a finite real array"):
        finite_real_array(InvalidArray(), "x")


def _npy_v2(values: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, values, version=(2, 0))
    return buffer.getvalue()


def test_noise_summary_accepts_version_two_complex128_and_zero_captures() -> None:
    values = np.array([[1 + 1j, 0j]], dtype=np.complex128)
    scale, sum_squares, count, shape = _noise_rms_summary(_npy_v2(values))
    assert scale == pytest.approx(1.0)
    assert count == 2 and shape == (1, 2)
    assert sum_squares == pytest.approx(2.0)  # |1+1j|^2 after scaling by the peak component
    scale, sum_squares, count, shape = _noise_rms_summary(
        _npy_v2(np.zeros((2,), dtype=np.complex64))
    )
    assert (scale, sum_squares, count, shape) == (0.0, 0.0, 2, (2,))


def test_noise_summary_rejects_empty_and_unknown_headers() -> None:
    with pytest.raises(ValueError, match="at least one complex value"):
        _noise_rms_summary(_npy_v2(np.zeros((0,), dtype=np.complex64)))
    corrupted = bytearray(_npy_v2(np.ones(2, dtype=np.complex64)))
    corrupted[6], corrupted[7] = 3, 0
    with pytest.raises(ValueError, match="unsupported NPY format version"):
        _noise_rms_summary(bytes(corrupted))
    with pytest.raises(ValueError, match="invalid NPY header"):
        _noise_rms_summary(b"\x93NUMPY\x02\x00\x10\x00truncated")
