"""Acquisition helpers: record indexes, JSON pointers, canonical parsing and NPY noise summaries."""

from __future__ import annotations

import io

import numpy as np
import pytest
from evaluation.acquisition.canonical import (
    _json_bytes,
    _parse_utc_timestamp,
    _strata_key_set,
)
from evaluation.acquisition.index import _pointer_item, _unique_index
from evaluation.acquisition.noise import _noise_rms_summary


def _npy(array: np.ndarray, version=(1, 0)) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, version=version)
    return buffer.getvalue()


def test_unique_index_requires_an_array() -> None:
    errors: list[str] = []
    assert _unique_index("not-a-list", key="id", label="records", errors=errors) == {}
    assert errors == ["records must be an array"]


def test_unique_index_error_order() -> None:
    errors: list[str] = []
    _unique_index(
        [{"id": "a"}, "not-object", {"id": ""}, {"id": "a"}],
        key="id",
        label="records",
        errors=errors,
    )
    assert errors == [
        "records contains a non-object record",
        "records record lacks a nonempty id",
        "duplicate records id: a",
    ]


def test_pointer_items_resolve_only_declared_prefixes() -> None:
    document = {"items": [{"name": "first"}]}
    assert _pointer_item(document, "/items/0", "/items/") == {"name": "first"}
    with pytest.raises(ValueError, match="pointer must start with /items/"):
        _pointer_item(document, "/other/0", "/items/")
    with pytest.raises(ValueError, match="nonnegative decimal integer"):
        _pointer_item(document, "/items/-1", "/items/")
    with pytest.raises(ValueError, match="container 'missing' is missing"):
        _pointer_item(document, "/missing/0", "/missing/")
    with pytest.raises(ValueError, match="target container is not an array"):
        _pointer_item({"items": {"a": 1}}, "/items/0", "/items/")


def test_utc_timestamps_require_an_explicit_offset() -> None:
    assert _parse_utc_timestamp("2026-01-01T12:00:00Z") is not None
    assert _parse_utc_timestamp("2026-01-01T12:00:00") is None
    assert _parse_utc_timestamp("not-a-time") is None
    assert _parse_utc_timestamp(20260101) is None


def test_json_bytes_reject_constants_duplicates_and_invalid_text() -> None:
    assert _json_bytes(b'{"a": 1}', "doc") == {"a": 1}
    with pytest.raises(ValueError, match="non-standard JSON constant NaN"):
        _json_bytes(b'{"a": NaN}', "doc")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        _json_bytes(b'{"a": 1, "a": 2}', "doc")
    with pytest.raises(ValueError):
        _json_bytes(b"\xff", "doc")


def test_strata_keys_are_a_set_of_unique_string_triples() -> None:
    rows = [
        {"material_id": "m", "polarization": "V", "incidence_stratum_id": "i"},
        {"material_id": "m", "polarization": "H", "incidence_stratum_id": "i"},
    ]
    assert _strata_key_set(rows) == {("m", "V", "i"), ("m", "H", "i")}
    assert _strata_key_set("not-a-list") is None
    assert _strata_key_set(["not-object"]) is None
    assert (
        _strata_key_set([{"material_id": 1, "polarization": "V", "incidence_stratum_id": "i"}])
        is None
    )
    assert _strata_key_set(rows + [rows[0]]) is None


def test_noise_summary_reports_scale_and_sum_of_squares() -> None:
    values = np.array([3 + 4j, 0 + 0j], dtype=np.complex128)
    scale, sum_squares, count, shape = _noise_rms_summary(_npy(values))
    assert scale == pytest.approx(4.0)
    assert count == 2 and shape == (2,)
    assert sum_squares == pytest.approx(1.25**2)


def test_noise_summary_accepts_version_two_headers_and_all_zero_estimates() -> None:
    zeros = np.zeros((2, 3), dtype=np.complex64)
    assert _noise_rms_summary(_npy(zeros, version=(2, 0))) == (0.0, 0.0, 6, (2, 3))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        pytest.param(b"not an npy file", "invalid NPY header", id="bad-magic"),
        pytest.param(
            _npy(np.ones(2, dtype=np.float64)), "must contain complex64 or complex128", id="real"
        ),
        pytest.param(
            _npy(np.ones(0, dtype=np.complex64)), "at least one complex value", id="empty"
        ),
        pytest.param(
            _npy(np.ones(2, dtype=np.complex64))[:-4],
            "truncated or has trailing bytes",
            id="truncated",
        ),
        pytest.param(
            _npy(np.array([np.nan + 0j], dtype=np.complex64)), "must be finite", id="nonfinite"
        ),
    ],
)
def test_noise_summary_rejects_unusable_captures(payload: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _noise_rms_summary(payload)


def test_noise_summary_rejects_unsupported_npy_versions() -> None:
    header = bytearray(_npy(np.ones(2, dtype=np.complex64)))
    header[6], header[7] = 3, 0
    with pytest.raises(ValueError, match="unsupported NPY format version"):
        _noise_rms_summary(bytes(header))
