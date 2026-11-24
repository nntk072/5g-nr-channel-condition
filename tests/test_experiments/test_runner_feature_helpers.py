"""Runner feature helpers: CFR feature domains, digests, benchmark inputs and group statistics."""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from experiments.runner.features import (
    _array_sha256,
    _identity_complex_cfr,
    _kish_effective_group_count,
    _log_power_per_link_subcarrier,
    _sparse_multipath_features,
    _stable_input_bytes,
    _stable_single_link_sha256,
    _validate_npz_snapshot,
)


def _cfr(links: int = 1, tones: int = 8, samples: int = 2) -> np.ndarray:
    rng = np.random.default_rng(4)
    return (
        rng.normal(size=(samples, links, tones)) + 1j * rng.normal(size=(samples, links, tones))
    ).astype(np.complex64)


def test_identity_and_log_power_features_keep_their_domains() -> None:
    cfr = _cfr()
    assert _identity_complex_cfr(cfr).shape == cfr.shape
    assert _log_power_per_link_subcarrier(cfr).shape == (2, 8)
    with pytest.raises(ValueError, match="power-only features require complex"):
        _log_power_per_link_subcarrier(np.ones((2, 8)))
    bad = _cfr()
    bad[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="power-only CFR input must be finite"):
        _log_power_per_link_subcarrier(bad)
    huge = np.full((1, 1, 8), 1e200 + 0j)
    with pytest.raises(ValueError, match="not representable in float64"):
        _log_power_per_link_subcarrier(huge)


def test_sparse_multipath_features_validate_their_domain() -> None:
    with pytest.raises(ValueError, match="sparse multipath features require complex"):
        _sparse_multipath_features(np.ones((2, 8)))
    with pytest.raises(ValueError, match="at least eight tones"):
        _sparse_multipath_features(_cfr(tones=7))
    bad = _cfr()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="sparse multipath CFR must be finite"):
        _sparse_multipath_features(bad)


def test_array_digest_is_layout_independent() -> None:
    array = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert _array_sha256(array) == _array_sha256(np.asfortranarray(array))


def test_single_link_model_digest_requires_a_regular_file(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    assert len(_stable_single_link_sha256(model)) == 64
    link = tmp_path / "linked.bin"
    link.symlink_to(model)
    with pytest.raises(ValueError, match="single-link regular file|changed|path"):
        _stable_single_link_sha256(link)


def test_kish_group_count_is_the_effective_sample_size() -> None:
    assert _kish_effective_group_count([4, 4, 4]) == pytest.approx(3.0)
    assert _kish_effective_group_count([10]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="group row counts must be positive"):
        _kish_effective_group_count([0, 4])


def test_stable_input_bytes_are_bounded(tmp_path: Path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"0123456789")
    assert _stable_input_bytes(source) == b"0123456789"
    with pytest.raises(ValueError, match="exceeds the 4-byte limit"):
        _stable_input_bytes(source, max_bytes=4)
    with pytest.raises(ValueError, match="single-link regular file"):
        _stable_input_bytes(tmp_path)


def test_npz_snapshot_rejects_bad_archives_and_oversized_members() -> None:
    with pytest.raises(ValueError, match="not a valid NPZ archive"):
        _validate_npz_snapshot(b"not a zip", label="dataset", max_uncompressed_bytes=10)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("a.npy", b"x" * 64)
    with pytest.raises(ValueError, match="exceeds the uncompressed dataset size limit"):
        _validate_npz_snapshot(buffer.getvalue(), label="dataset", max_uncompressed_bytes=10)
    crowded = BytesIO()
    with zipfile.ZipFile(crowded, "w") as archive:
        for index in range(33):
            archive.writestr(f"member{index}.npy", b"x")
    with pytest.raises(ValueError, match="invalid NPZ entry roster"):
        _validate_npz_snapshot(crowded.getvalue(), label="dataset", max_uncompressed_bytes=10_000)
