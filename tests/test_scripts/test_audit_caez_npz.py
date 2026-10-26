"""Tests for bounded-memory, pickle-free CAEZ NPZ auditing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from datasets.caez import audit
from scripts.evaluate_caez_magnitude_forecast import evaluate, forecast_pairs


def _write_fixture(path: Path, *, include_metadata_object: bool = True) -> None:
    values: dict[str, Any] = {
        "H": np.arange(24, dtype=np.float32).reshape(3, 2, 4),
        "noise_var": np.ones((3, 2), dtype=np.float32),
        "timestamps": np.array([10.0, 10.02, 10.04], dtype=np.float64),
        "pos": np.zeros((3, 2), dtype=np.float32),
    }
    if include_metadata_object:
        values["oru_pos_dict"] = np.array({"oru": [1.0, 2.0, 3.0]}, dtype=object)
    np.savez(path, **values)


def test_audit_reads_numeric_fields_without_unpickling_object_member(tmp_path: Path) -> None:
    archive = tmp_path / "caez_fixture.npz"
    _write_fixture(archive)

    report = audit(archive, chunk_rows=2)

    assert report["sample_count"] == 3
    assert report["fields"]["H"]["shape"] == [3, 2, 4]
    assert report["fields"]["H"]["all_finite"] is True
    assert report["channel_observation"]["stored_dtype"] == "float32"
    assert report["channel_observation"]["phase_information_available_in_archive"] is False
    assert (
        report["channel_observation"]["g3_complex_csi_disposition"]
        == "ineligible_no_explicit_complex_phase"
    )
    assert "oru_pos_dict.npy" in report["npz_members"]
    assert report["timestamps"]["strictly_increasing"] is True
    assert len(report["archive_sha256"]) == 64


def test_audit_does_not_treat_complex_dtype_as_phase_reference_evidence(tmp_path: Path) -> None:
    archive = tmp_path / "complex_fixture.npz"
    _write_fixture(archive, include_metadata_object=False)
    with np.load(archive, allow_pickle=False) as data:
        np.savez(
            archive,
            H=data["H"].astype(np.complex64) * (1.0 + 1.0j),
            noise_var=data["noise_var"],
            timestamps=data["timestamps"],
            pos=data["pos"],
        )

    report = audit(archive, chunk_rows=2)

    assert report["channel_observation"]["complex_coefficients_present"] is True
    assert report["channel_observation"]["phase_information_available_in_archive"] is True
    assert (
        report["channel_observation"]["g3_complex_csi_disposition"]
        == "not_established_phase_reference_unverified"
    )


def test_audit_rejects_missing_required_fields(tmp_path: Path) -> None:
    archive = tmp_path / "missing.npz"
    _write_fixture(archive, include_metadata_object=False)
    with np.load(archive) as data:
        np.savez(
            archive,
            noise_var=data["noise_var"],
            timestamps=data["timestamps"],
            pos=data["pos"],
        )

    with pytest.raises(ValueError, match="required CAEZ array is missing: H"):
        audit(archive)


def test_audit_rejects_nonpositive_chunk_size(tmp_path: Path) -> None:
    archive = tmp_path / "caez_fixture.npz"
    _write_fixture(archive)
    with pytest.raises(ValueError, match="chunk_rows must be positive"):
        audit(archive, chunk_rows=0)


def test_forecast_pairs_are_chronological_and_respect_test_boundary() -> None:
    timestamps = np.arange(100, dtype=np.float64) * 0.02
    anchors, targets, history = forecast_pairs(
        timestamps,
        horizon_s=0.1,
        lookback_s=0.04,
        tolerance_s=1e-6,
        test_fraction=0.3,
    )

    assert anchors.size > 0
    assert np.all(anchors >= 70)
    assert np.all(targets > anchors)
    assert np.all(history >= 70)
    assert np.all(history < anchors)
    assert np.allclose(timestamps[targets] - timestamps[anchors], 0.1)


def test_forecast_evaluation_compares_persistence_and_linear_trend(tmp_path: Path) -> None:
    archive = tmp_path / "trend.npz"
    timestamps = np.arange(120, dtype=np.float64) * 0.02
    levels = np.arange(120, dtype=np.float32) / 100.0 + 1.0
    channel = np.broadcast_to(levels[:, None, None, None], (120, 4, 2, 3)).copy()
    np.savez(
        archive,
        H=channel,
        noise_var=np.ones((120, 4), dtype=np.float32),
        timestamps=timestamps,
        pos=np.zeros((120, 2), dtype=np.float32),
    )

    result = evaluate(
        archive,
        horizon_s=0.1,
        lookback_s=0.04,
        tolerance_s=1e-6,
        test_fraction=0.3,
        chunk_pairs=16,
    )

    assert result["split"]["forecast_pairs"] > 0
    assert (
        result["scores"]["linear_trend"]["macro_normalized_rmse"]
        < result["scores"]["persistence"]["macro_normalized_rmse"]
    )
    compact = result["compact_summary_regression"]
    assert compact["split"]["train_pairs"] > compact["split"]["test_pairs"] > 0
    assert compact["scores"]["ridge"]["macro_nrmse_by_target_std"] >= 0
    assert compact["scores"]["ridge"]["per_oru"][0]["features"] > 0
