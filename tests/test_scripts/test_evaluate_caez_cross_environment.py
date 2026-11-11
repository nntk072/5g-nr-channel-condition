"""Tests for cross-environment CAEZ forecast splits and summaries."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from datasets.model_input_audit import audit_model_input_payload
from evaluation.caez_forecast import (
    CAEZForecastModelInputs,
    make_caez_forecast_artifact_contract,
    paired_time_block_bootstrap,
)
from scripts.evaluate_caez_cross_environment import evaluate_transfer


def test_caez_model_input_contract_rejects_truth_leakage_and_bad_pair_order():
    features = np.zeros((2, 2), dtype=np.float32)
    targets = np.ones((2, 1), dtype=np.float32)
    pairs = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    contract = make_caez_forecast_artifact_contract(
        train_features=features,
        train_targets=targets,
        train_pair_indices=pairs,
        evaluation_features=features,
        evaluation_targets=targets,
        evaluation_pair_indices=pairs,
        train_archive_sha256="a" * 64,
        evaluation_archive_sha256="b" * 64,
        feature_transform_sha256="c" * 64,
    )
    assert contract["training"]["inputs"]["sha256"] != contract["training"]["targets"]["sha256"]
    with pytest.raises(ValueError, match="strictly history/anchor/target ordered"):
        make_caez_forecast_artifact_contract(
            train_features=features,
            train_targets=targets,
            train_pair_indices=np.asarray([[1, 0, 2], [1, 2, 3]], dtype=np.int64),
            evaluation_features=features,
            evaluation_targets=targets,
            evaluation_pair_indices=pairs,
            train_archive_sha256="a" * 64,
            evaluation_archive_sha256="b" * 64,
            feature_transform_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="finite"):
        audit_model_input_payload(
            CAEZForecastModelInputs(feature_matrix=np.asarray([[np.nan]], dtype=np.float32))
        )


def _write_archive(path: Path, scale: float) -> None:
    n = 400
    timestamps = np.arange(n, dtype=np.float64) * 0.02 + 1_700_000_000.0
    levels = scale * (1.0 + np.arange(n, dtype=np.float32) * 0.001)
    channel = np.broadcast_to(levels[:, None, None, None], (n, 4, 2, 3)).copy()
    np.savez(
        path,
        H=channel,
        noise_var=np.ones((n, 4), dtype=np.float32),
        timestamps=timestamps,
        pos=np.zeros((n, 2), dtype=np.float32),
    )


def test_cross_environment_regression_uses_source_train_and_target_tail(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_indoor.npz"
    target = tmp_path / "target_outdoor.npz"
    _write_archive(source, 1.0)
    _write_archive(target, 0.6)

    result = evaluate_transfer(
        source,
        [source, target],
        horizon_s=0.1,
        lookback_s=0.04,
        tolerance_s=1e-6,
        chunk_rows=32,
        bootstrap_block_seconds=0.4,
        bootstrap_replicates=100,
    )

    assert result["training_domain"]["train_pairs"] > 0
    assert np.isclose(result["unused_source_temporal_gap_fraction"], 0.1)
    assert [item["environment_id"] for item in result["target_domains"]] == [
        "source_indoor",
        "target_outdoor",
    ]
    for domain in result["target_domains"]:
        assert domain["test_pairs"] > 0
        assert domain["scores"]["ridge_source_trained"]["per_oru"][0]["features"] > 0
        assert domain["scores"]["persistence"]["macro_nrmse_by_target_std"] >= 0
        assert domain["uncertainty"]["block_count"] > 0
        contract = domain["model_input_contract"]
        schema = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "datasets/schema/caez-ridge-model-input-contract.v1.schema.json"
            ).read_text(encoding="utf-8")
        )
        jsonschema.validate(contract, schema)
        assert contract["schema"] == "caez.ridge_model_input_contract.v1"
        assert contract["scope"].startswith("offline CAEZ research")
        assert (
            contract["training"]["inputs"]["shape"][0] == result["training_domain"]["train_pairs"]
        )
        assert contract["evaluation"]["inputs"]["shape"][0] == domain["test_pairs"]
        assert contract["evaluation"]["targets"]["shape"][0] == domain["test_pairs"]
        assert contract["evaluation_archive_sha256"] == domain["archive_sha256"]
    assert any("not LOS/NLOS validation" in item for item in result["interpretation_limits"])


def test_paired_time_block_bootstrap_is_deterministic_and_paired() -> None:
    anchors = np.arange(1000, 2000, dtype=np.int64)
    base = np.sin(np.arange(anchors.size, dtype=np.float64)[:, None] / 17.0)
    target = np.broadcast_to(base, (anchors.size, 8)).copy()
    predictions = {
        "ridge_source_trained": target + 0.1,
        "persistence": target + 0.2,
        "linear_trend": target + 0.3,
    }

    first = paired_time_block_bootstrap(
        target,
        predictions,
        anchors,
        test_start=1000,
        block_rows=100,
        n_orus=4,
        n_rx=2,
        n_bootstrap=100,
        seed=12,
    )
    second = paired_time_block_bootstrap(
        target,
        predictions,
        anchors,
        test_start=1000,
        block_rows=100,
        n_orus=4,
        n_rx=2,
        n_bootstrap=100,
        seed=12,
    )

    assert first == second
    assert first["block_count"] == 10
    contrast = first["contrasts"]["ridge_source_trained_minus_persistence"]
    assert contrast["point_difference"] < 0
    assert contrast["bootstrap_95_percentile_interval"][1] < 0
