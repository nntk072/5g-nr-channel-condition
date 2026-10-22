"""Tests for cross-environment CAEZ forecast splits and summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from evaluation.caez_forecast import paired_time_block_bootstrap
from scripts.evaluate_caez_cross_environment import evaluate_transfer


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
