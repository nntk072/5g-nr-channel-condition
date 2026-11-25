"""CAEZ forecast alignment, block bootstrap, scoring and contract binding."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from evaluation.scoring.caez_forecast import (
    align_forecast_pairs,
    make_caez_forecast_artifact_contract,
    paired_time_block_bootstrap,
    score_summary_predictions,
)

DIGEST = hashlib.sha256(b"fixture").hexdigest()


def _partition(rows: int = 4, features: int = 3):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(rows, features))
    y = rng.normal(size=(rows, 2))
    pairs = np.stack([np.arange(rows), np.arange(rows) + 1, np.arange(rows) + 2], axis=1)
    return x, y, pairs


def test_align_forecast_pairs_selects_exact_grid_targets_with_history() -> None:
    anchors, targets, history = align_forecast_pairs(
        np.arange(10.0), horizon_s=2.0, lookback_s=3.0, tolerance_s=0.5
    )
    # Anchors need a full lookback (t >= 3) and a target inside the record (t + 2 <= 9).
    assert anchors.tolist() == [3, 4, 5, 6, 7]
    assert targets.tolist() == [5, 6, 7, 8, 9]
    assert history.tolist() == [0, 1, 2, 3, 4]
    assert np.all(history < anchors) and np.all(anchors < targets)


def test_align_forecast_pairs_rejects_unusable_timestamps() -> None:
    with pytest.raises(ValueError, match="finite one-dimensional"):
        align_forecast_pairs(np.zeros((2, 2)), horizon_s=1, lookback_s=1, tolerance_s=0)
    with pytest.raises(ValueError, match="finite one-dimensional"):
        align_forecast_pairs(np.array([0.0, np.nan, 2.0]), horizon_s=1, lookback_s=1, tolerance_s=0)
    with pytest.raises(ValueError, match="strictly increasing"):
        align_forecast_pairs(np.array([0.0, 2.0, 1.0]), horizon_s=1, lookback_s=1, tolerance_s=0)
    with pytest.raises(ValueError, match="horizon/lookback"):
        align_forecast_pairs(np.arange(5.0), horizon_s=0, lookback_s=1, tolerance_s=0)
    with pytest.raises(ValueError, match="tolerance"):
        align_forecast_pairs(np.arange(5.0), horizon_s=1, lookback_s=1, tolerance_s=-1)


def test_align_forecast_pairs_drops_targets_outside_tolerance() -> None:
    # Observations every 1.0 except a gap, so the nearest sample to t+2 is 0.4 away.
    times = np.array([0.0, 1.0, 2.6, 3.0, 4.0, 5.0])
    anchors, targets, _ = align_forecast_pairs(
        times, horizon_s=2.0, lookback_s=1.0, tolerance_s=0.2
    )
    assert np.all(np.abs(times[targets] - (times[anchors] + 2.0)) <= 0.2)
    assert 0 not in anchors.tolist()


def test_contract_binds_partitions_and_digests() -> None:
    train_x, train_y, train_pairs = _partition()
    eval_x, eval_y, eval_pairs = _partition(rows=3)
    contract = make_caez_forecast_artifact_contract(
        train_features=train_x,
        train_targets=train_y,
        train_pair_indices=train_pairs,
        evaluation_features=eval_x,
        evaluation_targets=eval_y,
        evaluation_pair_indices=eval_pairs,
        train_archive_sha256=DIGEST,
        evaluation_archive_sha256=DIGEST,
        feature_transform_sha256=DIGEST,
    )
    assert contract["schema"] == "caez.ridge_model_input_contract.v1"
    assert contract["training"]["inputs"]["shape"] == [4, 3]
    assert contract["evaluation"]["targets"]["label"] == "held_out_truth"
    # The identity depends on the bytes, so changing one target changes the digest.
    changed_y = train_y.copy()
    changed_y[0, 0] += 1.0
    other = make_caez_forecast_artifact_contract(
        train_features=train_x,
        train_targets=changed_y,
        train_pair_indices=train_pairs,
        evaluation_features=eval_x,
        evaluation_targets=eval_y,
        evaluation_pair_indices=eval_pairs,
        train_archive_sha256=DIGEST,
        evaluation_archive_sha256=DIGEST,
        feature_transform_sha256=DIGEST,
    )
    assert other["training"]["targets"]["sha256"] != contract["training"]["targets"]["sha256"]


def _contract_kwargs(**overrides):
    train_x, train_y, train_pairs = _partition()
    eval_x, eval_y, eval_pairs = _partition(rows=3)
    kwargs = {
        "train_features": train_x,
        "train_targets": train_y,
        "train_pair_indices": train_pairs,
        "evaluation_features": eval_x,
        "evaluation_targets": eval_y,
        "evaluation_pair_indices": eval_pairs,
        "train_archive_sha256": DIGEST,
        "evaluation_archive_sha256": DIGEST,
        "feature_transform_sha256": DIGEST,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"evaluation_features": np.zeros((3, 2))}, "feature widths differ"),
        ({"train_targets": np.zeros(4)}, "targets must be finite"),
        ({"train_targets": np.full((4, 2), np.nan)}, "targets must be finite"),
        ({"train_targets": np.zeros((3, 2))}, "targets must be finite"),
        ({"evaluation_targets": np.zeros((3, 2), dtype=complex)}, "evaluation targets"),
        ({"train_pair_indices": -np.ones((4, 3), dtype=np.int64)}, "nonnegative integer"),
        ({"train_pair_indices": np.zeros((4, 3))}, "nonnegative integer"),
        (
            {
                "train_pair_indices": np.stack(
                    [np.arange(4), np.arange(4), np.arange(4) + 2], axis=1
                )
            },
            "strictly history/anchor/target",
        ),
        ({"train_archive_sha256": "ABC"}, "lowercase SHA-256"),
        ({"feature_transform_sha256": 7}, "lowercase SHA-256"),
    ],
)
def test_contract_rejects_inconsistent_inputs(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        make_caez_forecast_artifact_contract(**_contract_kwargs(**overrides))


def test_score_summary_reports_rmse_mae_and_normalized_error() -> None:
    target = np.array([[0.0, 1.0], [2.0, 3.0]])
    prediction = target + 0.5
    result = score_summary_predictions(prediction, target, n_orus=1, n_rx=2)
    score = result["per_oru"][0]
    assert score["rmse_db"] == pytest.approx(0.5)
    assert score["mae_db"] == pytest.approx(0.5)
    assert score["features"] == 4
    assert score["nrmse_by_target_std"] == pytest.approx(0.5 / np.std(target))
    assert result["macro_nrmse_by_target_std"] == pytest.approx(score["nrmse_by_target_std"])


def test_score_summary_excludes_constant_targets_from_macro_average() -> None:
    target = np.array([[1.0, 1.0, 0.0, 1.0]])
    prediction = target.copy()
    result = score_summary_predictions(prediction, target, n_orus=2, n_rx=2)
    assert result["per_oru"][0]["nrmse_by_target_std"] is None
    assert result["per_oru"][1]["nrmse_by_target_std"] is not None
    assert result["macro_nrmse_by_target_std"] == result["per_oru"][1]["nrmse_by_target_std"]

    all_constant = score_summary_predictions(np.ones((2, 2)), np.ones((2, 2)), n_orus=1, n_rx=2)
    assert all_constant["macro_nrmse_by_target_std"] is None


@pytest.mark.parametrize(
    ("prediction", "target", "n_orus", "n_rx", "message"),
    [
        (np.zeros((2, 4)), np.zeros((2, 3)), 2, 2, "matching two-dimensional"),
        (np.zeros(4), np.zeros(4), 2, 2, "matching two-dimensional"),
        (np.zeros((2, 4)), np.zeros((2, 4)), 0, 2, "feature width"),
        (np.zeros((2, 4)), np.zeros((2, 4)), 2, 3, "feature width"),
    ],
)
def test_score_summary_rejects_shape_mismatch(prediction, target, n_orus, n_rx, message) -> None:
    with pytest.raises(ValueError, match=message):
        score_summary_predictions(prediction, target, n_orus=n_orus, n_rx=n_rx)


def _bootstrap_fixture(rows: int = 6, width: int = 4):
    rng = np.random.default_rng(11)
    target = rng.normal(size=(rows, width))
    anchors = np.arange(rows) + 100
    return target, anchors


def test_identical_predictions_give_zero_paired_contrast() -> None:
    target, anchors = _bootstrap_fixture()
    predictions = {
        "ridge_source_trained": target + 0.1,
        "baseline_mean": target + 0.1,
    }
    result = paired_time_block_bootstrap(
        target,
        predictions,
        anchors,
        test_start=100,
        block_rows=2,
        n_orus=2,
        n_rx=2,
        n_bootstrap=50,
    )
    assert result["block_count"] == 3
    assert result["dropped_incomplete_block_rows"] == 0
    contrast = result["contrasts"]["ridge_source_trained_minus_baseline_mean"]
    assert contrast["point_difference"] == pytest.approx(0.0)
    assert contrast["bootstrap_95_percentile_interval"] == pytest.approx([0.0, 0.0])
    assert contrast["bootstrap_mean_difference"] == pytest.approx(0.0)


def test_bootstrap_reports_dropped_incomplete_blocks_and_is_seeded() -> None:
    target, anchors = _bootstrap_fixture()
    predictions = {"ridge_source_trained": target + 1.0, "baseline_mean": target}
    kwargs = {
        "test_start": 100,
        "block_rows": 4,
        "n_orus": 2,
        "n_rx": 2,
        "n_bootstrap": 30,
    }
    first = paired_time_block_bootstrap(target, predictions, anchors, **kwargs)
    second = paired_time_block_bootstrap(target, predictions, anchors, **kwargs)
    assert first["dropped_incomplete_block_rows"] == 2
    assert first == second
    contrast = first["contrasts"]["ridge_source_trained_minus_baseline_mean"]
    assert contrast["point_difference"] > 0
    assert contrast["bootstrap_95_percentile_interval"][0] <= contrast["point_difference"]


@pytest.mark.parametrize(
    ("kwargs", "predictions_fn", "message"),
    [
        ({"block_rows": 2}, None, "target width"),
        ({"block_rows": 0}, None, "must be positive"),
        ({"block_rows": 2, "n_bootstrap": 0}, None, "must be positive"),
        ({"block_rows": 10}, None, "no near-complete"),
        ({"block_rows": 2}, "empty", "at least one prediction"),
        ({"block_rows": 2}, "shape", "prediction shape mismatch"),
        ({"block_rows": 2}, "only_ridge", "at least one baseline"),
    ],
)
def test_bootstrap_rejects_invalid_designs(kwargs, predictions_fn, message) -> None:
    target, anchors = _bootstrap_fixture()
    predictions = {"ridge_source_trained": target, "baseline_mean": target}
    if predictions_fn == "empty":
        predictions = {}
    elif predictions_fn == "shape":
        predictions["baseline_mean"] = target[:-1]
    elif predictions_fn == "only_ridge":
        predictions = {"ridge_source_trained": target}
    full = {"test_start": 100, "n_orus": 2, "n_rx": 2, "n_bootstrap": 5, **kwargs}
    bad_target = target[:, :3] if message == "target width" else target
    with pytest.raises(ValueError, match=message):
        paired_time_block_bootstrap(bad_target, predictions, anchors, **full)


def test_bootstrap_rejects_anchor_length_mismatch() -> None:
    target, anchors = _bootstrap_fixture()
    with pytest.raises(ValueError, match="one entry per target row"):
        paired_time_block_bootstrap(
            target,
            {"ridge_source_trained": target, "baseline_mean": target},
            anchors[:-1],
            test_start=100,
            block_rows=2,
            n_orus=2,
            n_rx=2,
        )
