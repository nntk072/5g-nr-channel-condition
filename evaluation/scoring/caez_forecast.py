"""Shared temporal forecast alignment and compact regression metrics."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from datasets.model_input_audit import audit_model_input_payload


@dataclass(frozen=True, slots=True)
class CAEZForecastModelInputs:
    """Feature matrix admitted to the offline CAEZ Ridge forecast only."""

    feature_matrix: np.ndarray


def _array_identity(value: np.ndarray, *, label: str) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    metadata = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(metadata + b"\0" + array.tobytes(order="C")).hexdigest()
    return {"label": label, "dtype": array.dtype.str, "shape": list(array.shape), "sha256": digest}


def make_caez_forecast_artifact_contract(
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_pair_indices: np.ndarray,
    evaluation_features: np.ndarray,
    evaluation_targets: np.ndarray,
    evaluation_pair_indices: np.ndarray,
    train_archive_sha256: str,
    evaluation_archive_sha256: str,
    feature_transform_sha256: str,
) -> dict[str, Any]:
    """Bind CAEZ Ridge features, truth, chronological pairs, and archive sources."""
    train_inputs = audit_model_input_payload(CAEZForecastModelInputs(train_features))
    evaluation_inputs = audit_model_input_payload(CAEZForecastModelInputs(evaluation_features))
    if train_inputs.feature_matrix.shape[1] != evaluation_inputs.feature_matrix.shape[1]:
        raise ValueError("CAEZ train and evaluation feature widths differ")

    def validate_partition(
        features: np.ndarray, targets: np.ndarray, pairs: np.ndarray, *, label: str
    ) -> None:
        target_array = np.asarray(targets)
        if (
            target_array.ndim != 2
            or target_array.shape[0] != features.shape[0]
            or target_array.dtype.kind not in "fiu"
            or not np.isfinite(target_array).all()
        ):
            raise ValueError(
                f"CAEZ {label} targets must be finite dense real rows aligned to features"
            )
        pair_array = np.asarray(pairs)
        if (
            pair_array.dtype.kind not in "iu"
            or pair_array.ndim != 2
            or pair_array.shape != (features.shape[0], 3)
            or np.any(pair_array < 0)
        ):
            raise ValueError(f"CAEZ {label} pair indices must be nonnegative integer [row, 3]")
        history, anchor, target = pair_array.T
        if np.any(history >= anchor) or np.any(anchor >= target):
            raise ValueError(f"CAEZ {label} pairs must be strictly history/anchor/target ordered")

    validate_partition(train_features, train_targets, train_pair_indices, label="training")
    validate_partition(
        evaluation_features,
        evaluation_targets,
        evaluation_pair_indices,
        label="evaluation",
    )
    for name, value in (
        ("train_archive_sha256", train_archive_sha256),
        ("evaluation_archive_sha256", evaluation_archive_sha256),
        ("feature_transform_sha256", feature_transform_sha256),
    ):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return {
        "schema": "caez.ridge_model_input_contract.v1",
        "scope": "offline CAEZ research score report; not an NR receiver deployment artifact",
        "model": "StandardScaler + Ridge(alpha=1.0)",
        "feature_transform_sha256": feature_transform_sha256,
        "train_archive_sha256": train_archive_sha256,
        "evaluation_archive_sha256": evaluation_archive_sha256,
        "training": {
            "inputs": _array_identity(train_features, label="feature_matrix"),
            "targets": _array_identity(train_targets, label="supervision_targets"),
            "pair_indices": _array_identity(train_pair_indices, label="history_anchor_target_rows"),
        },
        "evaluation": {
            "inputs": _array_identity(evaluation_features, label="feature_matrix"),
            "targets": _array_identity(evaluation_targets, label="held_out_truth"),
            "pair_indices": _array_identity(
                evaluation_pair_indices, label="history_anchor_target_rows"
            ),
        },
    }


def align_forecast_pairs(
    timestamps: np.ndarray,
    *,
    horizon_s: float,
    lookback_s: float,
    tolerance_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align strictly chronological past/target pairs to nearest observations."""
    times = np.asarray(timestamps)
    if times.ndim != 1 or times.size < 3 or not np.isfinite(times).all():
        raise ValueError("timestamps must be a finite one-dimensional array with at least 3 rows")
    if not np.all(np.diff(times) > 0):
        raise ValueError("timestamps must be strictly increasing")
    if horizon_s <= 0 or lookback_s <= 0 or tolerance_s < 0:
        raise ValueError("horizon/lookback must be positive and tolerance nonnegative")

    anchors = np.arange(times.size, dtype=np.int64)
    target_time = times[anchors] + horizon_s
    target_right = np.searchsorted(times, target_time, side="left")
    right_valid = target_right < times.size
    left = np.clip(target_right - 1, 0, times.size - 1)
    right = np.clip(target_right, 0, times.size - 1)
    choose_right = np.abs(times[right] - target_time) < np.abs(times[left] - target_time)
    targets = np.where(choose_right, right, left)
    errors = np.abs(times[targets] - target_time)
    history = np.searchsorted(times, times[anchors] - lookback_s, side="right") - 1
    valid = (
        right_valid
        & (targets > anchors)
        & (errors <= tolerance_s)
        & (history >= 0)
        & (history < anchors)
    )
    return anchors[valid], targets[valid], history[valid]


def paired_time_block_bootstrap(
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    anchors: np.ndarray,
    *,
    test_start: int,
    block_rows: int,
    n_orus: int,
    n_rx: int,
    n_bootstrap: int = 1000,
    seed: int = 20261022,
) -> dict[str, Any]:
    """Estimate paired macro-NRMSE differences by resampling complete time blocks."""
    if target.ndim != 2 or target.shape[1] != n_orus * n_rx:
        raise ValueError("target width must equal n_orus * n_rx")
    if anchors.ndim != 1 or anchors.size != target.shape[0]:
        raise ValueError("anchors must have one entry per target row")
    if block_rows <= 0 or n_bootstrap <= 0:
        raise ValueError("block_rows and n_bootstrap must be positive")
    if not predictions:
        raise ValueError("at least one prediction method is required")
    for name, prediction in predictions.items():
        if prediction.shape != target.shape:
            raise ValueError(f"prediction shape mismatch for {name}")

    block_ids = (anchors - test_start) // block_rows
    unique_ids, inverse, counts = np.unique(block_ids, return_inverse=True, return_counts=True)
    complete = counts >= int(np.ceil(0.95 * block_rows))
    if not np.any(complete):
        raise ValueError("no near-complete time blocks are available for bootstrap")
    selected_rows = complete[inverse]
    target_rows = target[selected_rows]
    inverse_rows = inverse[selected_rows]
    selected_block_indices = np.flatnonzero(complete)
    remap = np.full(unique_ids.size, -1, dtype=np.int64)
    remap[selected_block_indices] = np.arange(selected_block_indices.size)
    group_rows = remap[inverse_rows]
    block_count = selected_block_indices.size
    feature_count = np.bincount(group_rows, minlength=block_count).astype(np.float64) * n_rx

    per_oru_target = target_rows.reshape(target_rows.shape[0], n_orus, n_rx)
    target_sum = np.zeros((block_count, n_orus), dtype=np.float64)
    target_square_sum = np.zeros_like(target_sum)
    for oru in range(n_orus):
        target_sum[:, oru] = np.bincount(
            group_rows, weights=per_oru_target[:, oru, :].sum(axis=1), minlength=block_count
        )
        target_square_sum[:, oru] = np.bincount(
            group_rows,
            weights=np.square(per_oru_target[:, oru, :], dtype=np.float64).sum(axis=1),
            minlength=block_count,
        )

    error_square_sums: dict[str, np.ndarray] = {}
    for name, prediction in predictions.items():
        error = (prediction[selected_rows] - target_rows).reshape(
            target_rows.shape[0], n_orus, n_rx
        )
        sums = np.zeros((block_count, n_orus), dtype=np.float64)
        for oru in range(n_orus):
            sums[:, oru] = np.bincount(
                group_rows,
                weights=np.square(error[:, oru, :], dtype=np.float64).sum(axis=1),
                minlength=block_count,
            )
        error_square_sums[name] = sums

    def macro_nrmse(indices: np.ndarray, error_sums: np.ndarray) -> np.ndarray:
        count = feature_count[indices].sum(axis=1)[:, None]
        sum_y = target_sum[indices].sum(axis=1)
        sum_y2 = target_square_sum[indices].sum(axis=1)
        sum_error = error_sums[indices].sum(axis=1)
        variance = np.maximum(sum_y2 / count - np.square(sum_y / count), 0.0)
        rmse = np.sqrt(sum_error / count)
        per_oru = np.divide(
            rmse, np.sqrt(variance), out=np.full_like(rmse, np.nan), where=variance > 0
        )
        return np.nanmean(per_oru, axis=1)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, block_count, size=(n_bootstrap, block_count))
    contrasts = {}
    baseline_names = [name for name in predictions if name != "ridge_source_trained"]
    if not baseline_names:
        raise ValueError(
            "predictions must include at least one baseline beside ridge_source_trained"
        )
    all_block_indices = np.arange(block_count)[None, :]
    for baseline in baseline_names:
        ridge_boot = macro_nrmse(draws, error_square_sums["ridge_source_trained"])
        baseline_boot = macro_nrmse(draws, error_square_sums[baseline])
        delta = ridge_boot - baseline_boot
        point = float(
            macro_nrmse(all_block_indices, error_square_sums["ridge_source_trained"])[0]
            - macro_nrmse(all_block_indices, error_square_sums[baseline])[0]
        )
        contrasts[f"ridge_source_trained_minus_{baseline}"] = {
            "point_difference": point,
            "bootstrap_95_percentile_interval": [
                float(np.percentile(delta, 2.5)),
                float(np.percentile(delta, 97.5)),
            ],
            "bootstrap_mean_difference": float(delta.mean()),
        }
    return {
        "method": "paired moving time-block bootstrap of macro RMSE / held-out target standard deviation",
        "block_rows": block_rows,
        "block_count": int(block_count),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "dropped_incomplete_block_rows": int(target.shape[0] - selected_rows.sum()),
        "contrasts": contrasts,
    }


def score_summary_predictions(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    n_orus: int,
    n_rx: int,
) -> dict[str, Any]:
    """Score dB summary predictions per O-RU and macro-average their normalized RMSE."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have matching two-dimensional shapes")
    if n_orus <= 0 or n_rx <= 0 or prediction.shape[1] != n_orus * n_rx:
        raise ValueError("feature width must equal n_orus * n_rx")
    scores = []
    for oru in range(n_orus):
        feature_slice = slice(oru * n_rx, (oru + 1) * n_rx)
        actual = target[:, feature_slice]
        error = prediction[:, feature_slice] - actual
        squared_error = float(np.square(error, dtype=np.float64).sum())
        rmse = float(np.sqrt(squared_error / actual.size))
        target_std = float(np.std(actual, dtype=np.float64))
        scores.append(
            {
                "rmse_db": rmse,
                "mae_db": float(np.abs(error).mean()),
                "nrmse_by_target_std": rmse / target_std if target_std > 0 else None,
                "features": int(actual.size),
            }
        )
    normalized = [item["nrmse_by_target_std"] for item in scores]
    available = [value for value in normalized if value is not None]
    return {
        "per_oru": scores,
        "macro_nrmse_by_target_std": float(np.mean(available)) if available else None,
    }
