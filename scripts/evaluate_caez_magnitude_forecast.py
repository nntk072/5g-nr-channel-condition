"""Evaluate baselines and compact regression on CAEZ CSI magnitudes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets.caez import _numeric_member, audit
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def forecast_pairs(
    timestamps: np.ndarray,
    *,
    horizon_s: float,
    lookback_s: float,
    tolerance_s: float,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build chronological test pairs without crossing into a shuffled split."""
    times = np.asarray(timestamps)
    if times.ndim != 1 or times.size < 3 or not np.isfinite(times).all():
        raise ValueError("timestamps must be a finite one-dimensional array with at least 3 rows")
    if not np.all(np.diff(times) > 0):
        raise ValueError("timestamps must be strictly increasing")
    if horizon_s <= 0 or lookback_s <= 0 or tolerance_s < 0:
        raise ValueError("horizon/lookback must be positive and tolerance nonnegative")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")

    anchors, targets, history = _aligned_pairs(
        times,
        horizon_s=horizon_s,
        lookback_s=lookback_s,
        tolerance_s=tolerance_s,
    )
    test_start = int(np.floor(times.size * (1.0 - test_fraction)))
    valid = (anchors >= test_start) & (history >= test_start)
    return anchors[valid], targets[valid], history[valid]


def _aligned_pairs(
    times: np.ndarray,
    *,
    horizon_s: float,
    lookback_s: float,
    tolerance_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align every valid past/target pair to the nearest time samples."""
    anchors = np.arange(times.size, dtype=np.int64)
    targets_right = np.searchsorted(times, times[anchors] + horizon_s, side="left")
    right_valid = targets_right < times.size
    left = np.clip(targets_right - 1, 0, times.size - 1)
    right = np.clip(targets_right, 0, times.size - 1)
    choose_right = np.abs(times[right] - (times[anchors] + horizon_s)) < np.abs(
        times[left] - (times[anchors] + horizon_s)
    )
    targets = np.where(choose_right, right, left)
    errors = np.abs(times[targets] - (times[anchors] + horizon_s))
    history = np.searchsorted(times, times[anchors] - lookback_s, side="right") - 1
    valid = right_valid & (targets > anchors) & (errors <= tolerance_s) & (history < anchors)
    return anchors[valid], targets[valid], history[valid]


def _score_summary_predictions(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    n_orus: int,
    n_rx: int,
) -> dict[str, Any]:
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
    return {
        "per_oru": scores,
        "macro_nrmse_by_target_std": float(
            np.mean([value for value in normalized if value is not None])
        ),
    }


def _accumulate(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    squared_error: np.ndarray,
    absolute_error: np.ndarray,
    target_energy: np.ndarray,
    element_count: np.ndarray,
) -> None:
    error = prediction - target
    squared_error += np.square(error, dtype=np.float64).sum(axis=(0, 2, 3))
    absolute_error += np.abs(error).sum(axis=(0, 2, 3), dtype=np.float64)
    target_energy += np.square(target, dtype=np.float64).sum(axis=(0, 2, 3))
    element_count += target.shape[0] * target.shape[2] * target.shape[3]


def evaluate(
    archive_path: Path,
    *,
    horizon_s: float = 1.0,
    lookback_s: float = 0.5,
    tolerance_s: float = 0.02,
    test_fraction: float = 0.2,
    chunk_pairs: int = 128,
) -> dict[str, Any]:
    """Audit and score held-out CSI-magnitude forecasts, reporting aggregates only."""
    if chunk_pairs <= 0:
        raise ValueError("chunk_pairs must be positive")
    integrity = audit(archive_path)
    channel, _ = _numeric_member(archive_path.resolve(), "H")
    timestamps, _ = _numeric_member(archive_path.resolve(), "timestamps")
    if channel.ndim != 4 or channel.shape[1] != 4:
        raise ValueError("H must have [sample, O-RU, Rx antenna, subcarrier] axes")
    if channel.shape[0] != timestamps.shape[0]:
        raise ValueError("H and timestamps sample counts differ")
    anchors, targets, history = forecast_pairs(
        timestamps,
        horizon_s=horizon_s,
        lookback_s=lookback_s,
        tolerance_s=tolerance_s,
        test_fraction=test_fraction,
    )
    if anchors.size == 0:
        raise ValueError("no test forecast pairs satisfy the requested timing tolerance")

    n_orus = channel.shape[1]
    accumulators = {
        method: [np.zeros(n_orus, dtype=np.float64) for _ in range(4)]
        for method in ("persistence", "linear_trend")
    }
    for start in range(0, anchors.size, chunk_pairs):
        stop = min(start + chunk_pairs, anchors.size)
        anchor_block = anchors[start:stop]
        target_block = targets[start:stop]
        history_block = history[start:stop]
        anchor_values = np.asarray(channel[anchor_block], dtype=np.float32)
        target_values = np.asarray(channel[target_block], dtype=np.float32)
        history_values = np.asarray(channel[history_block], dtype=np.float32)
        trend = (anchor_values - history_values) * (horizon_s / lookback_s)
        trend_prediction = np.maximum(anchor_values + trend, 0.0)
        for method, prediction in (
            ("persistence", anchor_values),
            ("linear_trend", trend_prediction),
        ):
            sums = accumulators[method]
            _accumulate(
                prediction,
                target_values,
                squared_error=sums[0],
                absolute_error=sums[1],
                target_energy=sums[2],
                element_count=sums[3],
            )

    scores: dict[str, Any] = {}
    for method, (squared, absolute, energy, count) in accumulators.items():
        scores[method] = {
            "per_oru": [
                {
                    "rmse_magnitude": float(np.sqrt(squared[i] / count[i])),
                    "mae_magnitude": float(absolute[i] / count[i]),
                    "normalized_rmse": float(np.sqrt(squared[i] / energy[i]))
                    if energy[i] > 0
                    else None,
                    "elements": int(count[i]),
                }
                for i in range(n_orus)
            ],
            "macro_normalized_rmse": float(
                np.mean([np.sqrt(squared[i] / energy[i]) for i in range(n_orus) if energy[i] > 0])
            ),
        }

    # A compact learned regression target averages each antenna's magnitude over
    # subcarriers. It is deliberately distinct from the full-tensor scores above.
    summary_db = np.empty((channel.shape[0], n_orus * channel.shape[2]), dtype=np.float32)
    for start in range(0, channel.shape[0], chunk_pairs):
        stop = min(start + chunk_pairs, channel.shape[0])
        magnitude_mean = np.asarray(channel[start:stop], dtype=np.float32).mean(axis=-1)
        summary_db[start:stop] = 20.0 * np.log10(
            np.maximum(magnitude_mean.reshape(stop - start, -1), 1e-8)
        )
    all_anchors, all_targets, all_history = _aligned_pairs(
        timestamps,
        horizon_s=horizon_s,
        lookback_s=lookback_s,
        tolerance_s=tolerance_s,
    )
    train_end = int(np.floor(0.7 * timestamps.size))
    test_start = int(np.floor(0.8 * timestamps.size))
    train_mask = all_targets < train_end
    test_mask = (all_anchors >= test_start) & (all_history >= test_start)
    train_anchors = all_anchors[train_mask]
    train_targets = all_targets[train_mask]
    train_history = all_history[train_mask]
    test_anchors = all_anchors[test_mask]
    test_targets = all_targets[test_mask]
    test_history = all_history[test_mask]
    if train_anchors.size == 0 or test_anchors.size == 0:
        raise ValueError("temporal split produced an empty train or test partition")

    train_features = np.concatenate((summary_db[train_anchors], summary_db[train_history]), axis=1)
    train_targets_db = summary_db[train_targets]
    test_features = np.concatenate((summary_db[test_anchors], summary_db[test_history]), axis=1)
    test_targets_db = summary_db[test_targets]
    regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    regressor.fit(train_features, train_targets_db)
    ridge_prediction = np.asarray(regressor.predict(test_features), dtype=np.float64)
    persistence_summary = summary_db[test_anchors]
    trend_summary = persistence_summary + (persistence_summary - summary_db[test_history]) * (
        horizon_s / lookback_s
    )
    compact_scores = {
        "ridge": _score_summary_predictions(
            ridge_prediction, test_targets_db, n_orus=n_orus, n_rx=channel.shape[2]
        ),
        "persistence": _score_summary_predictions(
            persistence_summary, test_targets_db, n_orus=n_orus, n_rx=channel.shape[2]
        ),
        "linear_trend": _score_summary_predictions(
            trend_summary, test_targets_db, n_orus=n_orus, n_rx=channel.shape[2]
        ),
    }
    return {
        "schema": "caez-measured-magnitude-forecast.v1",
        "source_archive": integrity["source_file"],
        "archive_sha256": integrity["archive_sha256"],
        "archive_bytes": integrity["archive_bytes"],
        "forecast_target": "downsampled absolute OFDM-domain CSI magnitude; not complex CSI or LOS/NLOS",
        "horizon_s": horizon_s,
        "lookback_s": lookback_s,
        "timestamp_tolerance_s": tolerance_s,
        "split": {
            "policy": "chronological tail holdout",
            "test_fraction": test_fraction,
            "forecast_pairs": int(anchors.size),
            "first_test_anchor": float(timestamps[anchors[0]]),
            "last_test_target": float(timestamps[targets[-1]]),
        },
        "scores": scores,
        "compact_summary_regression": {
            "target": "per-O-RU, per-Rx-antenna subcarrier-mean magnitude in dB",
            "model": "StandardScaler fit on train block + Ridge(alpha=1.0)",
            "inputs": "current summary and summary nearest 500 ms in the past",
            "split": {
                "train_target_before_sample_fraction": 0.7,
                "test_anchor_after_sample_fraction": 0.8,
                "unused_temporal_gap_fraction": 0.1,
                "train_pairs": int(train_anchors.size),
                "test_pairs": int(test_anchors.size),
            },
            "scores": compact_scores,
            "limitations": [
                "Only summary magnitude is predicted; subcarrier-selective detail and phase are omitted.",
                "The Ridge alpha is fixed in advance and is not evidence of hyperparameter tuning.",
                "Single indoor campaign; chronological holdout does not establish other-site transfer.",
            ],
        },
        "limitations": [
            "Single indoor measurement campaign; this is not cross-scene validation.",
            "The selected feature archive discards complex phase and has no LOS/NLOS truth.",
            "The baselines use past observations only and do not establish a learned-model advantage.",
            "Temporal dependence remains within the held-out contiguous block by design.",
        ],
        "license": integrity["license"],
        "required_citation": integrity["required_citation"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--horizon-s", type=float, default=1.0)
    parser.add_argument("--lookback-s", type=float, default=0.5)
    parser.add_argument("--tolerance-s", type=float, default=0.02)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--output", type=Path, help="optional aggregate JSON output path")
    args = parser.parse_args()
    result = evaluate(
        args.archive,
        horizon_s=args.horizon_s,
        lookback_s=args.lookback_s,
        tolerance_s=args.tolerance_s,
        test_fraction=args.test_fraction,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
