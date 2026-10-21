"""Evaluate a chronological persistence/trend baseline on CAEZ CSI magnitudes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets.caez import _numeric_member, audit


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

    test_start = int(np.floor(times.size * (1.0 - test_fraction)))
    anchors = np.arange(test_start, times.size, dtype=np.int64)
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
    valid = (
        right_valid
        & (targets > anchors)
        & (errors <= tolerance_s)
        & (history >= test_start)
        & (history < anchors)
    )
    return anchors[valid], targets[valid], history[valid]


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
