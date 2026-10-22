"""Evaluate zero-shot CAEZ magnitude-summary transfer across measurement domains."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from datasets.caez import _numeric_member, audit
from evaluation.caez_forecast import (
    align_forecast_pairs,
    paired_time_block_bootstrap,
    score_summary_predictions,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _load_summary(path: Path, *, chunk_rows: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    report = audit(path, chunk_rows=chunk_rows)
    channel, _ = _numeric_member(path.resolve(), "H")
    timestamps, _ = _numeric_member(path.resolve(), "timestamps")
    if channel.ndim != 4 or channel.shape[1] != 4 or channel.shape[2] < 1:
        raise ValueError("H must have [sample, 4 O-RU, Rx antenna, subcarrier] axes")
    if channel.shape[0] != timestamps.size:
        raise ValueError("H and timestamps sample counts differ")
    summary = np.empty((channel.shape[0], channel.shape[1] * channel.shape[2]), dtype=np.float32)
    for start in range(0, channel.shape[0], chunk_rows):
        stop = min(start + chunk_rows, channel.shape[0])
        magnitude_mean = np.asarray(channel[start:stop], dtype=np.float32).mean(axis=-1)
        summary[start:stop] = 20.0 * np.log10(
            np.maximum(magnitude_mean.reshape(stop - start, -1), 1e-8)
        )
    return report, summary, np.asarray(timestamps)


def _source_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    code_files = (
        Path(__file__).resolve(),
        root / "evaluation" / "caez_forecast.py",
    )
    source_hashes = {}
    for path in code_files:
        source_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_revision": revision,
        "source_sha256": source_hashes,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def evaluate_transfer(
    train_archive: Path,
    test_archives: list[Path],
    *,
    horizon_s: float = 1.0,
    lookback_s: float = 0.5,
    tolerance_s: float = 0.02,
    train_fraction: float = 0.7,
    test_start_fraction: float = 0.8,
    chunk_rows: int = 2048,
    bootstrap_block_seconds: float = 10.0,
    bootstrap_replicates: int = 1000,
) -> dict[str, Any]:
    """Fit only on an early source-domain block and score untouched target tails."""
    if not test_archives:
        raise ValueError("at least one test archive is required")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if bootstrap_block_seconds <= 0 or bootstrap_replicates <= 0:
        raise ValueError("bootstrap duration and replicate count must be positive")
    if not 0 < train_fraction < test_start_fraction < 1:
        raise ValueError("fractions must satisfy 0 < train < test_start < 1")

    train_report, train_summary, train_times = _load_summary(train_archive, chunk_rows=chunk_rows)
    train_anchors, train_targets, train_history = align_forecast_pairs(
        train_times,
        horizon_s=horizon_s,
        lookback_s=lookback_s,
        tolerance_s=tolerance_s,
    )
    train_end = int(np.floor(train_fraction * train_times.size))
    train_mask = train_targets < train_end
    train_anchors = train_anchors[train_mask]
    train_targets = train_targets[train_mask]
    train_history = train_history[train_mask]
    if train_anchors.size == 0:
        raise ValueError("source-domain temporal split produced no training pairs")

    features_train = np.concatenate(
        (train_summary[train_anchors], train_summary[train_history]), axis=1
    )
    target_train = train_summary[train_targets]
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(features_train, target_train)

    n_orus = 4
    n_rx = int(train_summary.shape[1] // n_orus)
    results: list[dict[str, Any]] = []
    source_key = train_archive.resolve()
    loaded_domains = {
        source_key: (train_report, train_summary, train_times),
    }
    for target_path in test_archives:
        target_key = target_path.resolve()
        if target_key not in loaded_domains:
            loaded_domains[target_key] = _load_summary(target_path, chunk_rows=chunk_rows)
        target_report, target_summary, target_times = loaded_domains[target_key]
        if target_report["fields"]["H"]["shape"][1:] != train_report["fields"]["H"]["shape"][1:]:
            raise ValueError("source and target archives have different channel feature dimensions")
        anchors, targets, history = align_forecast_pairs(
            target_times,
            horizon_s=horizon_s,
            lookback_s=lookback_s,
            tolerance_s=tolerance_s,
        )
        start_index = int(np.floor(test_start_fraction * target_times.size))
        test_mask = (anchors >= start_index) & (history >= start_index)
        anchors = anchors[test_mask]
        targets = targets[test_mask]
        history = history[test_mask]
        if anchors.size == 0:
            raise ValueError(f"target temporal split produced no test pairs: {target_path.name}")

        features = np.concatenate((target_summary[anchors], target_summary[history]), axis=1)
        actual = target_summary[targets]
        prediction = np.asarray(model.predict(features), dtype=np.float64)
        persistence = target_summary[anchors]
        trend = persistence + (persistence - target_summary[history]) * (horizon_s / lookback_s)
        predictions = {
            "ridge_source_trained": prediction,
            "persistence": persistence,
            "linear_trend": trend,
        }
        scores = {
            name: score_summary_predictions(values, actual, n_orus=n_orus, n_rx=n_rx)
            for name, values in predictions.items()
        }
        median_step = float(np.median(np.diff(target_times)))
        block_rows = max(1, int(round(bootstrap_block_seconds / median_step)))
        uncertainty = paired_time_block_bootstrap(
            actual,
            predictions,
            anchors,
            test_start=start_index,
            block_rows=block_rows,
            n_orus=n_orus,
            n_rx=n_rx,
            n_bootstrap=bootstrap_replicates,
            seed=20261022,
        )
        uncertainty["block_duration_seconds_nominal"] = bootstrap_block_seconds
        results.append(
            {
                "environment_id": target_path.stem,
                "archive_sha256": target_report["archive_sha256"],
                "sample_count": target_report["sample_count"],
                "archive_bytes": target_report["archive_bytes"],
                "integrity": target_report["zip_crc_check"],
                "channel_feature_shape": target_report["fields"]["H"]["shape"][1:],
                "test_pairs": int(anchors.size),
                "first_test_anchor": float(target_times[anchors[0]]),
                "last_test_target": float(target_times[targets[-1]]),
                "scores": scores,
                "uncertainty": uncertainty,
            }
        )

    return {
        "schema": "caez-cross-environment-magnitude-summary.v1",
        "provenance": _source_provenance(),
        "training_domain": {
            "environment_id": train_archive.stem,
            "archive_sha256": train_report["archive_sha256"],
            "sample_count": train_report["sample_count"],
            "archive_bytes": train_report["archive_bytes"],
            "integrity": train_report["zip_crc_check"],
            "channel_feature_shape": train_report["fields"]["H"]["shape"][1:],
            "train_target_before_sample_fraction": train_fraction,
            "train_pairs": int(train_anchors.size),
        },
        "target": "per-O-RU, per-Rx-antenna subcarrier-mean absolute CSI magnitude in dB",
        "model": "StandardScaler fit on source training block + Ridge(alpha=1.0)",
        "inputs": "target-domain current summary and summary nearest lookback_s in the past",
        "horizon_s": horizon_s,
        "lookback_s": lookback_s,
        "timestamp_tolerance_s": tolerance_s,
        "test_start_fraction": test_start_fraction,
        "unused_source_temporal_gap_fraction": test_start_fraction - train_fraction,
        "target_domains": results,
        "interpretation_limits": [
            "Zero-shot means no target labels or target-domain fitting; observed target-domain CSI at the forecast anchor is an allowed input shared with persistence and trend.",
            "The compact target averages away subcarrier-selective detail and phase.",
            "CAEZ positions do not provide geometric LOS/NLOS labels; this is not LOS/NLOS validation.",
            "Environment, UE, trajectory, and measurement-session changes are confounded in this cross-dataset comparison.",
            "Each target is one measurement campaign with a chronological tail holdout, not multiple independent sites or trajectories.",
            "The paired 10-second time-block bootstrap describes uncertainty within each held-out campaign; it is not across-site or across-trajectory uncertainty.",
        ],
        "license": "CAEZ Dataset License v1.0; attribution required; raw data and subsets must not be redistributed.",
        "required_citation": train_report["required_citation"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-archive", type=Path, required=True)
    parser.add_argument("--test-archive", type=Path, nargs="+", required=True)
    parser.add_argument("--horizon-s", type=float, default=1.0)
    parser.add_argument("--lookback-s", type=float, default=0.5)
    parser.add_argument("--tolerance-s", type=float, default=0.02)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--test-start-fraction", type=float, default=0.8)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--bootstrap-block-seconds", type=float, default=10.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_transfer(
        args.train_archive,
        args.test_archive,
        horizon_s=args.horizon_s,
        lookback_s=args.lookback_s,
        tolerance_s=args.tolerance_s,
        train_fraction=args.train_fraction,
        test_start_fraction=args.test_start_fraction,
        chunk_rows=args.chunk_rows,
        bootstrap_block_seconds=args.bootstrap_block_seconds,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
