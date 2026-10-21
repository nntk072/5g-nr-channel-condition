"""Evaluate fixed one-step CSI forecasts against the frozen RT scene split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from artifacts.provenance import package_source_sha256, sha256_file, versions
from datasets.nr_store import NRStoredTrajectory, read_nr_trajectory
from evaluation.independent_splits import IndependentSplit
from experiments.rt_trajectory_forecast import (
    evaluate_trajectory_forecasts,
    summarize_forecast_rows,
)


def _load_stored_trajectory(store: Path, trajectory_id: str) -> NRStoredTrajectory:
    key = hashlib.sha256(trajectory_id.encode()).hexdigest()
    path = store / "manifests" / "trajectories" / f"{key}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sample_ids"] = tuple(payload["sample_ids"])
    return NRStoredTrajectory(**payload)


def evaluate_frozen_trajectories(
    store: Path,
    split_lock_path: Path,
    spec_paths: list[Path],
) -> dict[str, Any]:
    """Evaluate all fixed baselines once; preserve split membership in every row."""
    raw_lock = json.loads(split_lock_path.read_text(encoding="utf-8"))
    split = IndependentSplit(
        sample_to_split=raw_lock["sample_to_split"],
        group_to_split=raw_lock["group_to_split"],
        seed=raw_lock["seed"],
        group_field=raw_lock["group_field"],
    )
    if not spec_paths:
        raise ValueError("at least one trajectory spec is required")

    all_rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    input_specs: list[dict[str, str]] = []
    observed_samples: set[str] = set()
    for spec_path in spec_paths:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        trajectory_id = str(spec["trajectory_id"])
        stored = _load_stored_trajectory(store, trajectory_id)
        trajectory = read_nr_trajectory(store, stored)
        if observed_samples.intersection(trajectory.sample_ids):
            raise ValueError(f"trajectory specs contain duplicate samples: {trajectory_id}")
        observed_samples.update(trajectory.sample_ids)
        sample_splits = {split.sample_to_split[sample_id] for sample_id in trajectory.sample_ids}
        if len(sample_splits) != 1:
            raise ValueError(f"trajectory crosses frozen splits: {trajectory_id}")
        assigned_split = next(iter(sample_splits))
        if split.group_field == "scene_group_id":
            group_id = (
                "procedural_urban_family"
                if "urban_" in Path(str(spec["scene"])).stem
                else str(spec["scene"])
            )
            if split.group_to_split.get(group_id) != assigned_split:
                raise ValueError(
                    f"trajectory scene group differs from frozen lock: {trajectory_id}"
                )
        rows = evaluate_trajectory_forecasts(trajectory, split=assigned_split)
        all_rows.extend(rows)
        trajectories.append(
            {
                "trajectory_id": trajectory_id,
                "scene": str(spec["scene"]),
                "split": assigned_split,
                "content_sha256": stored.content_sha256,
                "sample_count": len(trajectory.records),
            }
        )
        input_specs.append({"path": str(spec_path), "sha256": sha256_file(spec_path)})

    if observed_samples != set(split.sample_to_split):
        raise ValueError("trajectory specs must cover exactly the samples in the frozen split")

    metrics_by_split = {
        name: summarize_forecast_rows([row for row in all_rows if row["split"] == name])
        for name in ("train", "validation", "test")
    }
    predictions = sorted(
        all_rows,
        key=lambda row: (
            row["split"],
            row["trajectory_id"],
            row["forecast_target_s"],
            row["method"],
        ),
    )
    return {
        "schema": "loslab.rt_nr_forecast_pilot.v1",
        "status": "PILOT_FIXED_BASELINES_NOT_A_FORECASTING_CLAIM",
        "target": "future absolute-delay full RT CFR",
        "model_inputs": "past receiver-visible PUSCH DMRS channel estimates only",
        "methods": {
            "persistence": "reuse the latest past DMRS channel estimate",
            "linear_complex_extrapolation": "two latest estimates with timestamp-scaled complex linear trend",
        },
        "split_lock": str(split_lock_path),
        "split_lock_sha256": sha256_file(split_lock_path),
        "split_seed": split.seed,
        "trajectory_count": len(trajectories),
        "transition_count": len(predictions),
        "trajectories": trajectories,
        "specs": input_specs,
        "metrics_by_split": metrics_by_split,
        "predictions": predictions,
        "runtime_versions": versions(),
        "package_source_sha256": package_source_sha256(),
        "limitations": [
            "Only six short synthetic or built-in scene trajectories are available.",
            "The locked test split contains one independent scene group and three one-step persistence transitions.",
            "The linear complex trend is a simple baseline, not a Doppler or learned dynamics model.",
            "Targets are simulator CFR truth; no measured channel or within-slot dynamics are evaluated.",
            "Metrics are descriptive and do not support a statistical scene-generalization claim.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("split_lock", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("specs", nargs="+", type=Path)
    args = parser.parse_args()
    report = evaluate_frozen_trajectories(args.store, args.split_lock, args.specs)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    if args.output.exists():
        parser.error(f"refusing to overwrite existing report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(json.dumps({"report": str(args.output), "sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
