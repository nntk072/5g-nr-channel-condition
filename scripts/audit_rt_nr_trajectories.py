"""Freeze scene-family splits and audit stored NR RT trajectories for leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from artifacts.provenance import package_source_sha256, sha256_file, versions
from datasets.nr_store import NRStoredTrajectory, read_nr_trajectory
from evaluation.duplicate_audit import audit_cfr_near_duplicates, find_cross_split_temporal_overlaps
from evaluation.independent_splits import (
    audit_group_leakage,
    freeze_split_lock,
    make_independent_split,
)


def _stored_trajectory(store: Path, trajectory_id: str) -> NRStoredTrajectory:
    key = hashlib.sha256(trajectory_id.encode()).hexdigest()
    manifest_path = store / "manifests" / "trajectories" / f"{key}.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["sample_ids"] = tuple(payload["sample_ids"])
    return NRStoredTrajectory(**payload)


def audit_trajectories(
    store: Path,
    specs: list[Path],
    split_lock: Path,
    report_path: Path,
    *,
    split_seed: str,
    observation_window_s: float,
    similarity_threshold: float,
) -> dict[str, Any]:
    """Load every listed trajectory, freeze scene-group splits, and audit inputs."""
    if not specs:
        raise ValueError("at least one trajectory spec is required")
    if not np.isfinite(observation_window_s) or observation_window_s <= 0:
        raise ValueError("observation_window_s must be finite and positive")

    records: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    observations: list[np.ndarray] = []
    input_hashes: list[dict[str, str]] = []
    trajectory_summaries: list[dict[str, Any]] = []
    for spec_path in specs:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        trajectory_id = str(spec["trajectory_id"])
        stored = _stored_trajectory(store, trajectory_id)
        trajectory = read_nr_trajectory(store, stored)
        scene = str(spec["scene"])
        # Procedural urban scenes share a generator/material family and stay in
        # one split even though their per-scene XML assets and seeds differ.
        group_id = "procedural_urban_family" if "urban_" in Path(scene).stem else scene
        trajectory_summaries.append(
            {
                "trajectory_id": trajectory_id,
                "scene": scene,
                "scene_group_id": group_id,
                "content_sha256": stored.content_sha256,
                "sample_count": len(trajectory.records),
                "sample_times_s": list(trajectory.sample_times_s),
            }
        )
        input_hashes.append({"path": str(spec_path), "sha256": sha256_file(spec_path)})
        for timestamp, record in zip(trajectory.sample_times_s, trajectory.records, strict=True):
            sample_id = record.observation.sample_id
            sample_ids.append(sample_id)
            records.append(
                {
                    "sample_id": sample_id,
                    "scene_group_id": group_id,
                    "trajectory_id": trajectory_id,
                    "window_start_s": timestamp,
                    "window_end_s": timestamp + observation_window_s,
                }
            )
            h_hat = record.observation.channel_estimate.h_hat
            if hasattr(h_hat, "detach"):
                h_hat = h_hat.detach().cpu().numpy()
            observations.append(np.asarray(h_hat).reshape(-1))

    split = make_independent_split(
        records,
        seed=split_seed,
        group_field="scene_group_id",
        fractions={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
    )
    audit_group_leakage(split, records)
    overlaps = find_cross_split_temporal_overlaps(records, split)
    duplicate_audit = audit_cfr_near_duplicates(
        sample_ids,
        np.stack(observations),
        split,
        similarity_threshold=similarity_threshold,
    )
    split_sha256 = freeze_split_lock(split_lock, split)
    report = {
        "schema": "loslab.rt_nr_trajectory_audit.v1",
        "status": "SIMULATION_PILOT_NOT_FIELD_QUALIFIED",
        "split_seed": split_seed,
        "split_group_field": split.group_field,
        "group_to_split": dict(sorted(split.group_to_split.items())),
        "sample_counts_by_split": {
            name: sum(value == name for value in split.sample_to_split.values())
            for name in ("train", "validation", "test")
        },
        "group_counts_by_split": {
            name: sum(value == name for value in split.group_to_split.values())
            for name in ("train", "validation", "test")
        },
        "trajectory_count": len(trajectory_summaries),
        "sample_count": len(sample_ids),
        "observation_window_s": observation_window_s,
        "temporal_cross_split_overlap_count": len(overlaps),
        "temporal_cross_split_overlaps": [
            {
                "sample_id_a": item.sample_id_a,
                "sample_id_b": item.sample_id_b,
                "split_a": item.split_a,
                "split_b": item.split_b,
                "overlap_duration_s": item.overlap_duration_s,
            }
            for item in overlaps
        ],
        "cfr_similarity_threshold": similarity_threshold,
        "cross_split_cfr_near_duplicate_count": len(duplicate_audit.pairs),
        "cross_split_cfr_near_duplicates": [
            {
                "sample_id_a": item.sample_id_a,
                "sample_id_b": item.sample_id_b,
                "split_a": item.split_a,
                "split_b": item.split_b,
                "similarity": item.similarity,
            }
            for item in duplicate_audit.pairs
        ],
        "trajectories": trajectory_summaries,
        "specs": input_hashes,
        "split_lock": str(split_lock),
        "split_lock_sha256": split_sha256,
        "runtime_versions": versions(),
        "package_source_sha256": package_source_sha256(),
        "limitations": [
            "Six simulated scene assets form a pilot, not a powered estimate of scene generalization.",
            "Three urban scenes share a procedural asset family and are kept in one split.",
            "Validation and test each contain only one independent scene group in this pilot.",
            "Temporal windows are declared NR-slot observation durations; they do not imply a forecast horizon.",
            "CFR near-duplicate checks use receiver-visible DMRS estimates, not path-oracle targets.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    if report_path.exists() and report_path.read_bytes() != encoded:
        raise FileExistsError(f"refusing to overwrite audit report: {report_path}")
    if not report_path.exists():
        report_path.write_bytes(encoded)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("split_lock", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("specs", nargs="+", type=Path)
    parser.add_argument("--split-seed", default="rt-nr-scene-family-v1")
    parser.add_argument("--observation-window-s", type=float, default=0.0005)
    parser.add_argument("--similarity-threshold", type=float, default=0.9999)
    args = parser.parse_args()
    result = audit_trajectories(
        args.store,
        args.specs,
        args.split_lock,
        args.report,
        split_seed=args.split_seed,
        observation_window_s=args.observation_window_s,
        similarity_threshold=args.similarity_threshold,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
