"""Audit and summarize a fixed three-transmitter RT/DMRS holdout experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from artifacts.provenance import atomic_json, package_source_sha256, sha256_file, versions
from datasets.channel import ChannelDataset


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _repository_relative_path(path: Path) -> str:
    """Return a portable repository-relative artifact path when available."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def summarize_tx_holdouts(run_root: Path) -> dict[str, Any]:
    """Verify the three fixed TX domains and their matched holdout runs."""
    observation_path = run_root / "nr-dmrs-observations.npz"
    if not observation_path.is_file():
        raise ValueError(f"missing DMRS observation artifact: {observation_path}")
    observations = ChannelDataset.load(observation_path)
    if observations.label_semantics != "geometric_direct_path_oracle":
        raise ValueError("TX-location experiment requires geometric direct-path labels")

    ids_by_domain: dict[str, set[str]] = {}
    positions_by_id: dict[str, set[tuple[float, float, float]]] = {}
    receiver_regions_by_id: dict[str, set[tuple[tuple[float, float, float], float]]] = {}
    samples_by_id: dict[str, list[int]] = {}
    scene_ids: set[str] = set()
    for index, (domain, label, metadata) in enumerate(
        zip(observations.domain, observations.y, observations.metadata, strict=True)
    ):
        transmitter_id = str(metadata.get("transmitter_id") or "")
        if not transmitter_id:
            raise ValueError("every observation must carry a transmitter_id")
        scene_reference = str(metadata.get("scene") or "")
        scene_id = Path(scene_reference.replace("\\", "/")).stem
        if not scene_id:
            raise ValueError("every observation must identify its RT scene")
        scene_ids.add(scene_id)
        expected_domain = f"rt_{scene_id}_{transmitter_id}"
        if str(domain) != expected_domain:
            raise ValueError(f"transmitter/domain mismatch: {transmitter_id} vs {domain}")
        tx = tuple(float(value) for value in metadata.get("tx", ()))
        rx = tuple(float(value) for value in metadata.get("rx", ()))
        if len(tx) != 3 or len(rx) != 3 or not np.isfinite((*tx, *rx)).all():
            raise ValueError("every record must contain finite 3D TX and RX coordinates")
        scene_config = metadata.get("scene_config")
        if not isinstance(scene_config, dict):
            raise ValueError("every record must declare its receiver-region configuration")
        center = tuple(float(value) for value in scene_config.get("rx_center", ()))
        radius = float(scene_config.get("radius_m", float("nan")))
        if len(center) != 3 or not np.isfinite((*center, radius)).all() or radius <= 0.0:
            raise ValueError(
                "every record must declare a finite 3D receiver center and positive radius"
            )
        if np.linalg.norm(np.asarray(rx[:2]) - np.asarray(center[:2])) > radius + 1e-6:
            raise ValueError("receiver sample lies outside its declared receiver-region disk")
        ids_by_domain.setdefault(str(domain), set()).add(transmitter_id)
        positions_by_id.setdefault(transmitter_id, set()).add(tx)
        receiver_regions_by_id.setdefault(transmitter_id, set()).add((center, radius))
        samples_by_id.setdefault(transmitter_id, []).append(index)

    if len(ids_by_domain) != 3 or len(positions_by_id) != 3:
        raise ValueError("exactly three distinct transmitter-location domains are required")
    if len(scene_ids) != 1:
        raise ValueError("all transmitter domains must use one common RT scene")
    scene_id = next(iter(scene_ids))
    if any(len(values) != 1 for values in positions_by_id.values()):
        raise ValueError("a transmitter ID maps to multiple coordinates")
    if any(len(values) != 1 for values in receiver_regions_by_id.values()):
        raise ValueError("a transmitter ID maps to multiple receiver regions")
    receiver_regions = {
        transmitter_id: next(iter(values))
        for transmitter_id, values in receiver_regions_by_id.items()
    }
    region_separation: dict[str, float] = {}
    for index, first_id in enumerate(sorted(receiver_regions)):
        first_center, first_radius = receiver_regions[first_id]
        for second_id in sorted(receiver_regions)[index + 1 :]:
            second_center, second_radius = receiver_regions[second_id]
            center_distance = float(
                np.linalg.norm(np.asarray(first_center[:2]) - np.asarray(second_center[:2]))
            )
            gap = center_distance - first_radius - second_radius
            if gap <= 0.0:
                raise ValueError("receiver-region disks overlap across transmitter domains")
            region_separation[f"{first_id}__{second_id}"] = gap
    for transmitter_id, indices in samples_by_id.items():
        if len(indices) != 40 or np.bincount(observations.y[indices], minlength=2).tolist() != [
            20,
            20,
        ]:
            raise ValueError(f"{transmitter_id} must contain 20 links per geometric class")

    groups_by_id = {
        transmitter_id: set(observations.group[indices].tolist())
        for transmitter_id, indices in samples_by_id.items()
    }
    for first_id, first_groups in groups_by_id.items():
        for second_id, second_groups in groups_by_id.items():
            if first_id < second_id and first_groups & second_groups:
                raise ValueError("spatial-block group IDs overlap across transmitter domains")

    fold_reports: list[dict[str, Any]] = []
    expected_config: dict[str, Any] | None = None
    input_sha256 = sha256_file(observation_path)
    per_method: dict[str, list[float]] = {}
    rx_points = {
        transmitter_id: np.asarray(
            [observations.metadata[index]["rx"] for index in indices], dtype=float
        )
        for transmitter_id, indices in samples_by_id.items()
    }

    for transmitter_id in sorted(samples_by_id):
        holdout = f"rt_{scene_id}_{transmitter_id}"
        fold_dir = run_root / f"holdout-{transmitter_id}"
        manifest_path = fold_dir / "manifest.json"
        results_path = fold_dir / "results.json"
        summary_path = fold_dir / "evaluation" / "summary.json"
        complete_path = fold_dir / "COMPLETE"
        if not all(
            path.is_file() for path in (manifest_path, results_path, summary_path, complete_path)
        ):
            raise ValueError(f"incomplete holdout artifacts for {transmitter_id}")
        manifest = _read_json(manifest_path)
        config = manifest["config"]
        if config.get("holdout") != holdout:
            raise ValueError(f"unexpected held-out domain in {manifest_path}")
        if config.get("observation") != "precomputed":
            raise ValueError("all folds must use the same precomputed DMRS input")
        if expected_config is None:
            expected_config = config
            per_method = {str(method): [] for method in config["methods"]}
        else:

            def normalize(value: dict[str, Any]) -> dict[str, Any]:
                return {key: item for key, item in value.items() if key != "holdout"}

            if normalize(config) != normalize(expected_config):
                raise ValueError("holdout configurations differ across transmitter locations")
        inputs = manifest.get("inputs", [])
        if len(inputs) != 1 or inputs[0].get("sha256") != input_sha256:
            raise ValueError("holdout runs do not use the exact shared observation artifact")

        rows = _read_json(results_path)
        expected_rows = len(config["methods"]) * len(config["seeds"])
        if not isinstance(rows, list) or len(rows) != expected_rows:
            raise ValueError(f"unexpected result count in {results_path}")
        split_hashes: dict[int, str] = {}
        fold_metrics: dict[str, dict[str, float | int]] = {}
        for row in rows:
            split = row["split"]
            if split["test_domains"] != [holdout] or holdout in split["train_domains"]:
                raise ValueError(f"held-out transmitter leaked into training: {results_path}")
            if set(split["train_domains"]) | {holdout} != set(ids_by_domain):
                raise ValueError(f"fold domains do not reconcile: {results_path}")
            seed = int(row["seed"])
            split_hash = str(split["indices_sha256"])
            if seed in split_hashes and split_hashes[seed] != split_hash:
                raise ValueError(f"methods do not share splits for seed {seed}")
            split_hashes[seed] = split_hash
        for method in config["methods"]:
            method_rows = [row for row in rows if row["method"] == method]
            if sorted(int(row["seed"]) for row in method_rows) != sorted(config["seeds"]):
                raise ValueError(f"missing or duplicate {method} seed rows in {results_path}")
            scores = [float(row["metrics"]["auroc"]) for row in method_rows]
            if not np.isfinite(scores).all():
                raise ValueError(f"non-finite AUROC in {results_path}")
            mean = float(np.mean(scores))
            fold_metrics[str(method)] = {
                "mean_test_auroc_across_seeds": mean,
                "std_test_auroc_across_seeds": float(np.std(scores)),
                "n_seeds": len(scores),
                "n_test_samples": int(rows[0]["split"]["test_size"]),
            }
            per_method[str(method)].append(mean)

        fold_reports.append(
            {
                "transmitter_id": transmitter_id,
                "tx_position_m": list(next(iter(positions_by_id[transmitter_id]))),
                "held_out_domain": holdout,
                "n_samples": len(samples_by_id[transmitter_id]),
                "class_counts": {"geometric_nlos": 20, "geometric_los": 20},
                "n_spatial_groups": len(groups_by_id[transmitter_id]),
                "split_hashes_by_seed": {str(key): value for key, value in split_hashes.items()},
                "methods": fold_metrics,
                "manifest_sha256": sha256_file(manifest_path),
                "results_sha256": sha256_file(results_path),
                "summary_sha256": sha256_file(summary_path),
            }
        )

    rx_overlap: dict[str, dict[str, float | int]] = {}
    ids = sorted(rx_points)
    for index, first_id in enumerate(ids):
        for second_id in ids[index + 1 :]:
            distances = np.linalg.norm(
                rx_points[first_id][:, None, :] - rx_points[second_id][None, :, :], axis=2
            )
            nearest = distances.min(axis=1)
            rx_overlap[f"{first_id}__{second_id}"] = {
                "minimum_nearest_rx_distance_m": float(nearest.min()),
                "median_nearest_rx_distance_m": float(np.median(nearest)),
                "rx_samples_with_other_tx_within_1m": int(np.count_nonzero(nearest < 1.0)),
            }

    assert expected_config is not None
    return {
        "schema": "loslab.rt_transmitter_holdout_summary.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "study_scope": "joint within-scene transmitter-and-receiver location holdout",
        "scene_id": scene_id,
        "run_root": _repository_relative_path(run_root),
        "observation_dataset": {
            "path": observation_path.relative_to(run_root).as_posix(),
            "sha256": input_sha256,
            "sample_count": len(observations.y),
            "channel_shape": list(observations.h.shape),
            "label_semantics": observations.label_semantics,
            "source_datasets": [
                {
                    "path": f"{transmitter_id}.npz",
                    "sha256": sha256_file(run_root / f"{transmitter_id}.npz"),
                }
                for transmitter_id in ids
            ],
        },
        "scene_asset_set_sha256": sorted(
            {str(row["scene_asset_set_sha256"]) for row in observations.metadata}
        ),
        "transmitter_count": len(positions_by_id),
        "receiver_regions": {
            transmitter_id: {
                "center_m": list(center),
                "radius_m": radius,
            }
            for transmitter_id, (center, radius) in receiver_regions.items()
        },
        "minimum_receiver_region_separation_m": region_separation,
        "domains": sorted(ids_by_domain),
        "methods": expected_config["methods"],
        "seeds": expected_config["seeds"],
        "configuration": {key: value for key, value in expected_config.items() if key != "holdout"},
        "folds": fold_reports,
        "mean_across_heldout_tx_locations": {
            method: {
                "descriptive_mean_auroc": float(np.mean(scores)),
                "descriptive_std_across_three_tx_locations": float(np.std(scores)),
                "per_tx_means": scores,
            }
            for method, scores in per_method.items()
        },
        "receiver_position_overlap_diagnostic": rx_overlap,
        "runtime_versions": versions(),
        "summarizer_sha256": sha256_file(Path(__file__)),
        "package_source_sha256": package_source_sha256(),
        "limitations": [
            "One synthetic RT scene and three transmitter locations only.",
            "TX and RX regions move together, so this experiment cannot isolate transmitter-only effects.",
            "Disjoint receiver regions reduce spatial overlap but do not replace independent-scene or measured-data evaluation.",
            "Forty balanced links per held-out TX and seven training seeds do not support a benchmark or independent-scene inference.",
            "The scene geometry/materials are not independently qualified; results are not measured radio evidence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing report: {args.output}")
    report = summarize_tx_holdouts(args.run_root)
    atomic_json(args.output, report)
    print(json.dumps(report["mean_across_heldout_tx_locations"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
