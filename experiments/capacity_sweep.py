"""Immutable grouped-split sweep across the registered neural capacities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from artifacts.provenance import atomic_json, canonical_hash, sha256_file

from experiments.runner import run_experiment

CAPACITY_TIERS = ("small", "medium", "large")


def run_capacity_sweep(
    paths: Sequence[str | Path],
    output: str | Path,
    *,
    methods: Sequence[str] = ("mlp", "raw_mlp", "cnn", "cvnn", "coral", "coral0"),
    seeds: Sequence[int] = (11, 23, 47, 59, 71, 83),
    bootstrap: int = 200,
    holdout: str | None = None,
    include_phase: bool = False,
    observation: str = "ideal_cfr",
    pilot_spacing: int = 4,
    pilot_snr_db: float = 12.0,
    observation_seed: int = 2026,
    device: str = "cpu",
    coral_penalty: float = 0.1,
) -> dict:
    """Run all capacity tiers with shared inputs/seeds and verify split identity.

    The output directory is created once and is immutable after completion. A
    failed sweep retains its child runs and a failed manifest for inspection.
    """
    if not paths:
        raise ValueError("No datasets supplied")
    if not methods or not seeds:
        raise ValueError("methods and seeds must both be nonempty")
    if len(set(methods)) != len(methods) or len(set(seeds)) != len(seeds):
        raise ValueError("methods and seeds must not contain duplicates")
    input_hashes = [{"path": str(path), "sha256": sha256_file(path)} for path in paths]
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"capacity sweep output already exists: {root}")
    root.mkdir(parents=True)
    config = {
        "capacity_tiers": list(CAPACITY_TIERS),
        "methods": list(methods),
        "seeds": list(seeds),
        "bootstrap": bootstrap,
        "holdout": holdout,
        "include_phase": include_phase,
        "observation": observation,
        "pilot_spacing": pilot_spacing,
        "pilot_snr_db": pilot_snr_db,
        "observation_seed": observation_seed,
        "device": device,
        "coral_penalty": coral_penalty,
    }
    manifest: dict[str, Any] = {
        "schema": "loslab.capacity_sweep.v1",
        "state": "running",
        "config": config,
        "config_sha256": canonical_hash(config),
        "inputs": input_hashes,
    }
    atomic_json(root / "manifest.json", manifest)
    child_runs: dict[str, dict[str, Any]] = {}
    try:
        for tier in CAPACITY_TIERS:
            child = root / tier
            run_experiment(
                paths,
                child,
                holdout=holdout,
                methods=methods,
                seeds=seeds,
                bootstrap=bootstrap,
                include_phase=include_phase,
                observation=observation,
                pilot_spacing=pilot_spacing,
                pilot_snr_db=pilot_snr_db,
                observation_seed=observation_seed,
                device=device,
                coral_penalty=coral_penalty,
                capacity=tier,
            )
            child_manifest_path = child / "manifest.json"
            child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
            if child_manifest.get("state") != "complete" or not (child / "COMPLETE").is_file():
                raise ValueError(f"child run did not complete successfully: {tier}")
            child_runs[tier] = {
                "manifest_sha256": sha256_file(child_manifest_path),
                "results": json.loads((child / "results.json").read_text(encoding="utf-8")),
            }

        reference_splits: dict[tuple[int, str], str] = {}
        reference_keys: set[tuple[int, str]] | None = None
        points: list[dict] = []
        for tier in CAPACITY_TIERS:
            rows = child_runs[tier]["results"]
            keys = [(int(row["seed"]), str(row["method"])) for row in rows]
            if len(set(keys)) != len(keys):
                raise ValueError(f"duplicate seed/method result in capacity tier {tier}")
            if reference_keys is None:
                reference_keys = set(keys)
            elif set(keys) != reference_keys:
                raise ValueError(f"result methods or seeds differ across capacity tier {tier}")
            for row in rows:
                key = (int(row["seed"]), str(row["method"]))
                split_hash = str(row["split"]["indices_sha256"])
                if tier == CAPACITY_TIERS[0]:
                    reference_splits[key] = split_hash
                elif reference_splits.get(key) != split_hash:
                    raise ValueError(f"split identity differs across capacities for {key}")
            for method in methods:
                selected = [row for row in rows if row["method"] == method]
                parameter_counts = [
                    row["compute"]["learned_parameter_count"]
                    for row in selected
                    if row["compute"]["learned_parameter_count"] is not None
                ]
                points.append(
                    {
                        "capacity": tier,
                        "method": method,
                        "seeds": [int(row["seed"]) for row in selected],
                        "mean_test_auroc": float(
                            np.mean([row["metrics"]["auroc"] for row in selected])
                        ),
                        "std_test_auroc": (
                            float(np.std([row["metrics"]["auroc"] for row in selected], ddof=1))
                            if len(selected) > 1
                            else None
                        ),
                        "mean_training_seconds": float(
                            np.mean([row["compute"]["training_seconds"] for row in selected])
                        ),
                        "mean_learned_parameter_count": (
                            float(np.mean(parameter_counts)) if parameter_counts else None
                        ),
                    }
                )
        report = {
            "schema": "loslab.capacity_sweep_report.v1",
            "config_sha256": manifest["config_sha256"],
            "split_identity_count": len(reference_splits),
            "points": points,
            "child_runs": {
                tier: {"path": tier, "manifest_sha256": child_runs[tier]["manifest_sha256"]}
                for tier in CAPACITY_TIERS
            },
            "interpretation": (
                "Capacity comparisons use identical grouped split IDs and configured seeds. "
                "They are conditional on the supplied data and are not independent-scene inference. "
                "Wall times are machine-specific; this report does not equalize FLOPs or tuning budgets."
            ),
        }
        atomic_json(root / "capacity_sweep.json", report)
        manifest["state"] = "complete"
        manifest["report_sha256"] = sha256_file(root / "capacity_sweep.json")
        atomic_json(root / "manifest.json", manifest)
        atomic_json(
            root / "COMPLETE",
            {
                "config_sha256": manifest["config_sha256"],
                "report_sha256": manifest["report_sha256"],
            },
        )
        return report
    except Exception as error:
        manifest["state"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        atomic_json(root / "manifest.json", manifest)
        raise
