"""Immutable grouped-split sweep across the registered neural capacities."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Sequence

import jsonschema
import numpy as np
from artifacts.manifest import mark_complete, mark_failed
from artifacts.provenance import (
    atomic_json,
    canonical_hash,
    package_source_sha256,
    sha256_file,
    versions,
)
from artifacts.run_replay import build_run_replay_provenance
from referencing import Registry, Resource

from experiments.runner import run_experiment

CAPACITY_TIERS = ("small", "medium", "large")
_MANIFEST_SCHEMA_PATH = (
    Path(__file__).resolve().parent / "schema" / "capacity-sweep-manifest.v2.schema.json"
)


def _validate_manifest(value: dict[str, Any]) -> dict[str, Any]:
    schema = json.loads(_MANIFEST_SCHEMA_PATH.read_text(encoding="utf-8"))
    replay_schema_path = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "schema"
        / "run-replay-provenance.v1.schema.json"
    )
    registry = Registry().with_resource(
        replay_schema_path.as_uri(),
        Resource.from_contents(json.loads(replay_schema_path.read_text(encoding="utf-8"))),
    )
    jsonschema.Draft202012Validator(
        schema, registry=registry, format_checker=jsonschema.FormatChecker()
    ).validate(value)
    if value["run_replay"]["state"] != value["state"]:
        raise ValueError("capacity sweep and replay provenance states differ")
    if value["run_replay"]["config_sha256"] != value["config_sha256"]:
        raise ValueError("capacity sweep replay config digest differs from its manifest")
    if value["run_replay"]["inputs"] != value["inputs"]:
        raise ValueError("capacity sweep replay input roster differs from its manifest")
    return value


def _capacity_sweep_rng_streams(config: dict[str, Any]) -> list[dict[str, Any]]:
    streams = [
        {
            "name": f"capacity_child_seed_{seed}",
            "library": "numpy/scikit-learn/torch",
            "algorithm": "common_runner_seed_v1",
            "seed_material": [seed],
            "derived_seed": seed,
            "purpose": "child split, model fitting, and seed-clustered evaluation",
        }
        for seed in config["seeds"]
    ]
    if config["observation"] == "comb_ls":
        seed = int(config["observation_seed"])
        streams.append(
            {
                "name": "observation_noise",
                "library": "numpy",
                "algorithm": "comb_ls_estimate_rng_v1",
                "seed_material": [seed],
                "derived_seed": seed,
                "purpose": "comb-LS observation noise in each child run",
            }
        )
    seed = int(config["phase_randomization_seed"])
    streams.append(
        {
            "name": "phase_randomization_control",
            "library": "numpy",
            "algorithm": "default_rng_uniform_phase_v1",
            "seed_material": [seed],
            "derived_seed": seed,
            "purpose": "phase-randomized falsification features in each child run",
        }
    )
    if "label_permuted" in config["methods"]:
        seed = int(config["label_permutation_seed"])
        streams.append(
            {
                "name": "label_permutation_base",
                "library": "numpy",
                "algorithm": "seedsequence_run_plus_base_permutation_v1",
                "seed_material": [seed],
                "derived_seed": seed,
                "purpose": "training-label permutation control in each child run",
            }
        )
    if "row_shuffled" in config["methods"]:
        seed = int(config["row_shuffle_seed"])
        streams.append(
            {
                "name": "row_shuffle_base",
                "library": "numpy",
                "algorithm": "seedsequence_run_plus_base_permutation_v1",
                "seed_material": [seed],
                "derived_seed": seed,
                "purpose": "training-feature row-shuffle control in each child run",
            }
        )
    return streams


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
    phase_randomization_seed: int = 2027,
    label_permutation_seed: int = 2028,
    row_shuffle_seed: int = 2029,
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
    if any(isinstance(seed, bool) or not isinstance(seed, Integral) for seed in seeds):
        raise TypeError("seeds must be non-Boolean integers")
    seed_values = [int(seed) for seed in seeds]
    if any(not 0 <= seed < 2**63 for seed in seed_values):
        raise ValueError("seeds must be in [0, 2**63)")
    control_seeds = {
        "observation_seed": observation_seed,
        "phase_randomization_seed": phase_randomization_seed,
        "label_permutation_seed": label_permutation_seed,
        "row_shuffle_seed": row_shuffle_seed,
    }
    for name, seed in control_seeds.items():
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError(f"{name} must be a non-Boolean integer")
        if not 0 <= int(seed) < 2**63:
            raise ValueError(f"{name} must be in [0, 2**63)")
    input_hashes = [{"path": str(path), "sha256": sha256_file(path)} for path in paths]
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"capacity sweep output already exists: {root}")
    config = {
        "capacity_tiers": list(CAPACITY_TIERS),
        "methods": list(methods),
        "seeds": seed_values,
        "bootstrap": bootstrap,
        "holdout": holdout,
        "include_phase": include_phase,
        "observation": observation,
        "pilot_spacing": pilot_spacing,
        "pilot_snr_db": pilot_snr_db,
        "observation_seed": int(observation_seed),
        "phase_randomization_seed": int(phase_randomization_seed),
        "label_permutation_seed": int(label_permutation_seed),
        "row_shuffle_seed": int(row_shuffle_seed),
        "device": device,
        "coral_penalty": coral_penalty,
    }
    dependency_versions = versions()
    created_utc = datetime.now(timezone.utc).isoformat()
    source_revision = dependency_versions.get("git_sha", "unavailable") or "unavailable"
    run_replay = build_run_replay_provenance(
        created_utc=created_utc,
        command_argv=list(sys.argv) or ["capacity_sweep"],
        command_source="process_argv_snapshot",
        source_revision=source_revision,
        package_source_sha256=package_source_sha256(),
        config_sha256=canonical_hash(config),
        inputs=input_hashes,
        backend_identity={
            "requested_device": device,
            "platform": platform.platform(),
            "dependencies": dependency_versions,
        },
        rng_streams=_capacity_sweep_rng_streams(config),
    )
    manifest: dict[str, Any] = {
        "schema": "loslab.capacity_sweep.v2",
        "state": "running",
        "created_utc": created_utc,
        "config": config,
        "config_sha256": canonical_hash(config),
        "inputs": input_hashes,
        "run_replay": run_replay,
    }
    _validate_manifest(manifest)
    root.mkdir(parents=True)
    atomic_json(root / "manifest.json", manifest)
    running_manifest = manifest
    complete_marker_written = False
    child_runs: dict[str, dict[str, Any]] = {}
    try:
        for tier in CAPACITY_TIERS:
            child = root / tier
            run_experiment(
                paths,
                child,
                holdout=holdout,
                methods=methods,
                seeds=seed_values,
                bootstrap=bootstrap,
                include_phase=include_phase,
                observation=observation,
                pilot_spacing=pilot_spacing,
                pilot_snr_db=pilot_snr_db,
                observation_seed=observation_seed,
                phase_randomization_seed=phase_randomization_seed,
                label_permutation_seed=label_permutation_seed,
                row_shuffle_seed=row_shuffle_seed,
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
        for record in input_hashes:
            if sha256_file(record["path"]) != record["sha256"]:
                raise ValueError(f"capacity sweep input changed during execution: {record['path']}")
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
        manifest["report_sha256"] = sha256_file(root / "capacity_sweep.json")
        atomic_json(
            root / "COMPLETE",
            {
                "config_sha256": manifest["config_sha256"],
                "report_sha256": manifest["report_sha256"],
            },
            overwrite=False,
        )
        complete_marker_written = True
        manifest = mark_complete(manifest)
        _validate_manifest(manifest)
        atomic_json(root / "manifest.json", manifest)
        return report
    except Exception as error:
        if complete_marker_written:
            (root / "COMPLETE").unlink(missing_ok=True)
        manifest = mark_failed(
            running_manifest,
            error_type=type(error).__name__,
            error_message=str(error),
            stage="capacity_sweep",
        )
        manifest["error"] = f"{type(error).__name__}: {error}"
        _validate_manifest(manifest)
        atomic_json(root / "manifest.json", manifest)
        raise
