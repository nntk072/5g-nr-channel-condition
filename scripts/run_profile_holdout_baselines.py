"""Run profile-disjoint classical baselines on a manifest-backed proxy dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from artifacts.provenance import atomic_json, package_source_sha256, sha256_file, versions
from datasets.channel import ChannelDataset
from experiments.profile_holdout import BASELINE_CONFIGURATION, run_profile_holdout


def run(
    dataset_path: Path,
    output_path: Path,
    *,
    methods: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    prediction_path = output_path.with_suffix(output_path.suffix + ".predictions.npz")
    if output_path.exists() or prediction_path.exists():
        raise FileExistsError(f"refusing to overwrite result artifacts at {output_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    sidecar = dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"dataset provenance manifest is required: {sidecar}")
    import json

    source_manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if source_manifest.get("dataset_sha256") != sha256_file(dataset_path):
        raise ValueError("dataset hash does not match its provenance manifest")
    if source_manifest.get("evidence_grade") != "SIMULATED_3GPP_PROFILE_PROXY_NOT_GEOMETRIC_LOS":
        raise ValueError("input must be declared as a standardized profile-proxy dataset")

    data = ChannelDataset.load(dataset_path)
    result = run_profile_holdout(
        data,
        methods=methods,
        seed=seed,
        progress=lambda message: print(message, flush=True),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_arrays: dict[str, np.ndarray] = {
        "label": np.asarray(data.y, dtype=np.uint8),
        "profile": np.asarray([str(row["profile"]).upper() for row in data.metadata]),
        "family": np.asarray([str(row["family"]).lower() for row in data.metadata]),
        "group": np.asarray(data.group, dtype=str),
    }
    for method in methods:
        prediction_arrays[f"score_{method}"] = result["predictions"][method]
        prediction_arrays[f"threshold_{method}"] = result["thresholds"][method]
    np.savez_compressed(prediction_path, **prediction_arrays)  # type: ignore[arg-type]

    versions_used = versions()
    report = {
        "schema": result["schema"],
        "evidence_grade": "SIMULATED_PROFILE_PROXY_DIAGNOSTIC_NOT_GEOMETRIC_LOS",
        "configuration": {
            "seed": seed,
            "methods": list(methods),
            "baseline_configuration": {
                method: BASELINE_CONFIGURATION[method] for method in methods
            },
            "validation_fraction_of_training_groups": result[
                "validation_fraction_of_training_groups"
            ],
            "feature_semantics": result["feature_semantics"],
            "threshold_selection": "Youden J on group-disjoint training validation only",
            "mlp_early_stopping": result["mlp_early_stopping"],
        },
        "inputs": {
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "source_manifest_path": str(sidecar),
            "source_manifest_sha256": sha256_file(sidecar),
            "source_revision": source_manifest.get("source_revision"),
        },
        "source_revision": versions_used.get("git_sha"),
        "package_source_sha256": package_source_sha256(),
        "runner_sha256": sha256_file(__file__),
        "versions": versions_used,
        "sample_count": len(data.y),
        "profile_holdout_folds": result["folds"],
        "summary": result["summary"],
        "predictions_path": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "limitations": [
            "Each outer fold holds out one profile and therefore contains a single proxy class.",
            "Report within-profile recall and macro averages; pooled AUROC is undefined and omitted.",
            "Only five profile groups exist, so no profile-level confidence interval is claimed.",
            "Profile-proxy labels are not geometric LOS/NLOS, measured truth, or physical-domain transfer.",
            "Validation thresholds are selected from the other profiles; no held-out profile labels tune them.",
        ],
    }
    atomic_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--methods", nargs="+", default=["logreg", "mlp", "rf"])
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    try:
        report = run(args.dataset, args.output, methods=args.methods, seed=args.seed)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved profile-holdout baseline summary to {args.output}")
    print(f"Predictions: {report['predictions_path']}")


if __name__ == "__main__":
    main()
