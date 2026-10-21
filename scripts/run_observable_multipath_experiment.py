"""Train and compare two-path delay estimators on grouped synthetic pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from artifacts.provenance import package_source_sha256, sha256_file, versions
from evaluation.statistics import paired_group_regression_comparison
from evaluation.uncertainty_io import evaluate_prediction_file
from experiments.observable_multipath import (
    fit_two_path_decoder,
    generate_grouped_two_path_dataset,
    predict_decoder,
    profile_likelihood_delay,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    path.write_bytes(encoded)


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def run_experiment(
    output_dir: Path,
    *,
    num_groups: int = 800,
    samples_per_group: int = 4,
    n_tones: int = 64,
    seed: int = 20261031,
    epochs: int = 120,
    batch_size: int = 128,
    patience: int = 15,
    device: str = "auto",
    identifiability_weight: float = 0.1,
    bootstrap_draws: int = 2000,
) -> dict[str, Any]:
    """Run training, freeze predictions, and produce grouped calibration reports."""
    if bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be >= 100")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        return _run_experiment_in_staging(
            staging_dir,
            output_dir,
            num_groups=num_groups,
            samples_per_group=samples_per_group,
            n_tones=n_tones,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            device=device,
            identifiability_weight=identifiability_weight,
            bootstrap_draws=bootstrap_draws,
        )
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _run_experiment_in_staging(
    staging_dir: Path,
    output_dir: Path,
    *,
    num_groups: int,
    samples_per_group: int,
    n_tones: int,
    seed: int,
    epochs: int,
    batch_size: int,
    patience: int,
    device: str,
    identifiability_weight: float,
    bootstrap_draws: int,
) -> dict[str, Any]:
    """Create all files privately, then publish the complete run atomically."""
    dataset = generate_grouped_two_path_dataset(
        num_groups=num_groups,
        samples_per_group=samples_per_group,
        n_tones=n_tones,
        seed=seed,
    )
    model_off, training_off = fit_two_path_decoder(
        dataset,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        device=device,
        identifiability_weight=0.0,
    )
    model_on, training_on = fit_two_path_decoder(
        dataset,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        device=device,
        identifiability_weight=identifiability_weight,
    )
    actual_device = str(training_off["device"])
    eval_idx = np.concatenate((dataset.indices("calibration"), dataset.indices("test")), axis=0)
    neural_predictions = {
        "identifiability_off": predict_decoder(model_off, dataset, eval_idx, device=actual_device),
        "identifiability_on": predict_decoder(model_on, dataset, eval_idx, device=actual_device),
    }

    neural_rows: dict[str, list[dict[str, Any]]] = {
        "identifiability_off": [],
        "identifiability_on": [],
    }
    classical_rows: list[dict[str, Any]] = []
    for local_index, sample_index in enumerate(eval_idx):
        row: dict[str, Any] = {
            "sample_id": str(dataset.sample_id[sample_index]),
            "group_id": str(dataset.group_id[sample_index]),
            "split": str(dataset.split[sample_index]),
            "target": float(dataset.relative_delay_s[sample_index]),
            "identifiability_stratum": str(dataset.stratum[sample_index]),
            "condition": dataset.condition[int(str(dataset.group_id[sample_index]).split("-")[-1])],
        }
        for condition, (delay, scale, _reconstruction) in neural_predictions.items():
            neural_rows[condition].append(
                {
                    **row,
                    "prediction": float(delay[local_index]),
                    "uncertainty_scale": float(scale[local_index]),
                }
            )
        estimate, profile_scale = profile_likelihood_delay(
            dataset.observed_cfr[sample_index],
            dataset.pilot_mask[sample_index],
            dataset.frequencies_hz[sample_index],
            dataset.maximum_relative_delay_s,
            float(dataset.noise_variance[sample_index]),
        )
        classical_rows.append({**row, "prediction": estimate, "uncertainty_scale": profile_scale})

    common = {
        "schema": "loslab.scalar_regression_predictions.v1",
        "experiment_id": "E11-analytical-two-path-delay-pilot-v1",
        "target_name": "relative_two_path_delay",
        "target_unit": "seconds",
        "dataset_source_sha256": dataset.source_sha256,
        "seed": seed,
        "split_policy": "whole synthetic condition groups; 60/10/15/15 train/validation/calibration/test",
        "calibration_and_test_are_disjoint_groups": True,
        "research_status": "ANALYTICAL_SYNTHETIC_PILOT_NOT_EXTERNAL_EVIDENCE",
    }
    prediction_payloads: dict[str, dict[str, Any]] = {
        "identifiability_off": {
            **common,
            "method_name": "two_path_mlp_identifiability_off",
            "predictions": neural_rows["identifiability_off"],
        },
        "identifiability_on": {
            **common,
            "method_name": "two_path_mlp_identifiability_on",
            "identifiability_weight": identifiability_weight,
            "predictions": neural_rows["identifiability_on"],
        },
        "profile_likelihood": {
            **common,
            "method_name": "profile_likelihood_known_noise",
            "predictions": classical_rows,
        },
    }
    prediction_paths = {
        name: staging_dir / f"{name}_predictions.json" for name in prediction_payloads
    }
    method_rows = {
        "identifiability_off": neural_rows["identifiability_off"],
        "identifiability_on": neural_rows["identifiability_on"],
        "profile_likelihood": classical_rows,
    }
    test_rows = {
        method: [row for row in rows if row["split"] == "test"]
        for method, rows in method_rows.items()
    }
    reference_ids = [row["sample_id"] for row in test_rows["identifiability_off"]]
    if any([row["sample_id"] for row in rows] != reference_ids for rows in test_rows.values()):
        raise RuntimeError("paired test sample IDs differ across estimators")
    comparisons = []
    for control, treatment in (
        ("identifiability_off", "identifiability_on"),
        ("identifiability_off", "profile_likelihood"),
        ("identifiability_on", "profile_likelihood"),
    ):
        base = test_rows[control]
        treated = test_rows[treatment]
        if [row["group_id"] for row in base] != [row["group_id"] for row in treated]:
            raise RuntimeError("paired test group IDs differ across estimators")
        comparisons.append(
            paired_group_regression_comparison(
                [row["target"] for row in base],
                [row["prediction"] for row in base],
                [row["prediction"] for row in treated],
                [row["group_id"] for row in base],
                control=control,
                treatment=treatment,
                seed=seed,
                draws=bootstrap_draws,
            )
        )
    paired_path = staging_dir / "paired_test_comparisons.json"
    _write_json(
        paired_path,
        {
            "schema": "loslab.paired_group_regression_comparisons.v1",
            "split": "test",
            "comparisons": comparisons,
        },
    )
    for method, payload in prediction_payloads.items():
        path = prediction_paths[method]
        _write_json(path, payload)
        report = evaluate_prediction_file(path, seed=seed, bootstrap_draws=bootstrap_draws)
        report["prediction_file"] = str(output_dir / path.name)
        _write_json(staging_dir / f"{method}_uncertainty_report.json", report)

    model_paths = {
        "identifiability_off": staging_dir / "decoder_identifiability_off.npz",
        "identifiability_on": staging_dir / "decoder_identifiability_on.npz",
    }
    for condition, model in (
        ("identifiability_off", model_off),
        ("identifiability_on", model_on),
    ):
        state = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
        np.savez_compressed(model_paths[condition], **state)
    manifest = {
        "schema": "loslab.observable_multipath_experiment.v1",
        "status": "DESCRIPTIVE_ANALYTICAL_PILOT",
        "claim_status": "NO_SCIENTIFIC_ADVANTAGE_CLAIM",
        "dataset": {
            "groups": num_groups,
            "samples_per_group": samples_per_group,
            "samples": int(dataset.observed_cfr.shape[0]),
            "n_tones": n_tones,
            "maximum_relative_delay_s": dataset.maximum_relative_delay_s,
            "source_sha256": dataset.source_sha256,
            "group_counts": {
                split: int(np.unique(dataset.group_id[dataset.split == split]).size)
                for split in ("train", "validation", "calibration", "test")
            },
            "sample_counts": {
                split: int(dataset.indices(split).size)
                for split in ("train", "validation", "calibration", "test")
            },
            "group_conditions": list(dataset.condition),
        },
        "training": {
            "identifiability_off": training_off,
            "identifiability_on": training_on,
            "same_initialization_seed": seed,
            "identifiability_weight_on": identifiability_weight,
        },
        "model": {
            "architecture": "two_path_uncertainty_decoder",
            "hidden_dims": [256, 128],
            "checkpoint_format": "numpy_npz_state_dict_no_pickle",
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "epochs_requested": epochs,
            "batch_size": batch_size,
            "early_stopping_patience": patience,
        },
        "provenance": {
            "git_revision": _git_revision(Path(__file__).resolve().parents[1]),
            "package_source_sha256": package_source_sha256(),
            "runtime_versions": versions(),
            "seed": seed,
            "source_sha256": sha256_file(Path(__file__)),
        },
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (
                *prediction_paths.values(),
                paired_path,
                *(
                    staging_dir / f"{method}_uncertainty_report.json"
                    for method in prediction_payloads
                ),
                *model_paths.values(),
            )
        },
        "limitations": [
            "Synthetic fixed-order two-path CFR with known receiver noise variance.",
            "Random condition groups come from one declared mixture; no scene or domain transfer is tested.",
            "The profile-likelihood baseline knows model order and noise variance.",
            "The identifiability-on/off pair is a single CRLB-scale-floor ablation, not a validated novel method.",
            "Learned uncertainty is only a scale; calibration reports are descriptive on this synthetic split.",
            "No diffuse paths, channel-estimation errors, synchronization impairments, or measured data.",
        ],
    }
    manifest_path = staging_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)
    staging_dir.rename(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--groups", type=int, default=800)
    parser.add_argument("--samples-per-group", type=int, default=4)
    parser.add_argument("--n-tones", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--identifiability-weight", type=float, default=0.1)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing output directory: {args.output_dir}")
    try:
        manifest = run_experiment(
            args.output_dir,
            num_groups=args.groups,
            samples_per_group=args.samples_per_group,
            n_tones=args.n_tones,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            device=args.device,
            identifiability_weight=args.identifiability_weight,
            bootstrap_draws=args.bootstrap_draws,
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    manifest_path = args.output_dir / "run_manifest.json"
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "training_device": manifest["training"]["identifiability_off"]["device"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
