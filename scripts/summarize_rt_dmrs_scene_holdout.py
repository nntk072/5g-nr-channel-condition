"""Create a hash-indexed compact report from six-scene PUSCH/DMRS holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from artifacts.provenance import atomic_json, package_source_sha256, sha256_file, versions
from datasets.channel import ChannelDataset
from integrations.sionna.rt_dmrs import resample_centered_cfr

SCENES = (
    "box_one_screen",
    "simple_street_canyon",
    "etoile",
    "urban_00041",
    "urban_00042",
    "urban_00043",
)
METHODS = ("rf", "mlp", "logreg")


def _fold(root: Path, scene: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    fold_root = root / f"holdout-rt_{scene}"
    manifest_path = fold_root / "manifest.json"
    summary_path = fold_root / "summary.json"
    results_path = fold_root / "results.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    if manifest.get("observation_semantics") != "Sionna PUSCH DMRS LS channel estimate":
        raise ValueError(f"fold {scene} is not a precomputed PUSCH DMRS observation run")
    split_hashes = {
        f"{row['seed']}:{row['method']}": row["split"]["indices_sha256"] for row in rows
    }
    expected = {f"{seed}:{method}" for seed in manifest["config"]["seeds"] for method in METHODS}
    if set(split_hashes) != expected:
        raise ValueError(f"fold {scene} has incomplete or unexpected seed/method results")
    return (
        {
            "domain": f"rt_{scene}",
            "sample_count": int(manifest["n_samples"]),
            "test_group_count": int(rows[0]["split"]["test_groups"]),
            "aggregate_across_fixed_splits": summary["aggregate_across_fixed_splits"],
            "result_sha256": sha256_file(results_path),
            "summary_sha256": sha256_file(summary_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        manifest,
        split_hashes,
    )


def _channel_agreement(dataset: ChannelDataset) -> dict[str, float]:
    artifacts = dataset.generation_report["input_artifacts"]
    source_data = ChannelDataset.concatenate(
        [ChannelDataset.load(item["path"]) for item in artifacts]
    )
    grid = dataset.generation_report["input_grid"]
    truth = resample_centered_cfr(
        source_data.h,
        source_spacing_hz=float(grid["spacing_hz"]),
        target_subcarriers=dataset.h.shape[-1],
        target_spacing_hz=float(
            dataset.generation_report["receiver_metadata"]["subcarrier_spacing_khz"]
        )
        * 1_000.0,
    )
    truth = truth / np.sqrt(np.mean(np.abs(truth) ** 2, axis=-1, keepdims=True))
    estimate = dataset.h[:, 0, :]
    estimate = estimate / np.sqrt(np.mean(np.abs(estimate) ** 2, axis=-1, keepdims=True))
    correlation = np.abs(np.sum(np.conj(truth) * estimate, axis=-1)) / truth.shape[-1]
    nmse = np.mean(np.abs(truth - estimate) ** 2, axis=-1)
    return {
        "median_normalized_complex_channel_correlation": float(np.median(correlation)),
        "median_normalized_complex_channel_nmse": float(np.median(nmse)),
    }


def summarize(
    dataset_path: Path,
    dmrs_results_root: Path,
    output_path: Path,
    generic_reference_root: Path | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_path}")
    dataset = ChannelDataset.load(dataset_path)
    if dataset.generation_report.get("schema") != "loslab.rt_dmrs_observation.v1":
        raise ValueError("dataset is not an RT-to-DMRS observation artifact")

    folds: dict[str, Any] = {}
    split_hash_sets: list[dict[str, str]] = []
    baseline_folds: dict[str, Any] = {}
    baseline_hash_sets: list[dict[str, str]] = []
    for scene in SCENES:
        fold, manifest, split_hashes = _fold(dmrs_results_root, scene)
        folds[scene] = fold
        split_hash_sets.append(split_hashes)
        if generic_reference_root is not None:
            ref_root = generic_reference_root / f"holdout-rt_{scene}"
            ref_manifest = json.loads((ref_root / "manifest.json").read_text(encoding="utf-8"))
            ref_summary = json.loads((ref_root / "summary.json").read_text(encoding="utf-8"))
            ref_results = json.loads((ref_root / "results.json").read_text(encoding="utf-8"))
            baseline_folds[scene] = {
                "observation_semantics": ref_manifest["observation_semantics"],
                "aggregate_across_fixed_splits": ref_summary["aggregate_across_fixed_splits"],
                "result_sha256": sha256_file(ref_root / "results.json"),
                "summary_sha256": sha256_file(ref_root / "summary.json"),
                "manifest_sha256": sha256_file(ref_root / "manifest.json"),
            }
            baseline_hash_sets.append(
                {
                    f"{row['seed']}:{row['method']}": row["split"]["indices_sha256"]
                    for row in ref_results
                }
            )

    macro = {
        method: float(
            np.mean(
                [
                    folds[scene]["aggregate_across_fixed_splits"][method]["mean_test_auroc"]
                    for scene in SCENES
                ]
            )
        )
        for method in METHODS
    }
    reference_comparison: dict[str, Any] | None = None
    if generic_reference_root is not None:
        same_splits = split_hash_sets == baseline_hash_sets
        reference_macro = {
            method: float(
                np.mean(
                    [
                        baseline_folds[scene]["aggregate_across_fixed_splits"][method][
                            "mean_test_auroc"
                        ]
                        for scene in SCENES
                    ]
                )
            )
            for method in METHODS
        }
        reference_comparison = {
            "reference_observation_semantics": "generic noisy comb pilot estimate, NOT 3GPP DMRS",
            "reference_folds": baseline_folds,
            "macro_mean_test_auroc": reference_macro,
            "dmrs_minus_comb_macro_mean": {
                method: macro[method] - reference_macro[method] for method in METHODS
            },
            "split_indices_identical_for_every_seed_and_method": same_splits,
            "interpretation": "Descriptive paired-observation comparison on identical held-out indices; scene folds are not independent replications.",
        }

    report: dict[str, Any] = {
        "schema": "loslab.rt_dmrs_scene_holdout.v1",
        "evidence_grade": "SIX_SCENE_SIMULATION_DMRS_RECEIVER_STRESS_TEST",
        "source_revision": versions().get("git_sha"),
        "package_source_sha256": package_source_sha256(),
        "summary_script_sha256": sha256_file(__file__),
        "observation_dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "sample_count": len(dataset.y),
            "observation_semantics": dataset.generation_report["observation_semantics"],
            "generation_report": dataset.generation_report,
            "channel_estimate_agreement_with_resampled_rt_cfr": _channel_agreement(dataset),
        },
        "evaluation_protocol": {
            "holdout_scenes": [f"rt_{scene}" for scene in SCENES],
            "methods": list(METHODS),
            "seeds": list(
                json.loads(
                    (dmrs_results_root / f"holdout-rt_{SCENES[0]}" / "manifest.json").read_text()
                )["config"]["seeds"]
            ),
            "observation": "precomputed Sionna PUSCH DMRS LS channel estimate",
            "features": "normalized CFR features from DMRS estimates; path loss excluded",
            "macro_mean_test_auroc": macro,
        },
        "folds": folds,
        "generic_comb_pilot_comparison": reference_comparison,
        "limitations": [
            "All channels are Sionna RT simulation with geometric direct-path-oracle labels and simplified or built-in scenes; no measured channels are used.",
            "The PUSCH observation uses a static frequency-domain channel, ideal synchronization, unit per-link channel-power normalization, and fixed nominal SNR; CFO, timing error, CP overrun, mobility, and path loss are absent.",
            "The generic-comb and DMRS comparison uses identical train/validation/test indices when the report's split hash check passes, but training still varies by observation representation.",
            "Six leave-one-scene-out folds share training scenes and are not six independent replications; metrics are descriptive and establish no model advantage.",
            "Sionna RT used the LLVM CPU backend in the source-data generation; the PUSCH/DMRS receiver ran on CUDA.",
        ],
    }
    atomic_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("dmrs_results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--generic-reference", type=Path)
    args = parser.parse_args()
    try:
        report = summarize(
            args.dataset,
            args.dmrs_results,
            args.output,
            generic_reference_root=args.generic_reference,
        )
    except (FileNotFoundError, FileExistsError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved six-scene DMRS holdout report to {args.output}")
    print(json.dumps(report["evaluation_protocol"]["macro_mean_test_auroc"], indent=2))


if __name__ == "__main__":
    main()
