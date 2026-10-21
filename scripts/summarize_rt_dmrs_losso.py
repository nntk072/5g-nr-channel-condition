"""Validate and summarize a six-scene, precomputed-DMRS LOSO run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def summarize_loso_run(run_root: Path, source_evidence: Path | None = None) -> dict[str, Any]:
    """Verify all scene folds share inputs/config and summarize scene-macro AUROC."""
    fold_dirs = sorted(path for path in run_root.glob("holdout-*") if path.is_dir())
    if len(fold_dirs) < 3:
        raise ValueError("at least three held-out scene folds are required")

    folds: list[dict[str, Any]] = []
    expected_domains: set[str] | None = None
    expected_config: dict[str, Any] | None = None
    input_identity: dict[str, Any] | None = None
    per_method: dict[str, list[float]] = {}

    for fold_dir in fold_dirs:
        marker_path = fold_dir / "COMPLETE"
        manifest_path = fold_dir / "manifest.json"
        results_path = fold_dir / "results.json"
        summary_path = fold_dir / "evaluation" / "summary.json"
        if not all(
            path.is_file() for path in (marker_path, manifest_path, results_path, summary_path)
        ):
            raise ValueError(f"incomplete fold artifact: {fold_dir}")

        manifest = _read_json(manifest_path)
        config = manifest["config"]
        domains = set(manifest["domain_counts"])
        holdout = str(config["holdout"])
        if holdout not in domains:
            raise ValueError(f"invalid domain declaration in {manifest_path}")
        if expected_domains is None:
            expected_domains = domains
            expected_config = config
            per_method = {str(method): [] for method in config["methods"]}
        elif domains != expected_domains:
            raise ValueError("scene domains differ across folds")

        comparable_config = {key: value for key, value in config.items() if key not in {"holdout"}}
        assert expected_config is not None
        expected_comparable = {
            key: value for key, value in expected_config.items() if key != "holdout"
        }
        if comparable_config != expected_comparable:
            raise ValueError("evaluation configuration differs across folds")

        inputs = manifest.get("inputs", [])
        if len(inputs) != 1:
            raise ValueError("each fold must declare exactly one shared observation input")
        current_input = inputs[0]
        if input_identity is None:
            input_identity = current_input
        elif current_input.get("sha256") != input_identity.get("sha256"):
            raise ValueError("folds use different observation datasets")
        input_path = Path(str(current_input["path"]))
        if not input_path.is_absolute():
            input_path = input_path.resolve()
        if not input_path.is_file() or _sha256(input_path) != current_input.get("sha256"):
            raise ValueError(f"observation input is missing or has changed: {input_path}")

        rows = _read_json(results_path)
        expected_rows = len(config["methods"]) * len(config["seeds"])
        if not isinstance(rows, list) or len(rows) != expected_rows:
            raise ValueError(f"unexpected result row count in {results_path}")
        split_hash_by_seed: dict[int, str] = {}
        for row in rows:
            split = row["split"]
            if split["test_domains"] != [holdout]:
                raise ValueError(f"holdout leakage or mismatch in {results_path}")
            if holdout in split["train_domains"]:
                raise ValueError(f"held-out scene appears in training: {results_path}")
            if set(split["train_domains"]) | {holdout} != domains:
                raise ValueError(f"fold does not cover declared scene domains: {results_path}")
            seed = int(row["seed"])
            split_hash = str(split["indices_sha256"])
            previous_hash = split_hash_by_seed.setdefault(seed, split_hash)
            if previous_hash != split_hash:
                raise ValueError(f"methods do not share the same split for seed {seed}: {fold_dir}")

        fold_metrics: dict[str, dict[str, float]] = {}
        seed_metrics: dict[str, dict[str, float]] = {}
        for method in config["methods"]:
            method_rows = [row for row in rows if row["method"] == method]
            if sorted(row["seed"] for row in method_rows) != sorted(config["seeds"]):
                raise ValueError(f"missing or duplicate seed results for {method} in {fold_dir}")
            values = [float(row["metrics"]["auroc"]) for row in method_rows]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"nonfinite AUROC in {results_path}")
            fold_mean = sum(values) / len(values)
            fold_std = math.sqrt(sum((value - fold_mean) ** 2 for value in values) / len(values))
            fold_metrics[method] = {
                "mean_test_auroc_across_seeds": fold_mean,
                "std_test_auroc_across_seeds": fold_std,
                "seeds": len(values),
            }
            seed_metrics[method] = {
                str(row["seed"]): float(row["metrics"]["auroc"]) for row in method_rows
            }
            per_method[method].append(fold_mean)

        folds.append(
            {
                "held_out_scene": holdout,
                "n_test_samples": rows[0]["split"]["test_size"],
                "n_test_groups": rows[0]["split"]["test_groups"],
                "split_indices_sha256": rows[0]["split"]["indices_sha256"],
                "methods": fold_metrics,
                "seed_test_auroc": seed_metrics,
                "manifest_sha256": _sha256(manifest_path),
                "results_sha256": _sha256(results_path),
                "summary_sha256": _sha256(summary_path),
            }
        )

    assert expected_domains is not None and expected_config is not None
    if {fold["held_out_scene"] for fold in folds} != expected_domains:
        raise ValueError("the run must contain exactly one fold per scene domain")
    assert input_identity is not None
    source_record = None
    if source_evidence is not None:
        source_evidence = source_evidence.resolve()
        source_report = _read_json(source_evidence)
        if source_report["dataset"]["observation_dataset_sha256"] != input_identity.get("sha256"):
            raise ValueError("upstream source report references a different observation dataset")
        source_record = {
            "path": str(source_evidence),
            "sha256": _sha256(source_evidence),
            "source_scene_datasets": source_report["dataset"]["source_datasets"],
        }

    return {
        "schema": "loslab.dmrs_loso_summary.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "run_root": str(run_root),
        "input": input_identity,
        "source_data_evidence": source_record,
        "scene_count": len(expected_domains),
        "scene_domains": sorted(expected_domains),
        "methods": expected_config["methods"],
        "seeds": expected_config["seeds"],
        "observation": expected_config["observation"],
        "configuration": {
            key: value
            for key, value in expected_config.items()
            if key not in {"holdout", "methods", "seeds"}
        },
        "scene_macro_test_auroc": {
            method: {
                "mean_across_scenes": sum(values) / len(values),
                "std_across_scenes": math.sqrt(
                    sum((value - sum(values) / len(values)) ** 2 for value in values) / len(values)
                ),
                "minimum_scene_mean": min(values),
                "maximum_scene_mean": max(values),
                "scene_means": values,
            }
            for method, values in per_method.items()
        },
        "folds": folds,
        "reproduction": {
            "fold_template": (
                "python -m loslab experiment --input outputs/mvp-six-scene-dmrs-observations-20261022.npz "
                "--output <new-run-root>/holdout-<domain> --holdout <domain> "
                "--methods logreg rf raw_mlp --seeds 11 23 47 59 71 83 97 "
                "--bootstrap 200 --device cuda:0 --capacity small --observation precomputed"
            ),
            "summarize": (
                "python scripts/summarize_rt_dmrs_losso.py <run-root> <new-report.json> "
                "--source-evidence docs/research/results/"
                "mvp-six-scene-dmrs-holdout-etoile-seed20261022.json"
            ),
        },
        "interpretation": (
            "Descriptive scene-macro results over these selected synthetic scenes. "
            "Seed repetitions are not independent scenes; this is not measured-channel "
            "validation, material qualification, or evidence of general superiority."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-evidence", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    report = summarize_loso_run(args.run_root, args.source_evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["scene_macro_test_auroc"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
