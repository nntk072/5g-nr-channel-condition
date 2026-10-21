"""Audit and aggregate fixed-split neural capacity sweeps across held-out RT scenes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Sequence

from artifacts.provenance import atomic_json, package_source_sha256, sha256_file, versions

CAPACITY_TIERS = ("small", "medium", "large")
DEFAULT_DOMAINS = (
    "rt_box_one_screen",
    "rt_simple_street_canyon",
    "rt_etoile",
    "rt_urban_00041",
    "rt_urban_00042",
    "rt_urban_00043",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_fold(root: Path, domain: str) -> dict[str, Any]:
    fold = root / domain
    manifest_path = fold / "manifest.json"
    report_path = fold / "capacity_sweep.json"
    manifest = _json(manifest_path)
    report = _json(report_path)
    if manifest.get("schema") != "loslab.capacity_sweep.v1" or manifest.get("state") != "complete":
        raise ValueError(f"incomplete or unsupported capacity sweep: {fold}")
    if sha256_file(report_path) != manifest.get("report_sha256"):
        raise ValueError(f"capacity report hash mismatch: {fold}")
    config = manifest.get("config", {})
    if config.get("holdout") != domain:
        raise ValueError(f"holdout mismatch for {domain}")
    if tuple(config.get("capacity_tiers", ())) != CAPACITY_TIERS:
        raise ValueError(f"capacity tiers mismatch for {domain}")
    if config.get("observation") != "precomputed":
        raise ValueError(f"expected precomputed common observations for {domain}")
    inputs = manifest.get("inputs", [])
    if len(inputs) != 1:
        raise ValueError(f"expected one locked observation input for {domain}")
    input_path = Path(inputs[0]["path"])
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    if not input_path.is_file() or sha256_file(input_path) != inputs[0].get("sha256"):
        raise ValueError(f"locked observation artifact is missing or changed for {domain}")
    if report.get("config_sha256") != manifest.get("config_sha256"):
        raise ValueError(f"capacity config hash mismatch for {domain}")

    methods = tuple(config.get("methods", ()))
    seeds = tuple(config.get("seeds", ()))
    if not methods or len(methods) != len(set(methods)):
        raise ValueError(f"invalid method set for {domain}")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"invalid seed set for {domain}")
    points = {(point["capacity"], point["method"]): point for point in report.get("points", [])}
    expected_points = {(tier, method) for tier in CAPACITY_TIERS for method in methods}
    if report.get("split_identity_count") != len(seeds) * len(methods):
        raise ValueError(f"split identity count mismatch for {domain}")
    if set(points) != expected_points:
        raise ValueError(f"missing or duplicate capacity/method points for {domain}")

    split_by_key: dict[tuple[str, int, str], str] = {}
    scene_points: dict[str, dict[str, Any]] = {}
    for tier in CAPACITY_TIERS:
        child = fold / tier
        child_manifest_path = child / "manifest.json"
        child_manifest = _json(child_manifest_path)
        if child_manifest.get("state") != "complete":
            raise ValueError(f"incomplete {tier} child run for {domain}")
        if child_manifest.get("config", {}).get("capacity") != tier:
            raise ValueError(f"child capacity mismatch for {domain}/{tier}")
        if child_manifest.get("config", {}).get("holdout") != domain:
            raise ValueError(f"child holdout mismatch for {domain}/{tier}")
        if child_manifest.get("inputs") != inputs:
            raise ValueError(f"child input mismatch for {domain}/{tier}")
        complete_path = child / "COMPLETE"
        if not complete_path.is_file():
            raise ValueError(f"child completion marker missing for {domain}/{tier}")
        complete = _json(complete_path)
        if complete.get("config_identity") != child_manifest.get("config_identity"):
            raise ValueError(f"child completion marker does not match manifest for {domain}/{tier}")
        if report.get("child_runs", {}).get(tier, {}).get("manifest_sha256") != sha256_file(
            child_manifest_path
        ):
            raise ValueError(f"child manifest hash mismatch for {domain}/{tier}")

        rows = _json(child / "results.json")
        by_key = {(int(row["seed"]), str(row["method"])): row for row in rows}
        expected_keys = {(seed, method) for seed in seeds for method in methods}
        if len(by_key) != len(rows) or set(by_key) != expected_keys:
            raise ValueError(f"seed/method coverage mismatch for {domain}/{tier}")
        split_per_seed: dict[int, str] = {}
        for (seed, method), row in by_key.items():
            split = row.get("split", {})
            if split.get("test_domains") != [domain]:
                raise ValueError(f"test domain is not exclusively held out for {domain}")
            if domain in split.get("train_domains", []):
                raise ValueError(f"held-out scene appears in training for {domain}")
            split_hash = str(split.get("indices_sha256", ""))
            if not split_hash:
                raise ValueError(f"missing split identity for {domain}/{tier}/{seed}/{method}")
            if seed in split_per_seed and split_per_seed[seed] != split_hash:
                raise ValueError(f"methods use different splits for {domain}/{tier}/{seed}")
            split_per_seed[seed] = split_hash
            split_by_key[(tier, seed, method)] = split_hash
        for method in methods:
            values = [float(by_key[(seed, method)]["metrics"]["auroc"]) for seed in seeds]
            point = points[(tier, method)]
            if point.get("seeds") != list(seeds):
                raise ValueError(
                    f"reported seeds do not match child results for {domain}/{tier}/{method}"
                )
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
                raise ValueError(f"invalid AUROC in child results for {domain}/{tier}/{method}")
            if not math.isclose(
                float(point["mean_test_auroc"]), fmean(values), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    f"reported AUROC does not match child results for {domain}/{tier}/{method}"
                )
            scene_points[f"{tier}:{method}"] = {
                "capacity": tier,
                "method": method,
                "seed_aurocs": values,
                "mean_test_auroc": fmean(values),
                "std_test_auroc_across_seeds": stdev(values) if len(values) > 1 else None,
                "mean_training_seconds": float(point["mean_training_seconds"]),
                "mean_learned_parameter_count": point["mean_learned_parameter_count"],
                "mean_peak_cuda_memory_bytes": _mean_peak_memory(by_key, seeds, method),
                "test_samples": int(by_key[(seeds[0], method)]["split"]["test_size"]),
                "test_groups": int(by_key[(seeds[0], method)]["split"]["test_groups"]),
            }
    for seed in seeds:
        for method in methods:
            split_values = {split_by_key[(tier, seed, method)] for tier in CAPACITY_TIERS}
            if len(split_values) != 1:
                raise ValueError(
                    f"split identity changes across capacity for {domain}/{seed}/{method}"
                )
    return {
        "domain": domain,
        "input": inputs[0],
        "config": config,
        "manifest_sha256": sha256_file(manifest_path),
        "report_sha256": sha256_file(report_path),
        "split_identity_count": len(seeds) * len(methods),
        "points": list(scene_points.values()),
    }


def _mean_peak_memory(
    rows: dict[tuple[int, str], dict[str, Any]], seeds: Sequence[int], method: str
) -> float | None:
    values = [rows[(seed, method)]["compute"].get("peak_cuda_memory_bytes") for seed in seeds]
    finite = [float(value) for value in values if value is not None]
    return fmean(finite) if finite else None


def summarize_capacity_loso(
    input_root: Path, domains: Sequence[str] = DEFAULT_DOMAINS
) -> dict[str, Any]:
    """Verify six capacity sweeps and report scene-macro, paired summaries."""
    if not domains or len(domains) != len(set(domains)):
        raise ValueError("domains must be nonempty and unique")
    folds = [_verify_fold(input_root, domain) for domain in domains]
    reference = folds[0]
    for fold in folds[1:]:
        if fold["input"] != reference["input"]:
            raise ValueError("capacity folds do not share the same observation artifact hash")
        fold_config = {key: value for key, value in fold["config"].items() if key != "holdout"}
        reference_config = {
            key: value for key, value in reference["config"].items() if key != "holdout"
        }
        if fold_config != reference_config:
            raise ValueError(f"capacity fold configuration differs for {fold['domain']}")
    methods = reference["config"]["methods"]
    output_points = []
    for tier in CAPACITY_TIERS:
        for method in methods:
            by_scene = []
            for fold in folds:
                row = next(
                    point
                    for point in fold["points"]
                    if point["capacity"] == tier and point["method"] == method
                )
                by_scene.append({"domain": fold["domain"], **row})
            means = [row["mean_test_auroc"] for row in by_scene]
            output_points.append(
                {
                    "capacity": tier,
                    "method": method,
                    "scene_count": len(by_scene),
                    "scene_macro_mean_test_auroc": fmean(means),
                    "scene_sd_of_mean_test_auroc": stdev(means) if len(means) > 1 else None,
                    "scene_min_test_auroc": min(means),
                    "scene_max_test_auroc": max(means),
                    "mean_training_seconds_across_scenes": fmean(
                        row["mean_training_seconds"] for row in by_scene
                    ),
                    "mean_learned_parameter_count": fmean(
                        row["mean_learned_parameter_count"]
                        for row in by_scene
                        if row["mean_learned_parameter_count"] is not None
                    )
                    if any(row["mean_learned_parameter_count"] is not None for row in by_scene)
                    else None,
                    "mean_peak_cuda_memory_bytes": fmean(
                        row["mean_peak_cuda_memory_bytes"]
                        for row in by_scene
                        if row["mean_peak_cuda_memory_bytes"] is not None
                    )
                    if any(row["mean_peak_cuda_memory_bytes"] is not None for row in by_scene)
                    else None,
                    "scene_results": by_scene,
                }
            )
    paired = []
    for method in methods:
        small = {
            fold["domain"]: row["mean_test_auroc"]
            for fold in folds
            for row in fold["points"]
            if row["capacity"] == "small" and row["method"] == method
        }
        for tier in CAPACITY_TIERS[1:]:
            other = {
                fold["domain"]: row["mean_test_auroc"]
                for fold in folds
                for row in fold["points"]
                if row["capacity"] == tier and row["method"] == method
            }
            differences = [other[domain] - small[domain] for domain in domains]
            paired.append(
                {
                    "method": method,
                    "capacity_vs_small": tier,
                    "scene_macro_paired_auroc_delta": fmean(differences),
                    "scenes_improved": sum(delta > 1e-12 for delta in differences),
                    "scenes_tied": sum(abs(delta) <= 1e-12 for delta in differences),
                    "scenes_decreased": sum(delta < -1e-12 for delta in differences),
                    "scene_deltas": dict(zip(domains, differences, strict=True)),
                }
            )
    shared_config = dict(reference["config"])
    shared_config.pop("holdout", None)
    return {
        "schema": "loslab.capacity_loso_suite.v1",
        "source_revision": versions().get("git_sha"),
        "script_sha256": sha256_file(Path(__file__)),
        "package_source_sha256": package_source_sha256(),
        "runtime": versions(),
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED_EXPLORATORY",
        "input": reference["input"],
        "configuration": shared_config,
        "capacity_tiers": list(CAPACITY_TIERS),
        "domains": list(domains),
        "scene_count": len(domains),
        "seed_count_per_scene": len(reference["config"]["seeds"]),
        "methods": methods,
        "fit_count": len(domains)
        * len(reference["config"]["seeds"])
        * len(methods)
        * len(CAPACITY_TIERS),
        "folds": [
            {
                key: fold[key]
                for key in ("domain", "manifest_sha256", "report_sha256", "split_identity_count")
            }
            for fold in folds
        ],
        "scene_macro_results": output_points,
        "paired_capacity_deltas_vs_small": paired,
        "limitations": [
            "Six selected synthetic scene families are the aggregation units; seed repeats are not additional environments.",
            "Only 40 links are tested per held-out scene, with 4–20 spatial groups; estimates are exploratory and imprecise.",
            "Scene geometry/materials are not independently qualified and no measured-radio data are used.",
            "Training budgets, hyperparameter search, and effective compute are not equalized across architectures or capacity tiers.",
            "Per-scene block-bootstrap intervals are retained in child results but are not pooled into a scene-level inferential interval.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing report: {args.output}")
    report = summarize_capacity_loso(args.input_root, args.domains)
    atomic_json(args.output, report)
    print(f"Saved audited {report['fit_count']}-fit capacity summary to {args.output}")


if __name__ == "__main__":
    main()
