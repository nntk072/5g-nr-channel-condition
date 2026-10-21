"""Verify a completed analytical pilot and write a compact evidence summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def summarize_run(run_dir: Path, output: Path) -> dict[str, Any]:
    """Verify every run artifact hash and publish a compact, no-claim summary."""
    run_dir = run_dir.resolve()
    output = output.resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest not found: {manifest_path}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing summary: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "loslab.observable_multipath_experiment.v1":
        raise ValueError("unsupported experiment manifest schema")
    verified: dict[str, dict[str, Any]] = {}
    for name, descriptor in manifest.get("artifacts", {}).items():
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"run artifact is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != descriptor.get("sha256") or path.stat().st_size != descriptor.get("bytes"):
            raise ValueError(f"run artifact failed integrity check: {path}")
        verified[name] = {"sha256": digest, "bytes": path.stat().st_size}

    method_metrics: dict[str, dict[str, Any]] = {}
    for method in ("identifiability_off", "identifiability_on", "profile_likelihood"):
        report_path = run_dir / f"{method}_uncertainty_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "DESCRIPTIVE_EVALUATION_NOT_A_RESEARCH_CLAIM":
            raise ValueError(f"unexpected report status in {report_path}")
        metrics = report["metrics"]
        interval = metrics["group_conformal_intervals"]["0.95"]
        method_metrics[method] = {
            "group_macro_rmse_s": metrics["error_metrics"]["group_macro_rmse"],
            "group_macro_rmse_ci95_s": metrics["error_metrics"]["group_macro_rmse_ci95"],
            "test_group_count": metrics["test_group_count"],
            "test_sample_count": metrics["test_sample_count"],
            "group_conformal_95": {
                "finite_sample_unbounded": interval["finite_sample_unbounded"],
                "test_group_simultaneous_coverage": interval["test_group_simultaneous_coverage"],
                "test_sample_coverage": interval["test_sample_coverage"],
                "mean_interval_width_s": interval["mean_interval_width"],
            },
        }

    paired_path = run_dir / "paired_test_comparisons.json"
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    if paired.get("schema") != "loslab.paired_group_regression_comparisons.v1":
        raise ValueError("unsupported paired comparison schema")
    summary = {
        "schema": "loslab.observable_multipath_summary.v1",
        "status": "DESCRIPTIVE_ANALYTICAL_PILOT_NO_SCIENTIFIC_CLAIM",
        "run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source": manifest["provenance"],
        "dataset": {
            key: manifest["dataset"][key]
            for key in (
                "groups",
                "samples_per_group",
                "samples",
                "n_tones",
                "maximum_relative_delay_s",
                "source_sha256",
                "group_counts",
                "sample_counts",
            )
        },
        "training": {
            "device": manifest["training"]["identifiability_off"]["device"],
            "identifiability_off_best_epoch": manifest["training"]["identifiability_off"][
                "best_epoch"
            ],
            "identifiability_on_best_epoch": manifest["training"]["identifiability_on"][
                "best_epoch"
            ],
            "identifiability_weight_on": manifest["training"]["identifiability_weight_on"],
            "same_initialization_seed": manifest["training"]["same_initialization_seed"],
        },
        "model": manifest["model"],
        "methods": method_metrics,
        "paired_test_comparisons": paired["comparisons"],
        "verified_artifacts": verified,
        "limitations": manifest["limitations"],
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        summarize_run(args.run_dir, args.output)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "summary": str(args.output),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
