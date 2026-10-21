"""Matched comparisons for reproducible sample-domain link curves."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from artifacts.provenance import package_source_sha256, sha256_file, versions

from evaluation.statistics import paired_binary_error_comparison


def load_curve(path: str | Path) -> dict[str, Any]:
    """Load a supported, explicitly simulation-only sample-domain curve."""
    path = Path(path)
    result = json.loads(path.read_text(encoding="utf8"))
    if result.get("schema") != "loslab.tdl_pusch_time_bler_curve.v1":
        raise ValueError(f"unsupported sample-domain curve schema in {path}")
    if result.get("evidence_grade") != "SIMULATION_ONLY_UNVALIDATED":
        raise ValueError(f"unexpected evidence grade in {path}")
    return result


def compare_curves(control_path: str | Path, treatment_path: str | Path) -> dict[str, Any]:
    """Validate a matched mobility pair and compute paired codeword outcomes."""
    control_path = Path(control_path)
    treatment_path = Path(treatment_path)
    control = load_curve(control_path)
    treatment = load_curve(treatment_path)
    control_config = control["configuration"]
    treatment_config = treatment["configuration"]
    varying = {"speed_mps", "matched_realization_checks"}
    for key in set(control_config) | set(treatment_config):
        if key not in varying and control_config.get(key) != treatment_config.get(key):
            raise ValueError(f"curve configuration differs unexpectedly at {key}")
    if control_config["speed_mps"] == treatment_config["speed_mps"]:
        raise ValueError("control and treatment must have different mobility speeds")
    control_checks = control_config["matched_realization_checks"]
    treatment_checks = treatment_config["matched_realization_checks"]
    if len(control_checks) != len(treatment_checks):
        raise ValueError("matched realization batch counts differ")
    for left, right in zip(control_checks, treatment_checks, strict=True):
        if left["seed"] != right["seed"]:
            raise ValueError("batch seeds differ across conditions")
        if left.get("noise_seed") != right.get("noise_seed"):
            raise ValueError(f"noise seeds differ for matched batch {left['seed']}")
        if left["transmit_bits_sha256"] != right["transmit_bits_sha256"]:
            raise ValueError(f"transmitted bits differ for matched seed {left['seed']}")

    control_points = control["points"]
    treatment_points = treatment["points"]
    if len(control_points) != len(treatment_points):
        raise ValueError("SNR point counts differ")
    comparisons: list[dict[str, Any]] = []
    for left, right in zip(control_points, treatment_points, strict=True):
        if left["snr_db"] != right["snr_db"] or left["frames"] != right["frames"]:
            raise ValueError("SNR values or codeword counts differ")
        comparisons.append(
            {
                "snr_db": left["snr_db"],
                **paired_binary_error_comparison(
                    left["frame_error_mask"],
                    right["frame_error_mask"],
                    seed=1,
                    draws=20_000,
                ),
            }
        )
    ordered = sorted(
        range(len(comparisons)),
        key=lambda index: comparisons[index]["mcnemar_exact_two_sided_p_value"],
    )
    adjusted_max = 0.0
    point_count = len(comparisons)
    for rank, index in enumerate(ordered):
        raw_p = comparisons[index]["mcnemar_exact_two_sided_p_value"]
        adjusted_max = max(adjusted_max, (point_count - rank) * raw_p)
        comparisons[index]["mcnemar_exact_two_sided_p_value_holm"] = min(1.0, adjusted_max)

    return {
        "schema": "loslab.paired_tdl_pusch_time_bler.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "control": {
            "path": str(control_path),
            "sha256": sha256_file(control_path),
            "speed_mps": control_config["speed_mps"],
        },
        "treatment": {
            "path": str(treatment_path),
            "sha256": sha256_file(treatment_path),
            "speed_mps": treatment_config["speed_mps"],
        },
        "paired_design": {
            "same_configuration_except": "speed_mps",
            "same_transmitted_bits_by_seed": True,
            "same_batch_seeds_and_noise_seeds": True,
            "statistic": "paired codeword bootstrap CI and Holm-adjusted exact McNemar tests",
            "versions": versions(),
            "package_source_sha256": package_source_sha256(),
        },
        "points": comparisons,
        "limitations": [
            "Inference is conditional on one standardized TDL-A simulation configuration.",
            "Codeword-paired uncertainty does not represent independent-scene uncertainty.",
            "The decoder and synchronization configuration are fixed; the result is not standards certification.",
        ],
    }
