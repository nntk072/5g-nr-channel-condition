"""Reaudit path matching sensitivity without assigning identities to zero-order duplicates."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

import numpy as np
from artifacts.provenance import atomic_json, package_source_sha256, sha256_file, versions

_ATTRIBUTION = runpy.run_path(str(Path(__file__).with_name("audit_rt_path_budget_attribution.py")))
_path_signature = _ATTRIBUTION["_path_signature"]
analyze_pair = _ATTRIBUTION["analyze_pair"]


def _contribution(path: dict[str, Any]) -> np.ndarray:
    return np.asarray(path["cfr_contribution_real"], dtype=np.float64) + 1j * np.asarray(
        path["cfr_contribution_imag"], dtype=np.float64
    )


def _sum_contributions(paths: list[dict[str, Any]], width: int) -> np.ndarray:
    return sum((_contribution(path) for path in paths), start=np.zeros(width, np.complex128))


def _row_for_paths(row: dict[str, Any], paths: list[dict[str, Any]], width: int) -> dict[str, Any]:
    cfr = _sum_contributions(paths, width)
    return {
        **row,
        "path_diagnostics": paths,
        "valid_path_count": len(paths),
        "cfr_vector_real": cfr.real.tolist(),
        "cfr_vector_imag": cfr.imag.tolist(),
        "path_cfr_reconstruction_within_tolerance": True,
    }


def _zero_order_summary(row: dict[str, Any]) -> dict[str, Any]:
    paths = [path for path in row["path_diagnostics"] if not _path_signature(path)]
    width = len(row["cfr_vector_real"])
    coherent = _sum_contributions(paths, width)
    incoherent_power = sum(float(np.mean(np.abs(_contribution(path)) ** 2)) for path in paths)
    coefficients = {
        (
            round(float(path["coefficient_real"]), 12),
            round(float(path["coefficient_imag"]), 12),
        )
        for path in paths
    }
    delays = {round(float(path["absolute_delay_s"]) * 1e12) for path in paths}
    return {
        "record_count": len(paths),
        "unique_geometry_fingerprints_descriptive_only": len(
            {path.get("path_fingerprint") for path in paths}
        ),
        "unique_delays_at_1_ps_resolution": len(delays),
        "unique_coefficients_at_1e-12_resolution": len(coefficients),
        "zero_transmitter_angle_records": sum(
            abs(float(path["theta_t_rad"])) < 1e-12 and abs(float(path["phi_t_rad"])) < 1e-12
            for path in paths
        ),
        "coherent_component_mean_power": float(np.mean(np.abs(coherent) ** 2)),
        "incoherent_sum_path_mean_power": float(incoherent_power),
        "aggregate_cfr_real": coherent.real.tolist(),
        "aggregate_cfr_imag": coherent.imag.tolist(),
    }


def analyze_report(
    report: dict[str, Any],
    *,
    link_id: str,
    seed: int,
    max_depth: int,
    low_samples: int,
    high_samples: int,
    thresholds_m: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05),
) -> dict[str, Any]:
    """Compare existing path diagnostics over thresholds, isolating zero-order multiplicity."""
    selected = [
        row
        for row in report["runs"]
        if row["link_id"] == link_id
        and row["seed"] == seed
        and row["max_depth"] == max_depth
        and row["samples_per_src"] in (low_samples, high_samples)
    ]
    by_budget = {int(row["samples_per_src"]): row for row in selected}
    if set(by_budget) != {low_samples, high_samples}:
        raise ValueError("report lacks the requested matched budget rows")
    low, high = by_budget[low_samples], by_budget[high_samples]
    if (
        low.get("path_cfr_reconstruction_within_tolerance") is not True
        or high.get("path_cfr_reconstruction_within_tolerance") is not True
    ):
        raise ValueError("both rows must pass native path-CFR reconstruction")

    width = len(low["cfr_vector_real"])
    low_nonzero = [path for path in low["path_diagnostics"] if _path_signature(path)]
    high_nonzero = [path for path in high["path_diagnostics"] if _path_signature(path)]
    low_zero = _zero_order_summary(low)
    high_zero = _zero_order_summary(high)
    zero_low = np.asarray(low_zero["aggregate_cfr_real"]) + 1j * np.asarray(
        low_zero["aggregate_cfr_imag"]
    )
    zero_high = np.asarray(high_zero["aggregate_cfr_real"]) + 1j * np.asarray(
        high_zero["aggregate_cfr_imag"]
    )
    total_delta = (
        np.asarray(high["cfr_vector_real"])
        + 1j * np.asarray(high["cfr_vector_imag"])
        - np.asarray(low["cfr_vector_real"])
        - 1j * np.asarray(low["cfr_vector_imag"])
    )
    total_delta_norm = float(np.linalg.norm(total_delta))
    zero_delta_norm = float(np.linalg.norm(zero_high - zero_low))
    low_zero.pop("aggregate_cfr_real")
    low_zero.pop("aggregate_cfr_imag")
    high_zero.pop("aggregate_cfr_real")
    high_zero.pop("aggregate_cfr_imag")

    threshold_results = []
    for threshold in thresholds_m:
        result = analyze_pair(
            _row_for_paths(low, low_nonzero, width),
            _row_for_paths(high, high_nonzero, width),
            maximum_match_cost_m=threshold,
        )
        threshold_results.append(
            {
                "maximum_match_cost_m": threshold,
                "nonzero_interaction_matching": {
                    "low_record_count": len(low_nonzero),
                    "high_record_count": len(high_nonzero),
                    "matched_count": result["path_counts"]["matched"],
                    "high_only_count": result["path_counts"]["discovered_at_high_budget"],
                    "low_only_count": result["path_counts"]["present_only_at_low_budget"],
                    "matched_path_variation": result["matched_path_variation"],
                    "cfr_delta_decomposition_norm_relative_to_nonzero_delta": result[
                        "cfr_delta_decomposition_norm_relative_to_total"
                    ],
                },
            }
        )

    return {
        "schema": "loslab.rt_path_match_sensitivity.v1",
        "case": {
            "link_id": link_id,
            "seed": seed,
            "max_depth": max_depth,
            "low_samples_per_src": low_samples,
            "high_samples_per_src": high_samples,
            "thresholds_m": list(thresholds_m),
        },
        "interpretation": (
            "Zero-interaction rows are summarized as an aggregate contribution class and are "
            "never assigned one-to-one identities by Hungarian matching. Threshold sensitivity "
            "applies only to nonzero-interaction records. Fingerprint counts are descriptive, "
            "not proof of physical identity."
        ),
        "zero_interaction_class": {
            "low": low_zero,
            "high": high_zero,
            "aggregate_delta_norm_relative_to_total_cfr_delta": zero_delta_norm
            / max(total_delta_norm, 1e-300),
        },
        "nonzero_interaction_threshold_sensitivity": threshold_results,
    }


def _read_report(path: Path) -> dict[str, Any]:
    payload = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    report = json.loads(payload)
    if not isinstance(report, dict) or not isinstance(report.get("runs"), list):
        raise ValueError("input must be a convergence report containing runs")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--link-id", default="center-los")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--low-samples", type=int, default=100_000)
    parser.add_argument("--high-samples", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing output")
    result = analyze_report(
        _read_report(args.report),
        link_id=args.link_id,
        seed=args.seed,
        max_depth=args.max_depth,
        low_samples=args.low_samples,
        high_samples=args.high_samples,
    )
    result["provenance"] = {
        "source_report": str(args.report),
        "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "analysis_script_sha256": sha256_file(__file__),
        "package_source_sha256": package_source_sha256(),
        "versions": versions(),
    }
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
