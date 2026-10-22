"""Compare CFR/path outputs across ray budgets in compatible Sionna RT reports."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from artifacts.provenance import atomic_json, sha256_file

SCHEMA = "loslab.sionna_rt_solver_convergence.v4"


def _load(path: Path) -> dict[str, Any]:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                report = json.load(stream)
        else:
            report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read report {path}: {error}") from error
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise ValueError(f"report must use {SCHEMA}: {path}")
    configuration = report.get("configuration", {})
    if configuration.get("cfr_vectors_included") is not True:
        raise ValueError(f"report has no retained CFR vectors: {path}")
    if configuration.get("normalize_delays_for_cfr") is not False:
        raise ValueError(f"report does not retain absolute-delay CFR: {path}")
    if not isinstance(report.get("runs"), list) or not report["runs"]:
        raise ValueError(f"report has no solver runs: {path}")
    return report


def _compatible(low: dict[str, Any], reference: dict[str, Any]) -> bool:
    config_a = dict(low["configuration"])
    config_b = dict(reference["configuration"])
    for config in (config_a, config_b):
        config.pop("samples_per_src_values", None)
        config.pop("reference_samples_per_src", None)
    if config_a != config_b:
        return False
    if low.get("scene", {}).get("scene_tree_sha256") != reference.get("scene", {}).get(
        "scene_tree_sha256"
    ):
        return False
    if any(
        low.get(field) != reference.get(field)
        for field in ("script_sha256", "package_source_sha256")
    ):
        return False
    versions_a = {key: value for key, value in low.get("versions", {}).items() if key != "git_sha"}
    versions_b = {
        key: value for key, value in reference.get("versions", {}).items() if key != "git_sha"
    }
    return versions_a == versions_b


def _index(report: dict[str, Any], budget: int) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    expected_length = report["configuration"].get("n_subcarriers")
    for row in report["runs"]:
        if row.get("samples_per_src") != budget:
            continue
        try:
            key = (row["link_id"], row["seed"], row["max_depth"])
            real, imag = row["cfr_vector_real"], row["cfr_vector_imag"]
            if (
                not isinstance(key[0], str)
                or any(isinstance(value, bool) or not isinstance(value, int) for value in key[1:])
                or isinstance(expected_length, bool)
                or not isinstance(expected_length, int)
                or not isinstance(real, list)
                or not isinstance(imag, list)
                or len(real) != expected_length
                or len(imag) != expected_length
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in (*real, *imag)
                )
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid CFR row for budget {budget}") from error
        if key in rows:
            raise ValueError(f"duplicate link/seed/depth row for budget {budget}: {key}")
        rows[key] = row
    if not rows:
        raise ValueError(f"report contains no rows at {budget} samples/source")
    return rows


def _relative_complex_l2(value: dict[str, Any], reference: dict[str, Any]) -> float | None:
    value_real, value_imag = value["cfr_vector_real"], value["cfr_vector_imag"]
    ref_real, ref_imag = reference["cfr_vector_real"], reference["cfr_vector_imag"]
    reference_power = math.fsum(
        real**2 + imag**2 for real, imag in zip(ref_real, ref_imag, strict=True)
    )
    difference_power = math.fsum(
        (real - ref_r) ** 2 + (imag - ref_i) ** 2
        for real, imag, ref_r, ref_i in zip(value_real, value_imag, ref_real, ref_imag, strict=True)
    )
    if reference_power == 0.0:
        return 0.0 if difference_power == 0.0 else None
    return math.sqrt(difference_power / reference_power)


def _ratio(value: Any, reference: Any) -> float | None:
    if (
        isinstance(value, bool)
        or isinstance(reference, bool)
        or not isinstance(value, (int, float))
        or not isinstance(reference, (int, float))
        or not math.isfinite(value)
        or not math.isfinite(reference)
    ):
        return None
    if reference == 0.0:
        return 1.0 if value == 0.0 else None
    return float(value / reference)


def compare_ray_budgets(
    low_path: Path,
    reference_path: Path,
    *,
    low_budget: int,
    reference_budget: int,
) -> dict[str, Any]:
    """Compare matched geometries/seeds/depths against a larger finite budget."""
    if (
        isinstance(low_budget, bool)
        or isinstance(reference_budget, bool)
        or low_budget < 1
        or reference_budget <= low_budget
    ):
        raise ValueError("budgets must be positive and reference_budget must exceed low_budget")
    low, reference = _load(low_path), _load(reference_path)
    if not _compatible(low, reference):
        raise ValueError("reports differ in scene, solver configuration, code, or runtime")
    low_rows = _index(low, low_budget)
    ref_rows = _index(reference, reference_budget)
    matched = sorted(low_rows.keys() & ref_rows.keys())
    if not matched:
        raise ValueError("reports have no matched link/seed/depth cases at the requested budgets")

    comparisons: list[dict[str, Any]] = []
    for key in matched:
        value, ref = low_rows[key], ref_rows[key]
        error = _relative_complex_l2(value, ref)
        comparisons.append(
            {
                "link_id": key[0],
                "seed": key[1],
                "max_depth": key[2],
                "geometric_los": value["geometric_los"],
                "low_budget_valid_path_count": value["valid_path_count"],
                "reference_budget_valid_path_count": ref["valid_path_count"],
                "valid_path_count_equal": value["valid_path_count"] == ref["valid_path_count"],
                "low_budget_candidate_path_counts_per_source": value.get(
                    "candidate_path_counts_per_source_before_path_validation"
                ),
                "reference_budget_candidate_path_counts_per_source": ref.get(
                    "candidate_path_counts_per_source_before_path_validation"
                ),
                "reference_budget_cap_filled": ref.get("candidate_path_cap_filled"),
                "low_budget_diffuse_path_count": value.get("valid_diffuse_reflection_path_count"),
                "reference_budget_diffuse_path_count": ref.get(
                    "valid_diffuse_reflection_path_count"
                ),
                "low_to_reference_cfr_mean_power_ratio": _ratio(
                    value.get("cfr_mean_power"), ref.get("cfr_mean_power")
                ),
                "low_to_reference_non_diffuse_path_power_ratio": _ratio(
                    value.get("non_diffuse_path_power_sum"),
                    ref.get("non_diffuse_path_power_sum"),
                ),
                "low_to_reference_diffuse_path_power_ratio": _ratio(
                    value.get("diffuse_path_power_sum"), ref.get("diffuse_path_power_sum")
                ),
                "absolute_rms_delay_spread_difference_ns": abs(
                    value["power_weighted_rms_delay_spread_ns"]
                    - ref["power_weighted_rms_delay_spread_ns"]
                )
                if isinstance(value.get("power_weighted_rms_delay_spread_ns"), (int, float))
                and isinstance(ref.get("power_weighted_rms_delay_spread_ns"), (int, float))
                else None,
                "relative_complex_cfr_l2": error,
            }
        )

    groups: dict[tuple[int, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        groups[(row["max_depth"], row["geometric_los"])].append(row)
    summary_by_depth_condition = []
    for (depth, is_los), rows in sorted(groups.items()):
        errors = [
            row["relative_complex_cfr_l2"]
            for row in rows
            if row["relative_complex_cfr_l2"] is not None
        ]

        def median_metric(name: str) -> float | None:
            values = [row[name] for row in rows if row[name] is not None]
            return statistics.median(values) if values else None

        summary_by_depth_condition.append(
            {
                "max_depth": depth,
                "condition": "LOS" if is_los else "NLOS",
                "matched_runs": len(rows),
                "path_count_changes": sum(not row["valid_path_count_equal"] for row in rows),
                "reference_cap_filled_runs": sum(
                    row["reference_budget_cap_filled"] is True for row in rows
                ),
                "relative_complex_cfr_l2_median": statistics.median(errors) if errors else None,
                "relative_complex_cfr_l2_max": max(errors) if errors else None,
                "low_to_reference_cfr_mean_power_ratio_median": median_metric(
                    "low_to_reference_cfr_mean_power_ratio"
                ),
                "low_to_reference_non_diffuse_path_power_ratio_median": median_metric(
                    "low_to_reference_non_diffuse_path_power_ratio"
                ),
                "low_to_reference_diffuse_path_power_ratio_median": median_metric(
                    "low_to_reference_diffuse_path_power_ratio"
                ),
                "absolute_rms_delay_spread_difference_ns_median": median_metric(
                    "absolute_rms_delay_spread_difference_ns"
                ),
                "zero_energy_comparisons": len(rows) - len(errors),
            }
        )

    return {
        "schema": "loslab.rt_solver_ray_budget_comparison.v1",
        "evidence_grade": "matched synthetic-scene solver sensitivity; larger budget is a finite comparison reference, not truth",
        "budgets_samples_per_source": {"low": low_budget, "reference": reference_budget},
        "reports": [
            {
                "path": str(low_path),
                "sha256": sha256_file(low_path),
                "revision": low.get("source_revision"),
            },
            {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
                "revision": reference.get("source_revision"),
            },
        ],
        "compatibility": {
            "scene_tree_sha256_equal": True,
            "solver_script_sha256_equal": True,
            "package_source_sha256_equal": True,
            "runtime_versions_equal_excluding_git_revision": True,
        },
        "summary": {
            "matched_runs": len(comparisons),
            "missing_reference_rows": len(low_rows.keys() - ref_rows.keys()),
            "grouped_by_depth_and_geometric_condition": summary_by_depth_condition,
        },
        "comparisons": comparisons,
        "limitations": [
            "A finite larger ray budget is not proven ground truth.",
            "The experiment covers only the recorded synthetic scene, links, materials, frequencies, and backend.",
            "This audit measures solver sensitivity; it does not establish propagation-model accuracy or release acceptance.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("low_report", type=Path)
    parser.add_argument("reference_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--low-budget", type=int, required=True)
    parser.add_argument("--reference-budget", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    result = compare_ray_budgets(
        args.low_report,
        args.reference_report,
        low_budget=args.low_budget,
        reference_budget=args.reference_budget,
    )
    atomic_json(args.output, result)
    deepest_depth = max(
        row["max_depth"] for row in result["summary"]["grouped_by_depth_and_geometric_condition"]
    )
    deepest = {
        row["condition"]: {
            "median_cfr_l2": row["relative_complex_cfr_l2_median"],
            "path_count_changes": row["path_count_changes"],
            "median_cfr_mean_power_ratio": row["low_to_reference_cfr_mean_power_ratio_median"],
        }
        for row in result["summary"]["grouped_by_depth_and_geometric_condition"]
        if row["max_depth"] == deepest_depth
    }
    print(
        json.dumps(
            {
                "matched_runs": result["summary"]["matched_runs"],
                "deepest_depth": deepest_depth,
                "conditions": deepest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
