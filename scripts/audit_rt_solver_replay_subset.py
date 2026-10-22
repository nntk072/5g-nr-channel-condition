"""Compare overlapping deepest-depth rows from compatible CFR solver reports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from artifacts.provenance import atomic_json, sha256_file

SCHEMA = "loslab.sionna_rt_solver_convergence.v4"


def _load(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read report {path}: {error}") from error
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise ValueError(f"report must use {SCHEMA}: {path}")
    if report.get("configuration", {}).get("cfr_vectors_included") is not True:
        raise ValueError(f"report has no retained CFR vectors: {path}")
    if report["configuration"].get("normalize_delays_for_cfr") is not False:
        raise ValueError(f"report does not use absolute-delay CFR: {path}")
    if not isinstance(report.get("runs"), list) or not report["runs"]:
        raise ValueError(f"report has no solver runs: {path}")
    return report


def _same_except_budget(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_config = dict(left.get("configuration", {}))
    right_config = dict(right.get("configuration", {}))
    for configuration in (left_config, right_config):
        configuration.pop("samples_per_src_values", None)
        configuration.pop("reference_samples_per_src", None)
    if left_config != right_config:
        return False
    if left.get("scene", {}).get("scene_tree_sha256") != right.get("scene", {}).get(
        "scene_tree_sha256"
    ):
        return False
    if any(left.get(key) != right.get(key) for key in ("script_sha256", "package_source_sha256")):
        return False
    versions_a = {k: v for k, v in left.get("versions", {}).items() if k != "git_sha"}
    versions_b = {k: v for k, v in right.get("versions", {}).items() if k != "git_sha"}
    return versions_a == versions_b


def _index(report: dict[str, Any], depth: int) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    rows: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in report["runs"]:
        if row.get("max_depth") != depth:
            continue
        try:
            key = (
                row["link_id"],
                row["seed"],
                row["max_depth"],
                row["samples_per_src"],
            )
            if not isinstance(key[0], str) or any(
                isinstance(value, bool) or not isinstance(value, int) for value in key[1:]
            ):
                raise ValueError
            real = row["cfr_vector_real"]
            imag = row["cfr_vector_imag"]
            if (
                not isinstance(real, list)
                or not isinstance(imag, list)
                or not real
                or len(real) != len(imag)
                or len(real) != report["configuration"]["n_subcarriers"]
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in (*real, *imag)
                )
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid replay row at depth {depth}") from error
        if key in rows:
            raise ValueError(f"duplicate replay row: {key}")
        rows[key] = row
    return rows


def audit_replay_subset(path_a: Path, path_b: Path) -> dict[str, Any]:
    """Compare shared rows at the deepest common depth; no ground truth implied."""
    report_a, report_b = _load(path_a), _load(path_b)
    if not _same_except_budget(report_a, report_b):
        raise ValueError("reports differ in scene, solver configuration, code, or runtime")
    depth_a = {row["max_depth"] for row in report_a["runs"]}
    depth_b = {row["max_depth"] for row in report_b["runs"]}
    common_depths = depth_a & depth_b
    if not common_depths:
        raise ValueError("reports have no common solver depth")
    depth = max(common_depths)
    rows_a, rows_b = _index(report_a, depth), _index(report_b, depth)
    keys = sorted(rows_a.keys() & rows_b.keys())
    if not keys:
        raise ValueError(f"reports have no matching link/seed/depth/budget rows at depth {depth}")

    comparisons: list[dict[str, Any]] = []
    for key in keys:
        first, second = rows_a[key], rows_b[key]
        if len(first["cfr_vector_real"]) != len(second["cfr_vector_real"]):
            raise ValueError(f"CFR grid dimensions differ for replay row {key}")
        real_a, imag_a = first["cfr_vector_real"], first["cfr_vector_imag"]
        real_b, imag_b = second["cfr_vector_real"], second["cfr_vector_imag"]
        reference_power = math.fsum(a * a + b * b for a, b in zip(real_a, imag_a, strict=True))
        difference_power = math.fsum(
            (ar - br) ** 2 + (ai - bi) ** 2
            for ar, ai, br, bi in zip(real_a, imag_a, real_b, imag_b, strict=True)
        )
        relative_l2 = (
            math.sqrt(difference_power / reference_power)
            if reference_power > 0.0
            else 0.0
            if difference_power == 0.0
            else None
        )
        comparisons.append(
            {
                "link_id": key[0],
                "seed": key[1],
                "max_depth": key[2],
                "samples_per_src": key[3],
                "relative_complex_cfr_l2": relative_l2,
                "cfr_sha256_equal": first.get("cfr_sha256") == second.get("cfr_sha256"),
                "valid_path_count_equal": first.get("valid_path_count")
                == second.get("valid_path_count"),
                "candidate_path_count_equal": first.get(
                    "candidate_path_count_before_path_validation"
                )
                == second.get("candidate_path_count_before_path_validation"),
                "per_source_candidate_counts_equal": first.get(
                    "candidate_path_counts_per_source_before_path_validation"
                )
                == second.get("candidate_path_counts_per_source_before_path_validation"),
            }
        )
    values = [
        row["relative_complex_cfr_l2"]
        for row in comparisons
        if row["relative_complex_cfr_l2"] is not None
    ]
    return {
        "schema": "loslab.rt_solver_replay_subset_audit.v1",
        "depth_compared": depth,
        "reports": [
            {
                "path": str(path_a),
                "sha256": sha256_file(path_a),
                "revision": report_a.get("source_revision"),
            },
            {
                "path": str(path_b),
                "sha256": sha256_file(path_b),
                "revision": report_b.get("source_revision"),
            },
        ],
        "same_scene_solver_package_and_runtime": True,
        "summary": {
            "matched_runs": len(comparisons),
            "exact_cfr_hash_matches": sum(row["cfr_sha256_equal"] for row in comparisons),
            "valid_path_count_matches": sum(row["valid_path_count_equal"] for row in comparisons),
            "candidate_path_count_matches": sum(
                row["candidate_path_count_equal"] for row in comparisons
            ),
            "per_source_candidate_count_matches": sum(
                row["per_source_candidate_counts_equal"] for row in comparisons
            ),
            "relative_complex_cfr_l2_min": min(values) if values else None,
            "relative_complex_cfr_l2_median": statistics.median(values) if values else None,
            "relative_complex_cfr_l2_max": max(values) if values else None,
        },
        "comparisons": comparisons,
        "limitations": [
            "Different source revisions are accepted only because the experiment runner hash, package-source hash, scene-tree hash, solver configuration, and runtime versions match.",
            "This is a same-configuration numerical replay audit, not physical validation or a solver-convergence result.",
            "A finite ray budget is not ground truth.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_a", type=Path)
    parser.add_argument("report_b", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    audit = audit_replay_subset(args.report_a, args.report_b)
    atomic_json(args.output, audit)
    print(json.dumps(audit["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
