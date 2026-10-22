"""Compare two CFR-bearing Sionna RT reports from repeated identical runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from artifacts.provenance import atomic_json, sha256_file


def _read_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "loslab.sionna_rt_solver_convergence.v4":
        raise ValueError(f"replay audit requires a schema-v4 report: {path}")
    configuration = report.get("configuration", {})
    if configuration.get("cfr_vectors_included") is not True:
        raise ValueError(f"report does not retain complex CFR vectors: {path}")
    if configuration.get("normalize_delays_for_cfr") is not False:
        raise ValueError(f"report does not use absolute-delay CFR: {path}")
    subcarrier_count = configuration.get("n_subcarriers")
    if (
        isinstance(subcarrier_count, bool)
        or not isinstance(subcarrier_count, int)
        or subcarrier_count < 1
    ):
        raise ValueError(f"report has an invalid CFR subcarrier count: {path}")
    if not isinstance(report.get("runs"), list) or not report["runs"]:
        raise ValueError(f"report has no solver runs: {path}")
    return report


def _run_index(
    report: dict[str, Any], path: Path
) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    rows: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in report["runs"]:
        try:
            key = (
                row["link_id"],
                row["seed"],
                row["max_depth"],
                row["samples_per_src"],
            )
            real = row["cfr_vector_real"]
            imag = row["cfr_vector_imag"]
            cfr_hash = row["cfr_sha256"]
            valid_path_count = row["valid_path_count"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"run is missing its identity or CFR vector in {path}") from error
        if (
            not isinstance(key[0], str)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in key[1:])
            or not isinstance(real, list)
            or not isinstance(imag, list)
            or not real
            or len(real) != len(imag)
            or len(real) != report["configuration"]["n_subcarriers"]
            or not isinstance(cfr_hash, str)
            or len(cfr_hash) != 64
            or any(character not in "0123456789abcdef" for character in cfr_hash)
            or isinstance(valid_path_count, bool)
            or not isinstance(valid_path_count, int)
            or valid_path_count < 0
        ):
            raise ValueError(
                f"run identity or configured CFR-grid dimensions are invalid in {path}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (*real, *imag)
        ):
            raise ValueError(f"CFR vector contains a non-finite or non-numeric sample in {path}")
        if key in rows:
            raise ValueError(f"duplicate run identity {key} in {path}")
        rows[key] = row
    return rows


def audit_solver_replay(
    reference_path: Path,
    replay_path: Path,
    *,
    max_relative_cfr_l2_error: float = 1e-6,
) -> dict[str, Any]:
    """Measure numerical reproducibility between two otherwise identical runs."""
    if (
        isinstance(max_relative_cfr_l2_error, bool)
        or not math.isfinite(max_relative_cfr_l2_error)
        or max_relative_cfr_l2_error < 0.0
    ):
        raise ValueError("max_relative_cfr_l2_error must be finite and nonnegative")
    reference = _read_report(reference_path)
    replay = _read_report(replay_path)
    for field in (
        "scene",
        "configuration",
        "source_revision",
        "package_source_sha256",
        "script_sha256",
    ):
        if reference.get(field) != replay.get(field):
            raise ValueError(f"replay reports differ in {field}")
    reference_rows = _run_index(reference, reference_path)
    replay_rows = _run_index(replay, replay_path)
    if reference_rows.keys() != replay_rows.keys():
        raise ValueError("replay reports do not contain identical link/seed/depth/budget cases")

    comparisons: list[dict[str, Any]] = []
    for key in sorted(reference_rows):
        first = reference_rows[key]
        second = replay_rows[key]
        first_real = first["cfr_vector_real"]
        first_imag = first["cfr_vector_imag"]
        second_real = second["cfr_vector_real"]
        second_imag = second["cfr_vector_imag"]
        if len(first_real) != len(second_real):
            raise ValueError(f"CFR grid shape differs for replay case {key}")
        reference_power = math.fsum(
            real * real + imag * imag for real, imag in zip(first_real, first_imag, strict=True)
        )
        difference_power = math.fsum(
            (first_re - second_re) ** 2 + (first_im - second_im) ** 2
            for first_re, first_im, second_re, second_im in zip(
                first_real, first_imag, second_real, second_imag, strict=True
            )
        )
        relative_error = (
            math.sqrt(difference_power / reference_power)
            if reference_power > 0.0
            else 0.0
            if difference_power == 0.0
            else None
        )
        path_count_equal = first.get("valid_path_count") == second.get("valid_path_count")
        comparisons.append(
            {
                "link_id": key[0],
                "seed": key[1],
                "max_depth": key[2],
                "samples_per_src": key[3],
                "relative_cfr_l2_error": relative_error,
                "reference_cfr_zero_energy": reference_power == 0.0,
                "cfr_sha256_equal": first.get("cfr_sha256") == second.get("cfr_sha256"),
                "valid_path_count_equal": path_count_equal,
            }
        )

    errors = [
        row["relative_cfr_l2_error"]
        for row in comparisons
        if row["relative_cfr_l2_error"] is not None
    ]
    failures = [
        row
        for row in comparisons
        if not row["valid_path_count_equal"]
        or row["relative_cfr_l2_error"] is None
        or row["relative_cfr_l2_error"] > max_relative_cfr_l2_error
    ]
    return {
        "schema": "loslab.rt_solver_replay_audit.v1",
        "reference": {"path": str(reference_path), "sha256": sha256_file(reference_path)},
        "replay": {"path": str(replay_path), "sha256": sha256_file(replay_path)},
        "configuration": {
            "scene": reference["scene"]["name"],
            "scene_tree_sha256": reference["scene"]["scene_tree_sha256"],
            "mitsuba_variant": reference["configuration"]["mitsuba_variant"],
            "max_num_paths_per_src": reference["configuration"]["max_num_paths_per_src"],
            "cfr_subcarriers": reference["configuration"]["n_subcarriers"],
            "cfr_vectors_included": True,
        },
        "tolerance": {
            "max_relative_cfr_l2_error": max_relative_cfr_l2_error,
            "path_counts_must_match": True,
        },
        "summary": {
            "case_count": len(comparisons),
            "exact_cfr_hash_matches": sum(row["cfr_sha256_equal"] for row in comparisons),
            "path_count_matches": sum(row["valid_path_count_equal"] for row in comparisons),
            "max_relative_cfr_l2_error": max(errors) if errors else None,
            "median_relative_cfr_l2_error": statistics.median(errors) if errors else None,
            "passed": not failures,
        },
        "comparisons": comparisons,
        "failed_cases": [
            {
                "link_id": row["link_id"],
                "seed": row["seed"],
                "max_depth": row["max_depth"],
                "samples_per_src": row["samples_per_src"],
            }
            for row in failures
        ],
        "limitations": [
            "A passing tolerance applies only to the exact scene, solver settings, backend, and replayed cases in these reports.",
            "Numerical agreement within tolerance is not bitwise identity and does not establish physical-model accuracy.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-relative-cfr-l2-error", type=float, default=1e-6)
    args = parser.parse_args()
    result = audit_solver_replay(
        args.reference,
        args.replay,
        max_relative_cfr_l2_error=args.max_relative_cfr_l2_error,
    )
    atomic_json(args.output, result)
    print(
        f"Compared {result['summary']['case_count']} replays: "
        f"max CFR relative L2 error={result['summary']['max_relative_cfr_l2_error']}; "
        f"passed={result['summary']['passed']}"
    )
    if not result["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
