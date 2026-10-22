"""Audit a matched zero-vs-nonzero material-scattering Sionna RT control."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from artifacts.provenance import atomic_json, sha256_file


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "loslab.sionna_rt_solver_convergence.v4":
        raise ValueError(f"scattering audit requires a schema-v4 report: {path}")
    configuration = report.get("configuration", {})
    if configuration.get("cfr_vectors_included") is not True:
        raise ValueError(f"report must retain complex CFR vectors: {path}")
    if configuration.get("normalize_delays_for_cfr") is not False:
        raise ValueError(f"report must use absolute-delay CFR: {path}")
    if not isinstance(report.get("runs"), list) or not report["runs"]:
        raise ValueError(f"report contains no solver runs: {path}")
    return report


def _scattering_values(report: dict[str, Any]) -> dict[str, float]:
    materials = report["configuration"].get("materials", {})
    result: dict[str, float] = {}
    for name, material in materials.items():
        values = material.get("scattering_coefficient")
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0.0
                or value > 1.0
                for value in values
            )
        ):
            raise ValueError(f"material {name!r} has invalid scattering coefficients")
        if any(float(value) != float(values[0]) for value in values):
            raise ValueError(f"material {name!r} has nonuniform scattering coefficients")
        result[name] = float(values[0])
    return result


def _run_index(
    report: dict[str, Any], path: Path
) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    rows: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    subcarrier_count = report["configuration"].get("n_subcarriers")
    for row in report["runs"]:
        try:
            key = (
                row["link_id"],
                row["seed"],
                row["max_depth"],
                row["samples_per_src"],
            )
            diffuse_count = row["valid_diffuse_reflection_path_count"]
            diffuse_power = row["diffuse_path_power_sum"]
            total_count = row["valid_path_count"]
            real = row["cfr_vector_real"]
            imag = row["cfr_vector_imag"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"run identity or scattering metrics are missing in {path}") from error
        if (
            not isinstance(key[0], str)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in key[1:])
            or isinstance(diffuse_count, bool)
            or not isinstance(diffuse_count, int)
            or diffuse_count < 0
            or isinstance(total_count, bool)
            or not isinstance(total_count, int)
            or total_count < diffuse_count
            or isinstance(diffuse_power, bool)
            or not isinstance(diffuse_power, (int, float))
            or not math.isfinite(diffuse_power)
            or diffuse_power < 0.0
            or not isinstance(real, list)
            or not isinstance(imag, list)
            or not isinstance(subcarrier_count, int)
            or len(real) != subcarrier_count
            or len(imag) != subcarrier_count
            or not real
        ):
            raise ValueError(f"run identity, counts, power, or CFR grid is invalid in {path}")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (*real, *imag)
        ):
            raise ValueError(f"CFR vector contains a non-finite or non-numeric value in {path}")
        if (diffuse_count == 0) != (float(diffuse_power) == 0.0):
            raise ValueError(f"diffuse path count and power disagree in {path}")
        if key in rows:
            raise ValueError(f"duplicate solver case {key} in {path}")
        rows[key] = row
    return rows


def audit_scattering_control(
    zero_scattering_path: Path,
    nonzero_scattering_path: Path,
    *,
    material_name: str,
) -> dict[str, Any]:
    """Check a paired scattering intervention and quantify its channel effect."""
    if not material_name:
        raise ValueError("material_name must be non-empty")
    zero_report = _load_report(zero_scattering_path)
    active_report = _load_report(nonzero_scattering_path)
    for field in ("scene", "source_revision", "package_source_sha256", "script_sha256", "versions"):
        if zero_report.get(field) != active_report.get(field):
            raise ValueError(f"paired scattering reports differ in {field}")

    zero_configuration = zero_report["configuration"]
    active_configuration = active_report["configuration"]
    zero_materials = _scattering_values(zero_report)
    active_materials = _scattering_values(active_report)
    if material_name not in zero_materials or material_name not in active_materials:
        raise ValueError(f"intervened material {material_name!r} is absent from a report")
    if zero_materials[material_name] != 0.0 or active_materials[material_name] <= 0.0:
        raise ValueError("the selected material must change from zero to nonzero scattering")
    if any(value != 0.0 for value in zero_materials.values()):
        raise ValueError("the control report must have zero scattering on all materials")
    if any(
        zero_materials[name] != active_materials[name]
        for name in zero_materials
        if name != material_name
    ) or set(zero_materials) != set(active_materials):
        raise ValueError("the selected material must be the only changed material")
    zero_material_metadata = zero_configuration.get("materials", {})
    active_material_metadata = json.loads(json.dumps(active_configuration.get("materials", {})))
    if not isinstance(zero_material_metadata, dict) or not isinstance(
        active_material_metadata, dict
    ):
        raise ValueError("reports must include material metadata")
    for name, material in zero_material_metadata.items():
        active_material_metadata[name]["scattering_coefficient"] = material[
            "scattering_coefficient"
        ]
    if zero_material_metadata != active_material_metadata:
        raise ValueError("material metadata differs beyond the selected scattering coefficient")

    expected_active_overrides = {material_name: active_materials[material_name]}
    if zero_configuration.get("material_scattering_overrides") != {}:
        raise ValueError("the zero-scattering report must not have material overrides")
    if active_configuration.get("material_scattering_overrides") != expected_active_overrides:
        raise ValueError("active report does not declare exactly the requested material override")

    comparable_zero = dict(zero_configuration)
    comparable_active = dict(active_configuration)
    for configuration in (comparable_zero, comparable_active):
        configuration.pop("materials", None)
        configuration.pop("material_scattering_overrides", None)
    if comparable_zero != comparable_active:
        raise ValueError("paired scattering reports differ in solver or link configuration")

    zero_rows = _run_index(zero_report, zero_scattering_path)
    active_rows = _run_index(active_report, nonzero_scattering_path)
    if zero_rows.keys() != active_rows.keys():
        raise ValueError("paired reports do not contain identical link/seed/depth/budget cases")
    if any(row["valid_diffuse_reflection_path_count"] > 0 for row in zero_rows.values()):
        raise ValueError("zero scattering produced diffuse paths in the control report")

    comparisons: list[dict[str, Any]] = []
    for key in sorted(zero_rows):
        baseline = zero_rows[key]
        intervention = active_rows[key]
        if key[2] == 0 and (
            baseline["valid_diffuse_reflection_path_count"] != 0
            or intervention["valid_diffuse_reflection_path_count"] != 0
        ):
            raise ValueError(f"zero-interaction depth produced diffuse paths for case {key}")
        base_real = baseline["cfr_vector_real"]
        base_imag = baseline["cfr_vector_imag"]
        active_real = intervention["cfr_vector_real"]
        active_imag = intervention["cfr_vector_imag"]
        reference_power = math.fsum(
            float(real) ** 2 + float(imag) ** 2
            for real, imag in zip(base_real, base_imag, strict=True)
        )
        difference_power = math.fsum(
            (float(base_re) - float(active_re)) ** 2 + (float(base_im) - float(active_im)) ** 2
            for base_re, base_im, active_re, active_im in zip(
                base_real, base_imag, active_real, active_imag, strict=True
            )
        )
        relative_error = (
            math.sqrt(difference_power / reference_power) if reference_power > 0.0 else None
        )
        comparisons.append(
            {
                "link_id": key[0],
                "seed": key[1],
                "max_depth": key[2],
                "samples_per_src": key[3],
                "zero_scattering_diffuse_path_count": baseline[
                    "valid_diffuse_reflection_path_count"
                ],
                "nonzero_scattering_diffuse_path_count": intervention[
                    "valid_diffuse_reflection_path_count"
                ],
                "nonzero_scattering_diffuse_power_sum": intervention["diffuse_path_power_sum"],
                "zero_scattering_total_path_count": baseline["valid_path_count"],
                "nonzero_scattering_total_path_count": intervention["valid_path_count"],
                "relative_cfr_l2_error": relative_error,
            }
        )

    positive_rows = [row for row in comparisons if row["nonzero_scattering_diffuse_path_count"] > 0]
    if not positive_rows:
        raise ValueError("nonzero scattering produced no diffuse paths in any matched case")
    relative_errors = [
        row["relative_cfr_l2_error"]
        for row in comparisons
        if row["relative_cfr_l2_error"] is not None
    ]
    return {
        "schema": "loslab.rt_scattering_control_audit.v1",
        "zero_scattering_report": {
            "path": str(zero_scattering_path),
            "sha256": sha256_file(zero_scattering_path),
        },
        "nonzero_scattering_report": {
            "path": str(nonzero_scattering_path),
            "sha256": sha256_file(nonzero_scattering_path),
        },
        "scene": zero_report["scene"],
        "configuration": {
            "material": material_name,
            "zero_scattering_coefficients": zero_materials,
            "nonzero_scattering_coefficients": active_materials,
            "solver": zero_configuration,
        },
        "summary": {
            "matched_case_count": len(comparisons),
            "zero_scattering_cases_with_diffuse_paths": sum(
                row["zero_scattering_diffuse_path_count"] > 0 for row in comparisons
            ),
            "nonzero_scattering_cases_with_diffuse_paths": len(positive_rows),
            "nonzero_scattering_cases_with_diffuse_power": sum(
                row["nonzero_scattering_diffuse_power_sum"] > 0 for row in comparisons
            ),
            "max_relative_cfr_l2_error": max(relative_errors) if relative_errors else None,
            "all_depth_zero_cases_have_no_diffuse_paths": all(
                row["max_depth"] != 0
                or (
                    row["zero_scattering_diffuse_path_count"] == 0
                    and row["nonzero_scattering_diffuse_path_count"] == 0
                )
                for row in comparisons
            ),
            "passed": True,
        },
        "comparisons": comparisons,
        "limitations": [
            "This verifies Sionna RT diffuse-path mechanics for the exact synthetic screen scene and settings, not calibrated material behavior.",
            "One SISO scene family does not qualify scattering physics across materials, geometries, arrays, or external measurements.",
            "The largest requested ray budget is a finite comparison setting, not physical ground truth.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zero_scattering_report", type=Path)
    parser.add_argument("nonzero_scattering_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--material", required=True)
    args = parser.parse_args()
    result = audit_scattering_control(
        args.zero_scattering_report,
        args.nonzero_scattering_report,
        material_name=args.material,
    )
    atomic_json(args.output, result)
    print(
        f"Audited {result['summary']['matched_case_count']} paired cases: "
        f"diffuse paths in {result['summary']['nonzero_scattering_cases_with_diffuse_paths']} "
        "nonzero-scattering cases; passed=True"
    )


if __name__ == "__main__":
    main()
