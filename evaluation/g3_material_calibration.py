"""Evaluate held-out material-scattering measurements against the G3.03 margins.

This evaluates measurement-derived predictions only. Passing the numerical
margins is necessary but is not, by itself, RT scene or complex-CSI
qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.g3_joint_acceptance import (
    MATERIAL_CLAIMS,
    family_alpha_for_claim,
    validate_g3_joint_acceptance_family,
)
from evaluation.g3_numeric import finite_real_array

SCHEMA = "loslab.g3_material_calibration_predictions.v2"
SPLIT_MANIFEST_SCHEMA = "loslab.g3_material_calibration_split_manifest.v1"
TARGET_FREQUENCY_HZ = 3.5e9
MAX_POWER_ERROR_MEDIAN_DB = 1.5
MAX_POWER_ERROR_P95_DB = 3.0
MAX_ANGULAR_CDF_KS = 0.10
MIN_INTERVAL_COVERAGE = 0.90
MAX_INTERVAL_COVERAGE = 0.98
MAX_MEDIAN_INTERVAL_WIDTH_DB = 3.0
MIN_INDEPENDENT_MEASUREMENT_UNITS = 20


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical string without surrounding whitespace")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _spherical_cell_solid_angles(
    theta_edges_deg: tuple[float, ...], phi_edges_deg: tuple[float, ...]
) -> np.ndarray:
    if (
        len(theta_edges_deg) < 2
        or len(phi_edges_deg) < 2
        or theta_edges_deg[0] != 0.0
        or theta_edges_deg[-1] != 180.0
        or phi_edges_deg[0] != -180.0
        or phi_edges_deg[-1] != 180.0
        or any(right <= left for left, right in zip(theta_edges_deg, theta_edges_deg[1:]))
        or any(right <= left for left, right in zip(phi_edges_deg, phi_edges_deg[1:]))
    ):
        raise ValueError(
            "angular grid must cover zenith [0, 180] and azimuth [-180, 180] with increasing edges"
        )
    theta = np.radians(np.asarray(theta_edges_deg, dtype=np.float64))
    phi = np.radians(np.asarray(phi_edges_deg, dtype=np.float64))
    theta_measure = np.cos(theta[:-1]) - np.cos(theta[1:])
    phi_measure = np.diff(phi)
    return theta_measure[:, None] * phi_measure[None, :]


def _angular_joint_cdf_ks(measured: Any, predicted: Any) -> float:
    measured_array = finite_real_array(measured, "measured angular powers", ndim=2)
    predicted_array = finite_real_array(predicted, "predicted angular powers", ndim=2)
    if (
        measured_array.ndim != 2
        or measured_array.shape != predicted_array.shape
        or any(size == 0 for size in measured_array.shape)
        or not np.all(np.isfinite(measured_array))
        or not np.all(np.isfinite(predicted_array))
        or np.any(measured_array < 0)
        or np.any(predicted_array < 0)
        or measured_array.sum() <= 0
        or predicted_array.sum() <= 0
    ):
        raise ValueError(
            "angular bin powers must be same-shape 2D finite, nonnegative, and have positive sums"
        )
    measured_probability = measured_array / measured_array.sum()
    predicted_probability = predicted_array / predicted_array.sum()
    measured_cdf = np.cumsum(np.cumsum(measured_probability, axis=0), axis=1)
    predicted_cdf = np.cumsum(np.cumsum(predicted_probability, axis=0), axis=1)
    return float(np.max(np.abs(measured_cdf - predicted_cdf)))


def _cluster_bootstrap_intervals(
    units: dict[str, list[dict[str, float]]],
    *,
    seed: int,
    replicates: int,
    simultaneous_comparisons: int,
    familywise_alpha: float = 0.05,
) -> dict[str, Any]:
    """Resample independent acquisition units and keep their rows together."""
    unit_ids = sorted(units)
    report: dict[str, Any] = {
        "resampling_unit": "independent_measurement_unit_id",
        "independent_unit_count": len(unit_ids),
        "minimum_units_for_pilot_interval_assessment": MIN_INDEPENDENT_MEASUREMENT_UNITS,
        "minimum_is_pilot_floor_not_full_qualification_sample_size": True,
        "familywise_alpha": familywise_alpha,
        "simultaneous_comparison_count": simultaneous_comparisons,
        "interval_method": "Bonferroni-adjusted percentile bootstrap",
        "interpretation": "DIAGNOSTIC_ONLY_INDEPENDENCE_AND_SOURCE_AUTHENTICITY_UNVERIFIED",
        "seed": seed,
        "replicates": replicates,
        "bootstrap_computable_for_diagnostics": len(unit_ids) >= MIN_INDEPENDENT_MEASUREMENT_UNITS,
        "independence_verified_for_admission": False,
    }
    if len(unit_ids) < MIN_INDEPENDENT_MEASUREMENT_UNITS:
        report["reason"] = (
            "the project-defined material-measurement pilot floor is "
            f"{MIN_INDEPENDENT_MEASUREMENT_UNITS} independent acquisition units"
        )
        report["percentile_95_ci"] = None
        return report

    def summarize_unit(rows: list[dict[str, float]]) -> dict[str, float]:
        errors = np.asarray([row["absolute_power_error_db"] for row in rows])
        angular_ks = np.asarray([row["angular_cdf_ks"] for row in rows])
        widths = np.asarray([row["interval_width_db"] for row in rows])
        coverage = np.asarray([row["covered"] for row in rows])
        return {
            "median_absolute_power_error_db": float(np.median(errors)),
            "p95_absolute_power_error_db": float(np.percentile(errors, 95)),
            "maximum_angular_cdf_ks": float(np.max(angular_ks)),
            "nominal_95_interval_coverage": float(np.mean(coverage)),
            "median_prediction_interval_width_db": float(np.median(widths)),
        }

    unit_values = {unit_id: summarize_unit(rows) for unit_id, rows in units.items()}
    rng = np.random.default_rng(seed)
    metrics: dict[str, list[float]] = {
        "median_absolute_power_error_db": [],
        "p95_absolute_power_error_db": [],
        "maximum_angular_cdf_ks": [],
        "nominal_95_interval_coverage": [],
        "median_prediction_interval_width_db": [],
    }
    for selected_indices in rng.integers(0, len(unit_ids), size=(replicates, len(unit_ids))):
        sampled_units = [unit_values[unit_ids[int(index)]] for index in selected_indices]
        metrics["median_absolute_power_error_db"].append(
            float(np.median([unit["median_absolute_power_error_db"] for unit in sampled_units]))
        )
        metrics["p95_absolute_power_error_db"].append(
            float(
                np.percentile([unit["p95_absolute_power_error_db"] for unit in sampled_units], 95)
            )
        )
        metrics["maximum_angular_cdf_ks"].append(
            float(max(unit["maximum_angular_cdf_ks"] for unit in sampled_units))
        )
        metrics["nominal_95_interval_coverage"].append(
            float(np.mean([unit["nominal_95_interval_coverage"] for unit in sampled_units]))
        )
        metrics["median_prediction_interval_width_db"].append(
            float(
                np.median([unit["median_prediction_interval_width_db"] for unit in sampled_units])
            )
        )

    per_comparison_alpha = familywise_alpha / simultaneous_comparisons
    report["percentile_95_ci"] = {
        name: [
            float(np.percentile(values, 100 * per_comparison_alpha / 2)),
            float(np.percentile(values, 100 * (1 - per_comparison_alpha / 2))),
        ]
        for name, values in metrics.items()
    }
    report["per_comparison_alpha"] = per_comparison_alpha
    report["per_comparison_confidence_level"] = 1 - per_comparison_alpha
    return report


def evaluate_g3_material_calibration(payload: dict[str, Any]) -> dict[str, Any]:
    """Score pre-frozen, group-held-out predictions against documented G3.03 margins.

    The input must contain a complete test-group manifest. Test rows must cover
    exactly the sample IDs declared for each held-out incidence-angle group.
    Material and polarization strata are scored independently. The evaluator
    cannot establish the provenance of the measurements or prove that the
    predictor was frozen before test access; those facts must be attested in
    the required identifiers and external experiment record.
    """
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA!r}")
    frequency = _finite_number(payload.get("target_frequency_hz"), "target_frequency_hz")
    if not math.isclose(frequency, TARGET_FREQUENCY_HZ, rel_tol=0.0, abs_tol=1.0):
        raise ValueError("G3.03 material admission is currently fixed at 3.5 GHz")

    for field in ("model_id", "model_sha256", "measurement_calibration_id", "material_mapping_id"):
        _nonempty_string(payload.get(field), field)
    _nonempty_string(payload.get("surface_description"), "surface_description")
    model_hash = payload["model_sha256"]
    if len(model_hash) != 64 or any(ch not in "0123456789abcdef" for ch in model_hash):
        raise ValueError("model_sha256 must be a lowercase 64-character SHA-256 hex digest")
    if (
        not isinstance(payload.get("uncertainty_description"), str)
        or not payload["uncertainty_description"].strip()
    ):
        raise ValueError("uncertainty_description must document measurement/prediction uncertainty")
    measurement_unit_description = _nonempty_string(
        payload.get("measurement_unit_independence_description"),
        "measurement_unit_independence_description",
    )
    bootstrap_seed = payload.get("cluster_bootstrap_seed")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError(
            "cluster_bootstrap_seed must be a nonnegative integer fixed before evaluation"
        )
    bootstrap_replicates = payload.get("cluster_bootstrap_replicates")
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 1_000
    ):
        raise ValueError("cluster_bootstrap_replicates must be an integer of at least 1000")

    manifest = payload.get("incidence_groups")
    rows = payload.get("test_predictions")
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("incidence_groups must be a nonempty complete group manifest")
    if not isinstance(rows, list) or not rows:
        raise ValueError("test_predictions must be a nonempty list")

    expected_strata_payload = payload.get("expected_strata")
    if not isinstance(expected_strata_payload, list) or not expected_strata_payload:
        raise ValueError(
            "expected_strata must explicitly declare all material/polarization/angle strata"
        )
    expected_strata: set[tuple[str, str, str]] = set()
    for item in expected_strata_payload:
        if not isinstance(item, dict):
            raise ValueError("each expected_strata entry must be an object")
        material = _nonempty_string(item.get("material_id"), "expected material_id")
        polarization = _nonempty_string(item.get("polarization"), "expected polarization")
        if polarization not in {"V", "H"}:
            raise ValueError("expected polarization must be V or H")
        incidence_stratum = _nonempty_string(
            item.get("incidence_stratum_id"), "expected incidence_stratum_id"
        )
        stratum = (material, polarization, incidence_stratum)
        if stratum in expected_strata:
            raise ValueError(f"duplicate expected stratum: {stratum}")
        expected_strata.add(stratum)

    full_split_manifest = payload.get("full_split_manifest")
    if (
        not isinstance(full_split_manifest, dict)
        or full_split_manifest.get("schema") != SPLIT_MANIFEST_SCHEMA
    ):
        raise ValueError(f"full_split_manifest must use schema {SPLIT_MANIFEST_SCHEMA!r}")
    split_manifest_hash = _nonempty_string(
        payload.get("full_split_manifest_sha256"), "full_split_manifest_sha256"
    )
    if len(split_manifest_hash) != 64 or any(
        char not in "0123456789abcdef" for char in split_manifest_hash
    ):
        raise ValueError("full_split_manifest_sha256 must be a lowercase 64-character SHA-256 hex digest")
    if _canonical_sha256(full_split_manifest) != split_manifest_hash:
        raise ValueError("full_split_manifest_sha256 does not match canonical full_split_manifest")
    _nonempty_string(
        payload.get("split_manifest_external_verification_id"),
        "split_manifest_external_verification_id",
    )
    preregistered_strata_payload = full_split_manifest.get("expected_strata")
    if not isinstance(preregistered_strata_payload, list):
        raise ValueError("full_split_manifest must bind the preregistered expected_strata")
    preregistered_strata: set[tuple[str, str, str]] = set()
    for item in preregistered_strata_payload:
        if not isinstance(item, dict):
            raise ValueError("each split expected stratum must be an object")
        stratum = (
            _nonempty_string(item.get("material_id"), "split expected material_id"),
            _nonempty_string(item.get("polarization"), "split expected polarization"),
            _nonempty_string(item.get("incidence_stratum_id"), "split expected incidence stratum"),
        )
        if stratum in preregistered_strata:
            raise ValueError(f"duplicate split expected stratum: {stratum}")
        preregistered_strata.add(stratum)
    if preregistered_strata != expected_strata:
        raise ValueError(
            "expected_strata must exactly match the externally verified split manifest"
        )
    split_groups = full_split_manifest.get("groups")
    if not isinstance(split_groups, list) or not split_groups:
        raise ValueError("full_split_manifest.groups must declare every split group")
    split_group_ids: set[str] = set()
    split_sample_ids: set[str] = set()
    split_test_groups: dict[str, set[str]] = {}
    for item in split_groups:
        if not isinstance(item, dict):
            raise ValueError("each full split manifest group must be an object")
        group_id = _nonempty_string(item.get("incidence_group_id"), "split incidence_group_id")
        if group_id in split_group_ids:
            raise ValueError(f"duplicate split incidence group: {group_id}")
        split_group_ids.add(group_id)
        split = item.get("split")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split assignment for group {group_id}")
        sample_ids = item.get("sample_ids")
        if not isinstance(sample_ids, list) or not sample_ids:
            raise ValueError(f"split group {group_id} requires nonempty sample_ids")
        normalized = {_nonempty_string(sample_id, "split sample_id") for sample_id in sample_ids}
        if len(normalized) != len(sample_ids) or split_sample_ids.intersection(normalized):
            raise ValueError("full split manifest sample IDs must be globally unique")
        split_sample_ids.update(normalized)
        if split == "test":
            split_test_groups[group_id] = normalized

    test_sample_ids = set().union(*split_test_groups.values())
    test_assignments_payload = full_split_manifest.get("test_sample_assignments")
    if not isinstance(test_assignments_payload, list) or not test_assignments_payload:
        raise ValueError(
            "full_split_manifest must bind every test sample to its preregistered stratum"
        )
    test_assignments: dict[str, dict[str, Any]] = {}
    strata_from_assignments: set[tuple[str, str, str]] = set()
    for assignment in test_assignments_payload:
        if not isinstance(assignment, dict):
            raise ValueError("each test_sample_assignments entry must be an object")
        sample_id = _nonempty_string(assignment.get("sample_id"), "assigned sample_id")
        if sample_id in test_assignments:
            raise ValueError(f"duplicate test sample assignment: {sample_id}")
        group_id = _nonempty_string(
            assignment.get("incidence_group_id"), "assigned incidence_group_id"
        )
        material = _nonempty_string(assignment.get("material_id"), "assigned material_id")
        polarization = _nonempty_string(
            assignment.get("polarization"), "assigned polarization"
        )
        if polarization not in {"V", "H"}:
            raise ValueError("assigned polarization must be V or H")
        incidence_stratum = _nonempty_string(
            assignment.get("incidence_stratum_id"), "assigned incidence_stratum_id"
        )
        assignment["incidence_angle_deg"] = _finite_number(
            assignment.get("incidence_angle_deg"), "assigned incidence_angle_deg"
        )
        _nonempty_string(assignment.get("measurement_unit_id"), "assigned measurement_unit_id")
        _nonempty_string(
            assignment.get("measurement_unit_provenance_id"),
            "assigned measurement_unit_provenance_id",
        )
        if group_id not in split_test_groups or sample_id not in split_test_groups[group_id]:
            raise ValueError(
                f"test sample assignment {sample_id} is outside its declared test group"
            )
        assignment_stratum = (material, polarization, incidence_stratum)
        strata_from_assignments.add(assignment_stratum)
        test_assignments[sample_id] = assignment
    if set(test_assignments) != test_sample_ids:
        raise ValueError("test_sample_assignments must exactly cover all test sample IDs")
    if strata_from_assignments != expected_strata:
        raise ValueError("test sample assignments must exactly cover expected_strata")

    expected: dict[str, set[str]] = {}
    measurement_unit_by_group: dict[str, str] = {}
    measurement_unit_provenance_by_id: dict[str, str] = {}
    provenance_by_measurement_unit: dict[str, str] = {}
    group_strata: dict[str, str] = {}
    group_angles: dict[str, float] = {}
    for item in manifest:
        if not isinstance(item, dict):
            raise ValueError("each incidence group must be an object")
        group_id = _nonempty_string(item.get("incidence_group_id"), "incidence_group_id")
        if group_id in expected:
            raise ValueError(f"duplicate incidence group: {group_id}")
        if item.get("split") != "test":
            raise ValueError("incidence_groups may contain only complete held-out test groups")
        measurement_unit_by_group[group_id] = _nonempty_string(
            item.get("measurement_unit_id"), "measurement_unit_id"
        )
        measurement_unit_id = measurement_unit_by_group[group_id]
        provenance_id = _nonempty_string(
            item.get("measurement_unit_provenance_id"), "measurement_unit_provenance_id"
        )
        if (
            measurement_unit_id in measurement_unit_provenance_by_id
            and measurement_unit_provenance_by_id[measurement_unit_id] != provenance_id
        ):
            raise ValueError("one measurement_unit_id maps to conflicting provenance IDs")
        if (
            provenance_id in provenance_by_measurement_unit
            and provenance_by_measurement_unit[provenance_id] != measurement_unit_id
        ):
            raise ValueError("distinct measurement units cannot reuse one provenance record ID")
        measurement_unit_provenance_by_id[measurement_unit_id] = provenance_id
        provenance_by_measurement_unit[provenance_id] = measurement_unit_id
        group_strata[group_id] = _nonempty_string(
            item.get("incidence_stratum_id"), "incidence_stratum_id"
        )
        group_angles[group_id] = _finite_number(
            item.get("incidence_angle_deg"), "incidence_angle_deg"
        )
        sample_ids = item.get("sample_ids")
        if not isinstance(sample_ids, list) or not sample_ids:
            raise ValueError(f"incidence group {group_id} needs its complete nonempty sample_ids")
        normalized = {_nonempty_string(sample_id, "sample_id") for sample_id in sample_ids}
        if len(normalized) != len(sample_ids):
            raise ValueError(f"incidence group {group_id} repeats a sample_id")
        expected[group_id] = normalized

    if split_test_groups != expected:
        raise ValueError(
            "incidence_groups must exactly match all test groups in full_split_manifest"
        )

    actual: dict[str, set[str]] = {group_id: set() for group_id in expected}
    row_metrics: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    metrics_by_unit: dict[tuple[str, str, str], dict[str, list[dict[str, float]]]] = {}
    groups_by_stratum: dict[tuple[str, str, str], set[str]] = {}
    all_sample_ids: set[str] = set()
    angular_grids_by_stratum: dict[
        tuple[str, str, str], tuple[tuple[float, ...], tuple[float, ...], str]
    ] = {}
    angles_by_stratum: dict[tuple[str, str, str], set[float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each test prediction must be an object")
        sample_id = _nonempty_string(row.get("sample_id"), "sample_id")
        if sample_id in all_sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        all_sample_ids.add(sample_id)
        group_id = _nonempty_string(row.get("incidence_group_id"), "incidence_group_id")
        if group_id not in expected or sample_id not in expected[group_id]:
            raise ValueError(f"sample {sample_id} is not in a declared held-out incidence group")
        assignment = test_assignments[sample_id]
        if assignment["incidence_group_id"] != group_id:
            raise ValueError(f"sample {sample_id} disagrees with its hashed group assignment")
        incidence_stratum = _nonempty_string(
            row.get("incidence_stratum_id"), "incidence_stratum_id"
        )
        if (
            incidence_stratum != group_strata[group_id]
            or incidence_stratum != assignment["incidence_stratum_id"]
        ):
            raise ValueError(f"sample {sample_id} disagrees with its incidence-group stratum")
        incidence_angle = _finite_number(row.get("incidence_angle_deg"), "incidence_angle_deg")
        if (
            incidence_angle != group_angles[group_id]
            or incidence_angle != assignment["incidence_angle_deg"]
        ):
            raise ValueError(f"sample {sample_id} disagrees with its incidence-group angle")
        if row.get("split") != "test":
            raise ValueError("all evaluated predictions must be from the test split")
        actual[group_id].add(sample_id)
        material = _nonempty_string(row.get("material_id"), "material_id")
        polarization = _nonempty_string(row.get("polarization"), "polarization")
        if polarization not in {"V", "H"}:
            raise ValueError("polarization must be V or H")
        if material != assignment["material_id"] or polarization != assignment["polarization"]:
            raise ValueError(f"sample {sample_id} disagrees with its hashed material/polarization")
        stratum = (material, polarization, incidence_stratum)
        groups_by_stratum.setdefault(stratum, set()).add(group_id)
        measured_power = _finite_number(
            row.get("measured_integrated_power_db"), "measured_integrated_power_db"
        )
        predicted_power = _finite_number(
            row.get("predicted_integrated_power_db"), "predicted_integrated_power_db"
        )
        interval = row.get("prediction_interval_95_db")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("prediction_interval_95_db must be [lower_db, upper_db]")
        lower = _finite_number(interval[0], "prediction interval lower bound")
        upper = _finite_number(interval[1], "prediction interval upper bound")
        if lower > upper:
            raise ValueError("prediction interval lower bound exceeds upper bound")
        if row.get("angular_power_semantics") != "integrated_power_per_bin":
            raise ValueError("angular_power_semantics must be integrated_power_per_bin")
        if row.get("angular_array_order") != ["theta_bin", "phi_bin; azimuth varies fastest"]:
            raise ValueError("angular_array_order must declare theta rows and fastest-varying phi")
        if row.get("missing_angular_bin_indices") != []:
            raise ValueError(
                "angular patterns with missing bins are unresolved and cannot be scored"
            )
        measured_angular = row.get("measured_angular_power_per_bin_w")
        predicted_angular = row.get("predicted_angular_power_per_bin_w")
        supplied_solid_angles = row.get("solid_angle_sr")
        if not all(
            isinstance(value, list)
            for value in (measured_angular, predicted_angular, supplied_solid_angles)
        ):
            raise ValueError("angular powers and solid_angle_sr must be nested 2D arrays")
        theta_payload = row.get("theta_edges_deg")
        phi_payload = row.get("phi_edges_deg")
        if not isinstance(theta_payload, list) or not isinstance(phi_payload, list):
            raise ValueError("theta_edges_deg and phi_edges_deg must be arrays")
        theta_edges = tuple(_finite_number(value, "theta edge") for value in theta_payload)
        phi_edges = tuple(_finite_number(value, "phi edge") for value in phi_payload)
        expected_solid_angles = _spherical_cell_solid_angles(theta_edges, phi_edges)
        expected_shape = expected_solid_angles.shape
        frame_id = _nonempty_string(
            row.get("angular_coordinate_frame_id"), "angular_coordinate_frame_id"
        )
        for field_name, matrix in (
            ("measured_angular_power_per_bin_w", measured_angular),
            ("predicted_angular_power_per_bin_w", predicted_angular),
            ("solid_angle_sr", supplied_solid_angles),
        ):
            array = finite_real_array(matrix, field_name, ndim=2)
            if array.shape != expected_shape:
                raise ValueError(f"{field_name} shape must match theta/phi angular bins")
            if not np.all(np.isfinite(array)) or np.any(array < 0):
                raise ValueError(f"{field_name} must contain finite nonnegative values")
        if not np.allclose(
            finite_real_array(supplied_solid_angles, "solid_angle_sr", ndim=2),
            expected_solid_angles,
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError("solid_angle_sr does not match the spherical angular-bin geometry")
        measured_angular_array = finite_real_array(
            measured_angular, "measured_angular_power_per_bin_w", ndim=2
        )
        predicted_angular_array = finite_real_array(
            predicted_angular, "predicted_angular_power_per_bin_w", ndim=2
        )
        if measured_angular_array.sum() <= 0 or predicted_angular_array.sum() <= 0:
            raise ValueError(
                "measured and predicted integrated angular powers must have positive sums"
            )
        angular_grid = (theta_edges, phi_edges, frame_id)
        if (
            stratum in angular_grids_by_stratum
            and angular_grids_by_stratum[stratum] != angular_grid
        ):
            raise ValueError(f"angular grid or coordinate frame varies within stratum {stratum}")
        angular_grids_by_stratum[stratum] = angular_grid
        angles_by_stratum.setdefault(stratum, set()).add(incidence_angle)
        ks = _angular_joint_cdf_ks(measured_angular, predicted_angular)
        metrics = {
            "absolute_power_error_db": abs(predicted_power - measured_power),
            "covered": float(lower <= measured_power <= upper),
            "interval_width_db": upper - lower,
            "angular_cdf_ks": ks,
        }
        row_metrics.setdefault(stratum, []).append(metrics)
        unit_id = measurement_unit_by_group[group_id]
        metrics_by_unit.setdefault(stratum, {}).setdefault(unit_id, []).append(metrics)

    for group_id, sample_ids in expected.items():
        if actual[group_id] != sample_ids:
            missing = sorted(sample_ids - actual[group_id])
            extra = sorted(actual[group_id] - sample_ids)
            raise ValueError(
                f"incidence group {group_id} is incomplete (missing={missing}, extra={extra})"
            )

    for group_id, sample_ids in expected.items():
        for sample_id in sample_ids:
            assignment = test_assignments[sample_id]
            if (
                assignment["incidence_group_id"] != group_id
                or assignment["incidence_stratum_id"] != group_strata[group_id]
                or assignment["incidence_angle_deg"] != group_angles[group_id]
                or assignment["measurement_unit_id"] != measurement_unit_by_group[group_id]
                or assignment["measurement_unit_provenance_id"]
                != measurement_unit_provenance_by_id[measurement_unit_by_group[group_id]]
            ):
                raise ValueError(
                    f"test sample assignment disagrees with incidence group {group_id}"
                )

    actual_strata = set(row_metrics)
    if actual_strata != expected_strata:
        raise ValueError("evaluated strata must exactly match expected_strata")

    family_payload = payload.get("joint_acceptance_family")
    if family_payload is None:
        simultaneous_comparisons = len(expected_strata) * len(MATERIAL_CLAIMS)
        family_reference = None
    else:
        normalized_family = validate_g3_joint_acceptance_family(family_payload)
        _, simultaneous_comparisons = family_alpha_for_claim(
            normalized_family,
            evaluator="material_calibration",
            expected_strata=sorted(expected_strata),
        )
        family_reference = {
            "family_id": normalized_family["family_id"],
            "family_sha256": normalized_family["family_sha256"],
        }

    strata_reports: list[dict[str, Any]] = []
    all_pilot_margins_pass = True
    all_admission_criteria_pass = True
    independence_verified_for_admission = False
    for stratum_index, ((material, polarization, incidence_stratum), values) in enumerate(
        sorted(row_metrics.items())
    ):
        group_count = len(groups_by_stratum[(material, polarization, incidence_stratum)])
        units = metrics_by_unit[(material, polarization, incidence_stratum)]
        unit_summaries = []
        for unit_rows in units.values():
            unit_errors = np.asarray([value["absolute_power_error_db"] for value in unit_rows])
            unit_widths = np.asarray([value["interval_width_db"] for value in unit_rows])
            unit_angular_ks = np.asarray([value["angular_cdf_ks"] for value in unit_rows])
            unit_coverage = np.asarray([value["covered"] for value in unit_rows])
            unit_summaries.append(
                {
                    "median_absolute_power_error_db": float(np.median(unit_errors)),
                    "p95_absolute_power_error_db": float(np.percentile(unit_errors, 95)),
                    "maximum_angular_cdf_ks": float(np.max(unit_angular_ks)),
                    "nominal_95_interval_coverage": float(np.mean(unit_coverage)),
                    "median_prediction_interval_width_db": float(np.median(unit_widths)),
                }
            )
        measurement_unit_count = len(unit_summaries)
        measurement_unit_bootstrap = _cluster_bootstrap_intervals(
            units,
            seed=bootstrap_seed + stratum_index,
            replicates=bootstrap_replicates,
            simultaneous_comparisons=simultaneous_comparisons,
        )
        equal_unit_errors = np.asarray(
            [unit["median_absolute_power_error_db"] for unit in unit_summaries]
        )
        equal_unit_p95_errors = np.asarray(
            [unit["p95_absolute_power_error_db"] for unit in unit_summaries]
        )
        equal_unit_angular_ks = np.asarray(
            [unit["maximum_angular_cdf_ks"] for unit in unit_summaries]
        )
        equal_unit_coverages = np.asarray(
            [unit["nominal_95_interval_coverage"] for unit in unit_summaries]
        )
        equal_unit_widths = np.asarray(
            [unit["median_prediction_interval_width_db"] for unit in unit_summaries]
        )
        coverage = float(np.mean(equal_unit_coverages))
        checks = {
            "median_absolute_power_error_db": float(np.median(equal_unit_errors))
            <= MAX_POWER_ERROR_MEDIAN_DB,
            "p95_absolute_power_error_db": bool(
                np.percentile(equal_unit_p95_errors, 95) <= MAX_POWER_ERROR_P95_DB
            ),
            "maximum_angular_cdf_ks": float(np.max(equal_unit_angular_ks)) <= MAX_ANGULAR_CDF_KS,
            "nominal_95_interval_coverage": MIN_INTERVAL_COVERAGE
            <= coverage
            <= MAX_INTERVAL_COVERAGE,
            "median_prediction_interval_width_db": float(np.median(equal_unit_widths))
            <= MAX_MEDIAN_INTERVAL_WIDTH_DB,
        }
        point_margins_pass = all(checks.values())
        percentile_ci = measurement_unit_bootstrap["percentile_95_ci"]
        uncertainty_checks: dict[str, bool] | None = None
        if measurement_unit_bootstrap["bootstrap_computable_for_diagnostics"] and isinstance(
            percentile_ci, dict
        ):
            uncertainty_checks = {
                "median_absolute_power_error_db_upper_ci": percentile_ci[
                    "median_absolute_power_error_db"
                ][1]
                <= MAX_POWER_ERROR_MEDIAN_DB,
                "p95_absolute_power_error_db_upper_ci": percentile_ci[
                    "p95_absolute_power_error_db"
                ][1]
                <= MAX_POWER_ERROR_P95_DB,
                "maximum_angular_cdf_ks_upper_ci": percentile_ci["maximum_angular_cdf_ks"][1]
                <= MAX_ANGULAR_CDF_KS,
                "nominal_95_interval_coverage_ci": percentile_ci["nominal_95_interval_coverage"][0]
                >= MIN_INTERVAL_COVERAGE
                and percentile_ci["nominal_95_interval_coverage"][1] <= MAX_INTERVAL_COVERAGE,
                "median_prediction_interval_width_db_upper_ci": percentile_ci[
                    "median_prediction_interval_width_db"
                ][1]
                <= MAX_MEDIAN_INTERVAL_WIDTH_DB,
            }
        uncertainty_pass = uncertainty_checks is not None and all(uncertainty_checks.values())
        angular_grid = angular_grids_by_stratum[(material, polarization, incidence_stratum)]
        # Independence is a physical/provenance claim this pure evaluator cannot verify.
        # Bootstrap and margin results remain diagnostic until an external evidence gate exists.
        pilot_margins_pass = point_margins_pass and uncertainty_pass
        admission_criteria_pass = pilot_margins_pass and independence_verified_for_admission
        all_pilot_margins_pass = all_pilot_margins_pass and pilot_margins_pass
        all_admission_criteria_pass = all_admission_criteria_pass and admission_criteria_pass
        strata_reports.append(
            {
                "material_id": material,
                "polarization": polarization,
                "incidence_stratum_id": incidence_stratum,
                "incidence_angles_deg": sorted(
                    angles_by_stratum[(material, polarization, incidence_stratum)]
                ),
                "angular_theta_edges_deg": list(angular_grid[0]),
                "angular_phi_edges_deg": list(angular_grid[1]),
                "angular_coordinate_frame_id": angular_grid[2],
                "angular_grid_shape": [len(angular_grid[0]) - 1, len(angular_grid[1]) - 1],
                "test_group_count": group_count,
                "test_sample_count": len(values),
                "independent_measurement_unit_count": measurement_unit_count,
                "equal_unit_median_absolute_power_error_db": float(np.median(equal_unit_errors)),
                "equal_unit_p95_absolute_power_error_db": float(
                    np.percentile(equal_unit_p95_errors, 95)
                ),
                "equal_unit_maximum_angular_cdf_ks": float(np.max(equal_unit_angular_ks)),
                "equal_unit_nominal_95_interval_coverage": coverage,
                "equal_unit_median_prediction_interval_width_db": float(
                    np.median(equal_unit_widths)
                ),
                "measurement_unit_bootstrap_uncertainty": measurement_unit_bootstrap,
                # Compatibility-facing metric names below now use equal-unit estimates;
                # row-pooled values are deliberately excluded from admission.
                "median_absolute_power_error_db": float(np.median(equal_unit_errors)),
                "p95_absolute_power_error_db": float(np.percentile(equal_unit_p95_errors, 95)),
                "maximum_angular_cdf_ks": float(np.max(equal_unit_angular_ks)),
                "nominal_95_interval_coverage": coverage,
                "median_prediction_interval_width_db": float(np.median(equal_unit_widths)),
                "checks": checks,
                "point_estimate_margins_pass": point_margins_pass,
                "uncertainty_checks": uncertainty_checks,
                "uncertainty_margins_pass": uncertainty_pass,
                "pilot_margins_pass": pilot_margins_pass,
                "admission_criteria_pass": admission_criteria_pass,
            }
        )

    return {
        "schema": "loslab.g3_material_calibration_evaluation.v2",
        "status": "MATERIAL_MEASUREMENT_PILOT_ONLY_NOT_RT_SCENE_QUALIFICATION",
        "qualification_status": "NOT_ASSESSABLE_UNVERIFIED_INDEPENDENCE_AND_NO_PRECISION_POWERED_PLAN",
        "target_frequency_hz": TARGET_FREQUENCY_HZ,
        "criteria_source": "Unified Multipath plan G3.03 project-defined margins; not 3GPP requirements",
        "criterion_margins": {
            "median_absolute_integrated_power_error_db_max": MAX_POWER_ERROR_MEDIAN_DB,
            "p95_absolute_integrated_power_error_db_max": MAX_POWER_ERROR_P95_DB,
            "maximum_angular_power_cdf_ks": MAX_ANGULAR_CDF_KS,
            "nominal_95_prediction_interval_coverage_min": MIN_INTERVAL_COVERAGE,
            "nominal_95_prediction_interval_coverage_max": MAX_INTERVAL_COVERAGE,
            "median_prediction_interval_width_db_max": MAX_MEDIAN_INTERVAL_WIDTH_DB,
        },
        "angular_pattern_contract": {
            "coordinate_order": "zenith theta rows, azimuth phi columns",
            "coverage": "complete outgoing sphere: theta 0..180 degrees, phi -180..180 degrees",
            "power_semantics": "integrated linear power per angular bin",
            "solid_angle_validation": "each cell must match delta_phi * (cos(theta_low) - cos(theta_high))",
            "distance": "supremum absolute difference of the normalized 2D cumulative power grids",
            "missing_bins": "not scoreable; no missing bin is substituted with zero",
            "acceptance_limit": MAX_ANGULAR_CDF_KS,
        },
        "material_mapping_id": payload["material_mapping_id"],
        "surface_description": payload["surface_description"],
        "measurement_calibration_id": payload["measurement_calibration_id"],
        "model_id": payload["model_id"],
        "model_sha256": model_hash,
        "uncertainty_description": payload["uncertainty_description"],
        "measurement_unit_independence_description": measurement_unit_description,
        "measurement_unit_independence_status": "UNVERIFIED_EXTERNAL_ATTESTATION_ONLY",
        "measurement_unit_provenance_ids": sorted(measurement_unit_provenance_by_id.values()),
        "split_manifest_sha256": split_manifest_hash,
        "split_completeness_status": "CONDITIONAL_ON_EXTERNALLY_VERIFIED_RECORD",
        "split_manifest_external_verification_id": payload[
            "split_manifest_external_verification_id"
        ],
        "expected_strata_count": len(expected_strata),
        "held_out_incidence_group_count": len(expected),
        "cluster_bootstrap": {
            "resampling_unit": "independent_measurement_unit_id",
            "base_seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
            "all_reported_strata_bootstrap_computable": all(
                row["measurement_unit_bootstrap_uncertainty"][
                    "bootstrap_computable_for_diagnostics"
                ]
                for row in strata_reports
            ),
            "independence_verified_for_admission": independence_verified_for_admission,
            "minimum_units_per_material_polarization_incidence_stratum": MIN_INDEPENDENT_MEASUREMENT_UNITS,
            "minimum_is_pilot_floor_not_full_qualification_sample_size": True,
            "familywise_confidence_level": 0.95,
            "simultaneous_comparison_count": simultaneous_comparisons,
            "multiplicity_adjustment": "Bonferroni",
            "intervals_are_diagnostic_only": True,
            "qualification_requires_separate_precision_powered_preregistration": True,
            "joint_acceptance_family": family_reference,
        },
        "strata": strata_reports,
        "numerical_margins_pass_all_reported_strata": all(
            row["point_estimate_margins_pass"] for row in strata_reports
        ),
        "pilot_margins_pass_all_reported_strata": all_pilot_margins_pass,
        "admission_criteria_pass_all_reported_strata": all_admission_criteria_pass,
        "admission_is_not_assessable_without_minimum_independent_units": True,
        "pilot_margin_passes_are_not_qualification": True,
        "external_independence_review_verified_for_admission": independence_verified_for_admission,
        "measurement_margins_can_qualify_without_external_independence_review": False,
        "complex_csi_qualification": "NOT_EVALUATED_REQUIRES_PHASE_REFERENCED_HELD_OUT_FIELD_DATA",
        "synthetic_fixture_data_can_qualify_scenes": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="frozen G3.03 prediction JSON")
    parser.add_argument("--output", type=Path, help="write evaluation report JSON")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input JSON root must be an object")
    report = evaluate_g3_material_calibration(payload)
    report["prediction_input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
