"""Synthetic contract tests; these fixtures never qualify RT scenes."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
from evaluation.g3_material_calibration import (
    SCHEMA,
    SPLIT_MANIFEST_SCHEMA,
    evaluate_g3_material_calibration,
)


def _refresh_split_manifest_hash(payload: dict[str, object]) -> None:
    split_manifest = payload["full_split_manifest"]
    assert isinstance(split_manifest, dict)
    incidence_groups = payload["incidence_groups"]
    predictions = payload["test_predictions"]
    assert isinstance(incidence_groups, list)
    assert isinstance(predictions, list)
    groups_by_id = {group["incidence_group_id"]: group for group in incidence_groups}
    split_manifest["test_sample_assignments"] = [
        {
            "sample_id": row["sample_id"],
            "incidence_group_id": row["incidence_group_id"],
            "incidence_stratum_id": row["incidence_stratum_id"],
            "incidence_angle_deg": row["incidence_angle_deg"],
            "material_id": row["material_id"],
            "polarization": row["polarization"],
            "measurement_unit_id": groups_by_id[row["incidence_group_id"]]["measurement_unit_id"],
            "measurement_unit_provenance_id": groups_by_id[row["incidence_group_id"]][
                "measurement_unit_provenance_id"
            ],
        }
        for row in predictions
    ]
    encoded = json.dumps(split_manifest, sort_keys=True, separators=(",", ":"))
    payload["full_split_manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()


def _payload() -> dict[str, object]:
    groups = [f"incidence-{group}" for group in range(4)]
    rows: list[dict[str, object]] = []
    manifest = []
    for group in groups:
        sample_ids = [f"{group}-sample-{sample}" for sample in range(10)]
        incidence_stratum = (
            "angle-20deg" if group in {"incidence-0", "incidence-1"} else "angle-40deg"
        )
        incidence_angle = 20.0 if incidence_stratum == "angle-20deg" else 40.0
        manifest.append(
            {
                "incidence_group_id": group,
                "measurement_unit_id": f"acquisition-{group}",
                "measurement_unit_provenance_id": f"provenance-{group}",
                "incidence_stratum_id": incidence_stratum,
                "incidence_angle_deg": incidence_angle,
                "split": "test",
                "sample_ids": sample_ids,
            }
        )
        for index, sample_id in enumerate(sample_ids):
            measured = 10.0 + 0.01 * index
            # The fixture covers the 90% lower bound in the first angle stratum.
            uncovered = group in {"incidence-0", "incidence-1", "incidence-2"} and index == 0
            lower, upper = (
                (measured + 1.0, measured + 2.0) if uncovered else (measured - 1, measured + 1)
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "incidence_group_id": group,
                    "incidence_stratum_id": incidence_stratum,
                    "incidence_angle_deg": incidence_angle,
                    "split": "test",
                    "material_id": "concrete-a",
                    "polarization": "V",
                    "measured_integrated_power_db": measured,
                    "predicted_integrated_power_db": measured + 0.5,
                    "prediction_interval_95_db": [lower, upper],
                    "measured_angular_power": [1.0, 2.0, 1.0],
                    "predicted_angular_power": [1.0, 2.0, 1.0],
                    "angular_bin_edges_deg": [0.0, 60.0, 120.0, 180.0],
                }
            )
    expected_strata = [
        {
            "material_id": "concrete-a",
            "polarization": "V",
            "incidence_stratum_id": angle,
        }
        for angle in ("angle-20deg", "angle-40deg")
    ]
    split_manifest = {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "expected_strata": deepcopy(expected_strata),
        "groups": [
            {
                "incidence_group_id": group["incidence_group_id"],
                "split": "test",
                "sample_ids": list(group["sample_ids"]),
            }
            for group in manifest
        ]
        + [{"incidence_group_id": "fit-group", "split": "train", "sample_ids": ["fit-sample"]}],
        "test_sample_assignments": [],
    }
    groups_by_id = {group["incidence_group_id"]: group for group in manifest}
    split_manifest["test_sample_assignments"] = [
        {
            "sample_id": row["sample_id"],
            "incidence_group_id": row["incidence_group_id"],
            "incidence_stratum_id": row["incidence_stratum_id"],
            "incidence_angle_deg": row["incidence_angle_deg"],
            "material_id": row["material_id"],
            "polarization": row["polarization"],
            "measurement_unit_id": groups_by_id[row["incidence_group_id"]]["measurement_unit_id"],
            "measurement_unit_provenance_id": groups_by_id[row["incidence_group_id"]][
                "measurement_unit_provenance_id"
            ],
        }
        for row in rows
    ]
    split_hash = hashlib.sha256(
        json.dumps(split_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "target_frequency_hz": 3.5e9,
        "model_id": "frozen-model-1",
        "model_sha256": "a" * 64,
        "measurement_calibration_id": "calibration-chain-1",
        "material_mapping_id": "specimen-to-scene-map-1",
        "surface_description": "Concrete coupon, measured roughness documented in calibration record.",
        "uncertainty_description": "Instrument uncertainty and predictive interval method are recorded.",
        "measurement_unit_independence_description": (
            "Each declared acquisition is a separately prepared specimen and independently "
            "calibrated measurement session."
        ),
        "cluster_bootstrap_seed": 41023,
        "cluster_bootstrap_replicates": 2_000,
        "expected_strata": expected_strata,
        "full_split_manifest": split_manifest,
        "full_split_manifest_sha256": split_hash,
        "split_manifest_external_verification_id": "split-record-2026-01",
        "incidence_groups": manifest,
        "test_predictions": rows,
    }


def _assessable_payload(*, failing_coverage: bool = False) -> dict[str, object]:
    payload = _payload()
    original_rows = payload["test_predictions"]
    assert isinstance(original_rows, list)
    group_manifest: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    split_groups: list[dict[str, object]] = []
    for stratum, angle in (("angle-20deg", 20.0), ("angle-40deg", 40.0)):
        template = next(row for row in original_rows if row["incidence_stratum_id"] == stratum)
        for unit_index in range(20):
            group_id = f"{stratum}-unit-{unit_index}"
            sample_ids = [f"{group_id}-sample-{sample_index}" for sample_index in range(20)]
            group_manifest.append(
                {
                    "incidence_group_id": group_id,
                    "measurement_unit_id": f"measurement-unit-{stratum}-{unit_index}",
                    "measurement_unit_provenance_id": f"source-record-{stratum}-{unit_index}",
                    "incidence_stratum_id": stratum,
                    "incidence_angle_deg": angle,
                    "split": "test",
                    "sample_ids": sample_ids,
                }
            )
            split_groups.append(
                {
                    "incidence_group_id": group_id,
                    "split": "test",
                    "sample_ids": list(sample_ids),
                }
            )
            covered_count = 18 if unit_index < 10 else (16 if failing_coverage else 20)
            for sample_index, sample_id in enumerate(sample_ids):
                row = deepcopy(template)
                measured = 10.0 + 0.001 * sample_index
                row.update(
                    sample_id=sample_id,
                    incidence_group_id=group_id,
                    incidence_stratum_id=stratum,
                    incidence_angle_deg=angle,
                    measured_integrated_power_db=measured,
                    predicted_integrated_power_db=measured + 0.5,
                    prediction_interval_95_db=(
                        [measured - 1.0, measured + 1.0]
                        if sample_index < covered_count
                        else [measured + 1.0, measured + 2.0]
                    ),
                )
                rows.append(row)
    split_groups.append(
        {"incidence_group_id": "fit-group", "split": "train", "sample_ids": ["fit-sample"]}
    )
    split_manifest = {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "expected_strata": payload["expected_strata"],
        "groups": split_groups,
        "test_sample_assignments": [],
    }
    payload["incidence_groups"] = group_manifest
    payload["test_predictions"] = rows
    payload["full_split_manifest"] = split_manifest
    _refresh_split_manifest_hash(payload)
    return payload


def test_synthetic_fixture_meets_numeric_contract_but_never_qualifies_scene():
    report = evaluate_g3_material_calibration(_payload())

    assert report["numerical_margins_pass_all_reported_strata"] is True
    assert report["pilot_margins_pass_all_reported_strata"] is False
    assert report["status"] == "MATERIAL_MEASUREMENT_PILOT_ONLY_NOT_RT_SCENE_QUALIFICATION"
    assert report["qualification_status"].startswith("NOT_ASSESSABLE")
    assert report["pilot_margin_passes_are_not_qualification"] is True
    assert report["split_completeness_status"] == "CONDITIONAL_ON_EXTERNALLY_VERIFIED_RECORD"
    assert report["measurement_unit_independence_status"] == "UNVERIFIED_EXTERNAL_ATTESTATION_ONLY"
    assert report["synthetic_fixture_data_can_qualify_scenes"] is False
    assert report["complex_csi_qualification"].startswith("NOT_EVALUATED")
    assert report["criterion_margins"]["median_absolute_integrated_power_error_db_max"] == 1.5
    stratum = report["strata"][0]
    assert stratum["test_group_count"] == 2
    assert stratum["test_sample_count"] == 20
    assert stratum["incidence_stratum_id"] == "angle-20deg"
    assert stratum["incidence_angles_deg"] == [20.0]
    assert stratum["median_absolute_power_error_db"] == pytest.approx(0.5)
    assert stratum["maximum_angular_cdf_ks"] == pytest.approx(0.0)
    assert stratum["nominal_95_interval_coverage"] == pytest.approx(0.90)
    assert stratum["measurement_unit_bootstrap_uncertainty"]["resampling_unit"] == (
        "independent_measurement_unit_id"
    )
    assert stratum["measurement_unit_bootstrap_uncertainty"]["independent_unit_count"] == 2
    assert (
        stratum["measurement_unit_bootstrap_uncertainty"]["bootstrap_computable_for_diagnostics"]
        is False
    )
    assert (
        stratum["measurement_unit_bootstrap_uncertainty"][
            "minimum_is_pilot_floor_not_full_qualification_sample_size"
        ]
        is True
    )
    assert stratum["pilot_margins_pass"] is False
    assert stratum["measurement_unit_bootstrap_uncertainty"]["percentile_95_ci"] is None


def test_material_and_polarization_are_evaluated_as_separate_strata():
    payload = _payload()
    rows = payload["test_predictions"]
    assert isinstance(rows, list)
    for row in rows[:10]:
        row["polarization"] = "H"
        row["material_id"] = "glass-b"
    expected_strata = payload["expected_strata"]
    assert isinstance(expected_strata, list)
    expected_strata.append(
        {
            "material_id": "glass-b",
            "polarization": "H",
            "incidence_stratum_id": "angle-20deg",
        }
    )
    split_manifest = payload["full_split_manifest"]
    assert isinstance(split_manifest, dict)
    split_manifest["expected_strata"] = deepcopy(expected_strata)
    _refresh_split_manifest_hash(payload)

    report = evaluate_g3_material_calibration(payload)

    assert {
        (row["material_id"], row["polarization"], row["incidence_stratum_id"])
        for row in report["strata"]
    } == {
        ("concrete-a", "V", "angle-20deg"),
        ("concrete-a", "V", "angle-40deg"),
        ("glass-b", "H", "angle-20deg"),
    }
    assert report["numerical_margins_pass_all_reported_strata"] is True
    assert report["pilot_margins_pass_all_reported_strata"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update(schema="wrong"), "schema"),
        (lambda data: data.update(target_frequency_hz=8e9), "fixed at 3.5 GHz"),
        (lambda data: data.update(model_sha256="not-a-hash"), "SHA-256"),
        (lambda data: data.update(cluster_bootstrap_seed=True), "nonnegative integer"),
        (lambda data: data.update(cluster_bootstrap_replicates=999), "at least 1000"),
        (lambda data: data.update(expected_strata=[]), "expected_strata"),
        (lambda data: data.update(full_split_manifest_sha256="f" * 64), "does not match canonical"),
        (
            lambda data: data["expected_strata"].pop(),
            "exactly match the externally verified split manifest",
        ),
        (lambda data: data["incidence_groups"][0].update(split="fit"), "held-out test groups"),
        (lambda data: data["test_predictions"].pop(), "incomplete"),
    ],
)
def test_invalid_or_incomplete_evidence_fails_closed(mutate, message):
    payload = deepcopy(_payload())
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        evaluate_g3_material_calibration(payload)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ("invalid", "each split expected stratum must be an object"),
        ("duplicate", "duplicate split expected stratum"),
    ],
)
def test_split_manifest_rejects_malformed_or_duplicate_expected_strata(entry, message):
    payload = deepcopy(_payload())
    split_manifest = payload["full_split_manifest"]
    assert isinstance(split_manifest, dict)
    strata = split_manifest["expected_strata"]
    assert isinstance(strata, list)
    strata.append("invalid" if entry == "invalid" else deepcopy(strata[0]))
    _refresh_split_manifest_hash(payload)

    with pytest.raises(ValueError, match=message):
        evaluate_g3_material_calibration(payload)


def test_angular_distribution_mismatch_fails_angular_margin():
    payload = _payload()
    rows = payload["test_predictions"]
    assert isinstance(rows, list)
    for row in rows:
        row["predicted_angular_power"] = [0.0, 0.0, 1.0]

    report = evaluate_g3_material_calibration(payload)

    assert report["strata"][0]["maximum_angular_cdf_ks"] == pytest.approx(0.75)
    assert report["strata"][0]["checks"]["maximum_angular_cdf_ks"] is False
    assert report["numerical_margins_pass_all_reported_strata"] is False


def test_each_incidence_angle_stratum_must_pass_independently():
    payload = _payload()
    rows = payload["test_predictions"]
    assert isinstance(rows, list)
    for row in rows:
        if row["incidence_stratum_id"] == "angle-40deg":
            row["predicted_angular_power"] = [0.0, 0.0, 1.0]

    report = evaluate_g3_material_calibration(payload)
    by_angle = {row["incidence_stratum_id"]: row for row in report["strata"]}

    assert by_angle["angle-20deg"]["checks"]["maximum_angular_cdf_ks"] is True
    assert by_angle["angle-40deg"]["checks"]["maximum_angular_cdf_ks"] is False
    assert report["numerical_margins_pass_all_reported_strata"] is False


def test_coverage_width_and_power_error_margins_are_independently_reported():
    payload = _payload()
    rows = payload["test_predictions"]
    assert isinstance(rows, list)
    for row in rows:
        row["predicted_integrated_power_db"] = row["measured_integrated_power_db"] + 4.0
        measured = row["measured_integrated_power_db"]
        if row["incidence_group_id"] not in {
            "incidence-0",
            "incidence-1",
            "incidence-2",
        } or not row["sample_id"].endswith("-sample-0"):
            row["prediction_interval_95_db"] = [measured - 1.0, measured + 5.0]

    report = evaluate_g3_material_calibration(payload)
    checks = report["strata"][0]["checks"]

    assert checks["median_absolute_power_error_db"] is False
    assert checks["p95_absolute_power_error_db"] is False
    assert checks["nominal_95_interval_coverage"] is True
    assert checks["median_prediction_interval_width_db"] is False


def test_mixed_angular_grid_and_invalid_powers_are_rejected():
    payload = _payload()
    rows = payload["test_predictions"]
    assert isinstance(rows, list)
    rows[1]["measured_angular_power"] = [1.0, 2.0]
    rows[1]["angular_bin_edges_deg"] = [0.0, 90.0, 180.0]

    with pytest.raises(ValueError, match="angular bin count varies"):
        evaluate_g3_material_calibration(payload)

    rows[1]["measured_angular_power"] = [1.0, 2.0, 1.0]
    rows[1]["angular_bin_edges_deg"] = [0.0, 50.0, 120.0, 180.0]
    with pytest.raises(ValueError, match="angular bin edges vary"):
        evaluate_g3_material_calibration(payload)

    rows[1]["measured_angular_power"] = [-1.0, 2.0, 1.0]
    rows[1]["angular_bin_edges_deg"] = [0.0, 60.0, 120.0, 180.0]
    with pytest.raises(ValueError, match="angular powers"):
        evaluate_g3_material_calibration(payload)


def test_bootstrap_counts_independent_groups_not_repeated_rows():
    payload = _payload()
    report = evaluate_g3_material_calibration(payload)
    stratum = next(row for row in report["strata"] if row["incidence_stratum_id"] == "angle-20deg")
    rows = payload["test_predictions"]
    assert isinstance(rows, list)
    repeated = deepcopy(rows[0])
    repeated["sample_id"] = "extra-independent-looking-row"
    repeated["measured_integrated_power_db"] = 10.0
    repeated["predicted_integrated_power_db"] = 10.5
    repeated["prediction_interval_95_db"] = [9.0, 11.0]
    rows.append(repeated)
    manifest = payload["incidence_groups"]
    assert isinstance(manifest, list)
    first_group = next(item for item in manifest if item["incidence_group_id"] == "incidence-0")
    first_group["sample_ids"].append("extra-independent-looking-row")
    split_manifest = payload["full_split_manifest"]
    assert isinstance(split_manifest, dict)
    split_groups = split_manifest["groups"]
    assert isinstance(split_groups, list)
    split_first_group = next(
        item for item in split_groups if item["incidence_group_id"] == "incidence-0"
    )
    split_first_group["sample_ids"].append("extra-independent-looking-row")
    _refresh_split_manifest_hash(payload)

    repeated_report = evaluate_g3_material_calibration(payload)
    repeated_stratum = next(
        row for row in repeated_report["strata"] if row["incidence_stratum_id"] == "angle-20deg"
    )
    bootstrap = repeated_stratum["measurement_unit_bootstrap_uncertainty"]

    assert repeated_stratum["test_sample_count"] == stratum["test_sample_count"] + 1
    assert repeated_stratum["test_group_count"] == stratum["test_group_count"]
    assert bootstrap["independent_unit_count"] == 2
    assert bootstrap["resampling_unit"] == "independent_measurement_unit_id"


def test_single_held_out_group_is_reported_as_uncertainty_unassessable():
    payload = _payload()
    payload["incidence_groups"] = [
        group
        for group in payload["incidence_groups"]
        if group["incidence_group_id"] != "incidence-1"
    ]
    payload["test_predictions"] = [
        row for row in payload["test_predictions"] if row["incidence_group_id"] != "incidence-1"
    ]
    split_manifest = payload["full_split_manifest"]
    assert isinstance(split_manifest, dict)
    split_manifest["groups"] = [
        group for group in split_manifest["groups"] if group["incidence_group_id"] != "incidence-1"
    ]
    split_manifest["test_sample_assignments"] = [
        assignment
        for assignment in split_manifest["test_sample_assignments"]
        if assignment["incidence_group_id"] != "incidence-1"
    ]
    encoded = json.dumps(split_manifest, sort_keys=True, separators=(",", ":"))
    payload["full_split_manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()

    report = evaluate_g3_material_calibration(payload)
    angle_20 = next(row for row in report["strata"] if row["incidence_stratum_id"] == "angle-20deg")
    bootstrap = angle_20["measurement_unit_bootstrap_uncertainty"]

    assert bootstrap["bootstrap_computable_for_diagnostics"] is False
    assert bootstrap["percentile_95_ci"] is None
    assert report["cluster_bootstrap"]["all_reported_strata_bootstrap_computable"] is False


def test_full_split_manifest_must_include_every_declared_test_group():
    payload = _payload()
    split_manifest = payload["full_split_manifest"]
    assert isinstance(split_manifest, dict)
    split_manifest["groups"] = [
        group for group in split_manifest["groups"] if group["incidence_group_id"] != "incidence-1"
    ]
    split_manifest["test_sample_assignments"] = [
        assignment
        for assignment in split_manifest["test_sample_assignments"]
        if assignment["incidence_group_id"] != "incidence-1"
    ]
    encoded = json.dumps(split_manifest, sort_keys=True, separators=(",", ":"))
    payload["full_split_manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()

    with pytest.raises(ValueError, match="exactly match all test groups"):
        evaluate_g3_material_calibration(payload)


def test_assessable_bootstrap_uses_familywise_bonferroni_intervals():
    report = evaluate_g3_material_calibration(_assessable_payload())

    assert report["cluster_bootstrap"]["all_reported_strata_bootstrap_computable"] is True
    assert report["cluster_bootstrap"]["familywise_confidence_level"] == 0.95
    assert report["cluster_bootstrap"]["simultaneous_comparison_count"] == 10
    assert report["cluster_bootstrap"]["intervals_are_diagnostic_only"] is True
    for stratum in report["strata"]:
        bootstrap = stratum["measurement_unit_bootstrap_uncertainty"]
        assert bootstrap["bootstrap_computable_for_diagnostics"] is True
        assert bootstrap["independent_unit_count"] == 20
        assert bootstrap["per_comparison_alpha"] == pytest.approx(0.005)
        assert bootstrap["per_comparison_confidence_level"] == pytest.approx(0.995)
        assert bootstrap["interpretation"].startswith("DIAGNOSTIC_ONLY")
        assert stratum["uncertainty_checks"] is not None
        assert stratum["uncertainty_margins_pass"] is True
        assert stratum["pilot_margins_pass"] is False
    assert report["pilot_margins_pass_all_reported_strata"] is False
    assert report["cluster_bootstrap"]["independence_verified_for_admission"] is False
    assert report["qualification_status"].startswith("NOT_ASSESSABLE")


def test_assessable_bootstrap_rejects_coverage_below_frozen_margin():
    report = evaluate_g3_material_calibration(_assessable_payload(failing_coverage=True))
    angle_20 = next(row for row in report["strata"] if row["incidence_stratum_id"] == "angle-20deg")

    assert (
        angle_20["measurement_unit_bootstrap_uncertainty"]["bootstrap_computable_for_diagnostics"]
        is True
    )
    assert angle_20["uncertainty_checks"] is not None
    assert angle_20["uncertainty_checks"]["nominal_95_interval_coverage_ci"] is False
    assert angle_20["pilot_margins_pass"] is False


def test_large_favorable_unit_cannot_hide_a_failing_independent_unit():
    payload = _payload()
    manifest = payload["incidence_groups"]
    rows = payload["test_predictions"]
    assert isinstance(manifest, list)
    assert isinstance(rows, list)
    group_ids = {"incidence-0", "incidence-1"}
    group0 = next(item for item in manifest if item["incidence_group_id"] == "incidence-0")
    group1 = next(item for item in manifest if item["incidence_group_id"] == "incidence-1")
    rows[:] = [row for row in rows if row["incidence_group_id"] not in group_ids]
    group0_ids: list[str] = []
    group1_ids: list[str] = []
    original_rows = _payload()["test_predictions"]
    base0 = next(row for row in original_rows if row["incidence_group_id"] == "incidence-0")
    base1 = next(row for row in original_rows if row["incidence_group_id"] == "incidence-1")
    # Give one independent unit 100 covered observations and the other 3
    # uncovered observations. A pooled row metric would report 97.1% coverage.
    for group, base, count, covered, sample_ids in (
        ("incidence-0", base0, 100, True, group0_ids),
        ("incidence-1", base1, 3, False, group1_ids),
    ):
        for index in range(count):
            row = deepcopy(base)
            sample_id = f"{group}-unequal-{index}"
            row["sample_id"] = sample_id
            measured = 10.0 + index / 100.0
            row["measured_integrated_power_db"] = measured
            row["predicted_integrated_power_db"] = measured + 0.5
            row["prediction_interval_95_db"] = (
                [measured - 1.0, measured + 1.0] if covered else [measured + 1.0, measured + 2.0]
            )
            rows.append(row)
            sample_ids.append(sample_id)
    group0["sample_ids"] = group0_ids
    group1["sample_ids"] = group1_ids
    split_manifest = payload["full_split_manifest"]
    assert isinstance(split_manifest, dict)
    split_groups = split_manifest["groups"]
    assert isinstance(split_groups, list)
    for group_id, sample_ids in (("incidence-0", group0_ids), ("incidence-1", group1_ids)):
        split_group = next(item for item in split_groups if item["incidence_group_id"] == group_id)
        split_group["sample_ids"] = list(sample_ids)
    _refresh_split_manifest_hash(payload)

    report = evaluate_g3_material_calibration(payload)
    angle_20 = next(row for row in report["strata"] if row["incidence_stratum_id"] == "angle-20deg")

    assert angle_20["test_sample_count"] == 103
    assert angle_20["nominal_95_interval_coverage"] == pytest.approx(0.5)
    assert angle_20["checks"]["nominal_95_interval_coverage"] is False
    assert angle_20["pilot_margins_pass"] is False
