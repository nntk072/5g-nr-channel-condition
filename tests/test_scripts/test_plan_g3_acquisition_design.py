"""Tests for the reproducible G3 acquisition-design planning CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from evaluation.g3_joint_acceptance import (
    build_g3_joint_acceptance_family,
    build_g3_joint_acceptance_family_v2,
    build_g3_joint_acceptance_family_v3,
    build_g3_joint_acceptance_family_v4,
)
from scripts.plan_g3_acquisition_design import _complete_design_status, main

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "datasets/schema/g3.03-acquisition-design-assumptions.v3.schema.json"
EXAMPLE = (
    ROOT / "datasets/schema/examples/g3.03-acquisition-design-assumptions.synthetic-example.json"
)
V5_EXAMPLE = (
    ROOT / "datasets/schema/examples/g3.03-acquisition-design-assumptions.v5.synthetic-example.json"
)


def test_v5_status_requires_joint_composition_before_reporting_a_complete_assurance_design() -> None:
    status = _complete_design_status(
        schema_version="loslab.g3_acquisition_design_assumptions.v5",
        missing_designs=[],
        material={"selected_units_per_stratum_among_evaluated_candidates": None},
        complex_csi={"selected_clusters_per_stratum_among_evaluated_candidates": 40},
        receiver_impact={"selected_clusters_per_stratum_among_evaluated_candidates": 50},
        propagation={"selected_clusters_per_stratum_among_evaluated_candidates": 60},
    )

    assert status == "JOINT_SCENARIO_ASSURANCE_COMPOSITION_UNAVAILABLE"


def _build_v5_synthetic_configuration() -> dict[str, Any]:
    configuration = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    material_strata = [
        (item["material_id"], item["polarization"], item["incidence_stratum_id"])
        for item in configuration["expected_strata"]
    ]
    family = build_g3_joint_acceptance_family_v4(
        family_id="synthetic-v5-acquisition-design-example",
        material_expected_strata=material_strata,
        complex_csi_expected_strata=[("urban", "LOS")],
        propagation_expected_strata=[("urban", "LOS", "total")],
        receiver_impact_expected_strata=[("urban", "LOS")],
        path_equivalence_expected_scenarios=[("urban", "LOS", "episode-1")],
    )
    configuration["schema_version"] = "loslab.g3_acquisition_design_assumptions.v5"
    configuration["joint_acceptance_family"] = family
    configuration["complex_csi_bootstrap_assurance"]["scenarios_by_stratum"] = [
        {
            "family_id": "urban",
            "condition": "LOS",
            "assumed_iid_scenario": {
                "assumed_cluster_summaries": [
                    {
                        "cluster_median_e_h": 0.005,
                        "cluster_p95_e_h": 0.02,
                        "cluster_failed": False,
                    }
                ],
                "probabilities": [1.0],
            },
        }
    ]
    configuration["complex_csi_bootstrap_assurance"]["candidate_clusters_per_stratum"] = [20]
    configuration["receiver_impact_bootstrap_assurance"] = {
        "scenarios_by_stratum": [
            {
                "family_id": "urban",
                "condition": "LOS",
                "assumed_iid_scenario": {
                    "assumed_cluster_outcomes": [
                        {
                            "candidate_crc_errors": [[True, False, False]],
                            "challenger_crc_errors": [[True, False, False]],
                        }
                    ],
                    "probabilities": [1.0],
                },
            }
        ],
        "candidate_clusters_per_stratum": [20],
        "snr_grid_db": [0.0, 1.0, 2.0],
        "comparison_snr_db": 0.0,
        "bootstrap_replicates": 1000,
        "assurance_replicates": 100,
        "target_assurance": 0.8,
        "monte_carlo_confidence": 0.95,
        "seed": 17,
    }
    metric_names = (
        "first_arrival_delay_s",
        "rms_delay_spread_s",
        "arrival_angular_spread_deg",
        "departure_angular_spread_deg",
    )
    configuration["worst_stratum_assumptions"].update(
        {
            "planned_propagation_p95_pass_probabilities": {metric: 0.99 for metric in metric_names},
            "planned_propagation_cluster_failure_probabilities": {
                metric: 0.01 for metric in metric_names
            },
        }
    )
    values = {
        "first_arrival_delay_s": 120e-9,
        "rms_delay_spread_s": 80e-9,
        "arrival_angular_spread_deg": 20.0,
        "departure_angular_spread_deg": 25.0,
    }
    metrics = {
        name: {
            "candidate": value,
            "challenger": value,
            "availability": "resolved",
            "reason": None,
        }
        for name, value in values.items()
    }
    configuration["propagation_bootstrap_assurance"] = {
        "scenarios_by_stratum": [
            {
                "family_id": "urban",
                "condition": "LOS",
                "component": "total",
                "assumed_iid_scenario": {
                    "assumed_cluster_episodes": [{"episodes": [{"metrics": metrics}]}],
                    "probabilities": [1.0],
                },
            }
        ],
        "candidate_clusters_per_stratum": [125],
        "effective_bandwidth_hz": 100e6,
        "bootstrap_draws": 1000,
        "assurance_replicates": 100,
        "target_assurance": 0.8,
        "monte_carlo_confidence": 0.95,
        "seed": 19,
    }
    return configuration


def test_synthetic_example_matches_schema_and_emits_hash_bound_nonqualifying_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configuration_bytes = EXAMPLE.read_bytes()
    configuration = json.loads(configuration_bytes)
    jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(
        configuration
    )
    output = tmp_path / "design-report.json"

    assert main([str(EXAMPLE), "--output", str(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "SCENARIO_CONDITIONAL_PLANNING_ONLY_NOT_QUALIFICATION"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["admission_possible"] is False
    assert report["independent_unit_design"]["uses_measurement_or_pilot_data"] is False
    assert report["independent_unit_design"]["complete_g3_sample_size"] is None
    assert report["bootstrap_assurance_design"]["admission_possible"] is False
    claim_count = report["joint_acceptance_family"]["claim_count"]
    assert report["bootstrap_assurance_design"]["familywise_claim_count"] == claim_count
    assert report["bootstrap_assurance_design"]["per_comparison_alpha"] == pytest.approx(
        0.05 / claim_count
    )
    assert report["complex_csi_bootstrap_assurance_design"]["admission_possible"] is False
    assert (
        report["complex_csi_bootstrap_assurance_design"]["joint_acceptance_family"]["claim_count"]
        == claim_count
    )
    csi_design = report["complex_csi_bootstrap_assurance_design"]
    assert csi_design["selected_clusters_per_stratum_among_evaluated_candidates"] == 120
    assert [item["meets_target_assurance"] for item in csi_design["candidate_designs"]] == [
        False,
        True,
    ]
    assert (
        report["provenance"]["input_exact_bytes_sha256"]
        == hashlib.sha256(configuration_bytes).hexdigest()
    )
    assert len(report["provenance"]["planner_source_sha256"]) == 64
    assert len(report["provenance"]["runner_source_sha256"]) == 64
    assert capsys.readouterr().out == ""


def test_cli_preserves_missing_angular_and_bootstrap_assumptions_as_unresolved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configuration = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    configuration["worst_stratum_assumptions"]["planned_angular_unit_pass_probability"] = None
    configuration["bootstrap_assurance"] = None
    configuration["complex_csi_bootstrap_assurance"] = None
    input_path = tmp_path / "unresolved.json"
    input_path.write_text(json.dumps(configuration), encoding="utf-8")

    assert main([str(input_path)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["bootstrap_assurance_design"] is None
    assert (
        report["independent_unit_design"]["material_design"]["angular_maximum_claim"][
            "assumption_supplied"
        ]
        is False
    )
    assert report["complete_design_status"] == "MATERIAL_AND_CSI_BOOTSTRAP_ASSURANCE_NOT_SUPPLIED"


def test_v4_schema_and_cli_report_evaluator_aligned_receiver_assurance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configuration = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    configuration["schema_version"] = "loslab.g3_acquisition_design_assumptions.v4"
    material_strata = [
        (item["material_id"], item["polarization"], item["incidence_stratum_id"])
        for item in configuration["expected_strata"]
    ]
    configuration["joint_acceptance_family"] = build_g3_joint_acceptance_family_v3(
        family_id="v4-cli-test",
        material_expected_strata=material_strata,
        complex_csi_expected_strata=[("urban", "LOS")],
        propagation_expected_strata=[("urban", "LOS", "total")],
        receiver_impact_expected_strata=[("urban", "LOS")],
    )
    configuration["complex_csi_bootstrap_assurance"] = None
    metrics = {
        "first_arrival_delay_s",
        "rms_delay_spread_s",
        "arrival_angular_spread_deg",
        "departure_angular_spread_deg",
    }
    configuration["worst_stratum_assumptions"].update(
        {
            "planned_propagation_p95_pass_probabilities": {metric: 0.99 for metric in metrics},
            "planned_propagation_cluster_failure_probabilities": {
                metric: 0.01 for metric in metrics
            },
        }
    )
    configuration["receiver_impact_bootstrap_assurance"] = {
        "scenarios_by_stratum": [
            {
                "family_id": "urban",
                "condition": "LOS",
                "assumed_iid_scenario": {
                    "assumed_cluster_outcomes": [
                        {
                            "candidate_crc_errors": [[True, True, True]] * 3,
                            "challenger_crc_errors": [[True, True, True]] * 3,
                        }
                    ],
                    "probabilities": [1.0],
                },
            }
        ],
        "candidate_clusters_per_stratum": [20],
        "snr_grid_db": [0.0, 1.0, 2.0],
        "comparison_snr_db": 1.0,
        "bootstrap_replicates": 1000,
        "assurance_replicates": 100,
        "target_assurance": 0.8,
        "monte_carlo_confidence": 0.95,
        "seed": 12,
    }
    schema = json.loads(
        (ROOT / "datasets/schema/g3.03-acquisition-design-assumptions.v4.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(configuration)
    input_path = tmp_path / "v4-plan.json"
    input_path.write_text(json.dumps(configuration), encoding="utf-8")

    assert main([str(input_path)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "loslab.g3_acquisition_design_report.v4"
    assert report["joint_acceptance_family"]["claim_count"] == 24
    receiver = report["receiver_impact_bootstrap_assurance_design"]
    assert receiver["admission_possible"] is False
    assert receiver["joint_acceptance_family"]["claim_count"] == 24
    assert receiver["candidate_designs"][0]["stratum_assurance"][0]["joint_pass_count"] == 0
    assert (
        report["receiver_impact_assurance_status"]
        == "NO_EVALUATED_CANDIDATE_MEETS_TARGET_ASSURANCE"
    )
    assert report["complete_design_status"] == "ASSURANCE_DESIGNS_MISSING:complex_csi"
    assert report["independent_unit_design"]["complete_g3_sample_size"] is None


def test_cli_plans_v2_propagation_claims_with_shared_alpha_and_keeps_assurance_open(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configuration = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    v1 = configuration["joint_acceptance_family"]
    family = build_g3_joint_acceptance_family_v2(
        family_id="v2-planning-protocol",
        material_expected_strata=[tuple(item) for item in v1["material_expected_strata"]],
        complex_csi_expected_strata=[tuple(item) for item in v1["complex_csi_expected_strata"]],
        propagation_expected_strata=[("example-site", "LOS", "total")],
    )
    configuration["joint_acceptance_family"] = family
    assumptions = configuration["worst_stratum_assumptions"]
    metric_names = (
        "first_arrival_delay_s",
        "rms_delay_spread_s",
        "arrival_angular_spread_deg",
        "departure_angular_spread_deg",
    )
    assumptions["planned_propagation_p95_pass_probabilities"] = {
        metric: 0.99 for metric in metric_names
    }
    assumptions["planned_propagation_cluster_failure_probabilities"] = {
        metric: 0.01 for metric in metric_names
    }
    configuration["bootstrap_assurance"] = None
    configuration["complex_csi_bootstrap_assurance"] = None
    jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(
        configuration
    )
    input_path = tmp_path / "v2-plan.json"
    input_path.write_text(json.dumps(configuration), encoding="utf-8")

    assert main([str(input_path)]) == 0

    report = json.loads(capsys.readouterr().out)
    propagation = report["independent_unit_design"]["propagation_design"]
    assert propagation["claim_count"] == 8
    assert propagation["stratum_count"] == 1
    assert propagation["required_independent_clusters_per_stratum_for_supported_claims"]
    assert report["independent_unit_design"]["complete_g3_sample_size"] is None
    assert (
        report["propagation_assurance_status"]
        == "EVALUATOR_ALIGNED_BOOTSTRAP_ASSURANCE_NOT_SUPPLIED"
    )


def test_v5_cli_runs_evaluator_aligned_propagation_assurance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configuration = json.loads(V5_EXAMPLE.read_text(encoding="utf-8"))
    assert configuration == _build_v5_synthetic_configuration()
    schema_path = ROOT / "datasets/schema/g3.03-acquisition-design-assumptions.v5.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(configuration)
    input_path = tmp_path / "v5-plan.json"
    input_path.write_text(json.dumps(configuration), encoding="utf-8")

    assert main([str(input_path)]) == 0

    report = json.loads(capsys.readouterr().out)
    propagation = report["propagation_bootstrap_assurance_design"]
    assert report["schema"] == "loslab.g3_acquisition_design_report.v6"
    joint_design = report["joint_sample_size_design_composition"]
    assert joint_design["qualification_status"] == "NOT_QUALIFIED"
    assert joint_design["single_combined_sample_size_is_reported"] is False
    assert joint_design["branch_designs"]["path_equivalence"]["sampling_unit"] == (
        "physical_episode"
    )
    assert report["joint_acceptance_family"]["claim_count"] == 24
    assert propagation["joint_acceptance_family"]["claim_count"] == 24
    assert report["bootstrap_assurance_design"]["familywise_claim_count"] == 24
    assert report["bootstrap_assurance_design"]["joint_acceptance_family"] == {
        "family_id": report["joint_acceptance_family"]["family_id"],
        "family_sha256": report["joint_acceptance_family"]["family_sha256"],
    }
    assert {
        tuple(value)
        for value in report["bootstrap_assurance_design"]["stratum_roster_by_id"].values()
    } == {
        tuple(row["stratum"])
        for row in report["joint_sample_size_design_composition"]["branch_designs"][
            "material_calibration"
        ]["required_units_by_stratum"]
    }
    assert (
        report["complex_csi_bootstrap_assurance_design"]["joint_acceptance_family"]["claim_count"]
        == 24
    )
    assert (
        report["receiver_impact_bootstrap_assurance_design"]["joint_acceptance_family"][
            "claim_count"
        ]
        == 24
    )
    assert propagation["candidate_designs"][0]["stratum_assurance"][0]["joint_pass_count"] == 100
    assert (
        report["receiver_impact_bootstrap_assurance_design"]["candidate_designs"][0][
            "stratum_assurance"
        ][0]["joint_pass_count"]
        == 100
    ), json.dumps(
        report["receiver_impact_bootstrap_assurance_design"]["candidate_designs"][0][
            "stratum_assurance"
        ][0],
        indent=2,
    )
    assert (
        report["propagation_assurance_status"]
        == "SCENARIO_CONDITIONAL_BOOTSTRAP_ASSURANCE_REPORTED"
    )
    assert report["independent_unit_design"]["complete_g3_sample_size"] is None
    assert report["complete_design_status"] == (
        "SCENARIO_CONDITIONAL_POWERED_G3_DESIGN_COMPLETE_NOT_QUALIFIED"
        if joint_design["complete_precision_powered_g3_design"]
        else "SCENARIO_CONDITIONAL_G3_DESIGN_INCOMPLETE"
    )
    assert len(report["provenance"]["propagation_assurance_source_sha256"]) == 64
    assert len(report["provenance"]["joint_design_composition_schema_sha256"]) == 64
    assert set(report["provenance"]["assurance_source_groups"]) == {
        "material_calibration",
        "complex_csi",
        "propagation_metrics",
        "receiver_impact",
    }
    assert all(
        len(value["bundle_sha256"]) == 64
        and all(len(digest) == 64 for digest in value["files"].values())
        for value in report["provenance"]["assurance_source_groups"].values()
    )


def test_cli_rejects_duplicate_json_keys_without_returning_a_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "duplicate-key.json"
    input_path.write_text(
        '{"schema_version":"loslab.g3_acquisition_design_assumptions.v3",'
        '"schema_version":"loslab.g3_acquisition_design_assumptions.v3"}',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as error:
        main([str(input_path)])

    assert error.value.code == 2
    assert "duplicate JSON object key" in capsys.readouterr().err


def test_cli_reports_exact_location_for_malformed_stratum(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configuration = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    configuration["expected_strata"][0]["material_id"] = " material-a"
    input_path = tmp_path / "malformed-stratum.json"
    input_path.write_text(json.dumps(configuration), encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main([str(input_path)])

    assert error.value.code == 2
    assert "expected_strata[0].material_id must be nonempty and trimmed" in capsys.readouterr().err


def test_cli_does_not_allow_report_to_overwrite_assumptions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "configuration.json"
    input_path.write_bytes(EXAMPLE.read_bytes())

    with pytest.raises(SystemExit) as error:
        main([str(input_path), "--output", str(input_path)])

    assert error.value.code == 2
    assert "must not overwrite" in capsys.readouterr().err
    assert input_path.read_bytes() == EXAMPLE.read_bytes()


def test_cli_rejects_alpha_that_weakens_frozen_95_percent_familywise_criteria(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configuration = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    configuration["worst_stratum_assumptions"]["familywise_alpha"] = 0.10
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(configuration)
    input_path = tmp_path / "weakened-alpha.json"
    input_path.write_text(json.dumps(configuration), encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main([str(input_path)])

    assert error.value.code == 2
    assert "must remain 0.05" in capsys.readouterr().err


def test_cli_rejects_bootstrap_candidates_below_evaluator_unit_floor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configuration = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    configuration["bootstrap_assurance"]["candidate_units_per_stratum"] = [19, 30]
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(configuration)
    input_path = tmp_path / "below-unit-floor.json"
    input_path.write_text(json.dumps(configuration), encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main([str(input_path)])

    assert error.value.code == 2
    assert "evaluator floor of 20" in capsys.readouterr().err


def test_cli_rejects_joint_family_roster_that_differs_from_material_design(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configuration = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    configuration["joint_acceptance_family"] = build_g3_joint_acceptance_family(
        family_id="different-protocol",
        material_expected_strata=[("omitted-material", "V", "incidence-x")],
        complex_csi_expected_strata=[("example-site", "LOS"), ("example-site", "NLOS")],
    )
    input_path = tmp_path / "mismatched-family.json"
    input_path.write_text(json.dumps(configuration), encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main([str(input_path)])

    assert error.value.code == 2
    assert "material strata differ from expected_strata" in capsys.readouterr().err


def test_cli_rejects_csi_bootstrap_scenario_missing_joint_family_stratum(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configuration = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    configuration["complex_csi_bootstrap_assurance"]["scenarios_by_stratum"].pop()
    input_path = tmp_path / "missing-csi-stratum.json"
    input_path.write_text(json.dumps(configuration), encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main([str(input_path)])

    assert error.value.code == 2
    assert "exactly every joint-family CSI stratum" in capsys.readouterr().err


def test_v5_schema_and_planner_accept_joint_family_v4(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configuration = _build_v5_synthetic_configuration()
    family_v3 = configuration["joint_acceptance_family"]
    configuration["joint_acceptance_family"] = build_g3_joint_acceptance_family_v4(
        family_id=family_v3["family_id"],
        material_expected_strata=[
            tuple(item) for item in family_v3["material_expected_strata"]
        ],
        complex_csi_expected_strata=[
            tuple(item) for item in family_v3["complex_csi_expected_strata"]
        ],
        propagation_expected_strata=[
            tuple(item) for item in family_v3["propagation_expected_strata"]
        ],
        receiver_impact_expected_strata=[
            tuple(item) for item in family_v3["receiver_impact_expected_strata"]
        ],
        path_equivalence_expected_scenarios=[("urban", "LOS", "episode-1")],
    )
    schema_path = ROOT / "datasets/schema/g3.03-acquisition-design-assumptions.v5.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(configuration)
    input_path = tmp_path / "v5-joint-family-v4.json"
    input_path.write_text(json.dumps(configuration), encoding="utf-8")

    assert main([str(input_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["joint_acceptance_family"]["schema"] == "loslab.g3_joint_acceptance_family.v4"
    assert report["joint_acceptance_family"]["claim_count"] == 24
    assert report["joint_acceptance_family"]["path_equivalence_expected_scenarios"] == [
        ["urban", "LOS", "episode-1"]
    ]


def test_v5_planner_rejects_family_v3_without_path_episode_census(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configuration = _build_v5_synthetic_configuration()
    family_v4 = configuration["joint_acceptance_family"]
    configuration["joint_acceptance_family"] = build_g3_joint_acceptance_family_v3(
        family_id=family_v4["family_id"],
        material_expected_strata=[tuple(row) for row in family_v4["material_expected_strata"]],
        complex_csi_expected_strata=[tuple(row) for row in family_v4["complex_csi_expected_strata"]],
        propagation_expected_strata=[tuple(row) for row in family_v4["propagation_expected_strata"]],
        receiver_impact_expected_strata=[tuple(row) for row in family_v4["receiver_impact_expected_strata"]],
    )
    schema_path = ROOT / "datasets/schema/g3.03-acquisition-design-assumptions.v5.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(configuration)

    input_path = tmp_path / "v5-family-v3.json"
    input_path.write_text(json.dumps(configuration), encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main([str(input_path)])

    assert error.value.code == 2
    assert "acquisition-design v5 inputs require joint-family v4" in capsys.readouterr().err
