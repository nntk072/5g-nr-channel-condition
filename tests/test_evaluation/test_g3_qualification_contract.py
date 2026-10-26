from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from evaluation.g3_joint_acceptance import build_g3_joint_acceptance_family_v4
from evaluation.g3_qualification_contract import (
    build_g3_qualification_run_contract,
    build_g3_qualification_run_contract_v3,
    build_g3_qualification_run_contract_v4,
    verify_g3_qualification_run_contract,
)


def _spec() -> dict[str, object]:
    family = build_g3_joint_acceptance_family_v4(
        family_id="g3-frozen-run-family",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("site-a", "LOS")],
        propagation_expected_strata=[("site-a", "LOS", "total")],
        receiver_impact_expected_strata=[("site-a", "LOS")],
        path_equivalence_expected_scenarios=[("site-a", "LOS", "episode-01")],
    )
    return {
        "run_id": "g3-confirmation-001",
        "joint_acceptance_family": family,
        "task_scope": {
            "task_id": "complex-channel-and-receiver-impact",
            "scenario_ids": ["site-a"],
            "channel_conditions": ["LOS"],
            "carrier_hz": 3.5e9,
        },
        "candidate": {"config_id": "candidate-v1", "configuration": {"receiver": "ls"}},
        "challenger": {
            "config_id": "challenger-v1",
            "configuration": {"receiver": "dft", "max_delay_samples": 7},
        },
        "waveform": {
            "waveform_id": "nr-pusch-v1",
            "configuration": {"mu": 1, "prbs": 24, "carrier_hz": 3.5e9},
        },
        "link_budget_noise": {
            "tx_power_dbm": 23.0,
            "tx_gain_dbi": 0.0,
            "rx_gain_dbi": 0.0,
            "noise_figure_db": 7.0,
            "noise_temperature_k": 290.0,
            "noise_bandwidth_hz": 20e6,
        },
        "path_matching": {
            "delay_tolerance_s": 1e-9,
            "azimuth_tolerance_deg": 2.0,
            "elevation_tolerance_deg": 2.0,
            "coefficient_relative_tolerance": 0.05,
            "minimum_bidirectional_coverage": 0.95,
        },
        "sampling": {
            "episode_ids": ["episode-01"],
            "path_equivalence_scenarios": [["site-a", "LOS", "episode-01"]],
            "unit_by_branch": {
                "material_calibration": "independent_acquisition_unit",
                "complex_csi": "independent_physical_episode",
                "propagation_metrics": "independent_physical_episode",
                "receiver_impact": "independent_transport_block_cluster",
                "path_equivalence": "physical_episode",
            },
            "planned_units_by_branch": {
                "material_calibration": 30,
                "complex_csi": 60,
                "propagation_metrics": 60,
                "receiver_impact": 100,
                "path_equivalence": 1,
            },
            "planned_units_by_stratum": {
                "material_calibration": [
                    {"stratum": ["concrete", "V", "20deg"], "planned_units": 30}
                ],
                "complex_csi": [{"stratum": ["site-a", "LOS"], "planned_units": 60}],
                "propagation_metrics": [
                    {"stratum": ["site-a", "LOS", "total"], "planned_units": 60}
                ],
                "receiver_impact": [{"stratum": ["site-a", "LOS"], "planned_units": 100}],
                "path_equivalence": [
                    {"stratum": ["site-a", "LOS", "episode-01"], "planned_units": 1}
                ],
            },
            "bootstrap_seed": 1729,
            "bootstrap_draws": 10000,
        },
        "solver_budget": {
            "max_rays_per_link": 100000,
            "max_interactions": 5,
            "max_episodes": 200,
            "max_wall_seconds": 3600,
            "overflow_policy": "record_censored_episode",
        },
        "source_identity": {
            "git_revision": "a" * 40,
            "working_tree_dirty": True,
            "source_bundle_sha256": "b" * 64,
        },
    }


def _complete_design(spec: dict[str, object]) -> dict[str, object]:
    family = spec["joint_acceptance_family"]
    branch_rosters = {
        "material_calibration": ("material_expected_strata", 20),
        "complex_csi": ("complex_csi_expected_strata", 2),
        "propagation_metrics": ("propagation_expected_strata", 2),
        "receiver_impact": ("receiver_impact_expected_strata", 20),
        "path_equivalence": ("path_equivalence_expected_scenarios", 1),
    }
    units = {
        "material_calibration": "independent_acquisition_unit",
        "complex_csi": "independent_physical_episode",
        "propagation_metrics": "independent_physical_episode",
        "receiver_impact": "independent_transport_block_cluster",
        "path_equivalence": "physical_episode",
    }
    branches = {
        branch: {
            "sampling_unit": units[branch],
            "required_units_by_stratum": [
                {"stratum": list(stratum), "required_units": count}
                for stratum in family[field]
            ],
            "count_available": True,
            "component_status": "SCENARIO_CONDITIONAL_COUNT_AVAILABLE",
        }
        for branch, (field, count) in branch_rosters.items()
    }
    schemas = {
        "joint": "loslab.g3_joint_sample_size_planning.v1",
        "material_assurance": "loslab.g3_bootstrap_assurance_planning.v1",
        "complex_csi_assurance": "loslab.g3_complex_csi_bootstrap_assurance_planning.v1",
        "propagation_assurance": "loslab.g3_propagation_bootstrap_assurance_planning.v1",
        "receiver_assurance": "loslab.g3_receiver_impact_bootstrap_assurance_planning.v1",
    }
    design: dict[str, object] = {
        "schema": "loslab.g3_joint_sample_size_design_composition.v1",
        "status": "SCENARIO_CONDITIONAL_DESIGN_NOT_QUALIFICATION",
        "qualification_status": "NOT_QUALIFIED",
        "admission_possible": False,
        "joint_acceptance_family": {
            "family_id": family["family_id"],
            "family_sha256": family["family_sha256"],
        },
        "component_plans": {
            name: {"schema": schema, "sha256": "a" * 64}
            for name, schema in schemas.items()
        },
        "sampling_units_are_distinct": True,
        "single_combined_sample_size_is_reported": False,
        "branch_designs": branches,
        "all_branch_counts_available": True,
        "component_designs_claim_complete_powered_design": True,
        "branch_assurance_lower_bounds": {
            "material_calibration": 0.99,
            "complex_csi": 0.99,
            "propagation_metrics": 0.99,
            "receiver_impact": 0.99,
        },
        "joint_g3_assurance_lower_bound_union_bound": 0.96,
        "target_joint_g3_assurance": 0.8,
        "material_angular_maximum_assumption": {
            "supplied": True,
            "compatible_with_material_design": True,
            "maximum_units_per_stratum": 100,
        },
        "complete_precision_powered_g3_design": True,
        "complete_g3_sample_size": None,
        "blockers": [],
        "limitations": ["Synthetic contract unit-test fixture; assumptions are not verified."],
    }
    canonical = json.dumps(
        design, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    design["design_sha256"] = hashlib.sha256(canonical).hexdigest()
    return design


def _redigest_design(design: dict[str, object]) -> None:
    design.pop("design_sha256")
    canonical = json.dumps(
        design, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    design["design_sha256"] = hashlib.sha256(canonical).hexdigest()


def test_g3_qualification_contract_freezes_configs_and_sampling_design() -> None:
    contract = build_g3_qualification_run_contract(_spec())

    assert contract["qualification_status"] == "NOT_QUALIFIED"
    assert contract["candidate"]["configuration_sha256"]
    assert contract["challenger"]["configuration_sha256"]
    assert contract["sampling"]["episode_ids"] == ["episode-01"]
    assert contract["finite_solver_caps_are_ground_truth"] is False
    assert verify_g3_qualification_run_contract(contract)["status"] == (
        "PASS_CONTRACT_INTEGRITY_NOT_QUALIFIED"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda spec: spec["task_scope"].update(task_id=" task"),
        lambda spec: spec["task_scope"].update(scenario_ids=["site-a "]),
        lambda spec: spec["candidate"].update(config_id=" candidate-v1"),
        lambda spec: spec["waveform"].update(waveform_id="nr-pusch-v1 "),
        lambda spec: spec["sampling"].update(episode_ids=[" episode-01"]),
        lambda spec: spec["sampling"].update(
            path_equivalence_scenarios=[["site-a ", "LOS", "episode-01"]]
        ),
        lambda spec: spec["sampling"]["planned_units_by_stratum"][
            "complex_csi"
        ][0].update(stratum=["site-a", " LOS", "LOS"]),
    ],
)
def test_g3_qualification_contract_rejects_padded_identity_aliases(mutate) -> None:
    spec = deepcopy(_spec())
    mutate(spec)

    with pytest.raises(ValueError, match="canonical|nonempty|roster"):
        build_g3_qualification_run_contract(spec)


def test_g3_qualification_contract_v3_binds_complete_design_and_digest() -> None:
    spec = _spec()
    spec["sample_size_design"] = _complete_design(spec)

    contract = build_g3_qualification_run_contract_v3(spec)

    assert contract["schema"] == "loslab.g3_qualification_run_contract.v3"
    assert contract["sample_size_design_sha256"] == contract["sample_size_design"]["design_sha256"]
    assert verify_g3_qualification_run_contract(contract)["status"] == (
        "PASS_CONTRACT_INTEGRITY_NOT_QUALIFIED"
    )
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).parents[2]
        / "datasets/schema/g3-qualification-run-contract.v3.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(contract)


def test_g3_qualification_contract_v3_rejects_a_count_below_evaluator_floor() -> None:
    spec = _spec()
    spec["sample_size_design"] = _complete_design(spec)
    spec["sampling"]["planned_units_by_stratum"]["complex_csi"][0]["planned_units"] = 1

    with pytest.raises(ValueError, match="must be at least 2; this is an evaluator/pilot floor"):
        build_g3_qualification_run_contract_v3(spec)


def test_g3_qualification_contract_v3_rejects_incomplete_design() -> None:
    spec = _spec()
    design = _complete_design(spec)
    design["component_designs_claim_complete_powered_design"] = False
    design["complete_precision_powered_g3_design"] = False
    _redigest_design(design)
    spec["sample_size_design"] = design

    with pytest.raises(
        ValueError,
        match=(
            "requires a complete precision- and power-based design|"
            "component powered-design claim is inconsistent"
        ),
    ):
        build_g3_qualification_run_contract_v3(spec)


def test_g3_qualification_contract_v4_freezes_raw_path_analysis_configuration() -> None:
    spec = _spec()
    spec["task_scope"]["effective_bandwidth_hz"] = 20e6
    spec["path_matching"].update(
        {
            "vertex_tolerance_m": 0.02,
            "direction_tolerance_rad": 0.05,
            "h_floor": 1e-12,
            "transmit_power_per_re_w": 0.1,
            "noise_power_per_re_w": 1e-10,
        }
    )
    spec["sample_size_design"] = _complete_design(spec)

    contract = build_g3_qualification_run_contract_v4(spec)

    assert contract["schema"] == "loslab.g3_qualification_run_contract.v4"
    assert contract["path_matching"]["vertex_tolerance_m"] == 0.02
    assert contract["path_matching"]["direction_tolerance_rad"] == 0.05
    assert verify_g3_qualification_run_contract(contract)["status"] == (
        "PASS_CONTRACT_INTEGRITY_NOT_QUALIFIED"
    )
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).parents[2]
        / "datasets/schema/g3-qualification-run-contract.v4.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(contract)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vertex_tolerance_m", 0.0),
        ("direction_tolerance_rad", 4.0),
        ("coefficient_relative_tolerance", 0.0500000001),
        ("h_floor", -1.0),
        ("transmit_power_per_re_w", 0.0),
        ("noise_power_per_re_w", float("inf")),
    ],
)
def test_g3_qualification_contract_v4_rejects_invalid_raw_path_settings(field, value):
    spec = _spec()
    spec["task_scope"]["effective_bandwidth_hz"] = 20e6
    spec["path_matching"].update(
        {
            "vertex_tolerance_m": 0.02,
            "direction_tolerance_rad": 0.05,
            "h_floor": 1e-12,
            "transmit_power_per_re_w": 0.1,
            "noise_power_per_re_w": 1e-10,
        }
    )
    spec["path_matching"][field] = value
    spec["sample_size_design"] = _complete_design(spec)

    with pytest.raises(ValueError):
        build_g3_qualification_run_contract_v4(spec)


def test_g3_qualification_contract_v4_requires_positive_effective_bandwidth():
    spec = _spec()
    spec["task_scope"]["effective_bandwidth_hz"] = 20e6
    spec["task_scope"]["effective_bandwidth_hz"] = 0.0
    spec["path_matching"].update(
        {
            "vertex_tolerance_m": 0.02,
            "direction_tolerance_rad": 0.05,
            "h_floor": 1e-12,
            "transmit_power_per_re_w": 0.1,
            "noise_power_per_re_w": 1e-10,
        }
    )
    spec["sample_size_design"] = _complete_design(spec)

    with pytest.raises(ValueError, match="task_scope.effective_bandwidth_hz"):
        build_g3_qualification_run_contract_v4(spec)


def test_g3_qualification_contract_rejects_sampling_roster_drift() -> None:
    spec = _spec()
    spec["sampling"]["episode_ids"] = ["episode-02"]

    with pytest.raises(ValueError, match="exactly match"):
        build_g3_qualification_run_contract(spec)


def test_g3_qualification_contract_rejects_task_scope_that_omits_family_scenario() -> None:
    spec = _spec()
    spec["task_scope"]["scenario_ids"] = ["different-site"]

    with pytest.raises(ValueError, match="cover every acceptance-family scenario"):
        build_g3_qualification_run_contract(spec)


def test_g3_qualification_contract_rejects_missing_link_budget_term() -> None:
    spec = _spec()
    del spec["link_budget_noise"]["noise_figure_db"]

    with pytest.raises(ValueError, match="exactly the declared fields"):
        build_g3_qualification_run_contract(spec)


def test_g3_qualification_contract_rejects_finite_cap_as_ground_truth() -> None:
    contract = build_g3_qualification_run_contract(_spec())
    tampered = deepcopy(contract)
    tampered["finite_solver_caps_are_ground_truth"] = True

    with pytest.raises(ValueError, match="digest"):
        verify_g3_qualification_run_contract(tampered)


@pytest.mark.parametrize("field", ["delay_tolerance_s", "minimum_bidirectional_coverage"])
def test_g3_qualification_contract_rejects_nonfinite_matching_limits(field: str) -> None:
    spec = _spec()
    spec["path_matching"][field] = float("inf")

    with pytest.raises(ValueError, match="NaN|Infinity|Out of range"):
        build_g3_qualification_run_contract(spec)


@pytest.mark.parametrize("coverage", [0.90, 0.99, 1.0])
def test_g3_qualification_contract_rejects_path_coverage_drift_from_family(
    coverage: float,
) -> None:
    spec = _spec()
    spec["path_matching"]["minimum_bidirectional_coverage"] = coverage

    with pytest.raises(ValueError, match="must match the acceptance-family criterion"):
        build_g3_qualification_run_contract(spec)


@pytest.mark.parametrize(
    ("branch", "count", "minimum"),
    [
        ("material_calibration", 19, 20),
        ("complex_csi", 1, 2),
        ("propagation_metrics", 1, 2),
        ("receiver_impact", 19, 20),
    ],
)
def test_g3_qualification_contract_rejects_per_stratum_evaluator_floor(
    branch: str, count: int, minimum: int
) -> None:
    spec = _spec()
    spec["sampling"]["planned_units_by_branch"][branch] = count
    spec["sampling"]["planned_units_by_stratum"][branch][0]["planned_units"] = count

    with pytest.raises(ValueError, match=f"at least {minimum}"):
        build_g3_qualification_run_contract(spec)


def test_g3_qualification_contract_rejects_per_stratum_count_total_mismatch() -> None:
    spec = _spec()
    spec["sampling"]["planned_units_by_stratum"]["receiver_impact"][0][
        "planned_units"
    ] = 101

    with pytest.raises(ValueError, match="must equal the sum of its per-stratum counts"):
        build_g3_qualification_run_contract(spec)


def test_g3_qualification_contract_rejects_missing_acceptance_stratum_count() -> None:
    spec = _spec()
    spec["sampling"]["planned_units_by_stratum"]["complex_csi"] = []

    with pytest.raises(ValueError, match="must be a nonempty roster"):
        build_g3_qualification_run_contract(spec)


def test_g3_qualification_contract_binds_one_path_unit_per_frozen_episode() -> None:
    spec = _spec()
    spec["sampling"]["planned_units_by_branch"]["path_equivalence"] = 2
    spec["sampling"]["planned_units_by_stratum"]["path_equivalence"][0][
        "planned_units"
    ] = 2

    with pytest.raises(ValueError, match="exactly one per frozen episode"):
        build_g3_qualification_run_contract(spec)
