"""Bootstrap assurance planners: scenario distribution and material-roster admission rules."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from evaluation.acceptance.joint_acceptance import build_joint_acceptance_family
from evaluation.sample_size import bootstrap as bootstrap_module
from evaluation.sample_size.bootstrap import (
    plan_bootstrap_assurance,
    plan_complex_csi_bootstrap_assurance,
)

ATOM = {
    "median_absolute_power_error_db": 0.5,
    "p95_absolute_power_error_db": 1.0,
    "maximum_angular_cdf_ks": 0.05,
    "nominal_95_interval_coverage": 0.95,
    "median_prediction_interval_width_db": 1.0,
}


def _scenario(atoms=None, probabilities=None) -> dict[str, Any]:
    return {
        "assumed_unit_summaries": atoms if atoms is not None else [dict(ATOM), dict(ATOM)],
        "probabilities": probabilities if probabilities is not None else [0.5, 0.5],
    }


def _family() -> dict[str, Any]:
    return build_joint_acceptance_family(
        family_id="scenario-guard",
        material_expected_strata=[("brick", "V", "normal")],
        complex_csi_expected_strata=[("urban", "LOS")],
    )


def _plan(scenarios, **overrides) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "assumed_iid_scenarios_by_stratum": {"brick-V-normal": scenarios},
        "candidate_units_per_stratum": [20],
        "bootstrap_replicates": 1_000,
        "assurance_replicates": 100,
        "seed": 11,
    }
    kwargs.update(overrides)
    return plan_bootstrap_assurance(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"candidate_units_per_stratum": []}, "nonempty integer sequence"),
        ({"candidate_units_per_stratum": [20, 20]}, "must be unique"),
        ({"bootstrap_replicates": 999}, "bootstrap_replicates must be an integer"),
        ({"seed": -1}, "seed must be a nonnegative integer"),
        ({"familywise_claim_count": 1}, "must cover every material bootstrap claim"),
    ],
)
def test_assurance_design_rejects_invalid_counts_and_seed(overrides, message):
    with pytest.raises(ValueError, match=message):
        plan_bootstrap_assurance(**_roster_kwargs(**overrides))


def test_assurance_design_enforces_total_and_peak_bootstrap_work_bounds(monkeypatch):
    monkeypatch.setattr(bootstrap_module, "_MAX_BOOTSTRAP_RESAMPLING_CELLS", 1)
    with pytest.raises(ValueError, match="bounded bootstrap planning work limit"):
        plan_bootstrap_assurance(**_roster_kwargs())

    monkeypatch.setattr(bootstrap_module, "_MAX_BOOTSTRAP_RESAMPLING_CELLS", 10**9)
    monkeypatch.setattr(bootstrap_module, "_MAX_BOOTSTRAP_PEAK_CELLS", 1)
    with pytest.raises(ValueError, match="single bootstrap resample exceeds"):
        plan_bootstrap_assurance(**_roster_kwargs())


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        pytest.param(_scenario(probabilities=[0.5, 0.5, 0.0]), "incompatible atom", id="count"),
        pytest.param(_scenario(probabilities=[0.0, 1.0]), "positive and sum to one", id="zero"),
        pytest.param(_scenario(probabilities=[0.2, 0.2]), "positive and sum to one", id="sum"),
        pytest.param(_scenario(probabilities=[True, False]), "must be numeric weights", id="bool"),
        pytest.param(
            _scenario(probabilities=["a", "b"]), "must be numeric weights", id="non-numeric"
        ),
        pytest.param(
            _scenario(probabilities=[[0.5, 0.5], [1.0]]),
            "must be a numeric sequence",
            id="ragged",
        ),
        pytest.param(_scenario(atoms="abc", probabilities=[1.0]), "incompatible atom", id="text"),
        pytest.param(
            _scenario(atoms=[{"median_absolute_power_error_db": 0.5}], probabilities=[1.0]),
            "exactly the five metrics",
            id="missing-metric",
        ),
        pytest.param(
            _scenario(atoms=[dict(ATOM, median_absolute_power_error_db=True)], probabilities=[1.0]),
            "must be a finite number",
            id="bool-metric",
        ),
        pytest.param(
            _scenario(
                atoms=[dict(ATOM, p95_absolute_power_error_db=float("nan"))], probabilities=[1.0]
            ),
            "must be a finite number",
            id="nan-metric",
        ),
        pytest.param(
            _scenario(atoms=[dict(ATOM, maximum_angular_cdf_ks=1.5)], probabilities=[1.0]),
            "outside metric ranges",
            id="ks-range",
        ),
        pytest.param(
            _scenario(atoms=[dict(ATOM, nominal_95_interval_coverage=-0.1)], probabilities=[1.0]),
            "outside metric ranges",
            id="coverage-range",
        ),
        pytest.param(
            _scenario(
                atoms=[dict(ATOM, median_prediction_interval_width_db=-1.0)], probabilities=[1.0]
            ),
            "outside metric ranges",
            id="width-range",
        ),
    ],
)
def test_assumed_unit_scenarios_reject_malformed_distributions(scenario, message) -> None:
    with pytest.raises(ValueError, match=message):
        _plan(scenario)


@pytest.mark.parametrize(
    ("scenarios", "message"),
    [
        pytest.param({}, "must be a nonempty mapping", id="empty"),
        pytest.param(
            {"   ": _scenario()}, "scenario stratum names must be nonempty strings", id="blank"
        ),
        pytest.param(
            {"brick-V-normal": {"assumed_unit_summaries": [dict(ATOM)]}},
            "must contain exactly assumed_unit_summaries and probabilities",
            id="missing-probabilities",
        ),
    ],
)
def test_scenario_mapping_must_name_each_stratum_and_its_two_fields(scenarios, message) -> None:
    with pytest.raises(ValueError, match=message):
        plan_bootstrap_assurance(
            assumed_iid_scenarios_by_stratum=scenarios,
            candidate_units_per_stratum=[20],
            bootstrap_replicates=1_000,
            assurance_replicates=100,
        )


def _roster_kwargs(**overrides) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "assumed_iid_scenarios_by_stratum": {"brick-V-normal": _scenario()},
        "candidate_units_per_stratum": [20],
        "joint_acceptance_family": _family(),
        "stratum_roster_by_id": {"brick-V-normal": ["brick", "V", "normal"]},
        "bootstrap_replicates": 1_000,
        "assurance_replicates": 100,
        "seed": 5,
    }
    kwargs.update(overrides)
    return kwargs


def test_material_roster_binds_scenario_strata_to_the_frozen_family() -> None:
    result = plan_bootstrap_assurance(**_roster_kwargs())
    assert "brick-V-normal" in str(result)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"stratum_roster_by_id": None},
            "must be supplied together",
            id="family-without-roster",
        ),
        pytest.param(
            {"joint_acceptance_family": None},
            "must be supplied together",
            id="roster-without-family",
        ),
        pytest.param(
            {"stratum_roster_by_id": {"other": ["brick", "V", "normal"]}},
            "roster IDs must match the scenario strata",
            id="id-mismatch",
        ),
        pytest.param(
            {
                "stratum_roster_by_id": {
                    "brick-V-normal": ["brick", "V", "normal"],
                    " ": ["x", "y", "z"],
                }
            },
            "roster IDs must match the scenario strata",
            id="blank-id",
        ),
        pytest.param(
            {"stratum_roster_by_id": {"brick-V-normal": ["brick", "V"]}},
            "must have three nonempty strings",
            id="short-entry",
        ),
        pytest.param(
            {"stratum_roster_by_id": {"brick-V-normal": ["brick", "", "normal"]}},
            "must have three nonempty strings",
            id="empty-part",
        ),
        pytest.param(
            {
                "assumed_iid_scenarios_by_stratum": {
                    "brick-V-normal": _scenario(),
                    "brick-V-other": _scenario(),
                },
                "stratum_roster_by_id": {
                    "brick-V-normal": ["brick", "V", "normal"],
                    "brick-V-other": ["brick", "V", "normal"],
                },
            },
            "contains duplicate strata",
            id="duplicate-stratum",
        ),
        pytest.param(
            {
                "stratum_roster_by_id": {"brick-V-normal": ["brick", "H", "normal"]},
            },
            "differs from acceptance-family strata",
            id="unknown-stratum",
        ),
    ],
)
def test_material_roster_rejects_partial_or_inconsistent_bindings(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        plan_bootstrap_assurance(**_roster_kwargs(**overrides))


def _csi_family() -> dict[str, Any]:
    return build_joint_acceptance_family(
        family_id="csi-scenario-guard",
        material_expected_strata=[("brick", "V", "normal")],
        complex_csi_expected_strata=[("urban", "LOS")],
    )


def _csi_scenario(**overrides) -> dict[str, Any]:
    scenario = {
        "assumed_cluster_summaries": [
            {"cluster_median_e_h": 0.005, "cluster_p95_e_h": 0.02, "cluster_failed": False}
        ],
        "probabilities": [1.0],
    }
    scenario.update(overrides)
    return scenario


def _csi_plan(scenarios) -> dict[str, Any]:
    return plan_complex_csi_bootstrap_assurance(
        joint_acceptance_family=_csi_family(),
        assumed_iid_scenarios_by_stratum=scenarios,
        candidate_clusters_per_stratum=[20],
        bootstrap_replicates=1_000,
        assurance_replicates=100,
        seed=3,
    )


@pytest.mark.parametrize(
    ("key", "scenario", "message"),
    [
        pytest.param(
            ("urban", "LOS"),
            {"assumed_cluster_summaries": []},
            "must contain cluster summaries and probabilities",
            id="missing-probabilities",
        ),
        pytest.param(
            ("urban", "LOS"),
            _csi_scenario(
                assumed_cluster_summaries=[{"cluster_median_e_h": 0.005, "cluster_failed": False}]
            ),
            "invalid field set",
            id="atom-fields",
        ),
        pytest.param(
            ("urban", "LOS"),
            _csi_scenario(probabilities=[0.4]),
            "positive and sum to one",
            id="probability-sum",
        ),
    ],
)
def test_complex_csi_scenarios_reject_malformed_entries(key, scenario, message) -> None:
    with pytest.raises(ValueError, match=message):
        _csi_plan({key: scenario})


def test_complex_csi_scenario_must_not_be_mutated_by_the_planner() -> None:
    scenarios = {("urban", "LOS"): _csi_scenario()}
    snapshot = deepcopy(scenarios)
    _csi_plan(scenarios)
    assert scenarios == snapshot


@pytest.mark.parametrize(
    "key",
    [("urban", "MAYBE"), ("  ", "LOS"), ("urban", None)],
    ids=["bad-state", "blank-family", "non-string-state"],
)
def test_csi_scenario_keys_must_be_family_and_los_or_nlos(key) -> None:
    from evaluation.sample_size.bootstrap import _validate_complex_csi_assurance_scenarios

    with pytest.raises(ValueError, match="LOS or NLOS"):
        _validate_complex_csi_assurance_scenarios({key: _csi_scenario()})
