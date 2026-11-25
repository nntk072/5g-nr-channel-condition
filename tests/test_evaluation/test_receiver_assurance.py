"""Tests for assumption-only assurance of the paired receiver evaluator."""

from __future__ import annotations

from typing import Any

import pytest
from evaluation.acceptance import receiver_assurance as assurance
from evaluation.acceptance.joint_acceptance import (
    build_joint_acceptance_family_v2,
    build_joint_acceptance_family_v3,
)
from evaluation.acceptance.receiver_assurance import plan_receiver_impact_bootstrap_assurance


def _family() -> dict[str, Any]:
    return build_joint_acceptance_family_v3(
        family_id="receiver-assurance-test",
        material_expected_strata=[("concrete", "V", "0-30deg")],
        complex_csi_expected_strata=[("urban", "LOS")],
        propagation_expected_strata=[("urban", "LOS", "total")],
        receiver_impact_expected_strata=[("urban", "LOS")],
    )


def _scenario() -> dict[tuple[str, str], dict[str, Any]]:
    return {
        ("urban", "LOS"): {
            "assumed_cluster_outcomes": [
                {
                    "candidate_crc_errors": [
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                    ],
                    "challenger_crc_errors": [
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                    ],
                }
            ],
            "probabilities": [1.0],
        }
    }


def _plan(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "joint_acceptance_family": _family(),
        "assumed_iid_scenarios_by_stratum": _scenario(),
        "candidate_clusters_per_stratum": [20],
        "snr_grid_db": [0.0, 1.0, 2.0],
        "comparison_snr_db": 1.0,
        "bootstrap_replicates": 1_000,
        "assurance_replicates": 100,
        "seed": 77,
    }
    values.update(overrides)
    return plan_receiver_impact_bootstrap_assurance(**values)


def test_receiver_assurance_uses_shared_family_and_never_qualifies() -> None:
    report = _plan()

    assert report["schema"] == "loslab.receiver_impact_bootstrap_assurance_planning.v1"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["admission_possible"] is False
    assert report["complete_sample_size"] is None
    assert report["joint_acceptance_family"]["claim_count"] == 19
    assert report["joint_acceptance_family"]["per_claim_alpha_bonferroni"] == pytest.approx(
        0.05 / 19
    )
    design = report["candidate_designs"][0]
    assert design["stratum_assurance"][0]["joint_pass_count"] == 0
    assert design["joint_assurance_lower_bound_union_bound"] == 0.0
    assert design["meets_target_assurance"] is False
    assert report["selected_clusters_per_stratum_among_evaluated_candidates"] is None
    assert len(report["scenario_sha256"]) == 64


def test_receiver_assurance_is_reproducible_and_scenario_hash_bound() -> None:
    first = _plan()
    second = _plan()

    assert first == second


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"candidate_clusters_per_stratum": [19]}, "integers >= 20"),
        ({"snr_grid_db": [0.0, 2.0, 1.0]}, "strictly increasing"),
        ({"comparison_snr_db": 0.5}, "exact member"),
        ({"bootstrap_replicates": 999}, ">= 1000"),
        ({"assurance_replicates": 99}, ">= 100"),
        ({"assumed_iid_scenarios_by_stratum": {}}, "exactly match"),
    ],
)
def test_receiver_assurance_rejects_invalid_design_inputs(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _plan(**overrides)


def test_receiver_assurance_rejects_unpaired_cluster_atom_shapes() -> None:
    scenarios = _scenario()
    scenarios[("urban", "LOS")]["assumed_cluster_outcomes"][0]["challenger_crc_errors"] = [
        [True, True, True]
    ]

    with pytest.raises(ValueError, match="block counts must match"):
        _plan(assumed_iid_scenarios_by_stratum=scenarios)


@pytest.mark.parametrize(
    "snr_grid",
    [[0.0, "1.0", 2.0], [False, 1.0, 2.0], [0j, 1.0, 2.0]],
)
def test_receiver_assurance_rejects_coercible_snr_values(snr_grid: list[Any]) -> None:
    with pytest.raises(ValueError, match="snr_grid_db values must be a finite real number"):
        _plan(snr_grid_db=snr_grid)


@pytest.mark.parametrize("probabilities", [["1.0"], [True], [1.0 + 0.0j]])
def test_receiver_assurance_rejects_coercible_scenario_probabilities(
    probabilities: list[Any],
) -> None:
    scenarios = _scenario()
    scenarios[("urban", "LOS")]["probabilities"] = probabilities
    with pytest.raises(
        ValueError,
        match="receiver scenario probabilities values must be a finite real number",
    ):
        _plan(assumed_iid_scenarios_by_stratum=scenarios)


@pytest.mark.parametrize("value", [True, "0.5", 0, 1, float("nan")])
def test_receiver_assurance_probability_validation(value: Any):
    with pytest.raises(ValueError):
        assurance._probability(value, "probe")


@pytest.mark.parametrize(
    ("key", "scenario", "message"),
    [
        ("urban", {}, "keys must be"),
        (("urban", "INVALID"), {}, "keys must be"),
        (("urban", "LOS"), {"extra": 1}, "must contain cluster outcomes"),
        (
            ("urban", "LOS"),
            {"assumed_cluster_outcomes": [], "probabilities": []},
            "nonempty sequence",
        ),
        (
            ("urban", "LOS"),
            {
                "assumed_cluster_outcomes": [
                    _scenario()[("urban", "LOS")]["assumed_cluster_outcomes"][0]
                ],
                "probabilities": [],
            },
            "match the outcome atoms",
        ),
        (
            ("urban", "LOS"),
            {"assumed_cluster_outcomes": [{}], "probabilities": [1.0]},
            "invalid field set",
        ),
        (
            ("urban", "LOS"),
            {
                "assumed_cluster_outcomes": [
                    {
                        "candidate_crc_errors": [[True, True]],
                        "challenger_crc_errors": [[True, True]],
                    }
                ],
                "probabilities": [1.0],
            },
            "boolean matrix",
        ),
        (
            ("urban", "LOS"),
            {
                "assumed_cluster_outcomes": [
                    {
                        "candidate_crc_errors": [[1, 0, 0]],
                        "challenger_crc_errors": [[False, False, False]],
                    }
                ],
                "probabilities": [1.0],
            },
            "values must be booleans",
        ),
        (
            ("urban", "LOS"),
            {
                "assumed_cluster_outcomes": [
                    _scenario()[("urban", "LOS")]["assumed_cluster_outcomes"][0]
                ],
                "probabilities": [0.5],
            },
            "sum to one",
        ),
    ],
)
def test_receiver_scenario_validation_rejects_malformed_entries(key, scenario, message):
    with pytest.raises(ValueError, match=message):
        assurance._validate_scenarios({key: scenario}, expected_strata={key}, snr_count=3)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"snr_grid_db": [1.0]}, "at least two finite values"),
        ({"comparison_snr_db": True}, "exact member"),
        ({"candidate_clusters_per_stratum": []}, "unique ascending integers"),
        ({"candidate_clusters_per_stratum": [20, 20]}, "unique ascending integers"),
        ({"bootstrap_replicates": True}, ">= 1000"),
        ({"assurance_replicates": True}, ">= 100"),
        ({"target_assurance": 1.0}, r"in \(0, 1\)"),
        ({"monte_carlo_confidence": 0.0}, r"in \(0, 1\)"),
        ({"seed": True}, "nonnegative integer"),
    ],
)
def test_receiver_assurance_rejects_invalid_limits_before_simulation(overrides, message):
    with pytest.raises(ValueError, match=message):
        _plan(**overrides)


def test_receiver_assurance_rejects_work_and_single_study_limits(monkeypatch):
    monkeypatch.setattr(assurance, "_MAX_ASSURANCE_WORK", 1)
    with pytest.raises(ValueError, match="bounded work limit"):
        _plan()
    monkeypatch.setattr(assurance, "_MAX_ASSURANCE_WORK", 250_000_000)
    monkeypatch.setattr(assurance, "_MAX_STUDY_OUTCOME_CELLS", 1)
    with pytest.raises(ValueError, match="bounded size limit"):
        _plan()


def test_receiver_assurance_requires_receiver_capable_family_version():
    family = build_joint_acceptance_family_v2(
        family_id="legacy-family",
        material_expected_strata=[("concrete", "V", "0-30deg")],
        complex_csi_expected_strata=[("urban", "LOS")],
        propagation_expected_strata=[("urban", "LOS", "total")],
    )
    with pytest.raises(ValueError, match="requires a joint-family v3 or v4 roster"):
        _plan(joint_acceptance_family=family)
