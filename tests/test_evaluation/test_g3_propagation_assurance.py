"""Tests for evaluator-aligned propagation assurance planning."""

from __future__ import annotations

from typing import Any

import pytest
from evaluation.g3_joint_acceptance import build_g3_joint_acceptance_family_v3
from evaluation.g3_propagation_assurance import plan_g3_propagation_bootstrap_assurance

_STRATUM = ("urban", "LOS", "total")
_METRICS = (
    "first_arrival_delay_s",
    "rms_delay_spread_s",
    "arrival_angular_spread_deg",
    "departure_angular_spread_deg",
)


def _family(
    propagation_strata: list[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    return build_g3_joint_acceptance_family_v3(
        family_id="propagation-assurance-test",
        material_expected_strata=[("concrete", "V", "0-30deg")],
        complex_csi_expected_strata=[("urban", "LOS")],
        propagation_expected_strata=propagation_strata or [_STRATUM],
        receiver_impact_expected_strata=[("urban", "LOS")],
    )


def _episode(*, departure_error_deg: float = 0.0) -> dict[str, Any]:
    values = {
        "first_arrival_delay_s": (120e-9, 120e-9),
        "rms_delay_spread_s": (80e-9, 80e-9),
        "arrival_angular_spread_deg": (20.0, 20.0),
        "departure_angular_spread_deg": (25.0, 25.0 + departure_error_deg),
    }
    return {
        "metrics": {
            name: {
                "candidate": values[name][0],
                "challenger": values[name][1],
                "availability": "resolved",
                "reason": None,
            }
            for name in _METRICS
        }
    }


def _scenario(
    *,
    departure_error_deg: float = 0.0,
    strata: list[tuple[str, str, str]] | None = None,
    not_applicable_stratum: tuple[str, str, str] | None = None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for stratum in strata or [_STRATUM]:
        episode = _episode(departure_error_deg=departure_error_deg)
        if stratum == not_applicable_stratum:
            episode["metrics"]["departure_angular_spread_deg"] = {
                "candidate": None,
                "challenger": None,
                "availability": "not_applicable",
                "reason": "component has no departure-angle support",
            }
        result[stratum] = {
            "assumed_cluster_episodes": [{"episodes": [episode]}],
            "probabilities": [1.0],
        }
    return result


def _plan(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "joint_acceptance_family": _family(),
        "assumed_iid_scenarios_by_stratum": _scenario(),
        "candidate_clusters_per_stratum": [125],
        "effective_bandwidth_hz": 100e6,
        "bootstrap_draws": 1_000,
        "assurance_replicates": 100,
        "seed": 91,
    }
    values.update(overrides)
    return plan_g3_propagation_bootstrap_assurance(**values)


def test_propagation_assurance_uses_shared_evaluator_and_never_qualifies() -> None:
    report = _plan()

    assert report["schema"] == "loslab.g3_propagation_bootstrap_assurance_planning.v1"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["admission_possible"] is False
    assert report["complete_g3_sample_size"] is None
    assert report["joint_acceptance_family"]["claim_count"] == 19
    assert report["joint_acceptance_family"]["per_claim_alpha_bonferroni"] == pytest.approx(
        0.05 / 19
    )
    assert report["candidate_designs"][0]["stratum_assurance"][0]["joint_pass_count"] == 100
    assert report["selected_clusters_per_stratum_among_evaluated_candidates"] == 125
    assert len(report["scenario_sha256"]) == 64


def test_propagation_assurance_is_reproducible() -> None:
    assert _plan() == _plan()


def test_propagation_assurance_uses_union_bound_and_respects_not_applicable_metrics() -> None:
    second_stratum = ("urban", "NLOS", "diffuse")
    scenarios = _scenario(strata=[_STRATUM, second_stratum], not_applicable_stratum=second_stratum)
    report = _plan(
        joint_acceptance_family=_family([_STRATUM, second_stratum]),
        assumed_iid_scenarios_by_stratum=scenarios,
    )

    candidate = report["candidate_designs"][0]
    assert report["joint_acceptance_family"]["claim_count"] == 27
    assert candidate["joint_assurance_lower_bound_union_bound"] == pytest.approx(
        1.0
        - sum(
            1.0 - stratum["assurance_one_sided_exact_lower_bound"]
            for stratum in candidate["stratum_assurance"]
        )
    )
    assert candidate["meets_target_assurance"] is True
    assert report["selected_clusters_per_stratum_among_evaluated_candidates"] == 125


def test_propagation_assurance_counts_failed_metric_scenarios() -> None:
    report = _plan(
        assumed_iid_scenarios_by_stratum=_scenario(departure_error_deg=2.0),
        candidate_clusters_per_stratum=[2],
    )

    candidate = report["candidate_designs"][0]
    assert candidate["stratum_assurance"][0]["joint_pass_count"] == 0
    assert candidate["joint_assurance_lower_bound_union_bound"] == 0.0
    assert report["selected_clusters_per_stratum_among_evaluated_candidates"] is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"candidate_clusters_per_stratum": [1]}, "integers >= 2"),
        ({"bootstrap_draws": 999}, ">= 1000"),
        ({"assurance_replicates": 99}, ">= 100"),
        ({"assumed_iid_scenarios_by_stratum": {}}, "exactly match"),
    ],
)
def test_propagation_assurance_rejects_invalid_design(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _plan(**overrides)


def test_propagation_assurance_rejects_malformed_metric_state() -> None:
    scenario = _scenario()
    scenario[_STRATUM]["assumed_cluster_episodes"][0]["episodes"][0]["metrics"][
        "departure_angular_spread_deg"
    ]["availability"] = []

    with pytest.raises(ValueError, match="availability is invalid"):
        _plan(assumed_iid_scenarios_by_stratum=scenario)
