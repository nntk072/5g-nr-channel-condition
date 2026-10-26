from __future__ import annotations

from typing import Any

import pytest
from evaluation.g3_joint_acceptance import (
    build_g3_joint_acceptance_family_v2,
    build_g3_joint_acceptance_family_v3,
)
from evaluation.g3_propagation_metrics import evaluate_propagation_metrics

_METRIC_NAMES = (
    "first_arrival_delay_s",
    "rms_delay_spread_s",
    "arrival_angular_spread_deg",
    "departure_angular_spread_deg",
)


def _resolved(candidate: float, challenger: float) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "challenger": challenger,
        "availability": "resolved",
        "reason": None,
    }


def _masked(state: str, reason: str) -> dict[str, Any]:
    return {"candidate": None, "challenger": None, "availability": state, "reason": reason}


def _row(
    cluster: str,
    episode: str,
    metrics: dict[str, dict[str, Any]] | None = None,
    *,
    component: str = "total",
) -> dict[str, Any]:
    if metrics is None:
        metrics = {
            "first_arrival_delay_s": _resolved(20e-9, 21e-9),
            "rms_delay_spread_s": _resolved(10e-9, 11e-9),
            "arrival_angular_spread_deg": _resolved(10.0, 11.0),
            "departure_angular_spread_deg": _resolved(10.0, 11.0),
        }
    return {
        "family_id": "family-a",
        "condition": "LOS",
        "component": component,
        "cluster_id": cluster,
        "episode_id": episode,
        "candidate_configuration_id": "candidate-v1",
        "challenger_configuration_id": "challenger-v1",
        "metrics": metrics,
    }


def _run(
    records: list[dict[str, Any]],
    *,
    expected: list[tuple[str, str, str]] | None = None,
    joint_acceptance_family: dict[str, Any] | None = None,
):
    return evaluate_propagation_metrics(
        records,
        expected_strata=expected or [("family-a", "LOS", "total")],
        effective_bandwidth_hz=100e6,
        joint_acceptance_family=joint_acceptance_family,
        bootstrap_draws=1_000,
    )


def _metric(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in report["strata"][0]["metrics"] if item["metric"] == name)


def test_frozen_tolerance_boundaries_pass_with_100_zero_failure_clusters() -> None:
    report = _run([_row(f"cluster-{index}", "episode-0") for index in range(100)])

    assert report["status"] == "NUMERICAL_CRITERIA_ONLY_NOT_QUALIFIED"
    assert report["any_metric_assessable"] is True
    assert report["all_numeric_criteria_pass"] is True
    assert report["all_criteria_resolved_and_pass"] is True
    for metric_name in _METRIC_NAMES:
        result = _metric(report, metric_name)
        assert result["median_normalized_error"] == pytest.approx(1.0)
        assert result["p95_normalized_error"] == pytest.approx(1.0)
        assert result["p95_one_sided_bootstrap_upper_confidence_bound"] == pytest.approx(1.0)
        assert result["clusters_with_any_error_above_tolerance"] == 0
        assert result["cluster_failure_one_sided_exact_upper_bound"] <= 0.05
        assert result["confidence_level"] == pytest.approx(1.0 - 0.05 / 8)
        assert result["criteria_status"] == "PASS"

    configuration = report["configuration"]
    assert configuration["familywise_claim_count"] == 8
    assert configuration["per_claim_alpha_bonferroni"] == pytest.approx(0.05 / 8)
    assert configuration["familywise_confidence_level"] == 0.95


def test_59_zero_failure_clusters_do_not_pass_the_eight_claim_family() -> None:
    report = _run([_row(f"cluster-{index}", "episode-0") for index in range(59)])

    delay = _metric(report, "first_arrival_delay_s")
    assert delay["clusters_with_any_error_above_tolerance"] == 0
    assert delay["cluster_failure_one_sided_exact_upper_bound"] > 0.05
    assert delay["criterion_checks"]["cluster_failure_upper_bound_within_tolerance"] is False
    assert delay["criteria_status"] == "FAIL"


def test_familywise_roster_counts_all_metrics_and_expected_strata() -> None:
    records = []
    for family, condition in (("family-a", "LOS"), ("family-b", "NLOS")):
        for index in range(120):
            row = _row(f"{family}-cluster-{index}", "episode-0")
            row["family_id"] = family
            row["condition"] = condition
            records.append(row)

    report = _run(
        records,
        expected=[("family-a", "LOS", "total"), ("family-b", "NLOS", "total")],
    )

    assert report["configuration"]["familywise_claim_count"] == 16
    assert report["configuration"]["per_claim_alpha_bonferroni"] == pytest.approx(0.05 / 16)
    assert report["all_numeric_criteria_pass"] is True


def test_propagation_scorer_uses_shared_material_csi_receiver_and_delay_angular_family():
    records = [_row(f"cluster-{index}", "episode-0") for index in range(120)]
    family = build_g3_joint_acceptance_family_v3(
        family_id="protocol-v3",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("family-a", "LOS")],
        propagation_expected_strata=[("family-a", "LOS", "total")],
        receiver_impact_expected_strata=[("family-a", "LOS")],
    )

    report = _run(records, joint_acceptance_family=family)

    assert family["claim_count"] == 19
    assert report["configuration"]["familywise_claim_count"] == 19
    assert report["configuration"]["per_claim_alpha_bonferroni"] == pytest.approx(0.05 / 19)
    assert report["configuration"]["joint_acceptance_family"] == {
        "family_id": "protocol-v3",
        "family_sha256": family["family_sha256"],
    }
    assert report["all_numeric_criteria_pass"] is True


def test_propagation_scorer_rejects_strata_outside_shared_family():
    family = build_g3_joint_acceptance_family_v2(
        family_id="protocol-v2",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("family-a", "LOS")],
        propagation_expected_strata=[("family-b", "NLOS", "total")],
    )

    with pytest.raises(ValueError, match="expected strata differ"):
        _run([_row("cluster-0", "episode-0")], joint_acceptance_family=family)


def test_metric_exceedance_uses_cluster_failure_bound_and_fails() -> None:
    records = [_row(f"cluster-{index}", "episode-0") for index in range(59)]
    records[0]["metrics"]["rms_delay_spread_s"] = _resolved(10e-9, 12e-9)

    report = _run(records)

    rms = _metric(report, "rms_delay_spread_s")
    assert rms["clusters_with_any_error_above_tolerance"] == 1
    assert rms["criterion_checks"]["cluster_failure_upper_bound_within_tolerance"] is False
    assert rms["criteria_status"] == "FAIL"
    assert report["all_numeric_criteria_pass"] is False


def test_unresolvable_measurement_is_masked_and_keeps_criterion_unresolved() -> None:
    records = [_row(f"cluster-{index}", "episode-0") for index in range(3)]
    records[0]["metrics"]["first_arrival_delay_s"] = _masked(
        "unresolved", "direct-arrival below declared time resolution"
    )

    result = _metric(_run(records), "first_arrival_delay_s")

    assert result["availability_counts"]["unresolved"] == 1
    assert result["mask_reasons"] == {"direct-arrival below declared time resolution": 1}
    assert result["criteria_status"] == "UNRESOLVED"
    assert "median_normalized_error" in result


def test_absent_diffuse_component_is_not_scored_as_zero() -> None:
    metrics = {
        name: _masked("not_applicable", "diffuse component absent") for name in _METRIC_NAMES
    }
    records = [
        _row(f"cluster-{index}", "episode-0", metrics, component="diffuse") for index in range(2)
    ]
    report = _run(records, expected=[("family-a", "LOS", "diffuse")])

    assert report["any_metric_assessable"] is False
    assert report["all_numeric_criteria_pass"] is False
    assert report["all_criteria_resolved_and_pass"] is False
    assert all(
        item["criteria_status"] == "NOT_APPLICABLE" for item in report["strata"][0]["metrics"]
    )
    assert all(
        item["availability_counts"]["not_applicable"] == 2
        for item in report["strata"][0]["metrics"]
    )


def test_missing_preregistered_component_stratum_is_rejected() -> None:
    records = [_row("cluster-a", "episode-a"), _row("cluster-b", "episode-b")]
    with pytest.raises(ValueError, match="missing=.*NLOS"):
        _run(records, expected=[("family-a", "LOS", "total"), ("family-a", "NLOS", "total")])


@pytest.mark.parametrize(
    "invalid",
    [
        _masked("unresolved", ""),
        {
            "candidate": None,
            "challenger": 1.0,
            "availability": "unresolved",
            "reason": "not measured",
        },
        {"candidate": -1e-9, "challenger": 0.0, "availability": "resolved", "reason": None},
    ],
)
def test_invalid_mask_and_metric_values_fail_closed(invalid: dict[str, Any]) -> None:
    records = [_row("cluster-a", "episode-a"), _row("cluster-b", "episode-b")]
    records[0]["metrics"]["arrival_angular_spread_deg"] = invalid

    with pytest.raises(ValueError):
        _run(records)


def test_total_component_cannot_be_marked_not_applicable() -> None:
    metrics = {name: _masked("not_applicable", "not present") for name in _METRIC_NAMES}
    with pytest.raises(ValueError, match="total-component metrics"):
        _run([_row("cluster-a", "episode-a", metrics), _row("cluster-b", "episode-b", metrics)])
