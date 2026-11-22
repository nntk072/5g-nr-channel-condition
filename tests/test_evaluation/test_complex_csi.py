from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from evaluation.complex_csi import evaluate_complex_csi, symmetric_complex_csi_error
from evaluation.joint_acceptance import (
    build_g3_joint_acceptance_family,
    build_g3_joint_acceptance_family_v3,
)
from scipy.stats import beta


def _record(
    cluster: str,
    episode: str,
    candidate: np.ndarray,
    challenger: np.ndarray,
) -> dict[str, object]:
    return {
        "family_id": "scene-family-a",
        "condition": "LOS",
        "cluster_id": cluster,
        "episode_id": episode,
        "evaluation_grid_id": "grid-a",
        "candidate_configuration_id": "candidate-v1",
        "challenger_configuration_id": "challenger-v1",
        "candidate_cfr": candidate,
        "challenger_cfr": challenger,
        "h_floor": 1.0e-12,
    }


def _view_contract(view_id: str, *, phase_gauge_id: str = "gauge-a") -> dict[str, object]:
    return {
        "view_id": view_id,
        "link_id": "link-1",
        "frame_times_s": [0.0],
        "rx_port_ids": ["rx-1"],
        "tx_port_ids": ["tx-1"],
        "carrier_hz": 3.5e9,
        "subcarrier_offsets_hz": [-15_000.0, 0.0],
        "delay_reference": "absolute_propagation_delay",
        "power_scale": "absolute_power_gain",
        "phase_gauge_id": phase_gauge_id,
    }


def _score(records: list[dict[str, object]], **kwargs: object) -> dict[str, object]:
    expected = sorted({(row["family_id"], row["condition"]) for row in records})
    return evaluate_complex_csi(records, expected_strata=expected, **kwargs)  # type: ignore[arg-type]


def test_complex_csi_view_contract_schema_is_strict_and_requires_normalization_recipe() -> None:
    schema_path = (
        Path(__file__).parents[2]
        / "evaluation/schema/complex-csi-channel-view-contract.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(_view_contract("view-1"))

    with pytest.raises(jsonschema.ValidationError):
        validator.validate({**_view_contract("view-2"), "unreviewed_field": True})
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({**_view_contract("view-3"), "power_scale": "normalized_power"})


def test_symmetric_complex_csi_error_matches_frozen_formula_without_phase_alignment() -> None:
    candidate = np.asarray([1.0 + 0.0j, 0.0 + 1.0j])
    challenger = np.asarray([0.0 + 1.0j, -1.0 + 0.0j])

    expected = np.linalg.norm(candidate - challenger) / (
        (np.linalg.norm(candidate) + np.linalg.norm(challenger)) / 2.0
    )
    assert symmetric_complex_csi_error(candidate, challenger, h_floor=1e-12) == pytest.approx(
        expected
    )
    assert symmetric_complex_csi_error(candidate, candidate, h_floor=1e-12) == 0.0


@pytest.mark.parametrize(
    ("candidate", "challenger", "floor", "message"),
    [
        (np.ones(2), np.ones(2), 1e-12, "complex-valued"),
        (np.ones(2, dtype=complex), np.ones(3, dtype=complex), 1e-12, "same nonempty shape"),
        (np.asarray([complex(np.nan, 0)]), np.ones(1, dtype=complex), 1e-12, "finite values"),
        (np.zeros(1, dtype=complex), np.zeros(1, dtype=complex), 0.0, "positive real number"),
        (np.zeros(1, dtype=complex), np.zeros(1, dtype=complex), True, "positive real number"),
    ],
)
def test_symmetric_complex_csi_error_rejects_invalid_inputs(
    candidate: np.ndarray, challenger: np.ndarray, floor: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        symmetric_complex_csi_error(candidate, challenger, h_floor=floor)


def test_complex_csi_view_contract_uses_declared_phase_gauge_audit() -> None:
    exact = np.ones((1, 1, 1, 2), dtype=np.complex128)
    records = [
        {
            **_record(f"cluster-{index}", "episode-0", exact, exact),
            "candidate_view_contract": _view_contract(f"candidate-{index}"),
            "challenger_view_contract": _view_contract(f"challenger-{index}"),
        }
        for index in range(2)
    ]

    report = _score(records, bootstrap_draws=1_000)

    assert report["phase_reference_audit"] == {
        "status": "ALL_DECLARED_CONTRACTS_MATCH",
        "checked_record_count": 2,
        "total_record_count": 2,
        "disposition": "eligible_same_declared_gauge",
        "meaning": "declared view contracts match; calibration and physical phase truth remain unverified",
    }


def test_complex_csi_view_contract_rejects_gauge_mismatch_and_masked_channels() -> None:
    exact = np.ones((1, 1, 1, 2), dtype=np.complex128)
    mismatch = [
        {
            **_record("cluster-a", "episode-0", exact, exact),
            "candidate_view_contract": _view_contract("candidate", phase_gauge_id="gauge-a"),
            "challenger_view_contract": _view_contract("challenger", phase_gauge_id="gauge-b"),
        },
        {
            **_record("cluster-b", "episode-0", exact, exact),
            "candidate_view_contract": _view_contract("candidate-b"),
            "challenger_view_contract": _view_contract("challenger-b"),
        },
    ]
    with pytest.raises(ValueError, match="phase_gauge_mismatch"):
        _score(mismatch, bootstrap_draws=1_000)

    zeroed_masked = np.ones_like(exact)
    zeroed_masked[..., 1] = 0.0
    masked = [
        {
            **_record(f"cluster-{index}", "episode-0", zeroed_masked, exact),
            "candidate_view_contract": {
                **_view_contract(f"candidate-{index}"),
                "valid_mask": [[[[True, False]]]],
            },
            "challenger_view_contract": _view_contract(f"challenger-{index}"),
        }
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="fully valid channel views"):
        _score(masked, bootstrap_draws=1_000)


def test_complex_csi_evaluation_weights_clusters_equally_and_never_qualifies() -> None:
    exact = np.ones(4, dtype=np.complex128)
    slightly_different = exact * (1.0 + 1e-4)
    records = [
        *[_record("cluster-a", f"a-{i}", slightly_different, exact) for i in range(25)],
        _record("cluster-b", "b-0", exact, exact),
    ]

    report = _score(records, bootstrap_draws=2_000)

    stratum = report["strata"][0]
    assert stratum["independent_clusters"] == 2
    assert stratum["episodes"] == 26
    assert stratum["median_e_h_equal_cluster_weight"] == pytest.approx(5e-5, rel=1e-4)
    assert stratum["numeric_criteria_pass"] is False
    assert stratum["criterion_checks"]["cluster_failure_upper_confidence_bound"] is False
    assert report["all_numeric_criteria_pass"] is False
    assert report["status"] == "NUMERICAL_CRITERIA_ONLY_NOT_QUALIFIED"
    assert report["admission_possible"] is False
    assert report["configuration"]["phase_alignment"] is False
    assert report["configuration"]["channel_normalization"] is False
    assert report["configuration"]["familywise_claim_count"] == 2
    assert report["configuration"]["per_claim_alpha_bonferroni"] == pytest.approx(0.025)
    assert report["configuration"]["per_claim_confidence_level"] == pytest.approx(0.975)
    assert (
        "not jointly corrected with other G3 estimands"
        in report["configuration"]["confidence_scope"]
    )


def test_cluster_failure_bound_uses_one_sided_exact_binomial_interval() -> None:
    baseline = np.ones(2, dtype=np.complex128)
    records = [
        _record(
            f"cluster-{index}",
            "episode-0",
            baseline * (1.1 if index == 0 else 1),
            baseline,
        )
        for index in range(59)
    ]

    stratum = _score(records, bootstrap_draws=1_000)["strata"][0]

    assert stratum["cluster_failure_count"] == 1
    assert stratum["cluster_failure_rate_one_sided_upper_confidence_bound"] == pytest.approx(
        beta.ppf(0.975, 2, 58)
    )
    assert stratum["cluster_failure_upper_confidence_level"] == pytest.approx(0.975)
    assert stratum["criterion_checks"]["cluster_failure_upper_confidence_bound"] is False


def test_zero_failure_rate_needs_59_clusters_for_the_frozen_five_percent_bound() -> None:
    same = np.ones(2, dtype=np.complex128)
    for count, expected in ((71, False), (72, True)):
        records = [_record(f"cluster-{index}", "episode-0", same, same) for index in range(count)]
        stratum = _score(records, bootstrap_draws=1_000)["strata"][0]
        assert stratum["cluster_failure_count"] == 0
        assert stratum["criterion_checks"]["cluster_failure_upper_confidence_bound"] is expected


def test_complex_csi_familywise_bounds_cover_all_expected_strata() -> None:
    same = np.ones(2, dtype=np.complex128)
    records = [
        _record("los-a", "episode-0", same, same),
        _record("los-b", "episode-0", same, same),
        {**_record("nlos-a", "episode-0", same, same), "condition": "NLOS"},
        {**_record("nlos-b", "episode-0", same, same), "condition": "NLOS"},
    ]
    report = evaluate_complex_csi(
        records,
        expected_strata=[("scene-family-a", "LOS"), ("scene-family-a", "NLOS")],
        bootstrap_draws=1_000,
    )

    assert report["configuration"]["familywise_claim_count"] == 4
    assert report["configuration"]["per_claim_alpha_bonferroni"] == pytest.approx(0.0125)
    assert report["configuration"]["per_claim_confidence_level"] == pytest.approx(0.9875)
    assert len(report["strata"]) == 2
    assert all(
        item["p95_e_h_upper_confidence_level"] == pytest.approx(0.9875) for item in report["strata"]
    )


def test_complex_csi_uses_the_same_joint_family_as_material_calibration() -> None:
    same = np.ones(2, dtype=np.complex128)
    records = [_record(f"cluster-{index}", "episode", same, same) for index in range(3)]
    family = build_g3_joint_acceptance_family_v3(
        family_id="g3-protocol-2026-01",
        material_expected_strata=[("concrete-a", "V", "angle-20deg")],
        complex_csi_expected_strata=[("scene-family-a", "LOS")],
        propagation_expected_strata=[("scene-family-a", "LOS", "total")],
        receiver_impact_expected_strata=[("scene-family-a", "LOS")],
    )

    report = evaluate_complex_csi(
        records,
        expected_strata=[("scene-family-a", "LOS")],
        bootstrap_draws=1_000,
        joint_acceptance_family=family,
    )

    assert family["claim_count"] == 19
    assert report["configuration"]["familywise_claim_count"] == 19
    assert report["configuration"]["per_claim_alpha_bonferroni"] == pytest.approx(0.05 / 19)
    assert report["configuration"]["joint_acceptance_family"] == {
        "family_id": "g3-protocol-2026-01",
        "family_sha256": family["family_sha256"],
    }
    assert (
        "jointly corrected with the material-calibration claims"
        in report["configuration"]["confidence_scope"]
    )


def test_complex_csi_rejects_strata_that_do_not_match_joint_family() -> None:
    same = np.ones(2, dtype=np.complex128)
    family = build_g3_joint_acceptance_family(
        family_id="g3-protocol-2026-01",
        material_expected_strata=[("concrete-a", "V", "angle-20deg")],
        complex_csi_expected_strata=[("scene-family-a", "LOS")],
    )
    with pytest.raises(ValueError, match="expected strata differ"):
        evaluate_complex_csi(
            [_record("cluster-1", "episode", same, same)],
            expected_strata=[("different-family", "LOS")],
            bootstrap_draws=1_000,
            joint_acceptance_family=family,
        )


def test_complex_csi_evaluation_requires_unique_episodes_and_multiple_clusters() -> None:
    same = np.ones(2, dtype=np.complex128)
    duplicate = [_record("cluster-a", "episode-0", same, same)] * 2
    with pytest.raises(ValueError, match="duplicate episode ID"):
        _score(duplicate)

    with pytest.raises(ValueError, match="at least 2 clusters"):
        _score([_record("cluster-a", "episode-0", same, same)])


def test_complex_csi_evaluation_rejects_mismatched_grid_or_floor_within_stratum() -> None:
    same = np.ones(2, dtype=np.complex128)
    records = [
        _record("cluster-a", "episode-a", same, same),
        _record("cluster-b", "episode-b", same, same),
    ]
    records[1]["evaluation_grid_id"] = "different-grid"

    with pytest.raises(ValueError, match="must match within"):
        _score(records)

    records[1]["evaluation_grid_id"] = "grid-a"
    records[1]["h_floor"] = 2e-12
    with pytest.raises(ValueError, match="must match within"):
        _score(records)


def test_complex_csi_evaluation_rejects_non_string_condition() -> None:
    same = np.ones(2, dtype=np.complex128)
    records = [
        _record("cluster-a", "episode-a", same, same),
        _record("cluster-b", "episode-b", same, same),
    ]
    records[0]["condition"] = ["LOS"]

    with pytest.raises(ValueError, match="condition must be LOS or NLOS"):
        evaluate_complex_csi(records, expected_strata=[("scene-family-a", "LOS")])


def test_complex_csi_evaluation_rejects_missing_preregistered_strata() -> None:
    same = np.ones(2, dtype=np.complex128)
    records = [
        _record("cluster-a", "episode-a", same, same),
        _record("cluster-b", "episode-b", same, same),
    ]

    with pytest.raises(ValueError, match="missing=.*NLOS"):
        evaluate_complex_csi(
            records,
            expected_strata=[("scene-family-a", "LOS"), ("scene-family-a", "NLOS")],
        )

    with pytest.raises(ValueError, match="expected_strata must be nonempty"):
        evaluate_complex_csi(records, expected_strata=[])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bootstrap_seed": -1},
        {"bootstrap_draws": 999},
        {"confidence_level": 1.0},
    ],
)
def test_complex_csi_evaluation_rejects_invalid_procedure_settings(
    kwargs: dict[str, object],
) -> None:
    same = np.ones(2, dtype=np.complex128)
    records = [
        _record("cluster-a", "episode-a", same, same),
        _record("cluster-b", "episode-b", same, same),
    ]
    with pytest.raises(ValueError):
        _score(records, **kwargs)
