from __future__ import annotations

import pytest
from evaluation.acceptance.joint_acceptance import build_joint_acceptance_family_v3
from evaluation.acceptance.receiver_impact import _crossing_snr, evaluate_receiver_impact


def _records(
    *,
    challenger_error_rate_at_10_db: float = 0.1,
    candidate_error_rate_at_10_db: float = 0.1,
    block_count: int = 20,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counts = {
        "candidate": {
            0.0: block_count // 2,
            10.0: round(block_count * candidate_error_rate_at_10_db),
            20.0: 0,
        },
        "challenger": {
            0.0: block_count // 2,
            10.0: round(block_count * challenger_error_rate_at_10_db),
            20.0: 0,
        },
    }
    for cluster in range(20):
        for config, snr_counts in counts.items():
            for snr, errors in snr_counts.items():
                for block in range(block_count):
                    rows.append(
                        {
                            "family_id": "family-a",
                            "condition": "LOS",
                            "cluster_id": f"channel-{cluster}",
                            "transport_block_id": f"block-{block}",
                            "snr_db": snr,
                            "configuration_id": config,
                            "crc_error": block < errors,
                            "decoder_outcome": "crc_fail" if block < errors else "crc_pass",
                        }
                    )
    return rows


def _evaluate(records: list[dict[str, object]]) -> dict[str, object]:
    return evaluate_receiver_impact(
        records,
        expected_strata=[("family-a", "LOS")],
        candidate_config_id="candidate",
        challenger_config_id="challenger",
        snr_grid_db=[0.0, 10.0, 20.0],
        comparison_snr_db=10.0,
        bootstrap_seed=7,
        bootstrap_draws=1000,
    )


def test_equal_paired_receiver_outcomes_meet_numeric_criteria_only() -> None:
    report = _evaluate(_records(block_count=2500))
    stratum = report["strata"][0]  # type: ignore[index]
    assert report["status"] == "NUMERICAL_CRITERIA_ONLY_NOT_QUALIFIED"
    assert stratum["candidate_bler"] == pytest.approx(0.1)
    assert stratum["challenger_minus_candidate_bler"] == pytest.approx(0.0)
    assert stratum["candidate_snr_at_10_percent_bler_db"] == 10.0
    assert stratum["snr_at_10_percent_shift_confidence_interval_db"] == [0.0, 0.0]
    assert report["familywise_claim_count"] == 4
    assert report["per_claim_alpha_bonferroni"] == pytest.approx(0.05 / 4)
    assert report["outcome_record_count"] == 300000
    assert len(report["outcome_records_sha256"]) == 64  # type: ignore[arg-type]
    assert stratum["decoder_outcome_counts"] == {
        "candidate": {"crc_fail": 30000, "crc_pass": 120000},
        "challenger": {"crc_fail": 30000, "crc_pass": 120000},
    }


def test_bler_difference_over_margin_fails_even_when_clusters_are_paired() -> None:
    report = _evaluate(_records(challenger_error_rate_at_10_db=0.2))
    stratum = report["strata"][0]  # type: ignore[index]
    assert stratum["challenger_minus_candidate_bler"] == pytest.approx(0.1)
    assert stratum["bler_criteria_pass"] is False
    assert report["status"] == "NUMERICAL_CRITERIA_NOT_MET_OR_UNRESOLVED"


def test_unbracketed_ten_percent_crossing_is_unresolved_without_extrapolation() -> None:
    rows = _records()
    for row in rows:
        if row["snr_db"] in {10.0, 20.0}:
            error_rate = 0.3 if row["snr_db"] == 10.0 else 0.2
            row["crc_error"] = int(str(row["transport_block_id"]).split("-")[1]) < round(
                20 * error_rate
            )
            row["decoder_outcome"] = "crc_fail" if row["crc_error"] else "crc_pass"
    report = _evaluate(rows)
    stratum = report["strata"][0]  # type: ignore[index]
    assert stratum["candidate_snr_at_10_percent_bler_db"] is None
    assert stratum["snr_at_10_percent_shift_confidence_interval_db"] is None
    assert stratum["snr_at_10_percent_criteria_pass"] is False


def test_all_failure_endpoint_roundoff_preserves_valid_bootstrap_crossings() -> None:
    rows = [
        {
            "family_id": "family-a",
            "condition": "LOS",
            "cluster_id": f"channel-{cluster}",
            "transport_block_id": "block-0",
            "snr_db": snr,
            "configuration_id": config,
            "crc_error": snr == 0.0,
            "decoder_outcome": "crc_fail" if snr == 0.0 else "crc_pass",
        }
        for cluster in range(20)
        for snr in (0.0, 1.0, 2.0)
        for config in ("candidate", "challenger")
    ]
    report = evaluate_receiver_impact(
        rows,
        expected_strata=[("family-a", "LOS")],
        candidate_config_id="candidate",
        challenger_config_id="challenger",
        snr_grid_db=[0.0, 1.0, 2.0],
        comparison_snr_db=0.0,
        bootstrap_draws=1_000,
    )

    stratum = report["strata"][0]  # type: ignore[index]
    assert stratum["valid_paired_crossing_bootstrap_fraction"] == 1.0
    assert stratum["candidate_bler_cluster_block_bootstrap_confidence_interval"] == [1.0, 1.0]
    assert stratum["snr_at_10_percent_shift_confidence_interval_db"] == [0.0, 0.0]
    assert stratum["snr_at_10_percent_criteria_pass"] is True


def test_missing_paired_outcome_is_rejected() -> None:
    rows = _records()
    rows.pop()
    with pytest.raises(ValueError, match="not fully paired"):
        _evaluate(rows)


def test_insufficient_independent_clusters_are_rejected() -> None:
    rows = _records()
    rows = [row for row in rows if row["cluster_id"] != "channel-19"]
    with pytest.raises(ValueError, match="at least 20 independent clusters"):
        _evaluate(rows)


def test_duplicate_block_outcome_is_rejected() -> None:
    rows = _records()
    rows.append(rows[0].copy())
    with pytest.raises(ValueError, match="duplicate"):
        _evaluate(rows)


def test_receiver_identity_fields_reject_unhashable_values_as_contract_errors() -> None:
    rows = _records()
    rows[0]["condition"] = []
    with pytest.raises(ValueError, match="family_id and condition"):
        _evaluate(rows)

    rows = _records()
    rows[0]["configuration_id"] = []
    with pytest.raises(ValueError, match="configuration_id"):
        _evaluate(rows)

    with pytest.raises(ValueError, match="configuration IDs"):
        evaluate_receiver_impact(
            _records(),
            expected_strata=[("family-a", "LOS")],
            candidate_config_id=[],
            challenger_config_id="challenger",
            snr_grid_db=[0.0, 10.0, 20.0],
            comparison_snr_db=10.0,
            bootstrap_draws=1000,
        )


def test_expected_receiver_strata_reject_malformed_pairs() -> None:
    with pytest.raises(ValueError, match="family IDs paired"):
        evaluate_receiver_impact(
            _records(),
            expected_strata=[("family-a", [])],
            candidate_config_id="candidate",
            challenger_config_id="challenger",
            snr_grid_db=[0.0, 10.0, 20.0],
            comparison_snr_db=10.0,
            bootstrap_draws=1000,
        )

    with pytest.raises(ValueError, match="family IDs paired"):
        evaluate_receiver_impact(
            _records(),
            expected_strata=[(" family-a", "LOS")],
            candidate_config_id="candidate",
            challenger_config_id="challenger",
            snr_grid_db=[0.0, 10.0, 20.0],
            comparison_snr_db=10.0,
            bootstrap_draws=1000,
        )

    padded_record = _records()
    padded_record[0]["family_id"] = "family-a "
    with pytest.raises(ValueError, match="family_id, cluster_id, and transport_block_id"):
        _evaluate(padded_record)


def test_outcome_digest_is_order_independent_and_binds_decoder_results() -> None:
    rows = _records()
    baseline = _evaluate(rows)
    reordered = _evaluate(list(reversed(rows)))
    assert baseline["outcome_records_sha256"] == reordered["outcome_records_sha256"]
    rows[0]["decoder_outcome"] = "different_outcome"
    changed = _evaluate(rows)
    assert baseline["outcome_records_sha256"] != changed["outcome_records_sha256"]


def test_observed_strata_must_match_preregistered_set() -> None:
    with pytest.raises(ValueError, match="do not match"):
        evaluate_receiver_impact(
            _records(),
            expected_strata=[("family-a", "LOS"), ("family-a", "NLOS")],
            candidate_config_id="candidate",
            challenger_config_id="challenger",
            snr_grid_db=[0.0, 10.0, 20.0],
            comparison_snr_db=10.0,
            bootstrap_draws=1000,
        )


def test_receiver_scorer_rejects_bootstrap_design_over_output_allocation_limit() -> None:
    with pytest.raises(
        ValueError, match="bootstrap output arrays exceed the configured allocation limit"
    ):
        evaluate_receiver_impact(
            [],
            expected_strata=[("family-a", "LOS")],
            candidate_config_id="candidate",
            challenger_config_id="challenger",
            snr_grid_db=[float(index) for index in range(2001)],
            comparison_snr_db=0.0,
            bootstrap_draws=2000,
        )


@pytest.mark.parametrize(
    "snr_grid",
    [[0.0, "10.0", 20.0], [False, 10.0, 20.0], [0j, 10.0, 20.0]],
)
def test_receiver_scorer_rejects_coercible_snr_grid_values(snr_grid: list[object]) -> None:
    with pytest.raises(ValueError, match="snr_grid_db values must be a finite real number"):
        evaluate_receiver_impact(
            [],
            expected_strata=[("family-a", "LOS")],
            candidate_config_id="candidate",
            challenger_config_id="challenger",
            snr_grid_db=snr_grid,
            comparison_snr_db=10.0,
        )


@pytest.mark.parametrize("comparison_snr", ["10.0", True, 10.0 + 0j])
def test_receiver_scorer_rejects_coercible_comparison_snr(comparison_snr: object) -> None:
    with pytest.raises(ValueError, match="comparison_snr_db must be a finite real number"):
        evaluate_receiver_impact(
            [],
            expected_strata=[("family-a", "LOS")],
            candidate_config_id="candidate",
            challenger_config_id="challenger",
            snr_grid_db=[0.0, 10.0, 20.0],
            comparison_snr_db=comparison_snr,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("record_snr", ["0.0", True, 0.0 + 0j])
def test_receiver_scorer_rejects_coercible_record_snr(record_snr: object) -> None:
    record = {
        "family_id": "family-a",
        "condition": "LOS",
        "cluster_id": "cluster-1",
        "transport_block_id": "block-1",
        "snr_db": record_snr,
        "configuration_id": "candidate",
        "crc_error": False,
        "decoder_outcome": "crc_pass",
    }
    with pytest.raises(ValueError, match="snr_db must be a finite real number"):
        evaluate_receiver_impact(
            [record],
            expected_strata=[("family-a", "LOS")],
            candidate_config_id="candidate",
            challenger_config_id="challenger",
            snr_grid_db=[0.0, 10.0, 20.0],
            comparison_snr_db=10.0,
        )


def test_receiver_intervals_join_the_shared_v3_family() -> None:
    family = build_joint_acceptance_family_v3(
        family_id="v3-test",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("family-a", "LOS")],
        propagation_expected_strata=[("family-a", "LOS", "total")],
        receiver_impact_expected_strata=[("family-a", "LOS")],
    )

    report = evaluate_receiver_impact(
        _records(block_count=2500),
        expected_strata=[("family-a", "LOS")],
        candidate_config_id="candidate",
        challenger_config_id="challenger",
        snr_grid_db=[0.0, 10.0, 20.0],
        comparison_snr_db=10.0,
        joint_acceptance_family=family,
        bootstrap_seed=7,
        bootstrap_draws=1000,
    )

    stratum = report["strata"][0]  # type: ignore[index]
    assert family["claim_count"] == 19
    assert report["schema"] == "receiver_impact_diagnostic_v2"
    assert report["familywise_claim_count"] == 19
    assert report["per_claim_alpha_bonferroni"] == pytest.approx(0.05 / 19)
    assert report["joint_acceptance_family"] == {
        "family_id": "v3-test",
        "family_sha256": family["family_sha256"],
    }
    assert stratum["interval_confidence_level"] == pytest.approx(1.0 - 0.05 / 19)
    assert report["status"] == "NUMERICAL_CRITERIA_ONLY_NOT_QUALIFIED"


def test_receiver_scorer_rejects_shared_family_with_different_strata() -> None:
    family = build_joint_acceptance_family_v3(
        family_id="v3-test",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("family-a", "LOS")],
        propagation_expected_strata=[("family-a", "LOS", "total")],
        receiver_impact_expected_strata=[("family-b", "NLOS")],
    )

    with pytest.raises(ValueError, match="expected strata differ"):
        evaluate_receiver_impact(
            _records(),
            expected_strata=[("family-a", "LOS")],
            candidate_config_id="candidate",
            challenger_config_id="challenger",
            snr_grid_db=[0.0, 10.0, 20.0],
            comparison_snr_db=10.0,
            joint_acceptance_family=family,
            bootstrap_draws=1000,
        )


@pytest.mark.parametrize(
    ("records", "overrides", "message"),
    [
        ([], {"snr_grid_db": [10.0]}, "at least two finite values"),
        ([], {"snr_grid_db": [0.0, 20.0, 10.0]}, "strictly increasing"),
        ([], {"comparison_snr_db": 5.0}, "exact member of the frozen grid"),
        ([], {"bootstrap_seed": True}, "bootstrap_seed must be a nonnegative integer"),
        ([], {"bootstrap_draws": 999}, "bootstrap_draws must be an integer"),
        ([], {"expected_strata": "family-a"}, "expected_strata must be a sequence"),
        ([], {"expected_strata": [("family-a",)]}, "each expected stratum"),
        ([], {"expected_strata": []}, "expected_strata must contain family IDs"),
        (
            [],
            {"expected_strata": [("family-a", "LOS"), ("family-a", "LOS")]},
            "must not contain duplicates",
        ),
        ([], {}, "records must be nonempty"),
        ([{}], {}, "all receiver outcome fields"),
        (
            _records()[:1],
            {"field": "cluster_id", "value": " "},
            "cluster_id and transport_block_id",
        ),
        (_records()[:1], {"field": "snr_db", "value": 99.0}, "SNR is outside"),
        (
            _records()[:1],
            {"field": "configuration_id", "value": "other"},
            "unexpected configuration_id",
        ),
        (_records()[:1], {"field": "crc_error", "value": "false"}, "crc_error must be boolean"),
        (
            _records()[:1],
            {"field": "decoder_outcome", "value": " "},
            "decoder_outcome must be a nonempty string",
        ),
    ],
)
def test_receiver_impact_rejects_invalid_frozen_inputs(records, overrides, message):
    arguments = {
        "expected_strata": [("family-a", "LOS")],
        "candidate_config_id": "candidate",
        "challenger_config_id": "challenger",
        "snr_grid_db": [0.0, 10.0, 20.0],
        "comparison_snr_db": 10.0,
        "bootstrap_draws": 1000,
    }
    record_field = overrides.pop("field", None)
    record_value = overrides.pop("value", None)
    if record_field is not None:
        records[0][record_field] = record_value
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        evaluate_receiver_impact(records, **arguments)


def test_receiver_crossing_rejects_malformed_bler_vectors():
    import numpy as np

    grid = np.asarray([0.0, 1.0, 2.0])
    assert _crossing_snr(grid, np.asarray([[1.0, 0.1, 0.0]])) is None
    assert _crossing_snr(grid, np.asarray([1.0, np.nan, 0.0])) is None
    assert _crossing_snr(grid, np.asarray([1.1, 0.1, 0.0])) is None
    assert _crossing_snr(grid, np.asarray([1.0, 0.2, 0.3])) is None


def test_receiver_impact_keeps_expected_strata_cluster_inventories_separate():
    records = _records()
    second_stratum = _records()
    for record in second_stratum:
        record["family_id"] = "family-b"
        record["condition"] = "NLOS"
    records.extend(second_stratum)

    report = evaluate_receiver_impact(
        records,
        expected_strata=[("family-a", "LOS"), ("family-b", "NLOS")],
        candidate_config_id="candidate",
        challenger_config_id="challenger",
        snr_grid_db=[0.0, 10.0, 20.0],
        comparison_snr_db=10.0,
        bootstrap_seed=7,
        bootstrap_draws=1000,
    )

    assert [(cell["family_id"], cell["condition"]) for cell in report["strata"]] == [
        ("family-a", "LOS"),
        ("family-b", "NLOS"),
    ]
