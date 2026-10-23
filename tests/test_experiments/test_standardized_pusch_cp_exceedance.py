"""Tests for the bounded standardized PUSCH CP-exceedance diagnostic."""

from __future__ import annotations

import pytest
from experiments.standardized_pusch_cp_exceedance import (
    _paired_table,
    _validate_snr_grid,
    run_cp_exceedance_probe,
    run_cp_exceedance_snr_curve,
    wilson_interval,
)


def test_wilson_interval_has_valid_bounds_at_extreme_outcomes() -> None:
    assert wilson_interval(0, 16) == pytest.approx((0.0, 0.1936076805))
    assert wilson_interval(16, 16) == pytest.approx((0.8063923195, 1.0))
    with pytest.raises(ValueError, match="0 <= successes <= trials"):
        wilson_interval(17, 16)


def test_paired_pass_table_accounts_for_every_trial() -> None:
    table = _paired_table([True, True, False, False], [True, False, True, False])
    assert table == {
        "both_pass": 1,
        "left_only_pass": 1,
        "right_only_pass": 1,
        "both_fail": 1,
    }


def test_snr_grid_requires_ordered_distinct_finite_points() -> None:
    assert _validate_snr_grid((20, 40.0)) == (20.0, 40.0)
    with pytest.raises(ValueError, match="at least two points"):
        _validate_snr_grid((20.0,))
    with pytest.raises(ValueError, match="unique and strictly increasing"):
        _validate_snr_grid((40.0, 20.0))
    with pytest.raises(ValueError, match="unique and strictly increasing"):
        _validate_snr_grid((20.0, 20.0))
    with pytest.raises(ValueError, match="finite"):
        _validate_snr_grid((20.0, float("inf")))
    with pytest.raises(TypeError, match="booleans"):
        _validate_snr_grid((20.0, True))


@pytest.mark.native_sionna
def test_repeated_cp_exceedance_probe_keeps_parent_and_oracle_semantics() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sionna.phy")

    report = run_cp_exceedance_probe(samples=4)

    assert report["schema"] == "umcd.g5_cp_exceedance_receiver_probe.v5"
    assert report["status"].startswith("exploratory")
    assert report["configuration"]["trial_unit"].startswith("independent TDL draw")
    assert (
        report["configuration"]["max_excess_delay_s"]
        > report["configuration"]["cyclic_prefix_duration_s"]
    )
    assert len(report["configuration_sha256"]) == 64
    assert len(report["parent_lineage"]) == 4
    assert len({item["channel_truth_sha256"] for item in report["parent_lineage"]}) == 4
    for condition in report["outcomes"].values():
        assert condition["independent_channel_draws"] == 4
        assert len(condition["crc_pass_by_draw"]) == 4
        assert "payload_mismatch_by_draw" not in condition
        assert 0.0 <= condition["wilson_95_percent_interval"][0]
        assert condition["wilson_95_percent_interval"][1] <= 1.0
    for paired in report["paired_pass_table"].values():
        assert sum(paired.values()) == 4
    for comparison in report["paired_time_domain_minus_ideal_cfr_bler"].values():
        assert comparison["independent_groups"] == 4
        assert comparison["paired_trials"] == 4
        assert comparison["sampling_unit"].startswith("independent group")
        lower, upper = comparison["cluster_bootstrap_ci95"]
        assert lower <= comparison["treatment_minus_control_error_rate"] <= upper


@pytest.mark.native_sionna
def test_cpu_snr_curve_reuses_exact_parents_and_pairs_receiver_outcomes() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sionna.phy")

    report = run_cp_exceedance_snr_curve(samples=4, snr_db=(60.0, 80.0))

    assert report["schema"] == "umcd.g5_cp_exceedance_snr_curve.v3"
    assert report["status"].startswith("exploratory")
    assert report["configuration"]["snr_db"] == [60.0, 80.0]
    assert report["configuration"]["independent_parent_count"] == 4
    assert len(report["parent_lineage"]) == 4
    assert len(report["point_report_sha256"]) == 2
    assert len(set(report["point_report_sha256"])) == 2
    assert [point["snr_db"] for point in report["points"]] == [60.0, 80.0]
    for point in report["points"]:
        for comparison in point["paired_time_domain_minus_ideal_cfr_bler"].values():
            assert comparison["independent_groups"] == 4
            assert comparison["paired_trials"] == 4
            lower, upper = comparison["cluster_bootstrap_ci95"]
            assert lower <= comparison["treatment_minus_control_error_rate"] <= upper
