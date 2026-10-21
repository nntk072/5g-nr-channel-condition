"""Tests for leave-one-condition-out profile-likelihood calibration."""

from __future__ import annotations

import pytest
from experiments.identifiability_cross_condition import build_cross_condition_calibration_sweep


def test_cross_condition_calibration_excludes_target_and_is_reproducible():
    arguments = {
        "seed": 20261026,
        "trials": 8,
        "calibration_trials_per_condition": 20,
        "tone_counts": (32,),
        "tone_spacing_hz": 30_000.0,
        "snr_db_values": (20.0, 30.0),
        "separation_fractions": (0.1, 0.5),
        "gain_phase_differences_deg": (0.0, 90.0),
        "delay_grid_step_fraction": 1.0 / 16.0,
        "maximum_delay_fraction": 1.0,
    }

    first = build_cross_condition_calibration_sweep(**arguments)
    second = build_cross_condition_calibration_sweep(**arguments)

    assert first == second
    assert first["evidence_grade"] == ("FACTOR_LEVEL_HOLDOUT_ANALYTICAL_CALIBRATION_STRESS_TEST")
    assert len(first["cases"]) == 8
    for case in first["cases"]:
        policies = case["calibration_transfer_policies"]
        condition_policy = policies["leave_one_condition_out"]
        assert condition_policy["calibration_records"] == 7 * 20
        assert len(condition_policy["source_conditions"]) == 7
        assert len(case["calibration_nonconformity_scores"]) == 20
        assert len(case["evaluation_records"]) == 8
        assert all(
            not (
                source["snr_db"] == case["snr_db"]
                and source["separation_fraction"]
                == case["separation_fraction_of_reciprocal_bandwidth"]
                and source["gain_phase_difference_deg"] == case["gain_phase_difference_deg"]
            )
            for source in condition_policy["source_conditions"]
        )
        assert policies["leave_one_snr_level_out"]["calibration_records"] == 4 * 20
        assert policies["leave_one_separation_level_out"]["calibration_records"] == 4 * 20
        assert policies["leave_one_gain_phase_level_out"]["calibration_records"] == 4 * 20
        assert all(
            source["snr_db"] != case["snr_db"]
            for source in policies["leave_one_snr_level_out"]["source_conditions"]
        )
        assert all(
            source["separation_fraction"] != case["separation_fraction_of_reciprocal_bandwidth"]
            for source in policies["leave_one_separation_level_out"]["source_conditions"]
        )
        assert all(
            source["gain_phase_difference_deg"] != case["gain_phase_difference_deg"]
            for source in policies["leave_one_gain_phase_level_out"]["source_conditions"]
        )
        assert case["leave_one_condition_out_95_median_set_span_ns"] >= 0
        assert case["within_condition_95_median_set_span_ns"] >= 0


def test_cross_condition_sweep_requires_finite_95_percent_calibration():
    with pytest.raises(ValueError, match="at least 19"):
        build_cross_condition_calibration_sweep(
            seed=1,
            trials=2,
            calibration_trials_per_condition=18,
            tone_counts=(32,),
            tone_spacing_hz=30_000.0,
            snr_db_values=(20.0, 30.0),
            separation_fractions=(0.1, 0.5),
            gain_phase_differences_deg=(0.0, 90.0),
            maximum_delay_fraction=1.0,
        )
