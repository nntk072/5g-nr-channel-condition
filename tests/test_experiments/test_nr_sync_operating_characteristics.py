"""Small deterministic tests for noise-calibrated NR cell-search evaluation."""

from __future__ import annotations

from copy import deepcopy

import pytest
from experiments.nr_sync_operating_characteristics import (
    compare_nr_sync_operating_characteristics,
    run_nr_sync_operating_characteristics,
)


def test_nr_sync_operating_study_separates_calibration_and_validation():
    kwargs = {
        "seed": 17,
        "calibration_trials": 16,
        "validation_trials": 24,
        "signal_trials_per_point": 4,
        "snr_db_values": (6.0,),
    }
    result = run_nr_sync_operating_characteristics(**kwargs)
    repeated = run_nr_sync_operating_characteristics(**kwargs)

    assert result == repeated
    assert result["schema"] == "loslab.nr_sync_operating_characteristics.v1"
    assert result["configuration"]["calibration_trials"] == 16
    assert result["configuration"]["fractional_timing_refinement"]["bounds_samples"] == [
        -0.5,
        0.5,
    ]
    assert result["false_alarm_validation"]["trials"] == 24
    assert len(result["false_alarm_validation"]["calibration_pss_scores"]) == 16
    assert len(result["false_alarm_validation"]["validation_accepted"]) == 24
    assert len(result["signal_points"]) == 2
    for point in result["signal_points"]:
        assert point["trials"] == 4
        assert len(point["per_trial"]) == 4
        assert point["accepted"] == sum(row["accepted"] for row in point["per_trial"])
        assert point["correct_acquisitions"] == sum(
            row["correct_acquisition"] for row in point["per_trial"]
        )


def test_nr_sync_comparison_pairs_trials_and_adjusts_family():
    study = run_nr_sync_operating_characteristics(
        seed=19,
        calibration_trials=12,
        validation_trials=20,
        signal_trials_per_point=8,
        snr_db_values=(0.0, 12.0),
    )
    control = deepcopy(study)
    treatment = deepcopy(study)
    for control_point, treatment_point in zip(
        control["signal_points"], treatment["signal_points"], strict=True
    ):
        for control_trial, treatment_trial in zip(
            control_point["per_trial"][:2],
            treatment_point["per_trial"][:2],
            strict=True,
        ):
            control_trial["correct_acquisition"] = False
            treatment_trial["correct_acquisition"] = True
    comparison = compare_nr_sync_operating_characteristics(
        control, treatment, bootstrap_seed=31, bootstrap_draws=100
    )

    assert comparison["schema"] == "loslab.nr_sync_paired_comparison.v1"
    assert len(comparison["operating_points"]) == 4
    assert all(
        row["improved_trials"] == 2 and row["regressed_trials"] == 0
        for row in comparison["operating_points"]
    )
    assert all("mcnemar_holm_adjusted_p_value" in row for row in comparison["operating_points"])

    treatment["signal_points"][0]["per_trial"][0]["nid1"] += 1
    with pytest.raises(ValueError, match="trial inputs/order differ"):
        compare_nr_sync_operating_characteristics(control, treatment, bootstrap_draws=100)
