"""Small deterministic tests for noise-calibrated NR cell-search evaluation."""

from __future__ import annotations

from experiments.nr_sync_operating_characteristics import (
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
