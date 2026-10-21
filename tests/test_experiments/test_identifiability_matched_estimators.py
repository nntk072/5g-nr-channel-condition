"""Tests for matched delay-estimator calibration experiments."""

from __future__ import annotations

import pytest
from experiments.identifiability_matched_estimators import (
    run_matched_uncertainty_estimators,
)


def _run():
    return run_matched_uncertainty_estimators(
        seed=91,
        trials=8,
        calibration_trials=19,
        tone_counts=(32,),
        tone_spacing_hz=30_000.0,
        snr_db_values=(20.0,),
        separation_fractions=(0.2,),
        gain_phase_differences_deg=(45.0,),
        profile_grid_step_fraction=0.125,
        omp_grid_step_fraction=0.125,
        maximum_delay_fraction=1.0,
    )


def test_matched_uncertainty_study_is_paired_and_reproducible():
    result = _run()

    assert result == _run()
    assert result["schema"] == "loslab.identifiability_matched_estimators.v1"
    assert result["calibration_protocol"].startswith("per-condition calibration")
    assert result["paired_bootstrap_draws"] == 5_000
    assert len(result["cases"]) == 1
    case = result["cases"][0]
    assert len(case["calibration_trials_records"]) == 19
    assert len(case["evaluation_trials_records"]) == 8
    assert 0.0 <= case["profile_conformal_coverage"] <= 1.0
    assert 0.0 <= case["omp_conformal_coverage"] <= 1.0
    assert -1.0 <= case["paired_profile_minus_omp_coverage_change"] <= 1.0
    assert len(case["paired_profile_minus_omp_coverage_bootstrap_ci95"]) == 2
    assert len(case["paired_profile_minus_omp_mean_span_bootstrap_ci95_ns"]) == 2
    assert "paired_coverage_mcnemar_holm_p_value" in case


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"calibration_trials": 18}, "calibration_trials must be at least 19"),
        ({"confidence_level": 1.0}, "confidence_level must be finite"),
        ({"maximum_delay_fraction": 0.4}, "must exceed max separation"),
        ({"bootstrap_draws": 0}, "bootstrap_draws must be positive"),
    ],
)
def test_matched_uncertainty_study_rejects_invalid_protocol(overrides, message):
    kwargs = {
        "seed": 91,
        "trials": 8,
        "calibration_trials": 19,
        "tone_counts": (32,),
        "tone_spacing_hz": 30_000.0,
        "snr_db_values": (20.0,),
        "separation_fractions": (0.2,),
        "gain_phase_differences_deg": (45.0,),
        "profile_grid_step_fraction": 0.125,
        "omp_grid_step_fraction": 0.125,
        "maximum_delay_fraction": 1.0,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        run_matched_uncertainty_estimators(**kwargs)


def test_matched_uncertainty_study_rejects_boolean_bootstrap_draw_count():
    with pytest.raises(TypeError, match="bootstrap_draws must be an integer"):
        run_matched_uncertainty_estimators(
            seed=91,
            trials=8,
            calibration_trials=19,
            tone_counts=(32,),
            tone_spacing_hz=30_000.0,
            snr_db_values=(20.0,),
            separation_fractions=(0.2,),
            gain_phase_differences_deg=(45.0,),
            bootstrap_draws=True,
        )
