"""Tests for two-path profile-likelihood uncertainty diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from experiments.identifiability_profile_likelihood import (
    _split_conformal_threshold,
    build_profile_likelihood_sweep,
    profile_relative_delay_likelihood,
)


def test_profile_likelihood_recovers_well_separated_noiseless_relative_delay():
    frequencies = (np.arange(64) - 32) * 30_000.0
    grid = np.arange(0.0, 1.51, 1.0 / 128.0) / np.ptp(frequencies)
    truth = 0.5 / np.ptp(frequencies)
    gains = np.asarray([1.0 + 0j, 0.7j])
    delays = np.asarray([0.2, 0.2 + 0.5]) / np.ptp(frequencies)
    observed = np.exp(-2j * np.pi * frequencies[:, None] * delays[None, :]) @ gains

    result = profile_relative_delay_likelihood(observed, frequencies, 1e-8, grid)

    assert result["map_relative_delay_s"] == pytest.approx(truth, abs=1e-12)
    assert result["interval_95_mass"] >= 0.95
    assert result["profiled_pair_count"] > 0
    assert len(result["relative_delay_mass"]) == len(grid)
    assert sum(result["relative_delay_mass"]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("noise_variance", "grid", "message"),
    [
        (0.0, np.linspace(0.0, 1.0, 4), "noise_variance"),
        (1.0, np.asarray([0.0, 0.0, 1.0]), "strictly increasing"),
    ],
)
def test_profile_likelihood_rejects_invalid_inputs(noise_variance, grid, message):
    frequencies = np.arange(16, dtype=np.float64)
    observed = np.ones(16, dtype=np.complex128)
    with pytest.raises(ValueError, match=message):
        profile_relative_delay_likelihood(observed, frequencies, noise_variance, grid)


def test_profile_likelihood_sweep_is_reproducible_and_reports_empirical_coverage():
    arguments = {
        "seed": 20261025,
        "trials": 8,
        "calibration_trials": 20,
        "tone_counts": (32,),
        "tone_spacing_hz": 30_000.0,
        "snr_db_values": (30.0,),
        "separation_fractions": (0.5,),
        "gain_phase_differences_deg": (90.0,),
        "delay_grid_step_fraction": 1.0 / 16.0,
        "maximum_delay_fraction": 1.0,
    }

    first = build_profile_likelihood_sweep(**arguments)
    second = build_profile_likelihood_sweep(**arguments)

    assert first == second
    assert first["evidence_grade"] == (
        "CONDITION_CALIBRATED_ANALYTICAL_TWO_PATH_PROFILE_LIKELIHOOD_DIAGNOSTIC"
    )
    case = first["cases"][0]
    assert case["trials"] == 8
    assert case["calibration_trials"] == 20
    assert 0.0 <= case["interval_95_empirical_coverage"] <= 1.0
    assert case["interval_95_wilson_95_ci"][0] <= case["interval_95_empirical_coverage"]
    assert case["median_interval_95_width_ns"] >= case["median_interval_68_width_ns"]
    assert len(case["per_trial"]) == 8
    assert case["split_conformal_95_empirical_coverage"] == 1.0
    assert case["split_conformal_95_median_set_span_ns"] >= 0.0


def test_split_conformal_threshold_uses_finite_sample_order_statistic():
    scores = np.arange(1.0, 21.0)
    assert _split_conformal_threshold(scores, 0.95) == 20.0


def test_profile_likelihood_requires_enough_calibration_for_finite_95_percent_threshold():
    with pytest.raises(ValueError, match="at least 19"):
        build_profile_likelihood_sweep(
            seed=1,
            trials=2,
            calibration_trials=18,
            tone_counts=(32,),
            tone_spacing_hz=30_000.0,
            snr_db_values=(20.0,),
            separation_fractions=(0.5,),
            gain_phase_differences_deg=(90.0,),
            maximum_delay_fraction=1.0,
        )
