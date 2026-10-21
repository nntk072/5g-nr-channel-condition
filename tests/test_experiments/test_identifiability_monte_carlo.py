"""Tests for local Monte Carlo validation of two-path delay bounds."""

from __future__ import annotations

import pytest
from experiments.identifiability_monte_carlo import (
    build_global_mle_comparison,
    build_monte_carlo,
)


def test_local_mle_dispersion_is_reproducible_and_reports_crlb_ratio():
    arguments = {
        "seed": 4815,
        "trials": 8,
        "tone_counts": (32,),
        "tone_spacing_hz": 30_000.0,
        "snr_db_values": (40.0,),
        "separation_fractions": (0.5,),
        "gain_phase_differences_deg": (90.0,),
    }

    first = build_monte_carlo(**arguments)
    second = build_monte_carlo(**arguments)

    assert first == second
    assert first["evidence_grade"] == (
        "ORACLE_INITIALIZED_LOCAL_MLE_MONTE_CARLO_NOT_RECEIVER_QUALIFICATION"
    )
    case = first["cases"][0]
    assert case["optimizer_successes"] == 8
    assert case["relative_delay_crlb_std_ns"] > 0
    assert case["empirical_std_ns"] > 0
    assert case["empirical_std_to_crlb_ratio"] == pytest.approx(
        case["empirical_std_ns"] / case["relative_delay_crlb_std_ns"]
    )
    assert len(case["per_trial_relative_delay_errors_ns"]) == 8


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"trials": 1}, "at least 2"),
        ({"tone_counts": (8,)}, "integers >= 16"),
        ({"separation_fractions": (0.0,)}, "finite positive"),
    ],
)
def test_local_mle_sweep_rejects_invalid_configuration(arguments, message):
    options = {
        "seed": 1,
        "trials": 2,
        "tone_counts": (32,),
        "tone_spacing_hz": 30_000.0,
        "snr_db_values": (20.0,),
        "separation_fractions": (0.5,),
        "gain_phase_differences_deg": (90.0,),
    }
    options.update(arguments)

    with pytest.raises(ValueError, match=message):
        build_monte_carlo(**options)


def test_omp_initialized_comparison_is_paired_and_reproducible():
    arguments = {
        "seed": 9031,
        "trials": 8,
        "tone_counts": (32,),
        "tone_spacing_hz": 30_000.0,
        "snr_db_values": (40.0,),
        "separation_fractions": (0.5,),
        "gain_phase_differences_deg": (90.0,),
        "delay_grid_step_fraction": 1.0 / 16.0,
        "maximum_delay_fraction": 1.0,
    }

    first = build_global_mle_comparison(**arguments)
    second = build_global_mle_comparison(**arguments)

    assert first == second
    assert first["evidence_grade"] == (
        "PAIRED_ANALYTICAL_TWO_PATH_MONTE_CARLO_NOT_RECEIVER_QUALIFICATION"
    )
    case = first["cases"][0]
    assert case["trials"] == 8
    assert case["path_count_given_to_omp"] == 2
    assert case["oracle_local_successes"] == 8
    assert len(case["per_trial_omp_grid_errors_ns"]) == 8
    assert len(case["per_trial_omp_refined_errors_ns"]) == case["omp_refined_successes"]
    assert len(case["per_trial_oracle_local_errors_ns"]) == 8


def test_omp_comparison_requires_delay_grid_to_cover_true_paths():
    with pytest.raises(ValueError, match="must exceed max separation"):
        build_global_mle_comparison(
            seed=1,
            trials=2,
            tone_counts=(32,),
            tone_spacing_hz=30_000.0,
            snr_db_values=(20.0,),
            separation_fractions=(0.5,),
            gain_phase_differences_deg=(90.0,),
            maximum_delay_fraction=0.7,
        )
