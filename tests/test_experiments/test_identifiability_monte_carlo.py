"""Tests for local Monte Carlo validation of two-path delay bounds."""

from __future__ import annotations

import pytest
from experiments.identifiability_monte_carlo import build_monte_carlo


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
