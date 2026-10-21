"""Tests for paired estimator calibration-transfer stress analysis."""

from __future__ import annotations

import itertools

import pytest
from experiments.identifiability_matched_transfer import (
    analyze_matched_cross_condition_transfer,
)


def _matched_result():
    cases = []
    for tone_count, snr_db, separation, phase in itertools.product(
        (32,), (20.0, 30.0), (0.1, 1.0), (0.0, 90.0)
    ):
        bandwidth_hz = (tone_count - 1) * 30_000.0
        offset = int(snr_db == 30.0) + int(separation == 1.0) + int(phase == 90.0)
        calibration = [
            {
                "profile_nonconformity": float((index + offset) % 5),
                "omp_absolute_error_s": float((index + offset + 1) % 5) / bandwidth_hz,
            }
            for index in range(19)
        ]
        evaluation = [
            {
                "profile_nonconformity": float((index + offset) % 5),
                "omp_absolute_error_s": float((index + offset + 1) % 5) / bandwidth_hz,
            }
            for index in range(8)
        ]
        cases.append(
            {
                "tone_count": tone_count,
                "snr_db": snr_db,
                "separation_fraction_of_reciprocal_bandwidth": separation,
                "gain_phase_difference_deg": phase,
                "bandwidth_hz": bandwidth_hz,
                "calibration_trials_records": calibration,
                "evaluation_trials_records": evaluation,
            }
        )
    return {
        "schema": "loslab.identifiability_matched_estimators.v1",
        "source_revision": "test-revision",
        "cases": cases,
    }


def test_matched_transfer_excludes_target_and_is_reproducible():
    source = _matched_result()
    first = analyze_matched_cross_condition_transfer(source, seed=11, bootstrap_draws=100)
    second = analyze_matched_cross_condition_transfer(source, seed=11, bootstrap_draws=100)

    assert first == second
    assert first["evidence_grade"] == "MATCHED_ANALYTICAL_CROSS_CONDITION_CALIBRATION_STRESS_TEST"
    assert len(first["cases"]) == 40
    by_policy = {case["policy"]: case for case in first["cases"]}
    condition_holdout = next(
        case for case in first["cases"] if case["policy"] == "leave_one_condition_out"
    )
    assert condition_holdout["calibration_trials"] == 7 * 19
    target = (
        condition_holdout["tone_count"],
        condition_holdout["snr_db"],
        condition_holdout["separation_fraction_of_reciprocal_bandwidth"],
        condition_holdout["gain_phase_difference_deg"],
    )
    assert target not in [tuple(source) for source in condition_holdout["calibration_conditions"]]
    snr_holdout = next(
        case for case in first["cases"] if case["policy"] == "leave_one_snr_level_out"
    )
    assert snr_holdout["calibration_trials"] == 4 * 19
    assert all(
        source[1] != snr_holdout["snr_db"] for source in snr_holdout["calibration_conditions"]
    )
    within = next(case for case in first["cases"] if case["policy"] == "within_condition")
    assert within["calibration_trials"] == 19
    assert -1.0 <= within["paired_profile_minus_omp_coverage_change"] <= 1.0
    assert len(within["paired_coverage_bootstrap_ci95"]) == 2
    assert all(0.0 <= case["paired_mcnemar_holm_p_value"] <= 1.0 for case in first["cases"])
    assert len(by_policy) == 5


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda result: result.update(schema="wrong"), "matched-estimator schema"),
        (lambda result: result["cases"].pop(), "complete factorial grid"),
        (lambda result: result["cases"].append(result["cases"][0].copy()), "duplicate condition"),
    ],
)
def test_matched_transfer_rejects_incompatible_inputs(mutate, message):
    source = _matched_result()
    mutate(source)
    with pytest.raises(ValueError, match=message):
        analyze_matched_cross_condition_transfer(source, seed=11, bootstrap_draws=100)


def test_matched_transfer_rejects_invalid_bootstrap_draws():
    with pytest.raises(TypeError, match="bootstrap_draws must be an integer"):
        analyze_matched_cross_condition_transfer(_matched_result(), seed=11, bootstrap_draws=True)
    with pytest.raises(ValueError, match="bootstrap_draws must be positive"):
        analyze_matched_cross_condition_transfer(_matched_result(), seed=11, bootstrap_draws=0)
