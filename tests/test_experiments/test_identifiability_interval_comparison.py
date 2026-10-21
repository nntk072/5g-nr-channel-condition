"""Tests for paired delay-uncertainty interval comparison."""

from __future__ import annotations

import pytest
from experiments.identifiability_interval_comparison import compare_profile_intervals


def _study() -> dict:
    cases = []
    for phase in (0.0, 90.0):
        cases.append(
            {
                "tone_count": 256,
                "snr_db": 20.0,
                "separation_fraction_of_reciprocal_bandwidth": 0.1,
                "gain_phase_difference_deg": phase,
                "trials": 2,
                "per_trial": [
                    {"split": "calibration", "trial": 0},
                    {
                        "split": "evaluation",
                        "trial": 0,
                        "interval_95_covers_truth": False,
                        "interval_95_width_ns": 10.0,
                        "split_conformal_95_covers_truth": True,
                        "split_conformal_95_set_span_ns": 12.0,
                    },
                    {
                        "split": "evaluation",
                        "trial": 1,
                        "interval_95_covers_truth": True,
                        "interval_95_width_ns": 8.0,
                        "split_conformal_95_covers_truth": True,
                        "split_conformal_95_set_span_ns": 9.0,
                    },
                ],
            }
        )
    return {
        "schema": "loslab.identifiability_profile_likelihood.v1",
        "seed": 45,
        "source_revision": "fixture",
        "cases": cases,
    }


def test_comparison_pairs_coverage_and_uncertainty_spans_deterministically():
    result = compare_profile_intervals(_study(), seed=71, bootstrap_draws=200)
    repeated = compare_profile_intervals(_study(), seed=71, bootstrap_draws=200)

    assert result == repeated
    assert result["schema"] == "loslab.identifiability_interval_comparison.v1"
    assert result["pooled_profile_interval_coverage"] == 0.5
    assert result["pooled_conformal_set_coverage"] == 1.0
    assert len(result["conditions"]) == 2
    assert all(row["conformal_only_covers"] == 1 for row in result["conditions"])
    assert all(row["paired_mean_span_change_ns"] == 1.5 for row in result["conditions"])
    assert all("mcnemar_holm_adjusted_p_value" in row for row in result["conditions"])


def test_comparison_rejects_missing_paired_outcomes():
    study = _study()
    del study["cases"][0]["per_trial"][1]["split_conformal_95_covers_truth"]

    with pytest.raises(ValueError, match="missing paired coverage or width"):
        compare_profile_intervals(study, bootstrap_draws=10)


def test_comparison_rejects_non_boolean_coverage():
    study = _study()
    study["cases"][0]["per_trial"][1]["interval_95_covers_truth"] = "true"

    with pytest.raises(ValueError, match="coverage outcomes must be booleans"):
        compare_profile_intervals(study, bootstrap_draws=10)


@pytest.mark.parametrize("draws", [0, -1, True, 1.2])
def test_comparison_validates_bootstrap_draws(draws):
    with pytest.raises((TypeError, ValueError)):
        compare_profile_intervals(_study(), bootstrap_draws=draws)
