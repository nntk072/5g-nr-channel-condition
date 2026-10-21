"""Paired regression uncertainty resamples independent groups, not individual rows."""

from __future__ import annotations

import pytest
from evaluation.statistics import paired_group_regression_comparison


def test_group_bootstrap_preserves_paired_methods_and_equal_group_weight():
    report = paired_group_regression_comparison(
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 3.0, 3.0],
        [0.5, 0.5, 2.0, 2.0],
        ["a", "a", "b", "b"],
        control="unregularized",
        treatment="regularized",
        seed=4,
        draws=200,
    )

    assert report["comparison_unit"] == "independent_group"
    assert report["groups"] == 2
    assert report["samples"] == 4
    assert report["treatment_minus_control_group_macro_rmse"] == pytest.approx(-0.75)
    low, high = report["group_macro_rmse_delta_bootstrap_ci95"]
    assert low <= -0.75 <= high


def test_group_bootstrap_rejects_bad_alignment_and_nonfinite_values():
    with pytest.raises(ValueError, match="aligned"):
        paired_group_regression_comparison(
            [0.0], [1.0], [0.5, 0.2], ["a"], control="a", treatment="b"
        )
    with pytest.raises(ValueError, match="finite"):
        paired_group_regression_comparison(
            [0.0], [float("nan")], [0.5], ["a"], control="a", treatment="b"
        )
