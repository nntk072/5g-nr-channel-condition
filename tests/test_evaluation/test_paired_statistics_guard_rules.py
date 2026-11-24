"""Paired statistics: exact and Monte Carlo sign-flip p-values plus input guards for each estimator."""

from __future__ import annotations

import numpy as np
import pytest
from evaluation.scoring.statistics import (
    cluster_binary_error_rate_interval,
    paired_cluster_binary_error_comparison,
    paired_cluster_binary_error_family_comparison,
    paired_group_regression_comparison,
    paired_seed_delta,
)


def _rows(method: str, values: dict[int, float], split: str = "test") -> list[dict]:
    return [
        {"method": method, "seed": seed, "split": split, "metrics": {"auroc": value}}
        for seed, value in values.items()
    ]


def _seeds(count: int, delta: float) -> tuple[list[dict], list[dict]]:
    control = _rows("ctrl", {seed: 0.5 for seed in range(count)})
    treatment = _rows("trt", {seed: 0.5 + delta for seed in range(count)})
    return control, treatment


def test_exact_sign_flip_p_value_for_uniformly_positive_deltas() -> None:
    control, treatment = _seeds(4, 0.1)
    result = paired_seed_delta(control + treatment, "ctrl", "trt", draws=200)
    assert result["test"] == "exact paired sign-flip"
    # Only the all-plus and all-minus sign patterns reach |mean| = 0.1 of 2^4 patterns.
    assert result["two_sided_sign_flip_p_value"] == pytest.approx(2 / 16)
    assert result["paired_seeds"] == [0, 1, 2, 3]
    assert result["mean_delta"] == pytest.approx(0.1)


def test_monte_carlo_sign_flip_is_used_above_twenty_seeds() -> None:
    control, treatment = _seeds(21, 0.2)
    result = paired_seed_delta(control + treatment, "ctrl", "trt", draws=200, seed=9)
    assert result["test"].startswith("Monte Carlo paired sign-flip")
    # With 21 equal positive deltas, a sign pattern reaches the observed mean only when
    # every sign agrees: probability 2 / 2^21, so 20000 draws almost surely give zero.
    assert result["two_sided_sign_flip_p_value"] < 0.001
    assert result["warning"].startswith("Seed-level inference")


def test_short_seed_lists_carry_the_small_sample_warning() -> None:
    control, treatment = _seeds(3, 0.1)
    result = paired_seed_delta(control + treatment, "ctrl", "trt", draws=100)
    assert result["warning"].startswith("Fewer than six seeds")


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        pytest.param(
            lambda: _rows("ctrl", {1: 0.5}) + _rows("ctrl", {1: 0.6}) + _rows("trt", {1: 0.7}),
            "duplicate seed 1, method ctrl",
            id="duplicate-seed",
        ),
        pytest.param(
            lambda: _rows("ctrl", {1: 0.5}) + _rows("trt", {2: 0.7}),
            "Control/treatment seeds differ or empty",
            id="seed-sets-differ",
        ),
        pytest.param(
            lambda: _rows("ctrl", {1: 0.5}) + _rows("trt", {1: 0.7}, split="val"),
            "split does not match across methods",
            id="split-mismatch",
        ),
        pytest.param(
            lambda: _rows("ctrl", {1: float("nan")}) + _rows("trt", {1: 0.7}),
            "Nonfinite metric delta",
            id="nonfinite",
        ),
    ],
)
def test_seed_pairing_rejects_inconsistent_rows(rows, message) -> None:
    with pytest.raises(ValueError, match=message):
        paired_seed_delta(rows(), "ctrl", "trt")


def test_paired_binary_error_masks_must_be_one_dimensional_boolean_and_bounded() -> None:
    from evaluation.scoring.statistics import paired_binary_error_comparison as compare

    ok = [True, False, False]
    with pytest.raises(ValueError, match="one-dimensional"):
        compare(np.zeros((2, 2), bool), np.zeros((2, 2), bool))
    with pytest.raises(ValueError, match="only boolean outcomes"):
        compare([0, 2, 1], [0, 1, 1])
    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        compare(ok, ok, seed=True)
    with pytest.raises(ValueError, match="draws must be a positive integer"):
        compare(ok, ok, draws=0)
    with pytest.raises(ValueError, match="control_bits_per_success must be finite"):
        compare(ok, ok, control_bits_per_success=float("inf"))
    with pytest.raises(ValueError, match="treatment_bits_per_success must be finite"):
        compare(ok, ok, treatment_bits_per_success=-1.0)


def test_cluster_comparison_rejects_bad_masks_groups_and_seeds() -> None:
    controls = [True, False, True, False]
    treatments = [False, False, True, True]
    groups = ["a", "a", "b", "b"]
    with pytest.raises(ValueError, match="one-dimensional"):
        paired_cluster_binary_error_comparison(
            np.zeros((2, 2), bool), np.zeros((2, 2), bool), groups
        )
    with pytest.raises(ValueError, match="nonempty and equally sized"):
        paired_cluster_binary_error_comparison(controls, treatments[:2], groups)
    with pytest.raises(ValueError, match="only boolean outcomes"):
        paired_cluster_binary_error_comparison([0, 1, 1, 0], treatments, groups)
    with pytest.raises(ValueError, match="non-empty strings"):
        paired_cluster_binary_error_comparison(controls, treatments, ["a", " ", "b", "b"])
    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        paired_cluster_binary_error_comparison(controls, treatments, groups, seed=-2)
    with pytest.raises(ValueError, match="draws must be a positive integer"):
        paired_cluster_binary_error_comparison(controls, treatments, groups, draws=0)


def test_cluster_interval_rejects_bad_inputs_and_reports_the_pooled_rate() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        cluster_binary_error_rate_interval(np.zeros((2, 2), bool), ["a", "b", "c", "d"])
    with pytest.raises(ValueError, match="nonempty and equally sized"):
        cluster_binary_error_rate_interval([True], ["a", "b"])
    with pytest.raises(ValueError, match="only boolean values"):
        cluster_binary_error_rate_interval([1, 0], ["a", "b"])
    with pytest.raises(ValueError, match="non-empty strings"):
        cluster_binary_error_rate_interval([True, False], ["a", ""])
    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        cluster_binary_error_rate_interval([True, False], ["a", "b"], seed=False)
    with pytest.raises(ValueError, match="draws must be a positive integer"):
        cluster_binary_error_rate_interval([True, False], ["a", "b"], draws=0)


def test_family_comparison_needs_two_groups_and_bounded_grids() -> None:
    one_group = [[True, False]], [[False, False]], [["a", "a"]]
    with pytest.raises(ValueError, match="at least two independent groups"):
        paired_cluster_binary_error_family_comparison(*one_group)
    with pytest.raises(ValueError, match="same nonzero point count"):
        paired_cluster_binary_error_family_comparison([], [], [])
    two_groups = (
        [[True, False, True, False]],
        [[False, False, True, True]],
        [["a", "a", "b", "b"]],
    )
    with pytest.raises(ValueError, match="draw-by-SNR grid exceeds"):
        paired_cluster_binary_error_family_comparison(*two_groups, draws=5_000_001)
    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        paired_cluster_binary_error_family_comparison(*two_groups, seed=-1)
    with pytest.raises(ValueError, match="draws must be a positive integer"):
        paired_cluster_binary_error_family_comparison(*two_groups, draws=0)


def test_family_comparison_reports_one_row_per_snr_point() -> None:
    rows = paired_cluster_binary_error_family_comparison(
        [[True, True, False, False], [True, False, False, False]],
        [[False, False, False, False], [False, False, False, True]],
        [["a", "a", "b", "b"], ["a", "a", "b", "b"]],
        draws=200,
    )
    assert len(rows) == 2
    # Point 0: control 2/4 errors, treatment 0/4 errors -> pooled delta -0.5.
    # Point 1: control 1/4 and treatment 1/4 errors -> pooled delta 0.
    assert rows[0]["treatment_minus_control_error_rate"] == pytest.approx(-0.5)
    assert rows[1]["treatment_minus_control_error_rate"] == pytest.approx(0.0)
    assert rows[0]["independent_groups"] == 2
    assert rows[0]["paired_trials"] == 4


def test_group_regression_rejects_misaligned_or_nonfinite_inputs() -> None:
    kwargs = {"control": "ctrl", "treatment": "trt", "draws": 100}
    target = [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError, match="aligned nonempty vectors"):
        paired_group_regression_comparison(target, [1.0], [1.0] * 4, ["a"] * 4, **kwargs)
    with pytest.raises(ValueError, match="must be finite"):
        paired_group_regression_comparison(
            target, [1.0, np.nan, 3.0, 4.0], [1.0] * 4, ["a", "a", "b", "b"], **kwargs
        )
    with pytest.raises(ValueError, match="group IDs must be nonempty"):
        paired_group_regression_comparison(target, target, target, ["a", "", "b", "b"], **kwargs)
    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        paired_group_regression_comparison(
            target, target, target, ["a", "a", "b", "b"], control="ctrl", treatment="trt", seed=-1
        )
    with pytest.raises(ValueError, match="draws must be an integer >= 100"):
        paired_group_regression_comparison(
            target, target, target, ["a", "a", "b", "b"], control="ctrl", treatment="trt", draws=99
        )
