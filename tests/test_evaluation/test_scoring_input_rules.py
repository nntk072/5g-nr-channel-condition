"""Scoring helpers: numeric parsing, group-disjoint splits, metric and bootstrap contracts."""

from __future__ import annotations

import numpy as np
import pytest
from evaluation.scoring import metrics as metrics_module
from evaluation.scoring.metrics import (
    binary_metrics,
    block_bootstrap_auc,
    group_split,
    threshold_from_validation,
)
from evaluation.scoring.numeric import (
    finite_real_array,
    finite_real_number,
    finite_real_vector,
)


@pytest.mark.parametrize("value", [True, np.bool_(True), "1.0", None, [1.0]])
def test_scalar_parsing_refuses_coercions(value) -> None:
    with pytest.raises(ValueError, match="must be a finite real number"):
        finite_real_number(value, "x")


def test_scalar_parsing_accepts_numpy_reals_and_rejects_nonfinite() -> None:
    assert finite_real_number(np.float32(0.5), "x") == 0.5
    assert finite_real_number(np.int64(3), "x") == 3.0
    with pytest.raises(ValueError, match="must be a finite real number"):
        finite_real_number(float("inf"), "x")


def test_array_parsing_refuses_strings_and_ragged_inputs() -> None:
    with pytest.raises(ValueError, match="must be a finite real array"):
        finite_real_array("1 2 3", "x")
    with pytest.raises(ValueError, match="must be a finite real array"):
        finite_real_array(b"12", "x")
    with pytest.raises(ValueError, match="x values must be a finite real number"):
        finite_real_array([[1.0, 2.0], [3.0]], "x")


def test_array_parsing_enforces_dimension() -> None:
    with pytest.raises(ValueError, match="must be a 1-dimensional finite real array"):
        finite_real_array([[1.0, 2.0]], "x", ndim=1)
    assert finite_real_vector([1, 2.5], "x").tolist() == [1.0, 2.5]


def test_group_split_requires_a_valid_fraction_and_groups() -> None:
    y = np.array([0, 1] * 4)
    group = np.arange(8)
    with pytest.raises(ValueError, match="test_fraction must be in"):
        group_split(y, group, test_fraction=1.0)
    with pytest.raises(ValueError, match="At least four distinct groups"):
        group_split(y, np.array([0, 0, 1, 1, 2, 2, 2, 2]), test_fraction=0.25)


def test_group_split_reports_unbalanced_labels() -> None:
    y = np.zeros(12, dtype=int)
    group = np.arange(12)
    with pytest.raises(ValueError, match="Unable to create two-class"):
        group_split(y, group, test_fraction=0.25)


def test_threshold_defaults_when_no_point_is_valid() -> None:
    assert threshold_from_validation(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    with pytest.raises(ValueError, match="Need both classes in validation"):
        threshold_from_validation(np.array([1, 1]), np.array([0.1, 0.9]))


def test_threshold_uses_half_when_roc_has_no_finite_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sklearn.metrics

    monkeypatch.setattr(
        sklearn.metrics,
        "roc_curve",
        lambda *_args, **_kwargs: (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([np.inf, np.inf]),
        ),
    )

    assert threshold_from_validation(np.array([0, 1]), np.array([0.1, 0.9])) == 0.5


def test_binary_metrics_reject_empty_labels() -> None:
    with pytest.raises(ValueError, match="labels must be nonempty"):
        binary_metrics(np.array([]), np.array([]))


def test_bootstrap_needs_blocks_and_enough_draws() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="Need >=2 blocks, >=10 draws"):
        block_bootstrap_auc(y, p, np.array([0, 0, 0, 0]))
    with pytest.raises(ValueError, match="Need >=2 blocks, >=10 draws"):
        block_bootstrap_auc(y, p, np.array([0, 0, 1, 1]), draws=5)


def test_bootstrap_rejects_too_few_two_class_resamples(monkeypatch: pytest.MonkeyPatch) -> None:
    class FirstGroupOnly:
        def integers(self, low: int, high: int, *, size: int):
            return np.zeros(size, dtype=np.int64)

    monkeypatch.setattr(metrics_module.np.random, "default_rng", lambda _seed: FirstGroupOnly())
    with pytest.raises(ValueError, match="Too few bootstrap replicates contain both classes"):
        block_bootstrap_auc(
            np.array([0, 0, 1, 1]),
            np.array([0.1, 0.2, 0.8, 0.9]),
            np.array(["negative", "negative", "positive", "positive"]),
            draws=10,
        )
