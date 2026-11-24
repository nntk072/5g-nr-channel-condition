"""Complex circularity evaluator: input axes, error budgets and per-tap power rules."""

from __future__ import annotations

import numpy as np
import pytest
from evaluation.covariance.complex_circularity import evaluate_complex_circularity_family


def _draws(draws: int = 40, taps: int = 3) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(size=(draws, taps)) + 1j * rng.normal(size=(draws, taps))


def test_circular_family_is_reported_with_its_claim_budget() -> None:
    result = evaluate_complex_circularity_family(_draws(), family_claim_count=20)
    assert result is not None


def test_coefficients_must_be_a_numpy_array_of_finite_complex_axes() -> None:
    with pytest.raises(TypeError, match="NumPy array"):
        evaluate_complex_circularity_family([[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]])
    with pytest.raises(ValueError, match="draw>=2, tap>=1"):
        evaluate_complex_circularity_family(_draws(draws=1))
    with pytest.raises(TypeError, match="complex dtype"):
        evaluate_complex_circularity_family(np.ones((4, 2)))
    bad = _draws()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="only finite values"):
        evaluate_complex_circularity_family(bad)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        pytest.param(
            {"familywise_alpha": True},
            TypeError,
            "familywise_alpha must be a real scalar",
            id="alpha-bool",
        ),
        pytest.param(
            {"familywise_alpha": 0.0},
            ValueError,
            "familywise_alpha must be finite",
            id="alpha-zero",
        ),
        pytest.param(
            {"maximum_coordinate_half_width": "0.05"}, TypeError, "real scalar", id="width-type"
        ),
        pytest.param(
            {"maximum_coordinate_half_width": 0.0},
            ValueError,
            "must be finite and positive",
            id="width-zero",
        ),
        pytest.param(
            {"family_claim_count": 3}, ValueError, "at least four claims per tap", id="claims-small"
        ),
        pytest.param(
            {"family_claim_count": 4.5},
            ValueError,
            "at least four claims per tap",
            id="claims-float",
        ),
    ],
)
def test_error_budget_arguments_are_checked(kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        evaluate_complex_circularity_family(_draws(), **kwargs)


def test_each_tap_must_carry_positive_power() -> None:
    values = _draws()
    values[:, 1] = 0.0
    with pytest.raises(ValueError, match="positive finite power"):
        evaluate_complex_circularity_family(values)
