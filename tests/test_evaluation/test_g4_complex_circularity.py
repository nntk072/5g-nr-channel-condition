from __future__ import annotations

import numpy as np
import pytest
from evaluation.g4_complex_circularity import evaluate_complex_circularity_family


def test_complex_circularity_scorer_matches_direct_one_tap_formula() -> None:
    values = np.asarray(
        [[1.0 + 0.5j], [-0.5 + 1.0j], [0.75 - 0.25j], [-0.25 - 0.75j]],
        dtype=np.complex128,
    )
    result = evaluate_complex_circularity_family(
        values,
        familywise_alpha=0.05,
        maximum_coordinate_half_width=10.0,
    )
    power = np.mean(np.abs(values[:, 0]) ** 2)
    expected_mean = values[:, 0].mean() / np.sqrt(power)
    expected_pseudocovariance = np.mean(values[:, 0] ** 2) / power

    assert result["schema"] == "loslab.g4_complex_circularity_family.v1"
    assert result["draw_count"] == 4
    assert result["tap_count"] == 1
    assert result["claim_count"] == 4
    assert result["normalized_mean"][0] == pytest.approx(expected_mean)
    assert result["normalized_pseudocovariance"][0] == pytest.approx(expected_pseudocovariance)
    assert result["mean_coordinate_half_width"].shape == (1, 2)
    assert result["pseudocovariance_coordinate_half_width"].shape == (1, 2)


def test_complex_circularity_scorer_is_invariant_to_per_tap_power_scale() -> None:
    rng = np.random.default_rng(20261101)
    values = rng.normal(size=(4_096, 3)) + 1j * rng.normal(size=(4_096, 3))
    base = evaluate_complex_circularity_family(
        values.astype(np.complex128), maximum_coordinate_half_width=1.0
    )
    scaled = evaluate_complex_circularity_family(
        (values * 1e200).astype(np.complex128), maximum_coordinate_half_width=1.0
    )

    np.testing.assert_allclose(base["normalized_mean"], scaled["normalized_mean"])
    np.testing.assert_allclose(
        base["normalized_pseudocovariance"], scaled["normalized_pseudocovariance"]
    )
    np.testing.assert_allclose(
        base["mean_coordinate_half_width"], scaled["mean_coordinate_half_width"]
    )
    np.testing.assert_allclose(
        base["pseudocovariance_coordinate_half_width"],
        scaled["pseudocovariance_coordinate_half_width"],
    )


@pytest.mark.parametrize(
    ("values", "options", "error", "message"),
    [
        ([[1.0 + 1j]], {}, TypeError, "NumPy array"),
        (np.ones((8, 2)), {}, TypeError, "complex dtype"),
        (np.ones((1, 2), dtype=np.complex64), {}, ValueError, "draw>=2"),
        (np.ones((8, 0), dtype=np.complex64), {}, ValueError, "tap>=1"),
        (
            np.full((8, 1), 1 + 1j, dtype=np.complex64),
            {"family_claim_count": 3},
            ValueError,
            "four claims",
        ),
        (
            np.full((8, 1), 1 + 1j, dtype=np.complex64),
            {"family_claim_count": True},
            ValueError,
            "four claims",
        ),
        (
            np.full((8, 1), 1 + 1j, dtype=np.complex64),
            {"familywise_alpha": True},
            TypeError,
            "real scalar",
        ),
        (
            np.full((8, 1), 1 + 1j, dtype=np.complex64),
            {"maximum_coordinate_half_width": 0.0},
            ValueError,
            "positive",
        ),
        (np.zeros((8, 1), dtype=np.complex64), {}, ValueError, "positive finite power"),
    ],
)
def test_complex_circularity_scorer_rejects_invalid_inputs(values, options, error, message) -> None:
    with pytest.raises(error, match=message):
        evaluate_complex_circularity_family(values, **options)
