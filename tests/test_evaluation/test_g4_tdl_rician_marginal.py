from __future__ import annotations

import numpy as np
import pytest
from evaluation.g4_tdl_covariance_reference import fixed_tdl_first_tap_power_components
from evaluation.g4_tdl_rician_marginal import (
    rician_normalized_power_cdf,
    score_tdl_rician_marginal_family,
)
from scipy.stats import ncx2


def test_fixed_tdl_rician_power_components_are_independent_reference_values() -> None:
    d10 = fixed_tdl_first_tap_power_components("D10")
    d30 = fixed_tdl_first_tap_power_components("D30")
    assert d10[0] > 0.0 and d10[1] > d10[0]
    assert d30[0] > 0.0 and d30[1] > d30[0]
    assert d10 != d30


def test_rician_power_cdf_reduces_to_rayleigh_when_specular_power_is_zero() -> None:
    powers = np.asarray([0.0, 0.1, 0.5, 1.0, 3.0])
    expected = -np.expm1(-powers)
    observed = rician_normalized_power_cdf(
        powers, diffuse_power=0.37, specular_power=0.0
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-14)


def test_rician_power_cdf_matches_noncentral_chi_square_parameterization() -> None:
    powers = np.asarray([0.1, 0.5, 1.0, 2.0])
    diffuse, specular = 0.2, 0.8
    expected = ncx2.cdf(
        2.0 * powers * (diffuse + specular) / diffuse,
        df=2,
        nc=2.0 * specular / diffuse,
    )
    observed = rician_normalized_power_cdf(
        powers, diffuse_power=diffuse, specular_power=specular
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)


def test_synthetic_independent_rician_first_taps_pass_familywise_dkw_bound() -> None:
    rng = np.random.default_rng(1_880_003)
    draw_count = 32_768
    powers = {
        model: fixed_tdl_first_tap_power_components(model)
        for model in ("D10", "D30")
    }
    samples = {
        model: np.sqrt(diffuse / 2.0)
        * (rng.standard_normal(draw_count) + 1j * rng.standard_normal(draw_count))
        + np.sqrt(specular)
        * np.exp(2j * np.pi * rng.random(draw_count))
        for model, (diffuse, specular) in powers.items()
    }

    result = score_tdl_rician_marginal_family(samples, powers)

    assert tuple(row.model for row in result) == ("D10", "D30")
    assert all(row.draw_count == draw_count and row.passed for row in result)
    assert all(row.simultaneous_upper_bound <= 0.05 for row in result)


def test_rayleigh_first_tap_falsification_does_not_pass_rician_family() -> None:
    rng = np.random.default_rng(1_880_004)
    draw_count = 32_768
    reference = {model: fixed_tdl_first_tap_power_components(model) for model in ("D10", "D30")}
    samples = {
        model: np.sqrt(0.5)
        * (rng.standard_normal(draw_count) + 1j * rng.standard_normal(draw_count))
        for model in reference
    }

    results = score_tdl_rician_marginal_family(samples, reference)

    assert all(not row.passed for row in results)


def test_rician_family_rejects_noncanonical_model_keys_before_sorting() -> None:
    samples = {1: np.ones(8, dtype=np.complex128)}
    reference = {1: (0.2, 0.8)}

    with pytest.raises(ValueError, match="model identities"):
        score_tdl_rician_marginal_family(samples, reference)


@pytest.mark.parametrize(
    ("samples", "reference", "message"),
    [
        ({"D10": np.ones(8)}, {"D10": (0.2, 0.8)}, "complex vector"),
        ({"D10": np.ones(8, dtype=np.complex128)}, {"D30": (0.2, 0.8)}, "rosters"),
        ({"D10": np.ones(8, dtype=np.complex128)}, {"D10": (0.0, 0.8)}, "positive"),
    ],
)
def test_rician_family_rejects_invalid_samples_or_reference(
    samples: dict[str, np.ndarray],
    reference: dict[str, tuple[float, float]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        score_tdl_rician_marginal_family(samples, reference)
