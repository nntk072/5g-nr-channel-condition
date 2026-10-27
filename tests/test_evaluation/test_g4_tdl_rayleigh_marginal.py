from __future__ import annotations

import numpy as np
import pytest
from evaluation.g4_tdl_covariance_reference import (
    fixed_tdl_rayleigh_tap_power_fractions,
    scalable_tdl_rayleigh_tap_power_fractions,
)
from evaluation.g4_tdl_rayleigh_marginal import (
    rayleigh_marginal_dkw_radius,
    rayleigh_normalized_power_ks_distance,
    score_rayleigh_tap_marginal_family,
)


def test_rayleigh_marginal_reference_power_rosters_bind_only_diffuse_native_taps() -> None:
    scalable_a = scalable_tdl_rayleigh_tap_power_fractions("A")
    scalable_d = scalable_tdl_rayleigh_tap_power_fractions("D")
    fixed_a10 = fixed_tdl_rayleigh_tap_power_fractions("A10")
    fixed_d10 = fixed_tdl_rayleigh_tap_power_fractions("D10")

    assert len(scalable_a) == 23
    assert tuple(index for index, _ in scalable_d) == tuple(range(1, 13))
    assert len(fixed_a10) == 16
    assert tuple(index for index, _ in fixed_d10) == tuple(range(1, 10))
    assert all(0.0 < power < 1.0 for _, power in (*scalable_a, *scalable_d, *fixed_a10, *fixed_d10))


def test_dkw_radius_accounts_for_all_tap_cdfs_and_family_alpha() -> None:
    smaller_family = rayleigh_marginal_dkw_radius(32_768, 1, alpha=0.01)
    larger_family = rayleigh_marginal_dkw_radius(32_768, 177, alpha=0.01)
    assert larger_family > smaller_family
    assert rayleigh_marginal_dkw_radius(65_536, 177, alpha=0.01) < larger_family


def test_table_scaled_rayleigh_taps_pass_the_simultaneous_marginal_bound() -> None:
    rng = np.random.default_rng(178_000)
    draw_count = 16_384
    tap_powers = ((0, 0.4), (1, 0.6))
    samples = np.column_stack(
        [
            np.sqrt(power / 2.0)
            * (rng.standard_normal(draw_count) + 1j * rng.standard_normal(draw_count))
            for _, power in tap_powers
        ]
    )

    results = score_rayleigh_tap_marginal_family(samples, tap_powers)

    assert len(results) == 2
    assert all(result.passed for result in results)
    assert all(result.simultaneous_upper_bound <= 0.05 for result in results)


def test_proper_constant_magnitude_process_fails_rayleigh_marginal_check() -> None:
    phase = np.arange(32_768, dtype=np.float64) * (2.0 * np.pi / 32_768)
    constant_magnitude = np.exp(1j * phase)

    distance = rayleigh_normalized_power_ks_distance(constant_magnitude, expected_mean_power=1.0)
    result = score_rayleigh_tap_marginal_family(constant_magnitude[:, None], ((0, 1.0),))[0]

    assert distance > 0.3
    assert not result.passed


@pytest.mark.parametrize(
    ("samples", "power", "error", "message"),
    [
        (np.zeros(4, dtype=np.float64), 1.0, ValueError, "complex vector"),
        (np.zeros(4, dtype=np.complex128), 0.0, ValueError, "positive"),
        (np.asarray([1.0 + 1.0j, np.inf + 0.0j]), 1.0, ValueError, "finite"),
    ],
)
def test_rayleigh_marginal_rejects_invalid_samples_and_reference_power(
    samples: np.ndarray, power: float, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        rayleigh_normalized_power_ks_distance(samples, expected_mean_power=power)


def test_family_scorer_rejects_duplicate_or_out_of_range_tap_identity() -> None:
    samples = np.ones((64, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="unique"):
        score_rayleigh_tap_marginal_family(samples, ((0, 0.5), (0, 0.5)))
    with pytest.raises(ValueError, match="exceeds"):
        score_rayleigh_tap_marginal_family(samples, ((2, 0.5),))
