"""Propagation physics helpers: each input contract is enforced before any computation."""

from __future__ import annotations

import numpy as np
import pytest
from propagation import physics as physics_module
from propagation.identifiability import path_parameter_fisher_diagnostics
from propagation.physics import (
    add_estimation_noise,
    delay_k_proxy_features,
    frequency_response,
    observable_features,
    subcarrier_frequencies,
)


def test_frequency_response_requires_matching_complex_gains_and_delays() -> None:
    frequencies = np.linspace(0.0, 1.0, 4)
    with pytest.raises(ValueError, match="identically shaped, with complex a"):
        frequency_response(np.ones(2), np.ones(2), frequencies)
    with pytest.raises(ValueError, match="identically shaped, with complex a"):
        frequency_response(np.ones(2, dtype=complex), np.ones(3), frequencies)
    with pytest.raises(ValueError, match="finite"):
        frequency_response(np.array([np.nan + 0j]), np.array([0.0]), frequencies)


@pytest.mark.parametrize("count", [7, 0, -4])
def test_subcarrier_grid_needs_at_least_eight_tones(count: int) -> None:
    with pytest.raises(ValueError, match="invalid OFDM grid"):
        subcarrier_frequencies(count, 15e3)


@pytest.mark.parametrize("spacing", [True, "15e3"])
def test_subcarrier_spacing_must_be_real(spacing) -> None:
    with pytest.raises(TypeError, match="spacing_hz must be a real number"):
        subcarrier_frequencies(8, spacing)


def test_subcarrier_integer_must_not_be_boolean() -> None:
    with pytest.raises(TypeError, match="n_subcarriers must be an integer"):
        subcarrier_frequencies(True, 15e3)


def test_estimation_noise_requires_complex_cfr_and_finite_snr() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="complex CFR required"):
        add_estimation_noise(np.ones((1, 1, 8)), 10.0, rng)
    with pytest.raises(ValueError, match="finite SNR required"):
        add_estimation_noise(np.ones((1, 1, 8), dtype=complex), float("inf"), rng)


def test_observable_features_require_a_complex_three_axis_array() -> None:
    with pytest.raises(
        ValueError, match="expected \\[samples, antenna links, subcarriers\\] complex"
    ):
        observable_features(np.ones((2, 1, 8)))
    with pytest.raises(
        ValueError, match="expected \\[samples, antenna links, subcarriers\\] complex"
    ):
        observable_features(np.ones((2, 8), dtype=complex))


def test_observable_features_reject_nonfinite_outputs() -> None:
    h = np.ones((1, 1, 8), dtype=complex)
    h[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite observables"):
        observable_features(h)


def test_delay_proxy_requires_a_usable_complex_cfr() -> None:
    with pytest.raises(
        ValueError, match="expected \\[samples, antenna links, subcarriers\\] complex CFR"
    ):
        delay_k_proxy_features(np.ones((2, 1, 8)))
    with pytest.raises(ValueError, match="at least eight tones"):
        delay_k_proxy_features(np.ones((1, 1, 7), dtype=complex))
    bad = np.ones((1, 1, 8), dtype=complex)
    bad[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="requires finite CFR values"):
        delay_k_proxy_features(bad)


def test_delay_proxy_rejects_unrepresentable_power() -> None:
    huge = np.full((1, 1, 8), 1e200 + 0j)
    with pytest.raises(ValueError, match="not representable in float64"):
        delay_k_proxy_features(huge)


def test_delay_proxy_rejects_nonfinite_derived_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        physics_module.np.fft,
        "ifft",
        lambda value, axis: np.full(value.shape, complex(float("inf"), 0.0)),
    )

    with (
        np.errstate(over="ignore", invalid="ignore", divide="ignore"),
        pytest.raises(ValueError, match="produced nonfinite features"),
    ):
        delay_k_proxy_features(np.ones((1, 1, 8), dtype=np.complex128))


def test_fisher_diagnostics_validate_every_declared_input() -> None:
    gains = np.array([1.0 + 0j, 0.5 + 0j])
    delays = np.array([0.0, 1e-7])
    freqs = np.linspace(0.0, 1e6, 8)
    with pytest.raises(ValueError, match="non-empty finite 1D array"):
        path_parameter_fisher_diagnostics(
            np.array([], dtype=complex), delays, freqs, noise_variance=1.0
        )
    with pytest.raises(ValueError, match="excess_delays_s must be nonnegative"):
        path_parameter_fisher_diagnostics(gains, np.array([-1e-9, 0.0]), freqs, noise_variance=1.0)
    with pytest.raises(ValueError, match="at least two tones"):
        path_parameter_fisher_diagnostics(gains, delays, np.array([0.0]), noise_variance=1.0)
    with pytest.raises(ValueError, match="noise_variance"):
        path_parameter_fisher_diagnostics(gains, delays, freqs, noise_variance=0.0)
    with pytest.raises(TypeError, match="unknown_common_timing_offset must be a bool"):
        path_parameter_fisher_diagnostics(
            gains, delays, freqs, noise_variance=1.0, unknown_common_timing_offset=1
        )
    with pytest.raises(ValueError, match="rank_tolerance must be finite and positive"):
        path_parameter_fisher_diagnostics(
            gains, delays, freqs, noise_variance=1.0, rank_tolerance=0.0
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"building_width_min_m": 0.0}, "Invalid building widths"),
        ({"building_width_min_m": 30.0, "building_width_max_m": 20.0}, "Invalid building widths"),
        ({"height_min_m": 0.0}, "Invalid heights / vehicle count"),
        ({"height_min_m": 40.0, "height_max_m": 20.0}, "Invalid heights / vehicle count"),
        ({"n_vehicles": -1}, "Invalid heights / vehicle count"),
    ],
)
def test_city_configuration_rejects_invalid_dimensions(overrides, message) -> None:
    from propagation.geometry import CityConfig

    with pytest.raises(ValueError, match=message):
        CityConfig(**overrides)
