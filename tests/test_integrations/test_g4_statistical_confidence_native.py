"""Confidence-interval checks for standardized-channel observables.

Every statistical unit below is one independently generated batch draw. Tap,
antenna, and time coordinates within a draw are treated as paired values.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from integrations.sionna.cdl import generate_cdl_frequency_response  # noqa: E402
from integrations.sionna.tdl import generate_tdl_frequency_response  # noqa: E402

_Z_SIMULTANEOUS = 3.5


def _within_interval(estimate: Any, standard_error: Any, target: Any) -> bool:
    return bool((torch.abs(estimate - target) <= _Z_SIMULTANEOUS * standard_error).all())


@pytest.mark.native_sionna
@pytest.mark.parametrize("family", ("TDL", "CDL"))
def test_delay_profile_and_rms_have_draw_level_confidence_intervals(family: str) -> None:
    """Check profile PDP and RMS-delay estimates with batch-draw intervals."""
    batch_size = 4096
    channel: Any
    if family == "TDL":
        channel = generate_tdl_frequency_response(
            model="A",
            seed=7311,
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30_000.0,
            num_subcarriers=8,
            batch_size=batch_size,
            delay_spread_s=100e-9,
            spec_version="19.2",
            device="cpu",
        )
    else:
        channel = generate_cdl_frequency_response(
            model="A",
            seed=7311,
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30_000.0,
            num_subcarriers=8,
            batch_size=batch_size,
            delay_spread_s=100e-9,
            spec_version="19.2",
            device="cpu",
        )
    # [independent draw, profile tap/cluster]; all within-draw dimensions are
    # retained together when estimating the PDP and its uncertainty.
    tap_power = channel.h_path[:, 0, 0, 0, 0, :, 0].abs().square().to(torch.float64)
    empirical_power = tap_power.mean(dim=0)
    power_se = tap_power.std(dim=0, unbiased=True) / math.sqrt(batch_size)
    expected_values = channel.metadata.get(
        "mean_tap_powers", channel.metadata.get("normalized_cluster_powers")
    )
    expected_power = torch.as_tensor(expected_values, dtype=torch.float64)
    delays = torch.as_tensor(
        channel.metadata.get("nominal_tap_delays_s", channel.metadata.get("nominal_cluster_delays_s")),
        dtype=torch.float64,
    )
    assert empirical_power.shape == expected_power.shape == delays.shape
    # A 3.5-SE bound is conservative across the profile's multiple PDP taps.
    assert _within_interval(empirical_power, power_se, expected_power)

    total_power = tap_power.sum(dim=1)
    first_moment = (tap_power * delays).sum(dim=1)
    second_moment = (tap_power * delays.square()).sum(dim=1)
    mean_power = total_power.mean()
    mean_delay = first_moment.mean() / mean_power
    rms_variance = second_moment.mean() / mean_power - mean_delay.square()
    empirical_rms = torch.sqrt(rms_variance)
    expected_mean = (expected_power * delays).sum() / expected_power.sum()
    expected_rms = torch.sqrt(
        (expected_power * (delays - expected_mean).square()).sum() / expected_power.sum()
    )
    # Delta-method influence values are computed per independent realization;
    # this gives a CI for the RMS delay spread of the ensemble mean PDP.
    influence = (
        (tap_power * (delays - mean_delay).square()).sum(dim=1)
        - rms_variance * total_power
    ) / (2.0 * empirical_rms * mean_power)
    rms_se = influence.std(unbiased=True) / math.sqrt(batch_size)
    assert abs(float(empirical_rms - expected_rms)) <= _Z_SIMULTANEOUS * float(rms_se)
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("model", "los_angle_rad"),
    (("A", math.acos(0.7)), ("D", 0.0), ("D", math.pi / 2.0)),
)
def test_doppler_correlation_confidence_interval_uses_paired_independent_draws(
    model: str, los_angle_rad: float
) -> None:
    carrier_frequency_hz = 3.5e9
    speed_mps = 30.0
    maximum_doppler_hz = speed_mps * carrier_frequency_hz / 299_792_458.0
    lag_s = 1.0 / (4.0 * maximum_doppler_hz)
    batch_size = 8192
    channel = generate_tdl_frequency_response(
        model=model,
        seed=4031,
        carrier_frequency_hz=carrier_frequency_hz,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=batch_size,
        delay_spread_s=100e-9,
        num_time_steps=2,
        sampling_frequency_hz=1.0 / lag_s,
        min_speed_mps=speed_mps,
        max_speed_mps=speed_mps,
        los_angle_of_arrival_rad=los_angle_rad,
        spec_version="19.2",
        device="cpu",
    )
    tap = channel.h_path[:, 0, 0, 0, 0, 0, :]
    cross_product = tap[:, 0].conj() * tap[:, 1]
    reference_power = tap[:, 0].abs().square()
    mean_power = reference_power.mean()
    correlation = cross_product.mean() / mean_power
    # The two times belong to one draw. The influence function is formed from
    # each paired (t0,t1) draw, not from treating time samples as independent.
    influence = cross_product - correlation * reference_power
    denominator = mean_power * math.sqrt(batch_size)
    real_se = influence.real.std(unbiased=True) / denominator
    imag_se = influence.imag.std(unbiased=True) / denominator
    diffuse = torch.special.bessel_j0(torch.tensor(math.pi / 2.0, dtype=torch.float64))
    if model == "A":
        expected = diffuse.to(torch.complex128)
    else:
        k_factor = float(channel.metadata["k_factor_linear"])
        diffuse_fraction = 1.0 / (1.0 + k_factor)
        specular_fraction = 1.0 - diffuse_fraction
        specular = torch.exp(
            torch.tensor(
                1j * 2.0 * math.pi * maximum_doppler_hz * math.cos(los_angle_rad) * lag_s,
                dtype=torch.complex128,
            )
        )
        expected = specular_fraction * specular + diffuse_fraction * diffuse
    assert abs(float(correlation.real - expected.real)) <= _Z_SIMULTANEOUS * float(real_se)
    assert abs(float(correlation.imag - expected.imag)) <= _Z_SIMULTANEOUS * float(imag_se)
    assert float(real_se) < 0.02 and float(imag_se) < 0.02


@pytest.mark.native_sionna
def test_receive_spatial_covariance_confidence_intervals_match_configured_matrix() -> None:
    batch_size = 8192
    correlation = 0.6
    rx_corr_mat = ((1.0, correlation), (correlation, 1.0))
    channel = generate_tdl_frequency_response(
        model="A",
        seed=9107,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=batch_size,
        delay_spread_s=100e-9,
        num_rx_ant=2,
        rx_corr_mat=rx_corr_mat,
        spec_version="19.2",
        device="cpu",
    )
    # One vector per independent channel draw; antenna elements are paired.
    samples = channel.h_path[:, 0, :, 0, 0, 0, 0]
    outer_products = samples[:, :, None] * samples[:, None, :].conj()
    empirical_covariance = outer_products.mean(dim=0)
    covariance_real_se = outer_products.real.std(dim=0, unbiased=True) / math.sqrt(batch_size)
    covariance_imag_se = outer_products.imag.std(dim=0, unbiased=True) / math.sqrt(batch_size)
    tap_power = float(channel.metadata["mean_tap_powers"][0])
    expected_covariance = tap_power * torch.as_tensor(rx_corr_mat, dtype=torch.complex64)
    target_real = expected_covariance.real
    target_imag = expected_covariance.imag
    assert _within_interval(empirical_covariance.real, covariance_real_se, target_real)
    assert _within_interval(empirical_covariance.imag, covariance_imag_se, target_imag)
    metadata_real = torch.as_tensor(channel.metadata["rx_corr_mat_real"])
    torch.testing.assert_close(
        metadata_real,
        torch.tensor([[1.0, correlation], [correlation, 1.0]]),
        rtol=0.0,
        atol=1e-7,
    )
    assert channel.metadata["rx_corr_mat_imag"] == [[0.0, 0.0], [0.0, 0.0]]


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("matrix", "message"),
    (
        (((1.0,),), "shape"),
        (((1.0, 0.2), (0.1, 1.0)), "Hermitian"),
        (((1.0, 1.2), (1.2, 1.0)), "positive definite"),
        (((0.9, 0.0), (0.0, 1.0)), "unit diagonal"),
    ),
)
def test_tdl_rejects_invalid_receive_correlation_matrix(matrix, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate_tdl_frequency_response(
            model="A",
            seed=9,
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30_000.0,
            num_subcarriers=8,
            num_rx_ant=2,
            rx_corr_mat=matrix,
            spec_version="19.2",
            device="cpu",
        )
