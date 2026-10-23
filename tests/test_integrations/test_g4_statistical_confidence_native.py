"""Confidence-interval checks for standardized-channel observables.

Every statistical unit below is one independently generated batch draw. Tap,
antenna, and time coordinates within a draw are treated as paired values.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from integrations.sionna.cdl import CDLArrayConfig, generate_cdl_frequency_response  # noqa: E402
from integrations.sionna.tdl import generate_tdl_frequency_response  # noqa: E402

_Z_SIMULTANEOUS = 3.5
_FIXED_TDL_MODELS = ("A10", "A30", "B100", "C60", "C300", "D10", "D30")
_CDL_RAY_ANGLE_OFFSETS = (
    0.0447,
    -0.0447,
    0.1413,
    -0.1413,
    0.2492,
    -0.2492,
    0.3715,
    -0.3715,
    0.5129,
    -0.5129,
    0.6797,
    -0.6797,
    0.8844,
    -0.8844,
    1.1481,
    -1.1481,
    1.5195,
    -1.5195,
    2.1551,
    -2.1551,
)


def _within_interval(estimate: Any, standard_error: Any, target: Any) -> bool:
    return bool((torch.abs(estimate - target) <= _Z_SIMULTANEOUS * standard_error).all())


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("family", "model"),
    tuple((family, model) for family in ("TDL", "CDL") for model in "ABCDE")
    + tuple(("TDL", model) for model in _FIXED_TDL_MODELS),
)
def test_delay_profile_and_rms_have_draw_level_confidence_intervals(
    family: str, model: str
) -> None:
    """Check scalable and fixed profile PDP/RMS delay with draw-level intervals."""
    batch_size = 4096
    channel: Any
    if family == "TDL":
        channel = generate_tdl_frequency_response(
            model=model,
            seed=7311,
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30_000.0,
            num_subcarriers=8,
            batch_size=batch_size,
            delay_spread_s=None if model in _FIXED_TDL_MODELS else 100e-9,
            spec_version="19.2",
            device="cpu",
        )
    else:
        channel = generate_cdl_frequency_response(
            model=model,
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
        channel.metadata.get(
            "nominal_tap_delays_s", channel.metadata.get("nominal_cluster_delays_s")
        ),
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
        (tap_power * (delays - mean_delay).square()).sum(dim=1) - rms_variance * total_power
    ) / (2.0 * empirical_rms * mean_power)
    rms_se = influence.std(unbiased=True) / math.sqrt(batch_size)
    assert abs(float(empirical_rms - expected_rms)) <= _Z_SIMULTANEOUS * float(rms_se)
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("model", "los_angle_rad"),
    tuple((model, math.acos(0.7)) for model in "ABC")
    + tuple((model, angle) for model in "DE" for angle in (0.0, math.pi / 2.0))
    + tuple((model, math.acos(0.7)) for model in _FIXED_TDL_MODELS if model not in {"D10", "D30"})
    + tuple((model, angle) for model in ("D10", "D30") for angle in (0.0, math.pi / 2.0)),
)
def test_doppler_correlation_confidence_interval_uses_paired_independent_draws(
    model: str, los_angle_rad: float
) -> None:
    carrier_frequency_hz = 28e9 if model in {"A10", "C60", "D10"} else 3.5e9
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
        delay_spread_s=None if model in _FIXED_TDL_MODELS else 100e-9,
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
    if channel.metadata["k_factor_linear"] is None:
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
@pytest.mark.parametrize("model", ("A", "B", "C"))
def test_cdl_nlos_temporal_correlation_matches_independent_ray_doppler_reference(
    model: str,
) -> None:
    """Check CDL A-C Doppler covariance against the 38.901 ray-phase equation."""
    carrier_frequency_hz = 3.5e9
    speed_mps = 30.0
    lag_s = 0.7e-3
    batch_size = 4096
    channel = generate_cdl_frequency_response(
        model=model,
        seed=1984,
        carrier_frequency_hz=carrier_frequency_hz,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=batch_size,
        delay_spread_s=100e-9,
        num_time_steps=2,
        sampling_frequency_hz=1.0 / lag_s,
        ut_velocity_mps=(speed_mps, 0.0, 0.0),
        spec_version="19.2",
        device="cpu",
    )
    # Sum distinct delay clusters to form the narrowband channel. The two
    # observations in each row share one independent CDL realization.
    path = channel.h_path[:, 0, 0, 0, 0, :, :]
    h0 = path[:, :, 0].sum(dim=1).to(torch.complex128)
    h1 = path[:, :, 1].sum(dim=1).to(torch.complex128)
    cross_product = h0.conj() * h1
    reference_power = h0.abs().square()
    mean_power = reference_power.mean()
    empirical = cross_product.mean() / mean_power
    influence = cross_product - empirical * reference_power
    real_se = influence.real.std(unbiased=True) / (mean_power * math.sqrt(batch_size))
    imag_se = influence.imag.std(unbiased=True) / (mean_power * math.sqrt(batch_size))

    # Table 7.5-3 offsets are applied independently to azimuth and zenith
    # rays by the CDL random-coupling step. Average over their Cartesian ray
    # pairing, weighted by the profile's pinned cluster PDP.
    metadata = channel.metadata
    angles = metadata["nominal_cluster_angles_deg"]
    spreads = metadata["cluster_angle_spreads_deg"]
    powers = torch.as_tensor(metadata["normalized_cluster_powers"], dtype=torch.float64)
    offsets = torch.tensor(_CDL_RAY_ANGLE_OFFSETS, dtype=torch.float64)
    aoa = torch.deg2rad(
        torch.as_tensor(angles["aoa"], dtype=torch.float64)[:, None]
        + float(spreads["asa"]) * offsets
    )
    zoa = torch.deg2rad(
        torch.as_tensor(angles["zoa"], dtype=torch.float64)[:, None]
        + float(spreads["zsa"]) * offsets
    )
    x_direction = torch.sin(zoa)[:, :, None] * torch.cos(aoa)[:, None, :]
    ray_phase = (
        2.0 * math.pi * speed_mps * carrier_frequency_hz / 299_792_458.0 * lag_s * x_direction
    )
    expected = (powers * torch.exp(1j * ray_phase).mean(dim=(1, 2))).sum() / powers.sum()

    assert metadata["profile_los_component"] is False
    assert metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert abs(float(empirical.real - expected.real)) <= _Z_SIMULTANEOUS * float(real_se)
    assert abs(float(empirical.imag - expected.imag)) <= _Z_SIMULTANEOUS * float(imag_se)


@pytest.mark.native_sionna
def test_receive_spatial_covariance_confidence_intervals_match_configured_matrix() -> None:
    batch_size = 8192
    correlation = 0.55 + 0.25j
    rx_corr_mat = ((1.0 + 0.0j, correlation), (correlation.conjugate(), 1.0 + 0.0j))
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
    metadata_imag = torch.as_tensor(channel.metadata["rx_corr_mat_imag"])
    torch.testing.assert_close(
        metadata_real,
        torch.tensor([[1.0, correlation.real], [correlation.real, 1.0]]),
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        metadata_imag,
        torch.tensor([[0.0, correlation.imag], [-correlation.imag, 0.0]]),
        rtol=0.0,
        atol=1e-7,
    )


@pytest.mark.native_sionna
def test_four_element_tdl_covariance_uses_simultaneous_draw_level_intervals() -> None:
    """Check a nontrivial complex 4x4 RX covariance with familywise intervals."""
    batch_size = 8192
    rho = 0.45
    phase_rad = 0.23
    rx_corr_mat = tuple(
        tuple(
            rho ** abs(row - col)
            * complex(math.cos(phase_rad * (row - col)), math.sin(phase_rad * (row - col)))
            for col in range(4)
        )
        for row in range(4)
    )
    channel = generate_tdl_frequency_response(
        model="A",
        seed=5217,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=batch_size,
        delay_spread_s=100e-9,
        num_rx_ant=4,
        rx_corr_mat=rx_corr_mat,
        spec_version="19.2",
        device="cpu",
    )
    # Antenna elements remain paired within each independent channel draw.
    samples = channel.h_path[:, 0, :, 0, 0, 0, 0]
    outer_products = samples[:, :, None] * samples[:, None, :].conj()
    empirical = outer_products.mean(dim=0)
    real_se = outer_products.real.std(dim=0, unbiased=True) / math.sqrt(batch_size)
    imag_se = outer_products.imag.std(dim=0, unbiased=True) / math.sqrt(batch_size)
    expected = float(channel.metadata["mean_tap_powers"][0]) * torch.as_tensor(
        rx_corr_mat, dtype=torch.complex64
    )

    # Bonferroni controls the familywise two-sided error across 32 real/imag
    # components at 1%; each interval is asymptotic under independent draws.
    component_count = 2 * expected.numel()
    z_simultaneous = NormalDist().inv_cdf(1.0 - 0.01 / (2.0 * component_count))
    assert bool((torch.abs(empirical.real - expected.real) <= z_simultaneous * real_se).all())
    assert bool((torch.abs(empirical.imag - expected.imag) <= z_simultaneous * imag_se).all())
    torch.testing.assert_close(empirical, empirical.mH, rtol=0.0, atol=1e-7)
    assert float(torch.linalg.eigvalsh(empirical).min()) >= -1e-7
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"


@pytest.mark.native_sionna
def test_cdl_array_spatial_covariance_has_draw_level_confidence_intervals() -> None:
    """Estimate one CDL cluster's 2-element covariance with paired array draws."""
    batch_size = 4096
    channel = generate_cdl_frequency_response(
        model="A",
        seed=2381,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=batch_size,
        delay_spread_s=100e-9,
        ut_array=CDLArrayConfig(
            num_cols_per_panel=2,
            num_cols=1,
            element_horizontal_spacing=0.5,
        ),
        spec_version="19.2",
        device="cpu",
    )
    # The receive antenna elements are paired within each independent draw.
    samples = channel.h_path[:, 0, :, 0, 0, 0, 0]
    outer_products = samples[:, :, None] * samples[:, None, :].conj()
    covariance = outer_products.mean(dim=0)
    real_se = outer_products.real.std(dim=0, unbiased=True) / math.sqrt(batch_size)
    imag_se = outer_products.imag.std(dim=0, unbiased=True) / math.sqrt(batch_size)
    assert covariance.shape == (2, 2)
    torch.testing.assert_close(covariance, covariance.mH, rtol=0.0, atol=1e-7)
    assert float(torch.linalg.eigvalsh(covariance).min()) >= -1e-7
    assert float((_Z_SIMULTANEOUS * real_se).max()) < 0.1 * float(covariance.diagonal().real.mean())
    assert float((_Z_SIMULTANEOUS * imag_se).max()) < 0.1 * float(covariance.diagonal().real.mean())

    paired_power_difference = samples[:, 0].abs().square() - samples[:, 1].abs().square()
    paired_se = paired_power_difference.std(unbiased=True) / math.sqrt(batch_size)
    assert abs(float(paired_power_difference.mean())) <= _Z_SIMULTANEOUS * float(paired_se)
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("matrix", "message"),
    (
        (((1.0,),), "shape"),
        (((1.0, math.nan), (math.nan, 1.0)), "finite"),
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
