"""G4.07: dual-polarized CDL-A–E power follows independent XPR and K-factor references."""

from __future__ import annotations

import hashlib
import math
from statistics import NormalDist
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import CDLArrayConfig, generate_cdl_frequency_response  # noqa: E402
from sionna.phy.channel.tr38901 import models as profile_resources  # noqa: E402

_BATCH_SIZE = 4096
_SEED = 407
_FAMILYWISE_ALPHA = 0.01
_PRECISION_FRACTION_OF_CROSS_POWER = 0.25
_PRECISION_FRACTION_OF_LOS_TOTAL_POWER = 0.01
_PROFILE_SHA256 = {
    "A": "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7",
    "B": "7c87c092cb2bb44891d301ee3c1991cdda1c6b64094a03f18746ec5fa14ee2a2",
    "C": "f2044ecc3b72a1967221805ba70b1f0dab9634bb831478c1337b737ced6e4a28",
}
_PROFILE_LETTERS = tuple(_PROFILE_SHA256)
_LOS_PROFILE_SHA256 = {
    "D": "a8960f176347de457c0cb90fbc8cf137ac98bb1c5295b096f004d8d5c6e40176",
    "E": "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480",
}
_LOS_TX_ORIENTATION_RAD = (-0.3, 0.25, 0.5)
_LOS_RX_ORIENTATION_RAD = (0.6, 0.2, -0.4)
_RAY_BASIS_OFFSETS = (
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


def _independent_nlos_cdl_port_power_reference(model: str) -> tuple[Any, str]:
    """Calculate the four XPR-law expectations from the pinned table alone."""
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    table_bytes = table_path.read_bytes()
    digest = hashlib.sha256(table_bytes).hexdigest()
    assert digest == _PROFILE_SHA256[model]

    parameters = profile_resources.load_json(table_path)
    assert parameters["los"] == 0
    cluster_powers = [10.0 ** (float(power_db) / 10.0) for power_db in parameters["powers"]]
    total_pdp_power = math.fsum(cluster_powers)
    normalized_pdp_sum = math.fsum(power / total_pdp_power for power in cluster_powers)
    xpr_linear = 10.0 ** (float(parameters["xpr"]) / 10.0)

    # For a dual-VH pair, the TR 38.901 ray Jones matrix has unit expected
    # co-polar power and 1/XPR expected cross-polar power. Independent ray
    # phases remove cross-ray terms; the normalized cluster PDP sums to one.
    co_polar_power = normalized_pdp_sum
    cross_polar_power = normalized_pdp_sum / xpr_linear
    expected = torch.tensor(
        (
            (co_polar_power, cross_polar_power),
            (cross_polar_power, co_polar_power),
        ),
        dtype=torch.float64,
    )
    return expected, digest


def _los_displacement_angle_reference(
    orientation_rad: tuple[float, float, float], zenith_rad: float, azimuth_rad: float
) -> float:
    """Calculate the LOS field-basis displacement angle from TR 38.901 Eq. 7.1-15."""
    alpha, beta, gamma = orientation_rad
    real = math.sin(gamma) * math.cos(zenith_rad) * math.sin(azimuth_rad - alpha)
    real += math.cos(gamma) * (
        math.cos(beta) * math.sin(zenith_rad)
        - math.sin(beta) * math.cos(zenith_rad) * math.cos(azimuth_rad - alpha)
    )
    imag = math.sin(gamma) * math.cos(azimuth_rad - alpha)
    imag += math.sin(beta) * math.cos(gamma) * math.sin(azimuth_rad - alpha)
    return math.atan2(imag, real)


def _mean_dual_vh_field_component_powers(
    orientation_rad: tuple[float, float, float],
    azimuth_center_deg: float,
    zenith_center_deg: float,
    azimuth_spread_deg: float,
    zenith_spread_deg: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Average rotated V/H field powers over independent 20-ray angle couplings."""
    component_powers = [[0.0, 0.0], [0.0, 0.0]]
    for azimuth_offset in _RAY_BASIS_OFFSETS:
        azimuth = math.radians(azimuth_center_deg + azimuth_spread_deg * azimuth_offset)
        for zenith_offset in _RAY_BASIS_OFFSETS:
            zenith = math.radians(zenith_center_deg + zenith_spread_deg * zenith_offset)
            psi = _los_displacement_angle_reference(orientation_rad, zenith, azimuth)
            cos_squared = math.cos(psi) ** 2
            sin_squared = 1.0 - cos_squared
            component_powers[0][0] += cos_squared
            component_powers[0][1] += sin_squared
            component_powers[1][0] += sin_squared
            component_powers[1][1] += cos_squared

    normalization = float(len(_RAY_BASIS_OFFSETS) ** 2)
    return (
        (component_powers[0][0] / normalization, component_powers[0][1] / normalization),
        (component_powers[1][0] / normalization, component_powers[1][1] / normalization),
    )


def _independent_los_cdl_port_power_reference(
    model: str,
) -> tuple[Any, str]:
    """Calculate rotated dual-VH LOS-plus-diffuse powers from the profile table."""
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    table_bytes = table_path.read_bytes()
    digest = hashlib.sha256(table_bytes).hexdigest()
    assert digest == _LOS_PROFILE_SHA256[model]
    parameters = profile_resources.load_json(table_path)
    assert parameters["los"] == 1

    first_power = 10.0 ** (float(parameters["powers"][0]) / 10.0)
    diffuse_power = math.fsum(
        10.0 ** (float(power_db) / 10.0) for power_db in parameters["powers"][1:]
    )
    k_factor = first_power / diffuse_power
    xpr_linear = 10.0 ** (float(parameters["xpr"]) / 10.0)
    tx_psi = _los_displacement_angle_reference(
        _LOS_TX_ORIENTATION_RAD,
        math.radians(float(parameters["zod"][0])),
        math.radians(float(parameters["aod"][0])),
    )
    rx_psi = _los_displacement_angle_reference(
        _LOS_RX_ORIENTATION_RAD,
        math.radians(float(parameters["zoa"][0])),
        math.radians(float(parameters["aoa"][0])),
    )
    cross_los_power = math.sin(rx_psi + tx_psi) ** 2
    co_los_power = math.cos(rx_psi + tx_psi) ** 2
    los_power = torch.tensor(
        ((co_los_power, cross_los_power), (cross_los_power, co_los_power)),
        dtype=torch.float64,
    )
    raw_diffuse_cluster_powers = [
        10.0 ** (float(power_db) / 10.0) for power_db in parameters["powers"][1:]
    ]
    diffuse_cluster_power_sum = math.fsum(raw_diffuse_cluster_powers)
    diffuse_power_by_port_pair = [[0.0, 0.0], [0.0, 0.0]]
    for cluster_index, raw_power in enumerate(raw_diffuse_cluster_powers, start=1):
        rx_component_powers = _mean_dual_vh_field_component_powers(
            _LOS_RX_ORIENTATION_RAD,
            float(parameters["aoa"][cluster_index]),
            float(parameters["zoa"][cluster_index]),
            float(parameters["cASA"]),
            float(parameters["cZSA"]),
        )
        tx_component_powers = _mean_dual_vh_field_component_powers(
            _LOS_TX_ORIENTATION_RAD,
            float(parameters["aod"][cluster_index]),
            float(parameters["zod"][cluster_index]),
            float(parameters["cASD"]),
            float(parameters["cZSD"]),
        )
        normalized_cluster_power = raw_power / diffuse_cluster_power_sum
        for rx_port in range(2):
            for tx_port in range(2):
                expected_jones_power = math.fsum(
                    rx_component_powers[rx_port][rx_component]
                    * tx_component_powers[tx_port][tx_component]
                    * (1.0 if rx_component == tx_component else 1.0 / xpr_linear)
                    for rx_component in range(2)
                    for tx_component in range(2)
                )
                diffuse_power_by_port_pair[rx_port][tx_port] += (
                    normalized_cluster_power * expected_jones_power
                )
    diffuse_power_tensor = torch.tensor(diffuse_power_by_port_pair, dtype=torch.float64)
    expected = (k_factor * los_power + diffuse_power_tensor) / (k_factor + 1.0)
    return expected, digest


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _PROFILE_LETTERS)
def test_nlos_cdl_dual_vh_port_powers_match_independent_xpr_ray_law(model: str) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        expected, source_digest = _independent_nlos_cdl_port_power_reference(model)
        channel = generate_cdl_frequency_response(
            model=model,
            seed=_SEED + ord(model),
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30_000.0,
            num_subcarriers=8,
            batch_size=_BATCH_SIZE,
            delay_spread_s=100e-9,
            ut_array=CDLArrayConfig(
                polarization="dual",
                polarization_type="VH",
                antenna_pattern="omni",
            ),
            bs_array=CDLArrayConfig(
                polarization="dual",
                polarization_type="VH",
                antenna_pattern="omni",
            ),
            ut_orientation_rad=(0.0, 0.0, 0.0),
            bs_orientation_rad=(0.0, 0.0, 0.0),
            spec_version="19.2",
            device="cpu",
        )

        # One observation per independent draw and port pair, summing the
        # cluster tap powers. Axes after indexing: draw, RX polarization,
        # TX polarization, cluster.
        per_draw_taps = channel.h_path[:, 0, :, 0, :, :, 0].to(torch.complex128)
        assert tuple(per_draw_taps.shape[1:3]) == (2, 2)
        observations = per_draw_taps.abs().square().sum(dim=-1)
        estimates = observations.mean(dim=0)
        standard_errors = observations.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)

        # One Bonferroni family covers four port pairs across all three profiles.
        family_size = len(_PROFILE_LETTERS) * expected.numel()
        z_simultaneous = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * family_size))
        half_widths = z_simultaneous * standard_errors
        assert bool((torch.abs(estimates - expected) <= half_widths).all())
        assert float(half_widths.max()) <= float(expected.min()) * (
            _PRECISION_FRACTION_OF_CROSS_POWER
        )
        assert channel.metadata["parameter_table_sha256"] == source_digest
        assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", tuple(_LOS_PROFILE_SHA256))
def test_los_cdl_dual_vh_aggregate_power_matches_k_factor_reference(model: str) -> None:
    """Check the LOS-plus-diffuse Frobenius power without assuming an LOS port basis.

    For dual-VH omni ports, the LOS Jones/field transform is an orthogonal
    two-port transform, so its total port-pair power is 2 independent of the
    sampled LOS angles. The diffuse contribution has expected total power
    2*(1 + 1/XPR). The table's first-cluster K-factor then gives the reference
    below. This aggregate statistic deliberately does not claim per-port LOS
    mixing qualification.
    """
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
        table_bytes = table_path.read_bytes()
        source_digest = hashlib.sha256(table_bytes).hexdigest()
        assert source_digest == _LOS_PROFILE_SHA256[model]
        parameters = profile_resources.load_json(table_path)
        assert parameters["los"] == 1
        first_power = 10.0 ** (float(parameters["powers"][0]) / 10.0)
        diffuse_power = math.fsum(
            10.0 ** (float(power_db) / 10.0) for power_db in parameters["powers"][1:]
        )
        k_factor = first_power / diffuse_power
        xpr_linear = 10.0 ** (float(parameters["xpr"]) / 10.0)
        expected_total_power = 2.0 * (k_factor + 1.0 + 1.0 / xpr_linear) / (k_factor + 1.0)

        channel = generate_cdl_frequency_response(
            model=model,
            seed=_SEED + ord(model),
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30_000.0,
            num_subcarriers=8,
            batch_size=_BATCH_SIZE,
            delay_spread_s=100e-9,
            ut_array=CDLArrayConfig(
                polarization="dual",
                polarization_type="VH",
                antenna_pattern="omni",
            ),
            bs_array=CDLArrayConfig(
                polarization="dual",
                polarization_type="VH",
                antenna_pattern="omni",
            ),
            ut_orientation_rad=(0.0, 0.0, 0.0),
            bs_orientation_rad=(0.0, 0.0, 0.0),
            spec_version="19.2",
            device="cpu",
        )

        # Sum RX/TX port-pair powers and profile clusters per independent draw.
        per_draw_taps = channel.h_path[:, 0, :, 0, :, :, 0].to(torch.complex128)
        observations = per_draw_taps.abs().square().sum(dim=(1, 2, 3))
        estimate = observations.mean()
        standard_error = observations.std(unbiased=True) / math.sqrt(_BATCH_SIZE)
        z_simultaneous = NormalDist().inv_cdf(
            1.0 - _FAMILYWISE_ALPHA / (2.0 * len(_LOS_PROFILE_SHA256))
        )
        half_width = z_simultaneous * standard_error

        assert abs(float(estimate) - expected_total_power) <= float(half_width)
        assert float(half_width) <= _PRECISION_FRACTION_OF_LOS_TOTAL_POWER * expected_total_power
        assert channel.metadata["parameter_table_sha256"] == source_digest
        assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", tuple(_LOS_PROFILE_SHA256))
def test_los_cdl_rotated_dual_vh_port_powers_match_xpr_and_field_reference(
    model: str,
) -> None:
    """Check LOS port mixing and diffuse XPR under a frozen rotated topology."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        expected, source_digest = _independent_los_cdl_port_power_reference(model)
        channel = generate_cdl_frequency_response(
            model=model,
            seed=_SEED + ord(model),
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30_000.0,
            num_subcarriers=8,
            batch_size=_BATCH_SIZE,
            delay_spread_s=100e-9,
            ut_array=CDLArrayConfig(
                polarization="dual",
                polarization_type="VH",
                antenna_pattern="omni",
            ),
            bs_array=CDLArrayConfig(
                polarization="dual",
                polarization_type="VH",
                antenna_pattern="omni",
            ),
            ut_orientation_rad=_LOS_RX_ORIENTATION_RAD,
            bs_orientation_rad=_LOS_TX_ORIENTATION_RAD,
            spec_version="19.2",
            device="cpu",
        )

        per_draw_taps = channel.h_path[:, 0, :, 0, :, :, 0].to(torch.complex128)
        observations = per_draw_taps.abs().square().sum(dim=-1)
        estimates = observations.mean(dim=0)
        standard_errors = observations.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)
        family_size = len(_LOS_PROFILE_SHA256) * expected.numel()
        z_simultaneous = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * family_size))
        half_widths = z_simultaneous * standard_errors

        assert bool((torch.abs(estimates - expected) <= half_widths).all())
        assert float(half_widths.max()) <= 0.25 * float(expected.min())
        assert channel.metadata["parameter_table_sha256"] == source_digest
        assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    finally:
        torch.set_num_threads(previous_threads)
