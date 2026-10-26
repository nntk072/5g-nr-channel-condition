"""G4.07: dual-polarized CDL-A–E power follows independent XPR and K-factor references."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import CDLArrayConfig, generate_cdl_frequency_response  # noqa: E402

_BATCH_SIZE = 4096
_ALIGNED_LOS_CHUNK_SIZE = 32768
_ALIGNED_LOS_BATCH_SIZE = {"D": 262144, "E": 16384}
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
_ZERO_ORIENTATION_RAD = (0.0, 0.0, 0.0)
_LOS_TX_ORIENTATION_RAD = (-0.3, 0.25, 0.5)
_LOS_RX_ORIENTATION_RAD = (0.6, 0.2, -0.4)
_LOS_ORIENTATION_CASES = {
    "tx_tilted": (_ZERO_ORIENTATION_RAD, _LOS_TX_ORIENTATION_RAD),
    "rx_tilted": (_LOS_RX_ORIENTATION_RAD, _ZERO_ORIENTATION_RAD),
    "both_tilted": (_LOS_RX_ORIENTATION_RAD, _LOS_TX_ORIENTATION_RAD),
    "crossed": ((-0.2, 0.45, 0.8), (0.4, -0.3, 0.6)),
}
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
_ROOT = Path(__file__).resolve().parents[2]
_TAP_ANGLE_REFERENCE_PATH = (
    _ROOT / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
)
_PROFILE_SCALAR_REFERENCE_PATH = (
    _ROOT / "docs/research/G4_CDL_PROFILE_SCALAR_REFERENCE_2026-10-26.json"
)
_TAP_ANGLE_REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
_PROFILE_SCALAR_REFERENCE_SHA256 = (
    "3b5920e7c12b0e36c8e09d1c7f14025cfbb23b6d805e1044124bfcf147185620"
)
_ETSI_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"


def _independent_profile_references(model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    tap_bytes = _TAP_ANGLE_REFERENCE_PATH.read_bytes()
    scalar_bytes = _PROFILE_SCALAR_REFERENCE_PATH.read_bytes()
    assert hashlib.sha256(tap_bytes).hexdigest() == _TAP_ANGLE_REFERENCE_SHA256
    assert hashlib.sha256(scalar_bytes).hexdigest() == _PROFILE_SCALAR_REFERENCE_SHA256
    tap_reference = json.loads(tap_bytes)
    scalar_reference = json.loads(scalar_bytes)
    assert tap_reference["source_sha256"] == _ETSI_SOURCE_SHA256
    assert scalar_reference["source_sha256"] == _ETSI_SOURCE_SHA256
    tap_profiles = {row["profile_model"]: row for row in tap_reference["profiles"]}
    scalar_profiles = {row["profile_model"]: row for row in scalar_reference["profiles"]}
    assert set(tap_profiles) == set(scalar_profiles) == set("ABCDE")
    assert tap_profiles[model]["los"] == scalar_profiles[model]["los"]
    assert tap_profiles[model]["xpr_db"] == scalar_profiles[model]["xpr_db"]
    return tap_profiles[model], scalar_profiles[model]


def _independent_nlos_cdl_port_power_reference(model: str) -> Any:
    """Calculate the XPR expectations from the pinned ETSI table transcription."""
    profile, _ = _independent_profile_references(model)
    assert profile["los"] is False
    cluster_powers = [
        10.0 ** (float(component["power_db"]) / 10.0)
        for component in profile["components"]
        if component["component"] == "diffuse"
    ]
    total_pdp_power = math.fsum(cluster_powers)
    normalized_pdp_sum = math.fsum(power / total_pdp_power for power in cluster_powers)
    xpr_linear = 10.0 ** (float(profile["xpr_db"]) / 10.0)

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
    return expected


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
    *,
    rx_orientation_rad: tuple[float, float, float],
    tx_orientation_rad: tuple[float, float, float],
) -> Any:
    """Calculate rotated LOS-plus-diffuse powers from the pinned ETSI references."""
    profile, scalar = _independent_profile_references(model)
    assert profile["los"] is True
    k_factor = 10.0 ** (float(scalar["los_k_factor_db"]) / 10.0)
    xpr_linear = 10.0 ** (float(profile["xpr_db"]) / 10.0)
    specular = next(
        component
        for component in profile["components"]
        if component["component"] == "specular"
    )
    specular_angles = specular["angles_deg"]
    tx_psi = _los_displacement_angle_reference(
        tx_orientation_rad,
        math.radians(float(specular_angles["zod"])),
        math.radians(float(specular_angles["aod"])),
    )
    rx_psi = _los_displacement_angle_reference(
        rx_orientation_rad,
        math.radians(float(specular_angles["zoa"])),
        math.radians(float(specular_angles["aoa"])),
    )
    cross_los_power = math.sin(rx_psi + tx_psi) ** 2
    co_los_power = math.cos(rx_psi + tx_psi) ** 2
    los_power = torch.tensor(
        ((co_los_power, cross_los_power), (cross_los_power, co_los_power)),
        dtype=torch.float64,
    )
    diffuse_components = [
        component
        for component in profile["components"]
        if component["component"] == "diffuse"
    ]
    raw_diffuse_cluster_powers = [
        10.0 ** (float(component["power_db"]) / 10.0)
        for component in diffuse_components
    ]
    diffuse_cluster_power_sum = math.fsum(raw_diffuse_cluster_powers)
    diffuse_power_by_port_pair = [[0.0, 0.0], [0.0, 0.0]]
    spreads = profile["cluster_spreads_deg"]
    for component, raw_power in zip(diffuse_components, raw_diffuse_cluster_powers, strict=True):
        angles = component["angles_deg"]
        rx_component_powers = _mean_dual_vh_field_component_powers(
            rx_orientation_rad,
            float(angles["aoa"]),
            float(angles["zoa"]),
            float(spreads["asa"]),
            float(spreads["zsa"]),
        )
        tx_component_powers = _mean_dual_vh_field_component_powers(
            tx_orientation_rad,
            float(angles["aod"]),
            float(angles["zod"]),
            float(spreads["asd"]),
            float(spreads["zsd"]),
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
    return expected


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _PROFILE_LETTERS)
def test_nlos_cdl_dual_vh_port_powers_match_independent_xpr_ray_law(model: str) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        expected = _independent_nlos_cdl_port_power_reference(model)
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
        assert channel.metadata["parameter_table_sha256"] == _PROFILE_SHA256[model]
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
        profile, scalar = _independent_profile_references(model)
        k_factor = 10.0 ** (float(scalar["los_k_factor_db"]) / 10.0)
        xpr_linear = 10.0 ** (float(profile["xpr_db"]) / 10.0)
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
        assert channel.metadata["parameter_table_sha256"] == _LOS_PROFILE_SHA256[model]
        assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.native_sionna
@pytest.mark.parametrize("orientation_case", tuple(_LOS_ORIENTATION_CASES))
@pytest.mark.parametrize("model", tuple(_LOS_PROFILE_SHA256))
def test_los_cdl_rotated_dual_vh_port_powers_match_xpr_and_field_reference(
    model: str, orientation_case: str
) -> None:
    """Check LOS port mixing over a frozen orientation matrix and diffuse XPR."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rx_orientation_rad, tx_orientation_rad = _LOS_ORIENTATION_CASES[orientation_case]
        expected = _independent_los_cdl_port_power_reference(
            model,
            rx_orientation_rad=rx_orientation_rad,
            tx_orientation_rad=tx_orientation_rad,
        )
        channel = generate_cdl_frequency_response(
            model=model,
            seed=_SEED + ord(model) + tuple(_LOS_ORIENTATION_CASES).index(orientation_case) * 10,
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
            ut_orientation_rad=rx_orientation_rad,
            bs_orientation_rad=tx_orientation_rad,
            spec_version="19.2",
            device="cpu",
        )

        per_draw_taps = channel.h_path[:, 0, :, 0, :, :, 0].to(torch.complex128)
        observations = per_draw_taps.abs().square().sum(dim=-1)
        estimates = observations.mean(dim=0)
        standard_errors = observations.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)
        family_size = (
            len(_LOS_PROFILE_SHA256) * (len(_LOS_ORIENTATION_CASES) + 1) * expected.numel()
        )
        z_simultaneous = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * family_size))
        half_widths = z_simultaneous * standard_errors

        assert bool((torch.abs(estimates - expected) <= half_widths).all())
        assert float(half_widths.max()) <= 0.25 * float(expected.min())
        assert channel.metadata["parameter_table_sha256"] == _LOS_PROFILE_SHA256[model]
        assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", tuple(_LOS_PROFILE_SHA256))
def test_los_cdl_aligned_dual_vh_port_powers_meet_familywise_precision(model: str) -> None:
    """Qualify aligned LOS port powers at fixed, pilot-sized draw budgets."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        expected = _independent_los_cdl_port_power_reference(
            model,
            rx_orientation_rad=_ZERO_ORIENTATION_RAD,
            tx_orientation_rad=_ZERO_ORIENTATION_RAD,
        )
        draw_count = _ALIGNED_LOS_BATCH_SIZE[model]
        observations_by_chunk = []
        for chunk_index, start in enumerate(range(0, draw_count, _ALIGNED_LOS_CHUNK_SIZE)):
            batch_size = min(_ALIGNED_LOS_CHUNK_SIZE, draw_count - start)
            channel = generate_cdl_frequency_response(
                model=model,
                seed=1407 + ord(model) + chunk_index,
                carrier_frequency_hz=3.5e9,
                subcarrier_spacing_hz=30_000.0,
                num_subcarriers=8,
                batch_size=batch_size,
                delay_spread_s=100e-9,
                ut_array=CDLArrayConfig(
                    polarization="dual", polarization_type="VH", antenna_pattern="omni"
                ),
                bs_array=CDLArrayConfig(
                    polarization="dual", polarization_type="VH", antenna_pattern="omni"
                ),
                ut_orientation_rad=_ZERO_ORIENTATION_RAD,
                bs_orientation_rad=_ZERO_ORIENTATION_RAD,
                spec_version="19.2",
                device="cpu",
            )
            observations_by_chunk.append(
                channel.h_path[:, 0, :, 0, :, :, 0].to(torch.complex128).abs().square().sum(dim=-1)
            )

        observations = torch.cat(observations_by_chunk, dim=0)
        estimates = observations.mean(dim=0)
        standard_errors = observations.std(dim=0, unbiased=True) / math.sqrt(draw_count)
        family_size = (
            len(_LOS_PROFILE_SHA256) * (len(_LOS_ORIENTATION_CASES) + 1) * expected.numel()
        )
        z_simultaneous = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * family_size))
        half_widths = z_simultaneous * standard_errors

        assert observations.shape[0] == draw_count
        assert bool((torch.abs(estimates - expected) <= half_widths).all())
        assert float(half_widths.max()) <= 0.25 * float(expected.min())
        assert channel.metadata["parameter_table_sha256"] == _LOS_PROFILE_SHA256[model]
        assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    finally:
        torch.set_num_threads(previous_threads)
