"""Independent covariance references across the Core-1 TDL/CDL matrix.

The 25 TDL cells check the complete first-tap RX-by-TX covariance under an
explicit TR 38.901-style receive correlation matrix and the adapter's
uncorrelated transmit ports. The 25 CDL cells check receive-array covariance
for each topology's configured receive ULA, conditioning on transmit port 0.
CDL transmit-side covariance is deliberately outside this test's claim: the
existing independent ray-law references establish receive-angle covariance,
not the joint departure-angle/polarization covariance needed to qualify it.

Each estimate uses 16,384 independent channel draws. In each individual
matrix cell, two-sided normal intervals with empirical standard errors use a
single Bonferroni family over all 50 profile-topology cells and every unique
real covariance component. This gives nominal asymptotic 99% family-wise
confidence across the full matrix. Every half-width is bounded by 0.05 of the
normalized total channel power. References come from the analytic TDL
correlation law or the byte-pinned ETSI-derived CDL component fixture,
antenna-field equations, and Table 7.5-3 ray-offset/coupling rule, never from
generated channel outputs.
"""

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

from evaluation.g4_tdl_covariance_reference import (  # noqa: E402
    scalable_tdl_first_tap_covariance_reference,
)
from integrations.sionna.cdl import CDLArrayConfig, generate_cdl_frequency_response  # noqa: E402
from integrations.sionna.tdl import generate_tdl_frequency_response  # noqa: E402
from sionna.phy.channel.tr38901 import models as profile_resources  # noqa: E402

_BATCH_SIZE = 16384
_FAMILYWISE_ALPHA = 0.01
_MAX_INTERVAL_HALF_WIDTH = 0.05
_CARRIER_FREQUENCY_HZ = 3.5e9
_TOPOLOGIES = ((1, 1), (2, 1), (1, 2), (2, 2), (4, 4))
_CORE1_MATRIX_CELL_COUNT = 2 * len("ABCDE") * len(_TOPOLOGIES)
_CORE1_MATRIX_COMPARISON_COUNT = len("ABCDE") * sum(
    (num_rx_ant * num_tx_ant) ** 2 + num_rx_ant**2 for num_rx_ant, num_tx_ant in _TOPOLOGIES
)
_CDL_TABLE_SHA256 = {
    "A": "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7",
    "B": "7c87c092cb2bb44891d301ee3c1991cdda1c6b64094a03f18746ec5fa14ee2a2",
    "C": "f2044ecc3b72a1967221805ba70b1f0dab9634bb831478c1337b737ced6e4a28",
    "D": "a8960f176347de457c0cb90fbc8cf137ac98bb1c5295b096f004d8d5c6e40176",
    "E": "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480",
}
_CDL_ETSI_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "research"
    / "G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
)
_CDL_ETSI_REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
_CDL_ETSI_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
_RAY_ANGLE_OFFSETS = (
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


def _covariance_components(covariance: Any) -> Any:
    """Return the unique real components of a Hermitian matrix."""
    size = covariance.shape[0]
    components = [covariance[index, index].real for index in range(size)]
    for row in range(size):
        for column in range(row + 1, size):
            components.extend((covariance[row, column].real, covariance[row, column].imag))
    return torch.stack(components).to(torch.float64)


def _observed_covariance_components(samples: Any) -> tuple[Any, Any]:
    """Compute draw-level covariance observations and their empirical mean."""
    samples = samples.to(torch.complex128)
    outer_products = samples[:, :, None] * samples[:, None, :].conj()
    component_observations = [
        outer_products[:, index, index].real for index in range(samples.shape[1])
    ]
    for row in range(samples.shape[1]):
        for column in range(row + 1, samples.shape[1]):
            component_observations.extend(
                (outer_products[:, row, column].real, outer_products[:, row, column].imag)
            )
    observations = torch.stack(component_observations, dim=1)
    return observations, observations.mean(dim=0)


def _assert_simultaneous_covariance_match(samples: Any, expected: Any) -> None:
    """Apply the preregistered family-wise confidence and precision limits."""
    expected_components = _covariance_components(expected)
    observations, estimate = _observed_covariance_components(samples)
    standard_error = observations.std(dim=0, unbiased=True) / math.sqrt(samples.shape[0])
    assert observations.shape[1] == expected_components.numel()
    assert _CORE1_MATRIX_CELL_COUNT == 50
    assert _CORE1_MATRIX_COMPARISON_COUNT == 1535
    z_simultaneous = NormalDist().inv_cdf(
        1.0 - _FAMILYWISE_ALPHA / (2.0 * _CORE1_MATRIX_COMPARISON_COUNT)
    )
    half_width = z_simultaneous * standard_error
    assert bool((torch.abs(estimate - expected_components) <= half_width).all())
    assert float(half_width.max()) <= _MAX_INTERVAL_HALF_WIDTH
    complex_samples = samples.to(torch.complex128)
    empirical = complex_samples.mT @ complex_samples.conj()
    empirical = empirical / samples.shape[0]
    torch.testing.assert_close(empirical, empirical.mH, rtol=0.0, atol=1e-10)
    assert float(torch.linalg.eigvalsh(empirical).min()) >= -1e-8


def _tdl_rx_correlation(num_rx_ant: int) -> Any:
    """A positive-definite complex AR(1) spatial correlation reference."""
    rho = 0.45
    phase_rad = 0.23
    return torch.tensor(
        [
            [
                rho ** abs(row - column)
                * complex(
                    math.cos(phase_rad * (row - column)), math.sin(phase_rad * (row - column))
                )
                for column in range(num_rx_ant)
            ]
            for row in range(num_rx_ant)
        ],
        dtype=torch.complex128,
    )


def _vertical_field_components(zenith: Any, azimuth: Any) -> tuple[Any, Any]:
    """Compute omni/V field components after the pinned basis rotation.

    For an omni V element, ``AntennaElement.field`` returns (1, 0). The
    channel generator then rotates these components by displacement angle psi
    using its TR 38.901 LCS-to-GCS equation. With the explicit zero TX/RX
    orientations used below, psi=0 for the profile angles and the field is
    (1, 0). Computing psi makes the polarization gain explicit rather than
    silently treating it as a unit scalar.
    """
    real = torch.sin(zenith)
    imag = torch.zeros_like(real)
    psi = torch.atan2(imag, real)
    return torch.cos(psi), torch.sin(psi)


def _diffuse_polarization_power_gain(
    rx_field: tuple[Any, Any], tx_field: tuple[Any, Any], xpr_linear: float
) -> Any:
    """Evaluate E_phi |f_rx^T H_phi f_tx|^2 for four independent phases.

    The pinned phase matrix has unit-magnitude diagonal terms and
    XPR^-1/2-magnitude off-diagonal terms, each with an independent uniform
    phase. All cross terms therefore average to zero.
    """
    rx_theta, rx_phi = rx_field
    tx_theta, tx_phi = tx_field
    return (
        rx_theta.square() * tx_theta.square()
        + rx_theta.square() * tx_phi.square() / xpr_linear
        + rx_phi.square() * tx_theta.square() / xpr_linear
        + rx_phi.square() * tx_phi.square()
    )


def _independent_cdl_receive_covariance_reference(model: str, num_rx_ant: int) -> tuple[Any, str]:
    """Evaluate covariance from pinned profile, ray, field, and XPR laws.

    Source equations are pinned Sionna 2.2 files
    ``sionna/phy/channel/tr38901/antenna.py`` (``AntennaElement.field``) and
    ``channel_coefficients.py`` (``_compute_psi``,
    ``_step_11_phase_matrix``, ``_step_11_field_matrix``, and
    ``_step_11_los``). Profile values come from the byte-pinned ETSI-derived
    fixture; the installed Sionna table is checked separately as generator
    provenance.
    """
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    table_bytes = table_path.read_bytes()
    source_digest = hashlib.sha256(table_bytes).hexdigest()
    assert source_digest == _CDL_TABLE_SHA256[model]
    reference_bytes = _CDL_ETSI_REFERENCE_PATH.read_bytes()
    assert hashlib.sha256(reference_bytes).hexdigest() == _CDL_ETSI_REFERENCE_SHA256
    reference = json.loads(reference_bytes)
    assert reference["source_sha256"] == _CDL_ETSI_SOURCE_SHA256
    profile = next(row for row in reference["profiles"] if row["profile_model"] == model)
    components = profile["components"]
    is_los = profile["los"]
    specular_components = [item for item in components if item["component"] == "specular"]
    diffuse_components = [item for item in components if item["component"] == "diffuse"]
    assert len(specular_components) == int(is_los)
    if is_los:
        assert components[0] is specular_components[0]
        assert components[1:] == diffuse_components
    else:
        assert components == diffuse_components
    assert len(diffuse_components) == profile["cluster_count"]
    parameters = {
        "los": is_los,
        "powers": [item["power_db"] for item in components],
        "aoa": [item["angles_deg"]["aoa"] for item in components],
        "aod": [item["angles_deg"]["aod"] for item in components],
        "zoa": [item["angles_deg"]["zoa"] for item in components],
        "zod": [item["angles_deg"]["zod"] for item in components],
        "cASA": profile["cluster_spreads_deg"]["asa"],
        "cASD": profile["cluster_spreads_deg"]["asd"],
        "cZSA": profile["cluster_spreads_deg"]["zsa"],
        "cZSD": profile["cluster_spreads_deg"]["zsd"],
        "xpr": profile["xpr_db"],
    }
    raw_powers = torch.as_tensor(
        [10.0 ** (float(power_db) / 10.0) for power_db in parameters["powers"]],
        dtype=torch.float64,
    )
    powers = raw_powers / raw_powers.sum()
    covariance = torch.zeros((num_rx_ant, num_rx_ant), dtype=torch.complex128)
    element_indices = torch.arange(num_rx_ant, dtype=torch.float64)
    # With h[row] * conj(h[column]), local-y half-wave spacing gives
    # exp(-j*pi*(column-row)*sin(ZoA)*sin(AoA)).
    separation = element_indices[None, :] - element_indices[:, None]

    diffuse_start = 0
    if bool(parameters["los"]):
        los_projection = math.sin(math.radians(float(parameters["zoa"][0]))) * math.sin(
            math.radians(float(parameters["aoa"][0]))
        )
        los_aoa = torch.tensor(math.radians(float(parameters["aoa"][0])), dtype=torch.float64)
        los_zoa = torch.tensor(math.radians(float(parameters["zoa"][0])), dtype=torch.float64)
        los_aod = torch.tensor(math.radians(float(parameters["aod"][0])), dtype=torch.float64)
        los_zod = torch.tensor(math.radians(float(parameters["zod"][0])), dtype=torch.float64)
        f_rx_los = _vertical_field_components(los_zoa, los_aoa)
        f_tx_los = _vertical_field_components(los_zod, los_aod)
        # The LOS polarization matrix is diag(1,-1).
        los_gain = (f_rx_los[0] * f_tx_los[0] - f_rx_los[1] * f_tx_los[1]).square()
        los_phase = torch.exp(-1j * math.pi * los_projection * separation)
        covariance += powers[0].to(torch.complex128) * los_gain * los_phase
        diffuse_start = 1

    offsets = torch.as_tensor(_RAY_ANGLE_OFFSETS, dtype=torch.float64)
    aoa = torch.deg2rad(
        torch.as_tensor(parameters["aoa"][diffuse_start:], dtype=torch.float64)[:, None]
        + float(parameters["cASA"]) * offsets
    )
    zoa = torch.deg2rad(
        torch.as_tensor(parameters["zoa"][diffuse_start:], dtype=torch.float64)[:, None]
        + float(parameters["cZSA"]) * offsets
    )
    projection = torch.sin(zoa)[:, :, None] * torch.sin(aoa)[:, None, :]
    ray_phases = torch.exp(-1j * math.pi * projection[..., None, None] * separation)
    zod = torch.deg2rad(
        torch.as_tensor(parameters["zod"][diffuse_start:], dtype=torch.float64)[:, None]
        + float(parameters["cZSD"]) * offsets
    )
    aod = torch.deg2rad(
        torch.as_tensor(parameters["aod"][diffuse_start:], dtype=torch.float64)[:, None]
        + float(parameters["cASD"]) * offsets
    )
    f_rx = _vertical_field_components(zoa, aoa)
    f_tx = _vertical_field_components(zod, aod)
    xpr_linear = 10.0 ** (float(parameters["xpr"]) / 10.0)
    paired_ray_gain = _diffuse_polarization_power_gain(f_rx, f_tx, xpr_linear)
    # AoA, ZoA, AoD, and ZoD rays are independently coupled. In this explicit
    # zero-orientation omni/V setup the computed fields are all (1,0), so the
    # gain is exactly one for every one of the 20^4 cross-couplings. Therefore
    # the exact polarization-phase and ray-coupling expectation is this
    # constant, while spatial phase still averages over independent AoA/ZoA.
    torch.testing.assert_close(
        paired_ray_gain, torch.ones_like(paired_ray_gain), rtol=0.0, atol=1e-12
    )
    cluster_covariance = ray_phases.mean(dim=(1, 2))
    covariance += (
        powers[diffuse_start:].to(torch.complex128)[:, None, None] * cluster_covariance
    ).sum(dim=0)
    return covariance, source_digest


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", tuple("ABCDE"))
@pytest.mark.parametrize(("num_rx_ant", "num_tx_ant"), _TOPOLOGIES)
def test_core1_tdl_first_tap_covariance_matches_analytic_correlation(
    model: str, num_rx_ant: int, num_tx_ant: int
) -> None:
    """Check all TDL profiles and Core-1 topologies against the exact law."""
    rx_corr = _tdl_rx_correlation(num_rx_ant)
    channel = generate_tdl_frequency_response(
        model=model,
        seed=813_000 + ord(model) * 10 + num_rx_ant * 5 + num_tx_ant,
        carrier_frequency_hz=_CARRIER_FREQUENCY_HZ,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_BATCH_SIZE,
        delay_spread_s=100e-9,
        num_rx_ant=num_rx_ant,
        num_tx_ant=num_tx_ant,
        rx_corr_mat=rx_corr.tolist(),
        spec_version="19.2",
        device="cpu",
    )
    samples = channel.h_path[:, 0, :, 0, :, 0, 0].reshape(_BATCH_SIZE, -1)
    expected = torch.as_tensor(
        scalable_tdl_first_tap_covariance_reference(model, rx_corr.numpy(), num_tx_ant),
        dtype=torch.complex128,
    )
    assert tuple(samples.shape) == (_BATCH_SIZE, num_rx_ant * num_tx_ant)
    assert channel.metadata["profile_proxy"] == model
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    _assert_simultaneous_covariance_match(samples, expected)


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", tuple("ABCDE"))
@pytest.mark.parametrize(("num_rx_ant", "num_tx_ant"), _TOPOLOGIES)
def test_core1_cdl_receive_covariance_matches_independent_ray_law(
    model: str, num_rx_ant: int, num_tx_ant: int
) -> None:
    """Check CDL receive covariance for every profile/topology cell.

    The covariance is formed from the paired receive ports at transmit port 0.
    Configuring the requested BS array preserves each topology cell, but this
    independent reference only qualifies the receive-array marginal.
    """
    expected, source_digest = _independent_cdl_receive_covariance_reference(model, num_rx_ant)
    channel = generate_cdl_frequency_response(
        model=model,
        seed=923_000 + ord(model) * 10 + num_rx_ant * 5 + num_tx_ant,
        carrier_frequency_hz=_CARRIER_FREQUENCY_HZ,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_BATCH_SIZE,
        delay_spread_s=100e-9,
        ut_array=CDLArrayConfig(
            num_cols_per_panel=num_rx_ant,
            element_horizontal_spacing=0.5,
        ),
        bs_array=CDLArrayConfig(
            num_cols_per_panel=num_tx_ant,
            element_horizontal_spacing=0.5,
        ),
        ut_orientation_rad=(0.0, 0.0, 0.0),
        bs_orientation_rad=(0.0, 0.0, 0.0),
        spec_version="19.2",
        device="cpu",
    )
    samples = channel.h_path[:, 0, :, 0, 0, :, 0].sum(dim=-1)
    assert tuple(samples.shape) == (_BATCH_SIZE, num_rx_ant)
    assert channel.metadata["rx_antenna_count"] == num_rx_ant
    assert channel.metadata["tx_antenna_count"] == num_tx_ant
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["parameter_table_sha256"] == source_digest
    _assert_simultaneous_covariance_match(samples, expected)
