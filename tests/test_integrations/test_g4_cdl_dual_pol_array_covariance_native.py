"""NLOS CDL dual-polarized two-element covariance against pinned TR 38.901 rays.

The finite family covers CDL-A/B/C with a vertical/horizontal, two-element
UT array and one vertical TX port. The reference independently averages the
Table 7.5-3 azimuth/zenith ray sets, cluster powers, and XPR from the pinned
ETSI-derived fixture. One simultaneous 99% family covers all 16 unique real
Hermitian covariance components for all three profiles. This is a bounded
standardized-model check, not evidence for LOS D/E or arbitrary arrays.
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

from integrations.sionna.cdl import CDLArrayConfig, generate_cdl_frequency_response  # noqa: E402
from sionna.phy.channel.tr38901 import models as profile_resources  # noqa: E402

_BATCH_SIZE = 8192
_SEED_START = 1_700_000
_FAMILYWISE_ALPHA = 0.01
_MAX_INTERVAL_HALF_WIDTH = 0.05
_PROFILE_SHA256 = {
    "A": "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7",
    "B": "7c87c092cb2bb44891d301ee3c1991cdda1c6b64094a03f18746ec5fa14ee2a2",
    "C": "f2044ecc3b72a1967221805ba70b1f0dab9634bb831478c1337b737ced6e4a28",
}
_ETSI_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "research"
    / "G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
)
_ETSI_REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
_ETSI_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
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
_MODELS = tuple(_PROFILE_SHA256)


def _independent_dual_pol_array_covariance_reference(model: str) -> tuple[Any, str]:
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    table_digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    assert table_digest == _PROFILE_SHA256[model]
    reference_bytes = _ETSI_REFERENCE_PATH.read_bytes()
    assert hashlib.sha256(reference_bytes).hexdigest() == _ETSI_REFERENCE_SHA256
    document = json.loads(reference_bytes)
    assert document["source_sha256"] == _ETSI_SOURCE_SHA256
    profile = next(row for row in document["profiles"] if row["profile_model"] == model)
    assert profile["los"] is False
    components = profile["components"]
    assert all(row["component"] == "diffuse" for row in components)
    assert len(components) == profile["cluster_count"]

    powers = torch.as_tensor(
        [10.0 ** (float(row["power_db"]) / 10.0) for row in components],
        dtype=torch.float64,
    )
    powers = powers / powers.sum()
    offsets = torch.as_tensor(_RAY_ANGLE_OFFSETS, dtype=torch.float64)
    aoa = torch.deg2rad(
        torch.as_tensor(
            [row["angles_deg"]["aoa"] for row in components], dtype=torch.float64
        )[:, None]
        + float(profile["cluster_spreads_deg"]["asa"]) * offsets
    )
    zoa = torch.deg2rad(
        torch.as_tensor(
            [row["angles_deg"]["zoa"] for row in components], dtype=torch.float64
        )[:, None]
        + float(profile["cluster_spreads_deg"]["zsa"]) * offsets
    )

    # The receive ULA is on local y, so TR 38.901's arrival direction
    # projection is sin(ZoA)*sin(AoA). Independent azimuth/zenith coupling is
    # represented by the Cartesian product of their 20-ray sets.
    projection = torch.sin(zoa)[:, :, None] * torch.sin(aoa)[:, None, :]
    spatial_phase = torch.exp(-1j * math.pi * projection)
    xpr_linear = 10.0 ** (float(profile["xpr_db"]) / 10.0)
    # Use the TS 38.901 polarization-basis rotation at the frozen zero array
    # orientation. For V/H receive feeds the field powers are complementary.
    psi = torch.atan2(torch.zeros_like(zoa), torch.sin(zoa))
    vertical_power = torch.cos(psi).square() + torch.sin(psi).square() / xpr_linear
    horizontal_power = torch.sin(psi).square() + torch.cos(psi).square() / xpr_linear
    vertical_diagonal = (powers * vertical_power.mean(dim=1)).sum()
    horizontal_diagonal = (powers * horizontal_power.mean(dim=1)).sum()
    vertical_correlation = (
        powers.to(torch.complex128)
        * (vertical_power[:, :, None] * spatial_phase).mean(dim=(1, 2))
    ).sum()
    horizontal_correlation = (
        powers.to(torch.complex128)
        * (horizontal_power[:, :, None] * spatial_phase).mean(dim=(1, 2))
    ).sum()
    covariance = torch.zeros((4, 4), dtype=torch.complex128)
    covariance[0, 0] = covariance[1, 1] = vertical_diagonal
    covariance[0, 1] = vertical_correlation
    covariance[1, 0] = vertical_correlation.conj()
    covariance[2, 2] = covariance[3, 3] = horizontal_diagonal
    covariance[2, 3] = horizontal_correlation
    covariance[3, 2] = horizontal_correlation.conj()
    return covariance, table_digest


def _unique_hermitian_observations(samples: Any) -> Any:
    outer = samples[:, :, None] * samples[:, None, :].conj()
    observations = []
    for row in range(4):
        observations.append(outer[:, row, row].real)
        for column in range(row + 1, 4):
            observations.append(outer[:, row, column].real)
            observations.append(outer[:, row, column].imag)
    return torch.stack(observations, dim=1)


def _reference_components(reference: Any) -> Any:
    values = []
    for row in range(4):
        values.append(reference[row, row].real)
        for column in range(row + 1, 4):
            values.append(reference[row, column].real)
            values.append(reference[row, column].imag)
    return torch.stack(values).to(torch.float64)


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
def test_cdl_dual_polarized_array_covariance_matches_pinned_reference(model: str) -> None:
    reference, table_digest = _independent_dual_pol_array_covariance_reference(model)
    channel = generate_cdl_frequency_response(
        model=model,
        seed=_SEED_START + _MODELS.index(model),
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_BATCH_SIZE,
        delay_spread_s=100e-9,
        ut_array=CDLArrayConfig(
            num_cols_per_panel=2,
            element_horizontal_spacing=0.5,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="omni",
        ),
        bs_array=CDLArrayConfig(
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
        ),
        spec_version="19.2",
        device="cpu",
    )

    # Sum clusters into one narrowband receive vector per independent draw.
    samples = channel.h_path[:, 0, :, 0, 0, :, 0].sum(dim=-1).to(torch.complex128)
    assert tuple(samples.shape) == (_BATCH_SIZE, 4)
    empirical_components = _unique_hermitian_observations(samples)
    expected_components = _reference_components(reference)
    estimate = empirical_components.mean(dim=0)
    standard_error = empirical_components.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)
    z_simultaneous = NormalDist().inv_cdf(
        1.0 - _FAMILYWISE_ALPHA / (2.0 * expected_components.numel() * len(_MODELS))
    )
    half_width = z_simultaneous * standard_error

    assert bool((torch.abs(estimate - expected_components) <= half_width).all())
    assert float(half_width.max()) <= _MAX_INTERVAL_HALF_WIDTH
    assert channel.metadata["profile_model"] == model
    assert channel.metadata["profile_los_component"] is False
    assert channel.metadata["parameter_table_sha256"] == table_digest
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
