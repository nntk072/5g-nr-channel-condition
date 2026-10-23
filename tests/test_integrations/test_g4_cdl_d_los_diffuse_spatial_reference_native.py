"""Bounded CDL-D LOS-plus-diffuse spatial covariance ray reference.

This native CPU case checks CDL-D v19.2 at 3.5 GHz for a stationary
downlink and a two-element UT ULA along local y, with vertical-polarized
omnidirectional elements at half-wave spacing and zero orientation. Each of
8,192 independent channel draws yields one paired two-port channel vector;
four unique real covariance components
must contain the pinned 3GPP ray-law expectation in simultaneous two-sided
99% normal intervals, each no wider than 0.05 normalized mean power. The
reference combines the deterministic LOS steering term with diffuse cluster
ray sums over the independent azimuth/zenith ray pairings. This checks only
this profile and array configuration and provides no external channel claim.
"""

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

_BATCH_SIZE = 8192
_SEED = 1986
_CARRIER_FREQUENCY_HZ = 3.5e9
_FAMILYWISE_ALPHA = 0.01
_MAX_INTERVAL_HALF_WIDTH = 0.05
_CDL_D_TABLE_SHA256 = "a8960f176347de457c0cb90fbc8cf137ac98bb1c5295b096f004d8d5c6e40176"
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


def _independent_cdl_d_spatial_covariance_reference() -> tuple[Any, str]:
    """Evaluate the two-port covariance from the pinned CDL-D ray law."""
    table_path = profile_resources.parameter_file("CDL-D.json", "19.2")
    table_bytes = table_path.read_bytes()
    digest = hashlib.sha256(table_bytes).hexdigest()
    assert digest == _CDL_D_TABLE_SHA256
    parameters = profile_resources.load_json(table_path)
    assert parameters["los"] == 1
    assert int(parameters["num_clusters"]) == len(parameters["powers"]) - 1
    assert len(parameters["powers"]) == len(parameters["aoa"]) == len(parameters["zoa"])

    raw_powers = torch.as_tensor(
        [10.0 ** (float(power_db) / 10.0) for power_db in parameters["powers"]],
        dtype=torch.float64,
    )
    powers = raw_powers / raw_powers.sum()
    offsets = torch.as_tensor(_RAY_ANGLE_OFFSETS, dtype=torch.float64)

    # The first table entry is the LOS specular component. For a local-y
    # half-wave ULA, its inter-element phase uses sin(ZoA)*sin(AoA).
    los_aoa = math.radians(float(parameters["aoa"][0]))
    los_zoa = math.radians(float(parameters["zoa"][0]))
    los_projection = math.sin(los_zoa) * math.sin(los_aoa)
    off_diagonal = powers[0].to(torch.complex128) * torch.exp(
        torch.tensor(-1j * math.pi * los_projection, dtype=torch.complex128)
    )

    # CDL-D's remaining table entries are diffuse clusters. TR 38.901
    # couples the 20 azimuth offsets independently from the 20 zenith
    # offsets, so each cluster expectation averages their Cartesian product.
    aoa = torch.deg2rad(
        torch.as_tensor(parameters["aoa"][1:], dtype=torch.float64)[:, None]
        + float(parameters["cASA"]) * offsets
    )
    zoa = torch.deg2rad(
        torch.as_tensor(parameters["zoa"][1:], dtype=torch.float64)[:, None]
        + float(parameters["cZSA"]) * offsets
    )
    direction_projection = torch.sin(zoa)[:, :, None] * torch.sin(aoa)[:, None, :]
    diffuse_spatial_phase = torch.exp(-1j * math.pi * direction_projection).mean(dim=(1, 2))
    off_diagonal += (powers[1:].to(torch.complex128) * diffuse_spatial_phase).sum()

    covariance = torch.stack(
        (
            torch.stack((torch.ones_like(off_diagonal.real), off_diagonal)),
            torch.stack((off_diagonal.conj(), torch.ones_like(off_diagonal.real))),
        )
    )
    return covariance, digest


@pytest.mark.native_sionna
def test_cdl_d_spatial_covariance_matches_independent_los_diffuse_ray_reference() -> None:
    expected, source_digest = _independent_cdl_d_spatial_covariance_reference()
    channel = generate_cdl_frequency_response(
        model="D",
        seed=_SEED,
        carrier_frequency_hz=_CARRIER_FREQUENCY_HZ,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_BATCH_SIZE,
        delay_spread_s=100e-9,
        ut_array=CDLArrayConfig(
            num_cols_per_panel=2,
            element_horizontal_spacing=0.5,
        ),
        ut_orientation_rad=(0.0, 0.0, 0.0),
        spec_version="19.2",
        device="cpu",
    )

    # Axes are [draw, rx, rx_ant, tx, tx_ant, cluster, time]. Sum all
    # clusters to form one narrowband two-port vector per independent draw.
    samples = channel.h_path[:, 0, :, 0, 0, :, 0].sum(dim=-1).to(torch.complex128)
    assert tuple(samples.shape) == (_BATCH_SIZE, 2)
    outer_products = samples[:, :, None] * samples[:, None, :].conj()
    observations = torch.stack(
        (
            outer_products[:, 0, 0].real,
            outer_products[:, 1, 1].real,
            outer_products[:, 0, 1].real,
            outer_products[:, 0, 1].imag,
        ),
        dim=1,
    )
    expected_components = torch.stack(
        (expected[0, 0].real, expected[1, 1].real, expected[0, 1].real, expected[0, 1].imag)
    )
    estimate = observations.mean(dim=0)
    standard_error = observations.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)

    # Bonferroni controls the four unique real entries jointly at 99%.
    z_simultaneous = NormalDist().inv_cdf(
        1.0 - _FAMILYWISE_ALPHA / (2.0 * expected_components.numel())
    )
    half_width = z_simultaneous * standard_error
    assert bool((torch.abs(estimate - expected_components) <= half_width).all())
    assert float(half_width.max()) <= _MAX_INTERVAL_HALF_WIDTH
    assert channel.metadata["profile_model"] == "D"
    assert channel.metadata["profile_los_component"] is True
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["parameter_table_sha256"] == source_digest
