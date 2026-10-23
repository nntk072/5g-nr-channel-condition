"""Bounded CDL-C spatial covariance check against an independent ray sum.

This adds one standardized NLOS profile to the existing G4 spatial reference
scope: CDL-C v19.2 at 3.5 GHz, with a stationary downlink and a two-element,
vertical, omnidirectional UT ULA at half-wavelength spacing. The expected
covariance is calculated directly from the pinned 3GPP profile table and
Table 7.5-3 ray offsets, including the independent azimuth/zenith ray
coupling. Across 4,096 independent channel draws, the four unique real
components of the Hermitian covariance must contain that reference in
two-sided Bonferroni 99% confidence intervals, each with half-width at most
0.05 of normalized mean power. This check covers this configuration only;
it does not qualify other arrays, orientations, carriers, or CDL profiles.
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

_BATCH_SIZE = 4096
_SEED = 1322
_FAMILYWISE_ALPHA = 0.01
_MAX_INTERVAL_HALF_WIDTH = 0.05
_CDL_C_TABLE_SHA256 = "f2044ecc3b72a1967221805ba70b1f0dab9634bb831478c1337b737ced6e4a28"
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


def _independent_cdl_c_spatial_covariance_reference() -> tuple[Any, str]:
    """Evaluate the CDL-C 2x2 covariance from the pinned 3GPP ray law."""
    table_path = profile_resources.parameter_file("CDL-C.json", "19.2")
    table_bytes = table_path.read_bytes()
    digest = hashlib.sha256(table_bytes).hexdigest()
    assert digest == _CDL_C_TABLE_SHA256
    parameters = profile_resources.load_json(table_path)
    assert parameters["los"] == 0
    assert int(parameters["num_clusters"]) == len(parameters["powers"])

    powers = torch.as_tensor(
        [10.0 ** (float(power_db) / 10.0) for power_db in parameters["powers"]],
        dtype=torch.float64,
    )
    powers = powers / powers.sum()
    offsets = torch.as_tensor(_RAY_ANGLE_OFFSETS, dtype=torch.float64)
    aoa = torch.deg2rad(
        torch.as_tensor(parameters["aoa"], dtype=torch.float64)[:, None]
        + float(parameters["cASA"]) * offsets
    )
    zoa = torch.deg2rad(
        torch.as_tensor(parameters["zoa"], dtype=torch.float64)[:, None]
        + float(parameters["cZSA"]) * offsets
    )

    # A local-y half-wave ULA at zero orientation has ray projection
    # sin(ZoA)*sin(AoA). Azimuth and zenith ray sets are independently
    # coupled, so the ray expectation averages their Cartesian product.
    direction_projection = torch.sin(zoa)[:, :, None] * torch.sin(aoa)[:, None, :]
    spatial_phase = torch.exp(-1j * math.pi * direction_projection)
    off_diagonal = (powers.to(torch.complex128) * spatial_phase.mean(dim=(1, 2))).sum()
    covariance = torch.stack(
        (
            torch.stack((torch.ones_like(off_diagonal.real), off_diagonal)),
            torch.stack((off_diagonal.conj(), torch.ones_like(off_diagonal.real))),
        )
    )
    return covariance, digest


@pytest.mark.native_sionna
def test_cdl_c_spatial_covariance_matches_independent_38901_ray_reference() -> None:
    expected, source_digest = _independent_cdl_c_spatial_covariance_reference()
    channel = generate_cdl_frequency_response(
        model="C",
        seed=_SEED,
        carrier_frequency_hz=3.5e9,
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

    # Sum clusters for one narrowband channel vector per independent draw.
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

    # Bonferroni covers the four unique real entries of the Hermitian matrix.
    z_simultaneous = NormalDist().inv_cdf(
        1.0 - _FAMILYWISE_ALPHA / (2.0 * expected_components.numel())
    )
    half_width = z_simultaneous * standard_error
    assert bool((torch.abs(estimate - expected_components) <= half_width).all())
    assert float(half_width.max()) <= _MAX_INTERVAL_HALF_WIDTH
    assert channel.metadata["profile_model"] == "C"
    assert channel.metadata["profile_los_component"] is False
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["parameter_table_sha256"] == source_digest
