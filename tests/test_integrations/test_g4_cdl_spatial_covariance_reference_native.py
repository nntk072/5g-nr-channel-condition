"""Independent 3GPP ray-level checks of bounded CDL spatial covariance cases.

CDL-A and CDL-B each use a two-element half-wavelength UT ULA with vertical
omnidirectional elements on a stationary 3.5-GHz downlink. For each profile,
4,096 independent channel draws produce four unique real covariance components
that must contain the reference in two-sided Bonferroni 99% confidence
intervals. Each interval has a half-width at most 0.05 of unit normalized mean
power. The reference averages the Table 7.5-3 ray offsets and all independent
azimuth/zenith ray pairings, rather than reusing channel output or adapter
profile metadata as its expected answer."""

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
_FAMILYWISE_ALPHA = 0.01
_MAX_INTERVAL_HALF_WIDTH = 0.05
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


def _independent_cdl_spatial_covariance_reference(model: str) -> tuple[Any, str]:
    """Evaluate one profile ray sum directly from its pinned table."""
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    parameters = profile_resources.load_json(table_path)
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

    # For a local-y ULA at zero UT orientation, the arrival-direction
    # projection onto its half-wavelength element spacing is
    # sin(ZoA) * sin(AoA). TR 38.901 independently couples the azimuth and
    # zenith ray sets; averaging their Cartesian product is the expectation
    # over that random coupling, rather than assuming matching ray indices.
    direction_projection = torch.sin(zoa)[:, :, None] * torch.sin(aoa)[:, None, :]
    spatial_phase = torch.exp(-1j * math.pi * direction_projection)
    off_diagonal = (powers.to(torch.complex128) * spatial_phase.mean(dim=(1, 2))).sum()
    covariance = torch.stack(
        (
            torch.stack((torch.ones_like(off_diagonal.real), off_diagonal)),
            torch.stack((off_diagonal.conj(), torch.ones_like(off_diagonal.real))),
        )
    )
    digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    return covariance, digest


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", ("A", "B"))
def test_cdl_spatial_covariance_matches_independent_38901_ray_reference(model: str) -> None:
    expected, source_digest = _independent_cdl_spatial_covariance_reference(model)
    channel = generate_cdl_frequency_response(
        model=model,
        seed=1321,
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

    # Summing clusters forms one narrowband channel vector for each draw.
    # Keep both antenna ports paired within each independent realization.
    samples = channel.h_path[:, 0, :, 0, 0, :, 0].sum(dim=-1).to(torch.complex128)
    outer_products = samples[:, :, None] * samples[:, None, :].conj()
    empirical = outer_products.mean(dim=0)
    diagonal_power = torch.stack(
        (outer_products[:, 0, 0].real, outer_products[:, 1, 1].real), dim=1
    )
    off_diagonal = outer_products[:, 0, 1]
    observations = torch.stack(
        (
            diagonal_power[:, 0],
            diagonal_power[:, 1],
            off_diagonal.real,
            off_diagonal.imag,
        ),
        dim=1,
    )
    expected_components = torch.stack(
        (expected[0, 0].real, expected[1, 1].real, expected[0, 1].real, expected[0, 1].imag)
    )
    estimate = observations.mean(dim=0)
    standard_error = observations.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)

    # Bonferroni gives 99% family-wise coverage for the four unique real
    # entries of the Hermitian 2x2 covariance matrix.
    z_simultaneous = NormalDist().inv_cdf(
        1.0 - _FAMILYWISE_ALPHA / (2.0 * expected_components.numel())
    )
    half_width = z_simultaneous * standard_error
    assert bool((torch.abs(estimate - expected_components) <= half_width).all())
    assert float(half_width.max()) <= _MAX_INTERVAL_HALF_WIDTH
    torch.testing.assert_close(empirical, empirical.mH, rtol=0.0, atol=1e-7)

    assert channel.metadata["profile_model"] == model
    assert channel.metadata["profile_los_component"] is False
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["parameter_table_sha256"] == source_digest
