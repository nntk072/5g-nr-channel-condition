"""Independent ray-level qualification for a four-element CDL-A ULA.

The case and acceptance rule are fixed before comparing the channel estimate:
CDL-A at 3.5 GHz, four vertical omnidirectional UT elements on a local-y
half-wavelength ULA, and a stationary downlink. Across 8,192 independent
channel draws, the 16 unique real components of the Hermitian 4x4 covariance
must contain the independent TR 38.901 ray reference in two-sided Bonferroni
99% family-wise confidence intervals. Every interval half-width must be at
most 0.05 of normalized mean power. The ray expectation uses the pinned
CDL-A parameter table, its frozen SHA-256 digest, Table 7.5-3 ray offsets,
and the independent azimuth/zenith ray coupling rule.
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
_FAMILYWISE_ALPHA = 0.01
_MAX_INTERVAL_HALF_WIDTH = 0.05
_CDL_A_TABLE_SHA256 = "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7"
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


def _independent_cdl_a_spatial_covariance_reference() -> tuple[Any, str]:
    """Evaluate the 4x4 covariance by direct ray averaging from TR 38.901."""
    table_path = profile_resources.parameter_file("CDL-A.json", "19.2")
    table_bytes = table_path.read_bytes()
    digest = hashlib.sha256(table_bytes).hexdigest()
    assert digest == _CDL_A_TABLE_SHA256
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

    # For the local-y axis at zero UT orientation, the arrival-direction
    # projection is sin(ZoA)sin(AoA). The independent ray coupling pairs each
    # azimuth ray with each zenith ray. A half-wave element spacing gives a
    # pi phase increment per unit direction projection.
    direction_projection = torch.sin(zoa)[:, :, None] * torch.sin(aoa)[:, None, :]
    element_indices = torch.arange(4, dtype=torch.float64)
    # The channel covariance uses h[row] * conj(h[column]); matching that
    # convention makes the (0, 1) entry use the positive one-element offset.
    element_separation = element_indices[None, :] - element_indices[:, None]
    spatial_phase = torch.exp(
        -1j * math.pi * direction_projection[..., None, None] * element_separation
    )
    cluster_covariance = spatial_phase.mean(dim=(1, 2))
    covariance = (powers.to(torch.complex128)[:, None, None] * cluster_covariance).sum(dim=0)
    return covariance, digest


@pytest.mark.native_sionna
def test_cdl_a_four_element_spatial_covariance_matches_independent_ray_reference() -> None:
    expected, source_digest = _independent_cdl_a_spatial_covariance_reference()
    channel = generate_cdl_frequency_response(
        model="A",
        seed=6714,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_BATCH_SIZE,
        delay_spread_s=100e-9,
        ut_array=CDLArrayConfig(
            num_cols_per_panel=4,
            num_cols=1,
            element_horizontal_spacing=0.5,
        ),
        ut_orientation_rad=(0.0, 0.0, 0.0),
        spec_version="19.2",
        device="cpu",
    )

    # Keep array ports paired within each independent realization and form a
    # narrowband sample by summing all profile clusters, as in the 2-element
    # G4 reference qualification.
    samples = channel.h_path[:, 0, :, 0, 0, :, 0].sum(dim=-1).to(torch.complex128)
    assert samples.shape == (_BATCH_SIZE, 4)
    outer_products = samples[:, :, None] * samples[:, None, :].conj()
    empirical = outer_products.mean(dim=0)

    component_observations = []
    expected_components = []
    for row in range(4):
        component_observations.append(outer_products[:, row, row].real)
        expected_components.append(expected[row, row].real)
    for row in range(4):
        for column in range(row + 1, 4):
            component_observations.extend(
                (outer_products[:, row, column].real, outer_products[:, row, column].imag)
            )
            expected_components.extend((expected[row, column].real, expected[row, column].imag))

    observations = torch.stack(component_observations, dim=1)
    expected_vector = torch.stack(expected_components)
    estimate = observations.mean(dim=0)
    standard_error = observations.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)

    # Four diagonal entries plus real and imaginary parts of six unique
    # off-diagonal entries produce a 16-component family-wise CI.
    component_count = expected_vector.numel()
    assert component_count == 16
    z_simultaneous = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * component_count))
    half_width = z_simultaneous * standard_error
    assert bool((torch.abs(estimate - expected_vector) <= half_width).all())
    assert float(half_width.max()) <= _MAX_INTERVAL_HALF_WIDTH
    torch.testing.assert_close(empirical, empirical.mH, rtol=0.0, atol=1e-7)
    assert float(torch.linalg.eigvalsh(empirical).min()) >= -1e-7

    assert channel.metadata["profile_los_component"] is False
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["parameter_table_sha256"] == source_digest
