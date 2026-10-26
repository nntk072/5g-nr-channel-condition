"""Independent ray-level qualification for four-element CDL-A–E ULAs.

The configuration and acceptance rule are fixed before comparing channel
estimates: each profile at 3.5 GHz, four vertical omnidirectional UT elements
on a local-y half-wavelength ULA, and a stationary downlink. Across 8,192
independent channel draws per profile, the 16 unique real components of the
Hermitian 4x4 covariance must contain the independent TR 38.901 ray reference
in two-sided Bonferroni 99% family-wise confidence intervals. Every interval
half-width must be at most 0.05 of normalized mean power. The ray expectation
uses byte-pinned ETSI-derived profile component rows, Table 7.5-3 ray offsets,
and the independent azimuth/zenith ray coupling rule. CDL-D/E LOS steering and
diffuse contributions are accumulated separately.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

_BATCH_SIZE = 8192
_FAMILYWISE_ALPHA = 0.01
_MAX_INTERVAL_HALF_WIDTH = 0.05
_PROFILE_TABLE_SHA256 = {
    "A": "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7",
    "B": "7c87c092cb2bb44891d301ee3c1991cdda1c6b64094a03f18746ec5fa14ee2a2",
    "C": "f2044ecc3b72a1967221805ba70b1f0dab9634bb831478c1337b737ced6e4a28",
    "D": "a8960f176347de457c0cb90fbc8cf137ac98bb1c5295b096f004d8d5c6e40176",
    "E": "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480",
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


def _independent_cdl_spatial_covariance_reference(model: str) -> tuple[np.ndarray, str]:
    """Evaluate a 4x4 covariance from pinned ETSI profile component rows."""
    reference_bytes = _ETSI_REFERENCE_PATH.read_bytes()
    assert hashlib.sha256(reference_bytes).hexdigest() == _ETSI_REFERENCE_SHA256
    document = json.loads(reference_bytes)
    assert document["source_sha256"] == _ETSI_SOURCE_SHA256
    profile = next(item for item in document["profiles"] if item["profile_model"] == model)
    components = profile["components"]
    specular_components = [item for item in components if item["component"] == "specular"]
    diffuse_components = [item for item in components if item["component"] == "diffuse"]
    if profile["los"]:
        assert len(specular_components) == 1
        assert components[0] is specular_components[0]
        assert components[1:] == diffuse_components
    else:
        assert not specular_components
        assert components == diffuse_components
    assert len(diffuse_components) == profile["cluster_count"]

    component_powers = np.asarray(
        [10.0 ** (float(item["power_db"]) / 10.0) for item in components],
        dtype=np.float64,
    )
    component_powers = component_powers / component_powers.sum()
    offsets = np.asarray(_RAY_ANGLE_OFFSETS, dtype=np.float64)
    aoa = np.deg2rad(
        np.asarray([item["angles_deg"]["aoa"] for item in diffuse_components], dtype=np.float64)[
            :, None
        ]
        + float(profile["cluster_spreads_deg"]["asa"]) * offsets[None, :]
    )
    zoa = np.deg2rad(
        np.asarray([item["angles_deg"]["zoa"] for item in diffuse_components], dtype=np.float64)[
            :, None
        ]
        + float(profile["cluster_spreads_deg"]["zsa"]) * offsets[None, :]
    )

    # For the local-y axis at zero UT orientation, the arrival-direction
    # projection is sin(ZoA)sin(AoA). The independent ray coupling pairs each
    # azimuth ray with each zenith ray. A half-wave element spacing gives a
    # pi phase increment per unit direction projection.
    direction_projection = np.sin(zoa)[:, :, None] * np.sin(aoa)[:, None, :]
    element_indices = np.arange(4, dtype=np.float64)
    # The channel covariance uses h[row] * conj(h[column]); matching that
    # convention makes the (0, 1) entry use the positive one-element offset.
    element_separation = element_indices[None, :] - element_indices[:, None]
    spatial_phase = np.exp(
        -1j * math.pi * direction_projection[..., None, None] * element_separation
    )
    cluster_covariance = spatial_phase.mean(axis=(1, 2))
    diffuse_powers = component_powers[-len(diffuse_components) :]
    covariance = np.sum(diffuse_powers[:, None, None] * cluster_covariance, axis=0)
    if profile["los"]:
        los_component = specular_components[0]
        los_aoa = math.radians(float(los_component["angles_deg"]["aoa"]))
        los_zoa = math.radians(float(los_component["angles_deg"]["zoa"]))
        los_projection = math.sin(los_zoa) * math.sin(los_aoa)
        element_separation = element_indices[None, :] - element_indices[:, None]
        los_covariance = np.exp(-1j * math.pi * los_projection * element_separation)
        covariance = covariance + component_powers[0] * los_covariance
    return covariance.astype(np.complex128, copy=False), _ETSI_REFERENCE_SHA256


@pytest.mark.parametrize("model", tuple(_PROFILE_TABLE_SHA256))
def test_four_element_cdl_covariance_reference_is_hermitian_normalized_and_psd(
    model: str,
) -> None:
    covariance, reference_digest = _independent_cdl_spatial_covariance_reference(model)

    assert reference_digest == _ETSI_REFERENCE_SHA256
    assert covariance.shape == (4, 4)
    assert np.isfinite(covariance).all()
    np.testing.assert_allclose(covariance, covariance.conj().T, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.diag(covariance).real, np.ones(4), rtol=0.0, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", tuple(_PROFILE_TABLE_SHA256))
def test_cdl_four_element_spatial_covariance_matches_independent_ray_reference(
    model: str,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("sionna.phy")
    from integrations.sionna.cdl import (  # noqa: PLC0415
        CDLArrayConfig,
        generate_cdl_frequency_response,
    )
    from sionna.phy.channel.tr38901 import models as profile_resources  # noqa: PLC0415

    expected_array, _ = _independent_cdl_spatial_covariance_reference(model)
    expected = torch.as_tensor(expected_array, dtype=torch.complex128)
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    source_digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    assert source_digest == _PROFILE_TABLE_SHA256[model]
    channel = generate_cdl_frequency_response(
        model=model,
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
