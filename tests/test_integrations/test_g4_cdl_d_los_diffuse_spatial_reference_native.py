"""Bounded CDL-D/E LOS-plus-diffuse spatial covariance ray reference.

This native CPU case checks CDL-D and CDL-E v19.2 at 3.5 GHz for a stationary
downlink and a two-element UT ULA along local y, with vertical-polarized
omnidirectional elements at half-wave spacing and zero orientation. Each of
8,192 independent channel draws yields one paired two-port channel vector;
four unique real covariance components
must contain the pinned ETSI-derived ray-law expectation in simultaneous two-sided
99% normal intervals, each no wider than 0.05 normalized mean power. The
byte-pinned ETSI component rows supply the deterministic LOS steering term and
diffuse cluster means over independent azimuth/zenith ray pairings. This checks
only these profiles and this array configuration and provides no external channel claim.
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
_SEED = 1986
_CARRIER_FREQUENCY_HZ = 3.5e9
_FAMILYWISE_ALPHA = 0.01
_MAX_INTERVAL_HALF_WIDTH = 0.05
_CDL_TABLE_SHA256 = {
    "D": "a8960f176347de457c0cb90fbc8cf137ac98bb1c5295b096f004d8d5c6e40176",
    "E": "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480",
}
_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "research"
    / "G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
)
_REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
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


def _independent_cdl_los_spatial_covariance_reference(model: str) -> tuple[Any, str]:
    """Evaluate LOS-plus-diffuse covariance from pinned ETSI profile rows."""
    if model not in _CDL_TABLE_SHA256:
        raise ValueError(f"unsupported LOS spatial-covariance profile: {model}")
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    table_bytes = table_path.read_bytes()
    digest = hashlib.sha256(table_bytes).hexdigest()
    assert digest == _CDL_TABLE_SHA256[model]
    reference_bytes = _REFERENCE_PATH.read_bytes()
    assert hashlib.sha256(reference_bytes).hexdigest() == _REFERENCE_SHA256
    document = json.loads(reference_bytes)
    assert document["source_sha256"] == _ETSI_SOURCE_SHA256
    profile = next(item for item in document["profiles"] if item["profile_model"] == model)
    assert profile["los"] is True
    components = profile["components"]
    specular_components = [item for item in components if item["component"] == "specular"]
    diffuse_components = [item for item in components if item["component"] == "diffuse"]
    assert len(specular_components) == 1
    assert components[0] is specular_components[0]
    assert components[1:] == diffuse_components
    assert len(diffuse_components) == len(components) - 1

    raw_powers = torch.as_tensor(
        [10.0 ** (float(item["power_db"]) / 10.0) for item in components],
        dtype=torch.float64,
    )
    powers = raw_powers / raw_powers.sum()
    offsets = torch.as_tensor(_RAY_ANGLE_OFFSETS, dtype=torch.float64)

    # The first fixture component is the LOS specular component. For a local-y
    # half-wave ULA, its inter-element phase uses sin(ZoA)*sin(AoA).
    los_aoa = math.radians(float(specular_components[0]["angles_deg"]["aoa"]))
    los_zoa = math.radians(float(specular_components[0]["angles_deg"]["zoa"]))
    los_projection = math.sin(los_zoa) * math.sin(los_aoa)
    off_diagonal = powers[0].to(torch.complex128) * torch.exp(
        torch.tensor(-1j * math.pi * los_projection, dtype=torch.complex128)
    )

    # The remaining fixture components are diffuse clusters. TR 38.901
    # couples the 20 azimuth offsets independently from the 20 zenith
    # offsets, so each cluster expectation averages their Cartesian product.
    aoa = torch.deg2rad(
        torch.as_tensor(
            [item["angles_deg"]["aoa"] for item in diffuse_components], dtype=torch.float64
        )[:, None]
        + float(profile["cluster_spreads_deg"]["asa"]) * offsets
    )
    zoa = torch.deg2rad(
        torch.as_tensor(
            [item["angles_deg"]["zoa"] for item in diffuse_components], dtype=torch.float64
        )[:, None]
        + float(profile["cluster_spreads_deg"]["zsa"]) * offsets
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
@pytest.mark.parametrize("model", ("D", "E"))
def test_cdl_los_spatial_covariance_matches_independent_ray_reference(model: str) -> None:
    expected, source_digest = _independent_cdl_los_spatial_covariance_reference(model)
    channel = generate_cdl_frequency_response(
        model=model,
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
    assert channel.metadata["profile_model"] == model
    assert channel.metadata["profile_los_component"] is True
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["parameter_table_sha256"] == source_digest
