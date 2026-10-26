"""Bounded CDL-D LOS-plus-diffuse temporal covariance reference check.

The independent reference uses the byte-pinned ETSI-derived CDL-D v19.2 rows
and the TR 38.901 ray-offset Doppler phase. It checks one native CPU configuration:
single omnidirectional elements, a 3.5-GHz carrier, an x-directed 30-m/s UT,
and a 0.7-ms lag. With 8,192 independent channel draws, both real and
imaginary normalized covariance components must contain the independent
reference in simultaneous two-sided 99% intervals using a paired-observation
normal approximation, with interval half-width at most 0.05. This is a model-consistency check for this configuration; it
does not validate measured LOS channels or establish general CDL-D validity.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterator

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import generate_cdl_frequency_response  # noqa: E402

_SEED = 1985
_BATCH_SIZE = 8192
_CARRIER_FREQUENCY_HZ = 3.5e9
_SPEED_MPS = 30.0
_LAG_S = 0.7e-3
_C0_MPS = 299_792_458.0
_FAMILYWISE_ALPHA = 0.01
_MAX_CI_HALF_WIDTH = 0.05
_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "research"
    / "G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
)
_REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
_ETSI_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
_SIONNA_D_TABLE_SHA256 = "a8960f176347de457c0b90fbc8cf137ac98bb1c5295b096f004d8d5c6e40176"
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


@pytest.fixture(autouse=True)
def _limit_torch_threads() -> Iterator[None]:
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(original)


def _independent_cdl_d_temporal_reference() -> complex:
    """Sum LOS specular and diffuse ray terms from the pinned ETSI rows."""
    reference_bytes = _REFERENCE_PATH.read_bytes()
    assert hashlib.sha256(reference_bytes).hexdigest() == _REFERENCE_SHA256
    document = json.loads(reference_bytes)
    assert document["source_sha256"] == _ETSI_SOURCE_SHA256
    profile = next(item for item in document["profiles"] if item["profile_model"] == "D")
    assert profile["los"] is True
    components = profile["components"]
    specular_components = [item for item in components if item["component"] == "specular"]
    diffuse_components = [item for item in components if item["component"] == "diffuse"]
    assert len(specular_components) == 1
    powers = torch.as_tensor(
        [10.0 ** (float(item["power_db"]) / 10.0) for item in components],
        dtype=torch.float64,
    )
    powers = powers / powers.sum()
    offsets = torch.as_tensor(_RAY_ANGLE_OFFSETS, dtype=torch.float64)
    lag_doppler_phase = 2.0 * math.pi * _SPEED_MPS * _CARRIER_FREQUENCY_HZ / _C0_MPS * _LAG_S

    specular_component = specular_components[0]
    los_aoa = math.radians(float(specular_component["angles_deg"]["aoa"]))
    los_zoa = math.radians(float(specular_component["angles_deg"]["zoa"]))
    los_x_projection = math.sin(los_zoa) * math.cos(los_aoa)
    reference = powers[0].to(torch.complex128) * torch.exp(
        torch.tensor(1j * lag_doppler_phase * los_x_projection, dtype=torch.complex128)
    )

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
    x_projection = torch.sin(zoa)[:, :, None] * torch.cos(aoa)[:, None, :]
    diffuse_phase = torch.exp(1j * lag_doppler_phase * x_projection).mean(dim=(1, 2))
    reference += (powers[1:].to(torch.complex128) * diffuse_phase).sum()
    return complex(reference.item())


@pytest.mark.native_sionna
def test_cdl_d_los_diffuse_temporal_covariance_matches_independent_ray_reference() -> None:
    channel = generate_cdl_frequency_response(
        model="D",
        seed=_SEED,
        carrier_frequency_hz=_CARRIER_FREQUENCY_HZ,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_BATCH_SIZE,
        delay_spread_s=100e-9,
        num_time_steps=2,
        sampling_frequency_hz=1.0 / _LAG_S,
        ut_velocity_mps=(_SPEED_MPS, 0.0, 0.0),
        spec_version="19.2",
        device="cpu",
    )

    # Sum the profile's delay clusters into a narrowband scalar channel while
    # retaining the paired times within each independently generated draw.
    path = channel.h_path[:, 0, 0, 0, 0, :, :]
    h0 = path[:, :, 0].sum(dim=1).to(torch.complex128)
    h1 = path[:, :, 1].sum(dim=1).to(torch.complex128)
    cross_product = h0.conj() * h1
    reference_power = h0.abs().square()
    mean_power = reference_power.mean()
    estimate = cross_product.mean() / mean_power

    # The influence estimate respects paired two-time observations. Bonferroni
    # adjustment covers the real and imaginary covariance components jointly.
    influence = cross_product - estimate * reference_power
    standard_error = torch.stack(
        (
            influence.real.std(unbiased=True),
            influence.imag.std(unbiased=True),
        )
    ) / (mean_power * math.sqrt(_BATCH_SIZE))
    z_simultaneous = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / 4.0)
    half_width = z_simultaneous * standard_error
    expected = _independent_cdl_d_temporal_reference()
    error = torch.stack(
        (
            torch.abs(estimate.real - expected.real),
            torch.abs(estimate.imag - expected.imag),
        )
    )

    assert bool((error <= half_width).all())
    assert float(half_width.max()) <= _MAX_CI_HALF_WIDTH
    assert channel.metadata["profile_los_component"] is True
    assert channel.metadata["parameter_table_sha256"] == _SIONNA_D_TABLE_SHA256
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
