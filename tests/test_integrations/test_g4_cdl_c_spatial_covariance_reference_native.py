"""Oriented CDL-A/B/C covariance checks against independent TR 38.901 rays.

At 3.5 GHz, a two-element vertical omnidirectional UT ULA is checked under
three frozen three-axis orientations for NLOS CDL-A/B/C v19.2. The independent
reference uses byte-pinned ETSI component rows, Table 7.5-3 ray offsets,
independent azimuth/zenith coupling, the composite array-axis rotation, and
the single-V/XPR field law. One simultaneous two-sided Bonferroni 99% family
covers all real covariance entries across all nine cases; each interval must
contain its reference and have half-width at most 0.05 normalized power.
This verifies these named profiles/orientations and does not establish all
possible orientations, antenna patterns, or measured-channel validity.
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
_SEED = 1322
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
_MODELS = tuple(_PROFILE_SHA256)
_ORIENTATIONS = {
    "zero": (0.0, 0.0, 0.0),
    "tilted_xyz_positive": (0.2, -0.1, 0.3),
    "tilted_xyz_mixed": (-0.7, 0.35, 1.1),
}
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


def _independent_cdl_spatial_covariance_reference(
    model: str, orientation_rad: tuple[float, float, float]
) -> tuple[Any, str]:
    """Evaluate an NLOS 2x2 covariance from pinned ETSI CDL component rows."""
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    table_bytes = table_path.read_bytes()
    digest = hashlib.sha256(table_bytes).hexdigest()
    assert digest == _PROFILE_SHA256[model]
    reference_bytes = _ETSI_REFERENCE_PATH.read_bytes()
    assert hashlib.sha256(reference_bytes).hexdigest() == _ETSI_REFERENCE_SHA256
    document = json.loads(reference_bytes)
    assert document["source_sha256"] == _ETSI_SOURCE_SHA256
    profile = next(item for item in document["profiles"] if item["profile_model"] == model)
    assert profile["los"] is False
    components = profile["components"]
    assert all(item["component"] == "diffuse" for item in components)
    assert len(components) == profile["cluster_count"]

    powers = torch.as_tensor(
        [10.0 ** (float(item["power_db"]) / 10.0) for item in components],
        dtype=torch.float64,
    )
    powers = powers / powers.sum()
    offsets = torch.as_tensor(_RAY_ANGLE_OFFSETS, dtype=torch.float64)
    aoa = torch.deg2rad(
        torch.as_tensor(
            [item["angles_deg"]["aoa"] for item in components], dtype=torch.float64
        )[:, None]
        + float(profile["cluster_spreads_deg"]["asa"]) * offsets
    )
    zoa = torch.deg2rad(
        torch.as_tensor(
            [item["angles_deg"]["zoa"] for item in components], dtype=torch.float64
        )[:, None]
        + float(profile["cluster_spreads_deg"]["zsa"]) * offsets
    )

    alpha, beta, gamma = orientation_rad
    # TR 38.901 Eq. 7.1-15 rotates the single-V element field into GCS.
    delta_azimuth = aoa - alpha
    real = math.sin(gamma) * torch.cos(zoa)[:, :, None] * torch.sin(delta_azimuth)[:, None, :]
    real += math.cos(gamma) * (
        math.cos(beta) * torch.sin(zoa)[:, :, None]
        - math.sin(beta) * torch.cos(zoa)[:, :, None] * torch.cos(delta_azimuth)[:, None, :]
    )
    imag = math.sin(gamma) * torch.cos(delta_azimuth)[:, None, :]
    imag += math.sin(beta) * math.cos(gamma) * torch.sin(delta_azimuth)[:, None, :]
    psi = torch.atan2(imag, real)
    xpr_linear = 10.0 ** (float(profile["xpr_db"]) / 10.0)
    field_power = torch.cos(psi).square() + torch.sin(psi).square() / xpr_linear

    # Apply the TR 38.901 composite rotation to the local-y array axis.
    local_y_in_global = torch.tensor(
        (
            math.cos(alpha) * math.sin(beta) * math.sin(gamma) - math.sin(alpha) * math.cos(gamma),
            math.sin(alpha) * math.sin(beta) * math.sin(gamma) + math.cos(alpha) * math.cos(gamma),
            math.cos(beta) * math.sin(gamma),
        ),
        dtype=torch.float64,
    )
    sin_zoa = torch.sin(zoa)[:, :, None]
    direction_projection = (
        sin_zoa * torch.cos(aoa)[:, None, :] * local_y_in_global[0]
        + sin_zoa * torch.sin(aoa)[:, None, :] * local_y_in_global[1]
        + torch.cos(zoa)[:, :, None] * local_y_in_global[2]
    )
    spatial_phase = torch.exp(-1j * math.pi * direction_projection)
    diagonal = (powers * field_power.mean(dim=(1, 2))).sum()
    off_diagonal = (
        powers.to(torch.complex128) * (field_power * spatial_phase).mean(dim=(1, 2))
    ).sum()
    covariance = torch.stack(
        (
            torch.stack((diagonal, off_diagonal)),
            torch.stack((off_diagonal.conj(), diagonal)),
        )
    )
    return covariance, digest


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("orientation_case", tuple(_ORIENTATIONS))
def test_cdl_nlos_spatial_covariance_matches_independent_38901_ray_reference(
    model: str, orientation_case: str
) -> None:
    orientation_rad = _ORIENTATIONS[orientation_case]
    expected, source_digest = _independent_cdl_spatial_covariance_reference(model, orientation_rad)
    channel = generate_cdl_frequency_response(
        model=model,
        seed=_SEED
        + _MODELS.index(model) * len(_ORIENTATIONS)
        + tuple(_ORIENTATIONS).index(orientation_case),
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_BATCH_SIZE,
        delay_spread_s=100e-9,
        ut_array=CDLArrayConfig(
            num_cols_per_panel=2,
            element_horizontal_spacing=0.5,
        ),
        ut_orientation_rad=orientation_rad,
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

    # One Bonferroni family covers four components for every profile/orientation case.
    z_simultaneous = NormalDist().inv_cdf(
        1.0
        - _FAMILYWISE_ALPHA
        / (2.0 * expected_components.numel() * len(_ORIENTATIONS) * len(_MODELS))
    )
    half_width = z_simultaneous * standard_error
    assert bool((torch.abs(estimate - expected_components) <= half_width).all())
    assert float(half_width.max()) <= _MAX_INTERVAL_HALF_WIDTH
    assert channel.metadata["profile_model"] == model
    assert channel.metadata["profile_los_component"] is False
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["parameter_table_sha256"] == source_digest
