"""Rotated NLOS CDL-A/B/C dual-polarized receive covariance family."""

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
_SEED_START = 1_720_100
_ALPHA = 0.01
_MAX_HALF_WIDTH = 0.05
_PROFILE_SHA256 = {
    "A": "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7",
    "B": "7c87c092cb2bb44891d301ee3c1991cdda1c6b64094a03f18746ec5fa14ee2a2",
    "C": "f2044ecc3b72a1967221805ba70b1f0dab9634bb831478c1337b737ced6e4a28",
}
_MODELS = tuple(_PROFILE_SHA256)
_ORIENTATIONS_RAD = (
    (0.37, 0.28, -0.41),
    (0.0, 0.0, 0.65),
    (0.28, -0.36, 0.0),
)
_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2] / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
)
_REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
_RAY_OFFSETS = (
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


def _reference(model: str, orientation_rad: tuple[float, float, float]) -> tuple[Any, str]:
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    table_digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    assert table_digest == _PROFILE_SHA256[model]
    reference_bytes = _REFERENCE_PATH.read_bytes()
    assert hashlib.sha256(reference_bytes).hexdigest() == _REFERENCE_SHA256
    document = json.loads(reference_bytes)
    assert document["source_sha256"] == _SOURCE_SHA256
    profile = next(row for row in document["profiles"] if row["profile_model"] == model)
    assert profile["los"] is False
    components = profile["components"]
    assert all(row["component"] == "diffuse" for row in components)
    assert len(components) == profile["cluster_count"]

    alpha, beta, gamma = orientation_rad
    offsets = torch.tensor(_RAY_OFFSETS, dtype=torch.float64)
    powers = torch.tensor(
        [10.0 ** (float(row["power_db"]) / 10.0) for row in components],
        dtype=torch.float64,
    )
    powers /= powers.sum()
    spreads = profile["cluster_spreads_deg"]
    aoa = torch.deg2rad(
        torch.tensor(
            [float(row["angles_deg"]["aoa"]) for row in components],
            dtype=torch.float64,
        )[:, None, None]
        + float(spreads["asa"]) * offsets[None, None, :]
    )
    zoa = torch.deg2rad(
        torch.tensor(
            [float(row["angles_deg"]["zoa"]) for row in components],
            dtype=torch.float64,
        )[:, None, None]
        + float(spreads["zsa"]) * offsets[None, :, None]
    )
    delta = aoa - alpha
    psi_real = math.sin(gamma) * torch.cos(zoa) * torch.sin(delta)
    psi_real += math.cos(gamma) * (
        math.cos(beta) * torch.sin(zoa) - math.sin(beta) * torch.cos(zoa) * torch.cos(delta)
    )
    psi_imag = math.sin(gamma) * torch.cos(delta)
    psi_imag += math.sin(beta) * math.cos(gamma) * torch.sin(delta)
    psi = torch.atan2(psi_imag, psi_real)
    xpr = 10.0 ** (float(profile["xpr_db"]) / 10.0)
    v_power = torch.cos(psi).square() + torch.sin(psi).square() / xpr
    h_power = torch.sin(psi).square() + torch.cos(psi).square() / xpr

    local_y = torch.tensor(
        (
            math.cos(alpha) * math.sin(beta) * math.sin(gamma) - math.sin(alpha) * math.cos(gamma),
            math.sin(alpha) * math.sin(beta) * math.sin(gamma) + math.cos(alpha) * math.cos(gamma),
            math.cos(beta) * math.sin(gamma),
        ),
        dtype=torch.float64,
    )
    projection = (
        torch.sin(zoa) * torch.cos(aoa) * local_y[0]
        + torch.sin(zoa) * torch.sin(aoa) * local_y[1]
        + torch.cos(zoa) * local_y[2]
    )
    spatial_phase = torch.exp(-1j * math.pi * projection)
    covariance = torch.zeros((4, 4), dtype=torch.complex128)
    for weight, v_ray, h_ray, phase in zip(powers, v_power, h_power, spatial_phase, strict=True):
        covariance[0, 0] += weight * v_ray.mean()
        covariance[1, 1] += weight * v_ray.mean()
        covariance[0, 1] += weight * (v_ray * phase).mean()
        covariance[1, 0] += weight * (v_ray * phase).mean().conj()
        covariance[2, 2] += weight * h_ray.mean()
        covariance[3, 3] += weight * h_ray.mean()
        covariance[2, 3] += weight * (h_ray * phase).mean()
        covariance[3, 2] += weight * (h_ray * phase).mean().conj()
    return covariance, table_digest


def _observations(samples: Any) -> Any:
    outer = samples[:, :, None] * samples[:, None, :].conj()
    values = []
    for row in range(4):
        values.append(outer[:, row, row].real)
        for column in range(row + 1, 4):
            values.extend((outer[:, row, column].real, outer[:, row, column].imag))
    return torch.stack(values, dim=1)


def _components(matrix: Any) -> Any:
    values = []
    for row in range(4):
        values.append(matrix[row, row].real)
        for column in range(row + 1, 4):
            values.extend((matrix[row, column].real, matrix[row, column].imag))
    return torch.stack(values).to(torch.float64)


@pytest.mark.native_sionna
@pytest.mark.parametrize("model_index,model", tuple(enumerate(_MODELS)))
@pytest.mark.parametrize("orientation_index,orientation", tuple(enumerate(_ORIENTATIONS_RAD)))
def test_rotated_nlos_dual_polarized_covariance_matches_reference(
    model_index: int,
    model: str,
    orientation_index: int,
    orientation: tuple[float, float, float],
) -> None:
    reference, table_digest = _reference(model, orientation)
    seed = _SEED_START + orientation_index * len(_MODELS) + model_index
    channel = generate_cdl_frequency_response(
        model=model,
        seed=seed,
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
            polarization="single", polarization_type="V", antenna_pattern="omni"
        ),
        spec_version="19.2",
        device="cpu",
        ut_orientation_rad=orientation,
    )
    samples = channel.h_path[:, 0, :, 0, 0, :, 0].sum(dim=-1).to(torch.complex128)
    assert tuple(samples.shape) == (_BATCH_SIZE, 4)
    observed = _observations(samples)
    expected = _components(reference)
    estimate = observed.mean(dim=0)
    standard_error = observed.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)
    family_claims = expected.numel() * len(_MODELS) * len(_ORIENTATIONS_RAD)
    z_simultaneous = NormalDist().inv_cdf(1.0 - _ALPHA / (2.0 * family_claims))
    half_width = z_simultaneous * standard_error
    assert bool((torch.abs(estimate - expected) <= half_width).all())
    assert float(half_width.max()) <= _MAX_HALF_WIDTH
    assert channel.metadata["profile_model"] == model
    assert channel.metadata["profile_los_component"] is False
    assert channel.metadata["parameter_table_sha256"] == table_digest
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
