"""Rotated LOS CDL-D/E dual-polarized array covariance reference family."""

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

_BATCH_SIZE = 16_384
_SEED_START = 1_720_000
_ALPHA = 0.01
_MAX_HALF_WIDTH = 0.05
_PROFILE_SHA256 = {
    "D": "a8960f176347de457c0cb90fbc8cf137ac98bb1c5295b096f004d8d5c6e40176",
    "E": "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480",
}
_ROTATED_UT_ORIENTATIONS_RAD = (
    (0.37, 0.28, -0.41),
    (0.0, 0.0, 0.65),
    (0.28, -0.36, 0.0),
)
_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
)
_SCALAR_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/research/G4_CDL_PROFILE_SCALAR_REFERENCE_2026-10-26.json"
)
_REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
_SCALAR_SHA256 = "3b5920e7c12b0e36c8e09d1c7f14025cfbb23b6d805e1044124bfcf147185620"
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
_PORTS = 4  # V-element0, V-element1, H-element0, H-element1


def _reference(model: str, orientation_rad: tuple[float, float, float]) -> tuple[Any, str]:
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    table_digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    assert table_digest == _PROFILE_SHA256[model]

    reference_bytes = _REFERENCE_PATH.read_bytes()
    scalar_bytes = _SCALAR_PATH.read_bytes()
    assert hashlib.sha256(reference_bytes).hexdigest() == _REFERENCE_SHA256
    assert hashlib.sha256(scalar_bytes).hexdigest() == _SCALAR_SHA256
    reference = json.loads(reference_bytes)
    scalars = json.loads(scalar_bytes)
    assert reference["source_sha256"] == _SOURCE_SHA256
    assert scalars["source_sha256"] == _SOURCE_SHA256
    profile = next(row for row in reference["profiles"] if row["profile_model"] == model)
    scalar = next(row for row in scalars["profiles"] if row["profile_model"] == model)
    assert profile["los"] is True and scalar["los"] is True

    components = profile["components"]
    specular = next(row for row in components if row["component"] == "specular")
    diffuse = [row for row in components if row["component"] == "diffuse"]
    assert len(diffuse) == profile["cluster_count"]
    k_factor = 10.0 ** (float(scalar["los_k_factor_db"]) / 10.0)
    xpr = 10.0 ** (float(profile["xpr_db"]) / 10.0)
    spread = profile["cluster_spreads_deg"]
    alpha, beta, gamma = orientation_rad
    offsets = torch.tensor(_RAY_OFFSETS, dtype=torch.float64)
    powers = torch.tensor(
        [10.0 ** (float(row["power_db"]) / 10.0) for row in diffuse],
        dtype=torch.float64,
    )
    powers /= powers.sum()
    aoa = torch.deg2rad(
        torch.tensor(
            [float(row["angles_deg"]["aoa"]) for row in diffuse], dtype=torch.float64
        )[:, None, None]
        + float(spread["asa"]) * offsets[None, None, :]
    )
    zoa = torch.deg2rad(
        torch.tensor(
            [float(row["angles_deg"]["zoa"]) for row in diffuse], dtype=torch.float64
        )[:, None, None]
        + float(spread["zsa"]) * offsets[None, :, None]
    )
    delta_azimuth = aoa - alpha
    real = math.sin(gamma) * torch.cos(zoa) * torch.sin(delta_azimuth)
    real += math.cos(gamma) * (
        math.cos(beta) * torch.sin(zoa)
        - math.sin(beta) * torch.cos(zoa) * torch.cos(delta_azimuth)
    )
    imag = math.sin(gamma) * torch.cos(delta_azimuth)
    imag += math.sin(beta) * math.cos(gamma) * torch.sin(delta_azimuth)
    psi = torch.atan2(imag, real)
    v_power = torch.cos(psi).square() + torch.sin(psi).square() / xpr
    h_power = torch.sin(psi).square() + torch.cos(psi).square() / xpr

    local_y = torch.tensor(
        (
            math.cos(alpha) * math.sin(beta) * math.sin(gamma)
            - math.sin(alpha) * math.cos(gamma),
            math.sin(alpha) * math.sin(beta) * math.sin(gamma)
            + math.cos(alpha) * math.cos(gamma),
            math.cos(beta) * math.sin(gamma),
        ),
        dtype=torch.float64,
    )
    direction_projection = (
        torch.sin(zoa) * torch.cos(aoa) * local_y[0]
        + torch.sin(zoa) * torch.sin(aoa) * local_y[1]
        + torch.cos(zoa) * local_y[2]
    )
    spatial_phase = torch.exp(-1j * math.pi * direction_projection)
    covariance = torch.zeros((_PORTS, _PORTS), dtype=torch.complex128)
    for weight, v_ray_power, h_ray_power, ray_phase in zip(
        powers, v_power, h_power, spatial_phase, strict=True
    ):
        covariance[0, 0] += weight * v_ray_power.mean()
        covariance[1, 1] += weight * v_ray_power.mean()
        covariance[0, 1] += weight * (v_ray_power * ray_phase).mean()
        covariance[1, 0] += weight * (v_ray_power * ray_phase).mean().conj()
        covariance[2, 2] += weight * h_ray_power.mean()
        covariance[3, 3] += weight * h_ray_power.mean()
        covariance[2, 3] += weight * (h_ray_power * ray_phase).mean()
        covariance[3, 2] += weight * (h_ray_power * ray_phase).mean().conj()

    los_angles = specular["angles_deg"]
    los_aoa = math.radians(float(los_angles["aoa"]))
    los_zoa = math.radians(float(los_angles["zoa"]))
    los_delta = los_aoa - alpha
    los_real = math.sin(gamma) * math.cos(los_zoa) * math.sin(los_delta)
    los_real += math.cos(gamma) * (
        math.cos(beta) * math.sin(los_zoa)
        - math.sin(beta) * math.cos(los_zoa) * math.cos(los_delta)
    )
    los_imag = math.sin(gamma) * math.cos(los_delta)
    los_imag += math.sin(beta) * math.cos(gamma) * math.sin(los_delta)
    los_psi = math.atan2(los_imag, los_real)
    los_projection = (
        math.sin(los_zoa) * math.cos(los_aoa) * float(local_y[0])
        + math.sin(los_zoa) * math.sin(los_aoa) * float(local_y[1])
        + math.cos(los_zoa) * float(local_y[2])
    )
    steering = torch.exp(1j * math.pi * los_projection * torch.arange(2, dtype=torch.float64))
    los_vector = torch.cat(
        (math.cos(los_psi) * steering, math.sin(los_psi) * steering)
    ).to(torch.complex128)
    covariance /= k_factor + 1.0
    covariance += (k_factor / (k_factor + 1.0)) * (
        los_vector[:, None] * los_vector[None, :].conj()
    )
    return covariance, table_digest


def _hermitian_components(matrix: Any) -> Any:
    components = [matrix[index, index].real for index in range(_PORTS)]
    for row in range(_PORTS):
        for column in range(row + 1, _PORTS):
            components.extend((matrix[row, column].real, matrix[row, column].imag))
    return torch.stack(components).to(torch.float64)


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("orientation_index", "orientation_rad"),
    tuple(enumerate(_ROTATED_UT_ORIENTATIONS_RAD)),
)
@pytest.mark.parametrize("model", tuple(_PROFILE_SHA256))
def test_rotated_los_cdl_de_dual_polarized_array_covariance_matches_reference(
    model: str, orientation_index: int, orientation_rad: tuple[float, float, float]
) -> None:
    expected, table_digest = _reference(model, orientation_rad)
    profile_index = tuple(_PROFILE_SHA256).index(model)
    channel = generate_cdl_frequency_response(
        model=model,
        seed=_SEED_START + orientation_index * len(_PROFILE_SHA256) + profile_index,
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
        bs_array=CDLArrayConfig(polarization="single", polarization_type="V"),
        ut_orientation_rad=orientation_rad,
        bs_orientation_rad=(0.0, 0.0, 0.0),
        spec_version="19.2",
        device="cpu",
    )
    samples = channel.h_path[:, 0, :, 0, 0, :, 0].sum(dim=-1).to(torch.complex128)
    assert tuple(samples.shape) == (_BATCH_SIZE, _PORTS)
    outer = samples[:, :, None] * samples[:, None, :].conj()
    observations = torch.stack(
        [outer[:, index, index].real for index in range(_PORTS)]
        + [
            value
            for row in range(_PORTS)
            for column in range(row + 1, _PORTS)
            for value in (outer[:, row, column].real, outer[:, row, column].imag)
        ],
        dim=1,
    )
    estimate = observations.mean(dim=0)
    expected_components = _hermitian_components(expected)
    standard_error = observations.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)
    family_size = (
        len(_PROFILE_SHA256)
        * len(_ROTATED_UT_ORIENTATIONS_RAD)
        * expected_components.numel()
    )
    z_simultaneous = NormalDist().inv_cdf(1.0 - _ALPHA / (2.0 * family_size))
    half_width = z_simultaneous * standard_error

    assert bool((torch.abs(estimate - expected_components) <= half_width).all())
    assert float(half_width.max()) <= _MAX_HALF_WIDTH
    assert channel.metadata["profile_model"] == model
    assert channel.metadata["profile_los_component"] is True
    assert channel.metadata["parameter_table_sha256"] == table_digest
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
