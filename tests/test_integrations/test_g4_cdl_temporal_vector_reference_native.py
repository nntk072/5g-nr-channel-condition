"""ETSI-referenced CDL temporal-covariance checks across 3D velocity axes."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pytest
from evaluation.g4_seed_roster import cdl_temporal_velocity_seed_roster

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import generate_cdl_frequency_response  # noqa: E402
from sionna.phy.channel.tr38901 import models as profile_resources  # noqa: E402

_PROFILE_SHA256 = {
    "A": "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7",
    "B": "7c87c092cb2bb44891d301ee3c1991cdda1c6b64094a03f18746ec5fa14ee2a2",
    "C": "f2044ecc3b72a1967221805ba70b1f0dab9634bb831478c1337b737ced6e4a28",
    "D": "a8960f176347de457c0cb90fbc8cf137ac98bb1c5295b096f004d8d5c6e40176",
    "E": "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480",
}
_CDL_ETSI_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2] / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
)
_CDL_ETSI_REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
_ETSI_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
_MODELS = tuple("ABCDE")
_VELOCITY_AXES = {
    "x": (30.0, 0.0, 0.0),
    "y": (0.0, 30.0, 0.0),
    "z": (0.0, 0.0, 30.0),
    "oblique_122": (10.0, 20.0, 20.0),
}
_CARRIER_FREQUENCY_HZ = 3.5e9
_LAG_S = 0.7e-3
_SPEED_MPS = 30.0
_GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0
_ICOSAHEDRAL_DIRECTION_VECTORS = (
    (0.0, 1.0, _GOLDEN_RATIO),
    (0.0, 1.0, -_GOLDEN_RATIO),
    (0.0, -1.0, _GOLDEN_RATIO),
    (0.0, -1.0, -_GOLDEN_RATIO),
    (1.0, _GOLDEN_RATIO, 0.0),
    (1.0, -_GOLDEN_RATIO, 0.0),
    (-1.0, _GOLDEN_RATIO, 0.0),
    (-1.0, -_GOLDEN_RATIO, 0.0),
    (_GOLDEN_RATIO, 0.0, 1.0),
    (_GOLDEN_RATIO, 0.0, -1.0),
    (-_GOLDEN_RATIO, 0.0, 1.0),
    (-_GOLDEN_RATIO, 0.0, -1.0),
)
_ICOSAHEDRAL_NORM = math.sqrt(1.0 + _GOLDEN_RATIO**2)
_ICOSAHEDRAL_VELOCITIES: dict[str, tuple[float, float, float]] = {
    f"icosahedral_{index:02d}": (
        _SPEED_MPS * direction[0] / _ICOSAHEDRAL_NORM,
        _SPEED_MPS * direction[1] / _ICOSAHEDRAL_NORM,
        _SPEED_MPS * direction[2] / _ICOSAHEDRAL_NORM,
    )
    for index, direction in enumerate(_ICOSAHEDRAL_DIRECTION_VECTORS)
}
_VELOCITY_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    **_VELOCITY_AXES,
    **_ICOSAHEDRAL_VELOCITIES,
}
_SAMPLE_COUNT = 8192
_FAMILYWISE_ALPHA = 0.01
_CLAIM_COUNT = len(_MODELS) * len(_VELOCITY_DIRECTIONS) * 2
_MAXIMUM_HALF_WIDTH = 0.05
_SEED_ROSTER = cdl_temporal_velocity_seed_roster(_MODELS, tuple(_VELOCITY_DIRECTIONS))
_C0_MPS = 299_792_458.0
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


@lru_cache(maxsize=1)
def _load_etsi_cdl_reference() -> dict[str, Any]:
    """Load the one exact ETSI-derived table snapshot used by this family."""
    reference_bytes = _CDL_ETSI_REFERENCE_PATH.read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != _CDL_ETSI_REFERENCE_SHA256:
        raise ValueError("CDL temporal reference bytes do not match their pinned digest")
    reference = json.loads(reference_bytes)
    if (
        not isinstance(reference, dict)
        or reference.get("schema") != "loslab.g4_cdl_tap_angle_reference.v1"
        or reference.get("source_id") != "tr-138901-v19.2.0"
        or reference.get("source_sha256") != _ETSI_SOURCE_SHA256
        or not isinstance(reference.get("profiles"), list)
    ):
        raise ValueError("CDL temporal reference has an invalid pinned source contract")
    return reference


def _independent_temporal_covariance(
    model: str, velocity_mps: tuple[float, float, float]
) -> tuple[complex, str]:
    """Evaluate the 38.901 per-ray Doppler sum from ETSI-derived CDL rows."""
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    table_bytes = table_path.read_bytes()
    digest = hashlib.sha256(table_bytes).hexdigest()
    assert digest == _PROFILE_SHA256[model]
    reference = _load_etsi_cdl_reference()
    profile = next(row for row in reference["profiles"] if row["profile_model"] == model)
    components = profile["components"]
    parameters = {
        "los": int(profile["los"]),
        "powers": [row["power_db"] for row in components],
        "aoa": [row["angles_deg"]["aoa"] for row in components],
        "zoa": [row["angles_deg"]["zoa"] for row in components],
        "cASA": profile["cluster_spreads_deg"]["asa"],
        "cZSA": profile["cluster_spreads_deg"]["zsa"],
    }
    powers = torch.as_tensor(
        [10.0 ** (float(value) / 10.0) for value in parameters["powers"]],
        dtype=torch.float64,
    )
    powers /= powers.sum()
    velocity = torch.as_tensor(velocity_mps, dtype=torch.float64)
    phase_scale = 2.0 * math.pi * _CARRIER_FREQUENCY_HZ * _LAG_S / _C0_MPS
    offsets = torch.as_tensor(_RAY_ANGLE_OFFSETS, dtype=torch.float64)
    is_los = bool(parameters["los"])
    diffuse_start = 1 if is_los else 0
    expected = torch.zeros((), dtype=torch.complex128)

    if is_los:
        aoa_los = math.radians(float(parameters["aoa"][0]))
        zoa_los = math.radians(float(parameters["zoa"][0]))
        direction_los = torch.tensor(
            (
                math.sin(zoa_los) * math.cos(aoa_los),
                math.sin(zoa_los) * math.sin(aoa_los),
                math.cos(zoa_los),
            ),
            dtype=torch.float64,
        )
        los_phase = phase_scale * torch.dot(velocity, direction_los)
        expected += powers[0].to(torch.complex128) * torch.exp(1j * los_phase)

    for cluster in range(diffuse_start, len(parameters["powers"])):
        aoa = torch.deg2rad(float(parameters["aoa"][cluster]) + float(parameters["cASA"]) * offsets)
        zoa = torch.deg2rad(float(parameters["zoa"][cluster]) + float(parameters["cZSA"]) * offsets)
        # CDL couples azimuth and zenith ray offsets independently. The
        # Cartesian 20 x 20 mean is the exact expectation over that coupling.
        direction_x = torch.sin(zoa)[:, None] * torch.cos(aoa)[None, :]
        direction_y = torch.sin(zoa)[:, None] * torch.sin(aoa)[None, :]
        direction_z = torch.cos(zoa)[:, None].expand_as(direction_x)
        projection = (
            velocity[0] * direction_x + velocity[1] * direction_y + velocity[2] * direction_z
        )
        cluster_mean = torch.exp(1j * phase_scale * projection).mean()
        expected += powers[cluster].to(torch.complex128) * cluster_mean
    return complex(expected.item()), digest


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("velocity_direction", tuple(_VELOCITY_DIRECTIONS))
def test_cdl_temporal_covariance_matches_independent_reference_for_3d_velocity(
    model: str, velocity_direction: str
) -> None:
    """Check A–E temporal covariance for axis-aligned and spherical UT velocities."""
    velocity = _VELOCITY_DIRECTIONS[velocity_direction]
    expected, source_digest = _independent_temporal_covariance(model, velocity)
    channel = generate_cdl_frequency_response(
        model=model,
        seed=_SEED_ROSTER[(model, velocity_direction)],
        carrier_frequency_hz=_CARRIER_FREQUENCY_HZ,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_SAMPLE_COUNT,
        delay_spread_s=100e-9,
        num_time_steps=2,
        sampling_frequency_hz=1.0 / _LAG_S,
        ut_velocity_mps=velocity,
        spec_version="19.2",
        device="cpu",
    )

    taps = channel.h_path[:, 0, 0, 0, 0, :, :].to(torch.complex128)
    h0 = taps[:, :, 0].sum(dim=1)
    h1 = taps[:, :, 1].sum(dim=1)
    cross_product = h0.conj() * h1
    reference_power = h0.abs().square()
    mean_power = reference_power.mean()
    empirical = cross_product.mean() / mean_power
    influence = cross_product - empirical * reference_power
    standard_errors = torch.stack(
        (influence.real.std(unbiased=True), influence.imag.std(unbiased=True))
    )
    standard_errors /= mean_power * math.sqrt(_SAMPLE_COUNT)
    z_simultaneous = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * _CLAIM_COUNT))
    half_widths = z_simultaneous * standard_errors
    errors = torch.stack(
        (
            torch.abs(empirical.real - expected.real),
            torch.abs(empirical.imag - expected.imag),
        )
    )

    assert tuple(taps.shape) == (
        _SAMPLE_COUNT,
        len(channel.metadata["nominal_cluster_delays_s"]),
        2,
    )
    assert channel.metadata["profile_los_component"] is bool(model in {"D", "E"})
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["parameter_table_sha256"] == source_digest
    assert channel.metadata["ut_velocity_mps"] == list(velocity)
    assert math.isclose(math.sqrt(sum(value * value for value in velocity)), _SPEED_MPS)
    assert bool((errors <= half_widths).all()), (model, velocity_direction)
    assert float(half_widths.max()) <= _MAXIMUM_HALF_WIDTH
