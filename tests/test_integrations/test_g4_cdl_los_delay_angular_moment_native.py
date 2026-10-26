"""Bounded CDL-D/E delay/angular moment references with a composed LOS tap.

The statistic is the PDP-weighted correlation between normalized excess
delay and cluster-mean cosine of AoA, AoD, ZoA, or ZoD. For LOS profiles, the
zero-delay sample combines a specular component with the first diffuse
cluster. Its angular feature is defined by the power-weighted mixture of the
specular angle and the 20 diffuse ray angles; this is an ensemble-power
reference, not geometric ray truth after coherent tap composition.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterator

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import generate_cdl_frequency_response  # noqa: E402

_SEED = 241107
_BATCH_SIZE = 8192
_CARRIER_FREQUENCY_HZ = 3.5e9
_DELAY_SPREAD_S = 100e-9
_FAMILYWISE_ALPHA = 0.01
_FAMILY_CLAIM_COUNT = 20
_MAX_CI_HALF_WIDTH = 0.05
_MAX_ABSOLUTE_REFERENCE_ERROR = 0.05
_LOS_TABLE_SHA256 = {
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
_RAY_ANGLE_OFFSETS = np.asarray(
    [
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
    ],
    dtype=np.float64,
)
_ANGLE_SPREAD_FIELDS = {
    "aoa": "asa",
    "aod": "asd",
    "zoa": "zsa",
    "zod": "zsd",
}


@pytest.fixture(autouse=True)
def _limit_torch_threads() -> Iterator[None]:
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(original)


def _los_ray_reference(model: str, angle_name: str) -> tuple[float, np.ndarray]:
    """Compute a composed-tap angular moment from the pinned ETSI rows."""
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
    assert min(item["normalized_delay"] for item in diffuse_components) == 0.0

    raw_powers = np.power(
        10.0,
        np.asarray([item["power_db"] for item in components], dtype=np.float64) / 10.0,
    )
    specular_power = float(raw_powers[0] / raw_powers.sum())
    diffuse_powers = raw_powers[1:] / raw_powers.sum()
    delays = _DELAY_SPREAD_S * np.asarray(
        [item["normalized_delay"] for item in diffuse_components], dtype=np.float64
    )
    angle_cluster_mean = np.asarray(
        [item["angles_deg"][angle_name] for item in diffuse_components], dtype=np.float64
    )
    spread = float(profile["cluster_spreads_deg"][_ANGLE_SPREAD_FIELDS[angle_name]])
    ray_angles = np.deg2rad(
        angle_cluster_mean[:, None] + spread * _RAY_ANGLE_OFFSETS[None, :]
    )
    cluster_cos_angle = np.cos(ray_angles).mean(axis=1)
    los_angle = float(specular_components[0]["angles_deg"][angle_name])
    los_cos_angle = math.cos(math.radians(los_angle))

    cluster_powers = np.array(diffuse_powers, copy=True)
    cluster_powers[0] += specular_power
    cluster_cos_angle[0] = (
        specular_power * los_cos_angle + diffuse_powers[0] * cluster_cos_angle[0]
    ) / cluster_powers[0]

    order = np.argsort(delays, kind="stable")
    delays = delays[order]
    cluster_powers = cluster_powers[order]
    cluster_cos_angle = cluster_cos_angle[order]

    mean_delay = float(np.dot(cluster_powers, delays))
    mean_cos_angle = float(np.dot(cluster_powers, cluster_cos_angle))
    delay_variance = float(np.dot(cluster_powers, np.square(delays - mean_delay)))
    angular_variance = float(np.dot(cluster_powers, np.square(cluster_cos_angle - mean_cos_angle)))
    scores = (
        (delays - mean_delay)
        / math.sqrt(delay_variance)
        * (cluster_cos_angle - mean_cos_angle)
        / math.sqrt(angular_variance)
    )
    reference = float(np.dot(cluster_powers, scores))
    return reference, scores


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", ["D", "E"])
@pytest.mark.parametrize("angle_name", tuple(_ANGLE_SPREAD_FIELDS))
def test_cdl_los_delay_angular_moment_matches_composed_ray_reference(
    model: str, angle_name: str
) -> None:
    reference, cluster_scores = _los_ray_reference(model, angle_name)
    channel = generate_cdl_frequency_response(
        model=model,
        seed=_SEED + ord(model),
        carrier_frequency_hz=_CARRIER_FREQUENCY_HZ,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_BATCH_SIZE,
        delay_spread_s=_DELAY_SPREAD_S,
        spec_version="19.2",
        device="cpu",
    )

    tap_powers = channel.h_path[:, 0, 0, 0, 0, :, 0].abs().square().to(torch.float64)
    assert tap_powers.shape == (_BATCH_SIZE, cluster_scores.size)
    observations = (tap_powers * torch.as_tensor(cluster_scores)[None, :]).sum(dim=1)
    estimate = observations.mean()
    standard_error = observations.std(unbiased=True) / math.sqrt(_BATCH_SIZE)
    z_99 = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * _FAMILY_CLAIM_COUNT))
    half_width = z_99 * standard_error

    assert abs(float(estimate) - reference) <= _MAX_ABSOLUTE_REFERENCE_ERROR
    assert float(half_width) <= _MAX_CI_HALF_WIDTH
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["profile_model"] == model
    assert channel.metadata["parameter_table_sha256"] == _LOS_TABLE_SHA256[model]
    assert channel.metadata["profile_los_component"] is True


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", ["D", "E"])
def test_cdl_los_complex_mean_and_diffuse_variance_match_k_factor_reference(model: str) -> None:
    """Verify coherent LOS survives complex averaging while diffuse phase averages away."""
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
    specular_power = 10.0 ** (float(specular_components[0]["power_db"]) / 10.0)
    diffuse_powers = np.power(
        10.0,
        np.asarray([item["power_db"] for item in diffuse_components], dtype=np.float64) / 10.0,
    )
    diffuse_total = float(diffuse_powers.sum())
    total_power = specular_power + diffuse_total
    k_factor = float(specular_power / diffuse_total)
    los_power = specular_power / total_power
    diffuse_delays = np.asarray(
        [item["normalized_delay"] for item in diffuse_components], dtype=np.float64
    )
    diffuse_order = np.argsort(diffuse_delays, kind="stable")
    first_diffuse_power = float(diffuse_powers[diffuse_order[0]] / total_power)
    seed = _SEED + 1_000 + ord(model)
    channel = generate_cdl_frequency_response(
        model=model,
        seed=seed,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=8192,
        delay_spread_s=100e-9,
        spec_version="19.2",
        device="cpu",
    )
    los_tap = channel.h_path[:, 0, 0, 0, 0, 0, 0].to(torch.complex128)
    sample_count = los_tap.numel()
    sample_mean = los_tap.mean()
    centered = los_tap - sample_mean

    # Delta-method SE for |complex sample mean|^2, retaining real/imaginary
    # covariance. The diffuse residual-power SE uses independent draws as units.
    covariance = torch.stack((centered.real, centered.imag), dim=0)
    covariance = covariance @ covariance.T / (sample_count - 1)
    gradient = torch.stack((2.0 * sample_mean.real, 2.0 * sample_mean.imag))
    los_mean_power = sample_mean.abs().square()
    los_mean_power_se = torch.sqrt((gradient @ covariance @ gradient) / sample_count)
    diffuse_power_samples = centered.abs().square()
    diffuse_power = diffuse_power_samples.sum() / (sample_count - 1)
    diffuse_power_se = diffuse_power_samples.std(unbiased=True) / math.sqrt(sample_count)
    family_z = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / 8.0)
    # A finite-sample complex mean retains diffuse variance / n in its power.
    finite_sample_los_mean_reference = los_power + first_diffuse_power / sample_count

    assert abs(float(los_mean_power) - finite_sample_los_mean_reference) <= (
        family_z * float(los_mean_power_se) + 1e-6
    )
    assert abs(float(diffuse_power) - first_diffuse_power) <= (
        family_z * float(diffuse_power_se) + 1e-6
    )
    assert channel.metadata["profile_k_factor_linear"] == pytest.approx(k_factor)
    assert channel.metadata["parameter_table_sha256"] == _LOS_TABLE_SHA256[model]
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
