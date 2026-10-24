"""Bounded CDL delay/angular moment checks against independent ray references."""

from __future__ import annotations

import hashlib
import math
from statistics import NormalDist
from typing import Any, Iterator

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import generate_cdl_frequency_response  # noqa: E402
from sionna.phy.channel.tr38901 import models as profile_resources  # noqa: E402

_SEED = 90317
_BATCH_SIZE = 4096
_CARRIER_FREQUENCY_HZ = 3.5e9
_DELAY_SPREAD_S = 100e-9
_FAMILYWISE_ALPHA = 0.01
_MAX_CI_HALF_WIDTH = 0.05
_MAX_ABSOLUTE_REFERENCE_ERROR = 0.05
_NLOS_PROFILES = ("A", "B", "C")
_FAMILY_CLAIM_COUNT = 20
_ANGLE_SPREAD_FIELDS = {
    "aoa": "cASA",
    "aod": "cASD",
    "zoa": "cZSA",
    "zod": "cZSD",
}
_CDL_TABLE_SHA256 = {
    "A": "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7",
    "B": "7c87c092cb2bb44891d301ee3c1991cdda1c6b64094a03f18746ec5fa14ee2a2",
    "C": "f2044ecc3b72a1967221805ba70b1f0dab9634bb831478c1337b737ced6e4a28",
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


@pytest.fixture(autouse=True)
def _limit_torch_threads() -> Iterator[None]:
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(original)


def _ray_reference_and_cluster_scores(model: str, angle_name: str) -> tuple[float, Any]:
    """Compute one delay/angular correlation directly from the profile table."""
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    table_bytes = table_path.read_bytes()
    assert hashlib.sha256(table_bytes).hexdigest() == _CDL_TABLE_SHA256[model]
    parameters = profile_resources.load_json(table_path)
    assert not bool(parameters["los"])

    powers = torch.as_tensor(
        [10.0 ** (float(power_db) / 10.0) for power_db in parameters["powers"]],
        dtype=torch.float64,
    )
    powers = powers / powers.sum()
    delays = _DELAY_SPREAD_S * torch.as_tensor(parameters["delays"], dtype=torch.float64)
    offsets = torch.as_tensor(_RAY_ANGLE_OFFSETS, dtype=torch.float64)
    angle_rays = torch.deg2rad(
        torch.as_tensor(parameters[angle_name], dtype=torch.float64)[:, None]
        + float(parameters[_ANGLE_SPREAD_FIELDS[angle_name]]) * offsets
    )
    # Independent ray pairing gives 20 equal-probability rays per
    # cluster. Delay is common to every ray in that cluster.
    ray_cos_angle = torch.cos(angle_rays)
    ray_weights = powers[:, None] / ray_cos_angle.shape[1]
    mean_delay = (ray_weights * delays[:, None]).sum()
    mean_cos_angle = (ray_weights * ray_cos_angle).sum()
    cluster_cos_angle = ray_cos_angle.mean(dim=1)
    delay_variance = (powers * (delays - mean_delay).square()).sum()
    angular_variance = (powers * (cluster_cos_angle - mean_cos_angle).square()).sum()
    delay_scale = torch.sqrt(delay_variance)
    angular_scale = torch.sqrt(angular_variance)

    cluster_scores = (
        (delays - mean_delay) / delay_scale * (cluster_cos_angle - mean_cos_angle) / angular_scale
    )
    # The angular feature is ray-derived, then averaged within each profile
    # cluster so that its resolution matches the cluster-power observable.
    reference = float((powers * cluster_scores).sum())
    return reference, cluster_scores


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _NLOS_PROFILES)
@pytest.mark.parametrize("angle_name", tuple(_ANGLE_SPREAD_FIELDS))
def test_cdl_nlos_delay_angular_moment_matches_independent_ray_reference(
    model: str, angle_name: str
) -> None:
    reference, cluster_scores = _ray_reference_and_cluster_scores(model, angle_name)
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

    # Each independent draw contributes a cluster-power-weighted standardized
    # delay/angle score. The expected values come directly from the profile
    # table rather than adapter metadata.
    cluster_power = channel.h_path[:, 0, 0, 0, 0, :, 0].abs().square().to(torch.float64)
    assert cluster_power.shape == (_BATCH_SIZE, cluster_scores.numel())
    observations = (cluster_power * cluster_scores).sum(dim=1)
    estimate = observations.mean()
    standard_error = observations.std(unbiased=True) / math.sqrt(_BATCH_SIZE)
    z_99 = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * _FAMILY_CLAIM_COUNT))
    half_width = z_99 * standard_error

    assert abs(float(estimate) - reference) <= _MAX_ABSOLUTE_REFERENCE_ERROR
    assert float(half_width) <= _MAX_CI_HALF_WIDTH
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["profile_model"] == model
    assert channel.metadata["parameter_table_sha256"] == _CDL_TABLE_SHA256[model]
    assert channel.metadata["profile_los_component"] is False
