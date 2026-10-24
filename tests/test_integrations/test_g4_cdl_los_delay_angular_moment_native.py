"""Bounded CDL-D/E delay/AoA moment references with a composed LOS tap.

The statistic is the PDP-weighted correlation between normalized excess
delay and cluster-mean cos(AoA). For LOS profiles, the zero-delay sample
combines a specular component with the first diffuse cluster. Its angular
feature is defined by the power-weighted mixture of the specular AoA and the
20 diffuse ray AoAs; this is an ensemble-power reference, not a claim that
the simulator exposes geometric ray truth after coherent tap composition.
"""

from __future__ import annotations

import hashlib
import math
from statistics import NormalDist
from typing import Any, Iterator

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import generate_cdl_frequency_response  # noqa: E402
from sionna.phy.channel.tr38901 import models as profile_resources  # noqa: E402

_SEED = 241107
_BATCH_SIZE = 8192
_CARRIER_FREQUENCY_HZ = 3.5e9
_DELAY_SPREAD_S = 100e-9
_FAMILYWISE_ALPHA = 0.01
_FAMILY_PROFILE_COUNT = 5
_MAX_CI_HALF_WIDTH = 0.05
_MAX_ABSOLUTE_REFERENCE_ERROR = 0.05
_LOS_TABLE_SHA256 = {
    "D": "a8960f176347de457c0cb90fbc8cf137ac98bb1c5295b096f004d8d5c6e40176",
    "E": "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480",
}
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


@pytest.fixture(autouse=True)
def _limit_torch_threads() -> Iterator[None]:
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(original)


def _los_ray_reference(model: str) -> tuple[float, np.ndarray]:
    """Compute the composed-tap angular moment from the pinned CDL table."""
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    table_bytes = table_path.read_bytes()
    expected_hash = _LOS_TABLE_SHA256[model]
    assert expected_hash
    assert hashlib.sha256(table_bytes).hexdigest() == expected_hash
    parameters: dict[str, Any] = profile_resources.load_json(table_path)
    assert bool(parameters["los"])

    raw_powers = np.power(10.0, np.asarray(parameters["powers"], dtype=np.float64) / 10.0)
    k_factor = raw_powers[0] / raw_powers[1:].sum()
    diffuse_powers = raw_powers[1:] / raw_powers[1:].sum() / (k_factor + 1.0)
    specular_power = k_factor / (k_factor + 1.0)

    delays = _DELAY_SPREAD_S * np.asarray(parameters["delays"][1:], dtype=np.float64)
    aoa_cluster_mean = np.asarray(parameters["aoa"][1:], dtype=np.float64)
    ray_aoa = np.deg2rad(
        aoa_cluster_mean[:, None] + float(parameters["cASA"]) * _RAY_ANGLE_OFFSETS[None, :]
    )
    cluster_cos_aoa = np.cos(ray_aoa).mean(axis=1)
    los_cos_aoa = math.cos(math.radians(float(parameters["aoa"][0])))

    cluster_powers = np.array(diffuse_powers, copy=True)
    cluster_powers[0] += specular_power
    cluster_cos_aoa[0] = (
        specular_power * los_cos_aoa + diffuse_powers[0] * cluster_cos_aoa[0]
    ) / cluster_powers[0]

    order = np.argsort(delays, kind="stable")
    delays = delays[order]
    cluster_powers = cluster_powers[order]
    cluster_cos_aoa = cluster_cos_aoa[order]

    mean_delay = float(np.dot(cluster_powers, delays))
    mean_cos_aoa = float(np.dot(cluster_powers, cluster_cos_aoa))
    delay_variance = float(np.dot(cluster_powers, np.square(delays - mean_delay)))
    angular_variance = float(np.dot(cluster_powers, np.square(cluster_cos_aoa - mean_cos_aoa)))
    scores = (
        (delays - mean_delay)
        / math.sqrt(delay_variance)
        * (cluster_cos_aoa - mean_cos_aoa)
        / math.sqrt(angular_variance)
    )
    reference = float(np.dot(cluster_powers, scores))
    return reference, scores


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", ["D", "E"])
def test_cdl_los_delay_angular_moment_matches_composed_ray_reference(model: str) -> None:
    reference, cluster_scores = _los_ray_reference(model)
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
    z_99 = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * _FAMILY_PROFILE_COUNT))
    half_width = z_99 * standard_error

    assert abs(float(estimate) - reference) <= _MAX_ABSOLUTE_REFERENCE_ERROR
    assert float(half_width) <= _MAX_CI_HALF_WIDTH
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["profile_model"] == model
    assert channel.metadata["parameter_table_sha256"] == _LOS_TABLE_SHA256[model]
    assert channel.metadata["profile_los_component"] is True
