"""Bounded CDL angular checks against byte-pinned ETSI-derived ray references."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterator

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import generate_cdl_frequency_response  # noqa: E402

_SEED = 90317
_BATCH_SIZE = 4096
_CARRIER_FREQUENCY_HZ = 3.5e9
_DELAY_SPREAD_S = 100e-9
_FAMILYWISE_ALPHA = 0.01
_MAX_CI_HALF_WIDTH = 0.05
_MAX_ABSOLUTE_REFERENCE_ERROR = 0.05
_NLOS_PROFILES = ("A", "B", "C")
_LOS_PROFILES = ("D", "E")
_FAMILY_CLAIM_COUNT = 20
_ANGLE_SPREAD_FIELDS = {
    "aoa": "asa",
    "aod": "asd",
    "zoa": "zsa",
    "zod": "zsd",
}
_CDL_TABLE_SHA256 = {
    "A": "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7",
    "B": "7c87c092cb2bb44891d301ee3c1991cdda1c6b64094a03f18746ec5fa14ee2a2",
    "C": "f2044ecc3b72a1967221805ba70b1f0dab9634bb831478c1337b737ced6e4a28",
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


@pytest.fixture(autouse=True)
def _limit_torch_threads() -> Iterator[None]:
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(original)


def _reference_profile(model: str) -> dict[str, Any]:
    """Load one CDL profile from the byte-pinned ETSI-derived fixture."""
    reference_bytes = _REFERENCE_PATH.read_bytes()
    assert hashlib.sha256(reference_bytes).hexdigest() == _REFERENCE_SHA256
    document = json.loads(reference_bytes)
    assert document["source_sha256"] == _ETSI_SOURCE_SHA256
    return next(item for item in document["profiles"] if item["profile_model"] == model)


def _diffuse_components_and_powers(
    model: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    profile = _reference_profile(model)
    diffuse_components = [
        item for item in profile["components"] if item["component"] == "diffuse"
    ]
    assert diffuse_components
    powers = torch.as_tensor(
        [10.0 ** (float(item["power_db"]) / 10.0) for item in diffuse_components],
        dtype=torch.float64,
    )
    powers = powers / powers.sum()
    return profile, diffuse_components, powers


def _ray_reference_and_cluster_scores(model: str, angle_name: str) -> tuple[float, Any]:
    """Compute one delay/angular correlation from pinned ETSI profile rows."""
    profile, diffuse_components, powers = _diffuse_components_and_powers(model)
    assert profile["los"] is False
    delays = _DELAY_SPREAD_S * torch.as_tensor(
        [item["normalized_delay"] for item in diffuse_components], dtype=torch.float64
    )
    offsets = torch.as_tensor(_RAY_ANGLE_OFFSETS, dtype=torch.float64)
    angle_rays = torch.deg2rad(
        torch.as_tensor(
            [item["angles_deg"][angle_name] for item in diffuse_components],
            dtype=torch.float64,
        )[:, None]
        + float(profile["cluster_spreads_deg"][_ANGLE_SPREAD_FIELDS[angle_name]]) * offsets
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


def _native_cdl_rays(model: str) -> Any:
    """Create native Sionna rays with the adapter's pinned SISO-omni setup."""
    from sionna.phy.channel.tr38901 import CDL, PanelArray

    array_config = {
        "num_rows_per_panel": 1,
        "num_cols_per_panel": 1,
        "polarization": "single",
        "polarization_type": "V",
        "antenna_pattern": "omni",
        "num_rows": 1,
        "num_cols": 1,
        "carrier_frequency": _CARRIER_FREQUENCY_HZ,
        "device": "cpu",
    }
    ut_array = PanelArray(**array_config)
    bs_array = PanelArray(**array_config)
    model_instance = CDL(
        model=model,
        delay_spread=_DELAY_SPREAD_S,
        carrier_frequency=_CARRIER_FREQUENCY_HZ,
        ut_array=ut_array,
        bs_array=bs_array,
        direction="downlink",
        ut_orientation=[math.pi, 0.0, 0.0],
        bs_orientation=[0.0, 0.0, 0.0],
        device="cpu",
        spec_version="19.2",
    )
    from integrations.sionna.random import call_with_sionna_seed

    return call_with_sionna_seed(
        "cpu", _SEED + ord(model), lambda: model_instance._create_rays(batch_size=1)
    )


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
    # delay/angle score. The expectation comes from the pinned ETSI-derived
    # fixture rather than the adapter's packaged profile table.
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


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _NLOS_PROFILES)
def test_cdl_nlos_full_ray_angle_distribution_matches_pinned_etsi_rows(model: str) -> None:
    profile, diffuse_components, expected_powers = _diffuse_components_and_powers(model)
    assert profile["los"] is False
    rays = _native_cdl_rays(model)
    actual_powers = rays.powers[0, 0, 0].to(torch.float64)
    assert actual_powers.shape == expected_powers.shape
    assert torch.allclose(actual_powers, expected_powers, atol=1e-7, rtol=1e-6)

    for angle_name in _ANGLE_SPREAD_FIELDS:
        expected_angles = torch.as_tensor(
            [item["angles_deg"][angle_name] for item in diffuse_components], dtype=torch.float64
        )[:, None] + (
            float(profile["cluster_spreads_deg"][_ANGLE_SPREAD_FIELDS[angle_name]])
            * torch.as_tensor(_RAY_ANGLE_OFFSETS, dtype=torch.float64)
        )
        actual_angles = torch.rad2deg(getattr(rays, angle_name)[0, 0, 0]).to(torch.float64)
        assert actual_angles.shape == expected_angles.shape
        assert torch.allclose(
            torch.sort(actual_angles, dim=-1).values,
            torch.sort(expected_angles, dim=-1).values,
            atol=1e-4,
            rtol=0.0,
        )


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _LOS_PROFILES)
def test_cdl_los_diffuse_ray_geometry_matches_etsi_rows_after_separate_los_component(
    model: str,
) -> None:
    """Check the diffuse CDL-D/E rays without treating the specular LOS as a ray cluster."""
    profile, diffuse_components, diffuse_powers = _diffuse_components_and_powers(model)
    assert profile["los"] is True
    rays = _native_cdl_rays(model)

    # _create_rays returns diffuse rows; the coherent LOS component is composed elsewhere.
    actual_powers = rays.powers[0, 0, 0].to(torch.float64)
    assert actual_powers.shape == diffuse_powers.shape
    assert torch.allclose(actual_powers, diffuse_powers, atol=1e-7, rtol=1e-6)

    for angle_name in _ANGLE_SPREAD_FIELDS:
        expected_angles = torch.as_tensor(
            [item["angles_deg"][angle_name] for item in diffuse_components], dtype=torch.float64
        )[:, None] + (
            float(profile["cluster_spreads_deg"][_ANGLE_SPREAD_FIELDS[angle_name]])
            * torch.as_tensor(_RAY_ANGLE_OFFSETS, dtype=torch.float64)
        )
        actual_angles = torch.rad2deg(getattr(rays, angle_name)[0, 0, 0]).to(torch.float64)
        assert actual_angles.shape == expected_angles.shape
        assert torch.allclose(
            torch.sort(actual_angles, dim=-1).values,
            torch.sort(expected_angles, dim=-1).values,
            atol=1e-4,
            rtol=0.0,
        )
