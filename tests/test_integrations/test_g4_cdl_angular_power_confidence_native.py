"""Draw-level CDL angular-power marginals against pinned ETSI references."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterator

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import generate_cdl_frequency_response  # noqa: E402

_BATCH_SIZE = 8192
_SEED = 239_100
_FAMILYWISE_ALPHA = 0.01
_MAX_INTERVAL_HALF_WIDTH = 0.05
_MODELS = ("A", "B", "C", "D", "E")
_ANGLE_SPREAD_FIELDS = {
    "aoa": ("aoa", "asa"),
    "aod": ("aod", "asd"),
    "zoa": ("zoa", "zsa"),
    "zod": ("zod", "zsd"),
}
_ANGLE_DIRECTIONS = tuple(_ANGLE_SPREAD_FIELDS)
_PROFILE_SHA256 = {
    "A": "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7",
    "B": "7c87c092cb2bb44891d301ee3c1991cdda1c6b64094a03f18746ec5fa14ee2a2",
    "C": "f2044ecc3b72a1967221805ba70b1f0dab9634bb831478c1337b737ced6e4a28",
    "D": "a8960f176347de457c0cb90fbc8cf137ac98bb1c5295b096f004d8d5c6e40176",
    "E": "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480",
}
_ROOT = Path(__file__).resolve().parents[2]
_TAP_ANGLE_REFERENCE_PATH = (
    _ROOT / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
)
_PROFILE_SCALAR_REFERENCE_PATH = (
    _ROOT / "docs/research/G4_CDL_PROFILE_SCALAR_REFERENCE_2026-10-26.json"
)
_TAP_ANGLE_REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
_PROFILE_SCALAR_REFERENCE_SHA256 = (
    "3b5920e7c12b0e36c8e09d1c7f14025cfbb23b6d805e1044124bfcf147185620"
)
_ETSI_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
_RAY_ANGLE_OFFSETS = torch.tensor(
    (
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
    ),
    dtype=torch.float64,
)
_COSINE_SECTOR_EDGES = torch.tensor((-0.5, 0.0, 0.5), dtype=torch.float64)
_NUM_SECTORS = 4
_RANK_BINS_PER_ANGLE = 4
_FAMILY_CLAIM_COUNT = len(_MODELS) * len(_ANGLE_SPREAD_FIELDS) * _NUM_SECTORS
_JOINT_FAMILY_CLAIM_COUNT = (
    len(_MODELS) * math.comb(len(_ANGLE_DIRECTIONS), 2) * _RAY_ANGLE_OFFSETS.numel() ** 2
)
_FOUR_ANGLE_FAMILY_CLAIM_COUNT = len(_MODELS) * _RANK_BINS_PER_ANGLE ** len(_ANGLE_DIRECTIONS)
_MAX_FOUR_ANGLE_INTERVAL_HALF_WIDTH = 0.001


@pytest.fixture(autouse=True)
def _limit_torch_threads() -> Iterator[None]:
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(original)


def _independent_profile_references(model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    tap_bytes = _TAP_ANGLE_REFERENCE_PATH.read_bytes()
    scalar_bytes = _PROFILE_SCALAR_REFERENCE_PATH.read_bytes()
    assert hashlib.sha256(tap_bytes).hexdigest() == _TAP_ANGLE_REFERENCE_SHA256
    assert hashlib.sha256(scalar_bytes).hexdigest() == _PROFILE_SCALAR_REFERENCE_SHA256
    tap_reference = json.loads(tap_bytes)
    scalar_reference = json.loads(scalar_bytes)
    assert tap_reference["source_sha256"] == _ETSI_SOURCE_SHA256
    assert scalar_reference["source_sha256"] == _ETSI_SOURCE_SHA256
    tap_profiles = {row["profile_model"]: row for row in tap_reference["profiles"]}
    scalar_profiles = {row["profile_model"]: row for row in scalar_reference["profiles"]}
    assert set(tap_profiles) == set(scalar_profiles) == set(_MODELS)
    assert tap_profiles[model]["los"] == scalar_profiles[model]["los"]
    return tap_profiles[model], scalar_profiles[model]


def _profile_angular_sector_reference(
    profile: dict[str, Any], scalar: dict[str, Any], angle_name: str
) -> tuple[Any, Any]:
    """Return expected angular marginals from the pinned independent ETSI rows."""
    angle_field, spread_field = _ANGLE_SPREAD_FIELDS[angle_name]
    is_los = bool(profile["los"])
    diffuse_components = sorted(
        (
            component
            for component in profile["components"]
            if component["component"] == "diffuse"
        ),
        key=lambda component: float(component["normalized_delay"]),
    )
    diffuse_powers = torch.tensor(
        [10.0 ** (float(component["power_db"]) / 10.0) for component in diffuse_components],
        dtype=torch.float64,
    )
    diffuse_powers /= diffuse_powers.sum()
    if is_los:
        k_factor = 10.0 ** (float(scalar["los_k_factor_db"]) / 10.0)
        expected_tap_powers = diffuse_powers / (k_factor + 1.0)
        ray_angles_deg = torch.tensor(
            [float(component["angles_deg"][angle_field]) for component in diffuse_components],
            dtype=torch.float64,
        )[:, None]
    else:
        k_factor = 0.0
        expected_tap_powers = diffuse_powers
        ray_angles_deg = torch.tensor(
            [float(component["angles_deg"][angle_field]) for component in diffuse_components],
            dtype=torch.float64,
        )[:, None]
    cluster_spread = float(profile["cluster_spreads_deg"][spread_field])
    ray_angles_deg = ray_angles_deg + cluster_spread * _RAY_ANGLE_OFFSETS[None, :]
    ray_cosines = torch.cos(torch.deg2rad(ray_angles_deg))
    sector_indices = torch.bucketize(ray_cosines.contiguous(), _COSINE_SECTOR_EDGES)
    ray_sector_fractions = (
        torch.nn.functional.one_hot(sector_indices, num_classes=_NUM_SECTORS)
        .to(torch.float64)
        .mean(dim=1)
    )
    if is_los:
        specular_power = k_factor / (k_factor + 1.0)
        expected_tap_powers[0] += specular_power
        specular_component = next(
            component
            for component in profile["components"]
            if component["component"] == "specular"
        )
        los_angle = float(specular_component["angles_deg"][angle_field])
        los_cosine = math.cos(math.radians(los_angle))
        los_sector_index = torch.bucketize(torch.tensor(los_cosine), _COSINE_SECTOR_EDGES)
        los_sector_fraction = torch.nn.functional.one_hot(
            los_sector_index, num_classes=_NUM_SECTORS
        ).to(torch.float64)
        first_diffuse_power = expected_tap_powers[0] - specular_power
        ray_sector_fractions[0] = (
            first_diffuse_power * ray_sector_fractions[0] + specular_power * los_sector_fraction
        ) / expected_tap_powers[0]
    sector_reference = expected_tap_powers @ ray_sector_fractions
    assert math.isclose(float(sector_reference.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    return sector_reference, ray_sector_fractions


def _native_profile_rays(model: str) -> Any:
    """Generate the pinned CDL ray-angle coupling draws from the native model."""
    from sionna.phy.channel.tr38901 import CDL, PanelArray

    array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        num_rows=1,
        num_cols=1,
        carrier_frequency=3.5e9,
        device="cpu",
    )
    model_instance = CDL(
        model=model,
        delay_spread=100e-9,
        carrier_frequency=3.5e9,
        ut_array=array,
        bs_array=array,
        direction="downlink",
        device="cpu",
        spec_version="19.2",
    )
    from integrations.sionna.random import call_with_sionna_seed

    return call_with_sionna_seed(
        "cpu", _SEED + ord(model), lambda: model_instance._create_rays(batch_size=_BATCH_SIZE)
    )


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
def test_cdl_angular_sector_power_matches_pinned_profile_confidence(
    model: str,
) -> None:
    """Check angular power in all four directions with simultaneous draw-level intervals."""
    profile, scalar = _independent_profile_references(model)
    is_los = bool(profile["los"])

    channel = generate_cdl_frequency_response(
        model=model,
        seed=_SEED + ord(model),
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_BATCH_SIZE,
        delay_spread_s=100e-9,
        spec_version="19.2",
        device="cpu",
    )
    cluster_power_draws = channel.h_path[:, 0, 0, 0, 0, :, 0].abs().square().to(torch.float64)
    expected_cluster_count = profile["cluster_count"]
    assert cluster_power_draws.shape == (_BATCH_SIZE, expected_cluster_count)
    assert channel.metadata["parameter_table_sha256"] == _PROFILE_SHA256[model]
    z_simultaneous = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * _FAMILY_CLAIM_COUNT))

    for angle_name in _ANGLE_SPREAD_FIELDS:
        etsi_sector_reference, ray_sector_fractions = _profile_angular_sector_reference(
            profile, scalar, angle_name
        )
        assert int(torch.count_nonzero(etsi_sector_reference)) >= 2
        torch.testing.assert_close(
            ray_sector_fractions.sum(dim=1),
            torch.ones(expected_cluster_count, dtype=torch.float64),
        )
        observed_by_draw = cluster_power_draws @ ray_sector_fractions
        estimate = observed_by_draw.mean(dim=0)
        standard_error = observed_by_draw.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)

        assert bool(
            (torch.abs(estimate - etsi_sector_reference) <= z_simultaneous * standard_error).all()
        )
        assert float((z_simultaneous * standard_error).max()) <= _MAX_INTERVAL_HALF_WIDTH

    z_joint = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * _JOINT_FAMILY_CLAIM_COUNT))
    rays = _native_profile_rays(model)
    ray_ranks = {
        angle: torch.argsort(torch.argsort(getattr(rays, angle)[:, 0, 0], dim=-1), dim=-1)
        for angle in _ANGLE_DIRECTIONS
    }
    for first_angle, second_angle in itertools.combinations(_ANGLE_DIRECTIONS, 2):
        first_ranks = ray_ranks[first_angle]
        second_ranks = ray_ranks[second_angle]
        pair_indices = first_ranks * _RAY_ANGLE_OFFSETS.numel() + second_ranks
        pair_weights = (
            rays.powers[:, 0, 0, :, None]
            .to(torch.float64)
            .expand(-1, -1, _RAY_ANGLE_OFFSETS.numel())
            / _RAY_ANGLE_OFFSETS.numel()
        )
        joint_draws = torch.zeros(
            (_BATCH_SIZE, _RAY_ANGLE_OFFSETS.numel() ** 2), dtype=torch.float64
        ).scatter_add(
            dim=1,
            index=pair_indices.reshape(_BATCH_SIZE, -1),
            src=pair_weights.reshape(_BATCH_SIZE, -1),
        )
        joint_reference = torch.full(
            (_RAY_ANGLE_OFFSETS.numel() ** 2,),
            1.0 / (_RAY_ANGLE_OFFSETS.numel() ** 2),
            dtype=torch.float64,
        )
        joint_estimate = joint_draws.mean(dim=0)
        joint_standard_error = joint_draws.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)
        assert bool(
            (
                torch.abs(joint_estimate - joint_reference)
                <= z_joint * joint_standard_error + 1e-12
            ).all()
        ), f"{model} {first_angle}/{second_angle} pairwise diffuse angle rank coupling differs"
        assert float((z_joint * joint_standard_error).max()) <= _MAX_INTERVAL_HALF_WIDTH

    four_angle_indices = torch.zeros_like(ray_ranks[_ANGLE_DIRECTIONS[0]])
    for angle in _ANGLE_DIRECTIONS:
        four_angle_indices = four_angle_indices * _RANK_BINS_PER_ANGLE + ray_ranks[angle] // (
            _RAY_ANGLE_OFFSETS.numel() // _RANK_BINS_PER_ANGLE
        )
    four_angle_bins = _RANK_BINS_PER_ANGLE ** len(_ANGLE_DIRECTIONS)
    diffuse_ray_weights = (
        rays.powers[:, 0, 0, :, None].to(torch.float64).expand(-1, -1, _RAY_ANGLE_OFFSETS.numel())
        / _RAY_ANGLE_OFFSETS.numel()
    )
    four_angle_draws = torch.zeros((_BATCH_SIZE, four_angle_bins), dtype=torch.float64).scatter_add(
        dim=1,
        index=four_angle_indices.reshape(_BATCH_SIZE, -1),
        src=diffuse_ray_weights.reshape(_BATCH_SIZE, -1),
    )
    torch.testing.assert_close(
        four_angle_draws.sum(dim=1), torch.ones(_BATCH_SIZE, dtype=torch.float64)
    )
    four_angle_reference = torch.full(
        (four_angle_bins,), 1.0 / four_angle_bins, dtype=torch.float64
    )
    four_angle_z = NormalDist().inv_cdf(
        1.0 - _FAMILYWISE_ALPHA / (2.0 * _FOUR_ANGLE_FAMILY_CLAIM_COUNT)
    )
    four_angle_estimate = four_angle_draws.mean(dim=0)
    four_angle_standard_error = four_angle_draws.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)
    four_angle_half_width = four_angle_z * four_angle_standard_error
    assert bool(
        (
            torch.abs(four_angle_estimate - four_angle_reference) <= four_angle_half_width + 1e-12
        ).all()
    ), f"{model} four-angle rank-quartile dependence differs from independent coupling"
    assert float(four_angle_half_width.max()) <= _MAX_FOUR_ANGLE_INTERVAL_HALF_WIDTH

    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["parameter_table_sha256"] == _PROFILE_SHA256[model]
