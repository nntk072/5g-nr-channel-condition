"""Independent directional-element power checks for standardized CDL profiles."""

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

_BATCH_SIZE = 16_384
_FAMILYWISE_ALPHA = 0.01
_MAX_INTERVAL_FRACTION_OF_TOTAL_POWER = 0.05
_ONE_END_VECTOR_FIELD_SEED_START = 2_700_000
_BOTH_ENDS_VECTOR_FIELD_SEED_START = 2_701_000
_PROFILE_ANTENNA_PATTERNS = (
    ("38.901", 65.0, 65.0, 30.0, 30.0, 8.0),
    ("38.901-handheld", 125.0, 125.0, 22.5, 22.5, 5.3),
)
_PROFILE_CLUSTER_COUNTS = (23, 23, 24, 13, 14)
_ORIENTATIONS = {
    "zero": (0.0, 0.0, 0.0),
    "tilted_xyz_positive": (0.2, -0.1, 0.3),
    "tilted_xyz_mixed": (-0.7, 0.35, 1.1),
}
_BOTH_ENDS_ORIENTATIONS = {
    "zero": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "independent_tilts": ((0.2, -0.1, 0.3), (-0.7, 0.35, 1.1)),
}
_BOTH_ENDS_PATTERN_PAIRS = tuple(
    (ut_pattern, bs_pattern)
    for ut_pattern in _PROFILE_ANTENNA_PATTERNS
    for bs_pattern in _PROFILE_ANTENNA_PATTERNS
)
_NONZERO_ORIENTATION_CASES = tuple(name for name in _ORIENTATIONS if name != "zero")
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
_CDL_TABLE_SHA256 = {
    "A": "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7",
    "B": "7c87c092cb2bb44891d301ee3c1991cdda1c6b64094a03f18746ec5fa14ee2a2",
    "C": "f2044ecc3b72a1967221805ba70b1f0dab9634bb831478c1337b737ced6e4a28",
    "D": "a8960f176347de457c0cb90fbc8cf137ac98bb1c5295b096f004d8d5c6e40176",
    "E": "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480",
}
_ROOT = Path(__file__).resolve().parents[2]
_TAP_ANGLE_REFERENCE_PATH = _ROOT / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
_PROFILE_SCALAR_REFERENCE_PATH = (
    _ROOT / "docs/research/G4_CDL_PROFILE_SCALAR_REFERENCE_2026-10-26.json"
)
_TAP_ANGLE_REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
_PROFILE_SCALAR_REFERENCE_SHA256 = (
    "3b5920e7c12b0e36c8e09d1c7f14025cfbb23b6d805e1044124bfcf147185620"
)
_ETSI_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
_CLUSTER_CLAIM_COUNT = len(_PROFILE_ANTENNA_PATTERNS) * sum(_PROFILE_CLUSTER_COUNTS)
_ORIENTATION_CLAIM_COUNT = (
    len(_PROFILE_ANTENNA_PATTERNS) * sum(_PROFILE_CLUSTER_COUNTS) * len(_NONZERO_ORIENTATION_CASES)
)
_BOTH_ENDS_CLAIM_COUNT = (
    sum(_PROFILE_CLUSTER_COUNTS) * len(_BOTH_ENDS_PATTERN_PAIRS) * len(_BOTH_ENDS_ORIENTATIONS)
)


def _element_power_pattern(
    zenith_deg: Any,
    azimuth_deg: Any,
    *,
    orientation_rad: tuple[float, float, float],
    zenith_3db_deg: float,
    azimuth_3db_deg: float,
    vertical_sidelobe_limit_db: float,
    maximum_attenuation_db: float,
    maximum_element_gain_db: float,
) -> Any:
    """Evaluate the independent TR 38.901 Table 7.3-1/-2 power law.

    Global ray directions are inverse-rotated into the element's local frame
    with the pinned TR 38.901 composite Euler convention. This mirrors the
    coordinate equation, not Sionna's antenna implementation.
    """
    theta = torch.deg2rad(zenith_deg)
    phi = torch.deg2rad(azimuth_deg)
    x_global = torch.sin(theta) * torch.cos(phi)
    y_global = torch.sin(theta) * torch.sin(phi)
    z_global = torch.cos(theta)
    alpha, beta, gamma = orientation_rad
    cos_alpha, sin_alpha = math.cos(alpha), math.sin(alpha)
    cos_beta, sin_beta = math.cos(beta), math.sin(beta)
    cos_gamma, sin_gamma = math.cos(gamma), math.sin(gamma)
    # Apply the transpose of the TR 38.901 composite Rz(alpha)Ry(beta)Rx(gamma)
    # rotation to express a global ray direction in the antenna's local frame.
    x_local = (
        cos_alpha * cos_beta * x_global + sin_alpha * cos_beta * y_global - sin_beta * z_global
    )
    y_local = (
        (cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma) * x_global
        + (sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma) * y_global
        + cos_beta * sin_gamma * z_global
    )
    z_local = (
        (cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma) * x_global
        + (sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma) * y_global
        + cos_beta * cos_gamma * z_global
    )
    theta_local = torch.acos(torch.clamp(z_local, -1.0, 1.0))
    phi_local = torch.atan2(y_local, x_local)
    attenuation_vertical = -torch.minimum(
        12.0 * ((theta_local - math.pi / 2.0) / math.radians(zenith_3db_deg)).square(),
        torch.full_like(theta_local, vertical_sidelobe_limit_db),
    )
    attenuation_horizontal = -torch.minimum(
        12.0 * (phi_local / math.radians(azimuth_3db_deg)).square(),
        torch.full_like(phi_local, maximum_attenuation_db),
    )
    attenuation_db = (
        -torch.minimum(
            -(attenuation_vertical + attenuation_horizontal),
            torch.full_like(theta_local, maximum_attenuation_db),
        )
        + maximum_element_gain_db
    )
    return torch.pow(10.0, attenuation_db / 10.0)


def _element_field_gcs_components(
    zenith_deg: Any,
    azimuth_deg: Any,
    *,
    orientation_rad: tuple[float, float, float],
    pattern: tuple[str, float, float, float, float, float] | None,
) -> Any:
    """Return the independent single-V field vector in global coordinates.

    The local V field is ``[sqrt(power_pattern), 0]``. The final 2-D rotation
    is the TR 38.901 polarization displacement (7.1-11/15), implemented
    independently from Sionna's channel coefficient generator. ``None`` is
    the isotropic omni pattern with unit field gain.
    """
    theta = torch.deg2rad(zenith_deg)
    phi = torch.deg2rad(azimuth_deg)
    if pattern is None:
        power = torch.ones_like(theta)
    else:
        _, theta_3db, phi_3db, sla_v, a_max, g_max = pattern
        power = _element_power_pattern(
            zenith_deg,
            azimuth_deg,
            orientation_rad=orientation_rad,
            zenith_3db_deg=theta_3db,
            azimuth_3db_deg=phi_3db,
            vertical_sidelobe_limit_db=sla_v,
            maximum_attenuation_db=a_max,
            maximum_element_gain_db=g_max,
        )
    alpha, beta, gamma = orientation_rad
    real = math.sin(gamma) * torch.cos(theta) * torch.sin(phi - alpha)
    real = real + math.cos(gamma) * (
        math.cos(beta) * torch.sin(theta)
        - math.sin(beta) * torch.cos(theta) * torch.cos(phi - alpha)
    )
    imag = math.sin(gamma) * torch.cos(phi - alpha) + math.sin(beta) * math.cos(gamma) * torch.sin(
        phi - alpha
    )
    psi = torch.atan2(imag, real)
    amplitude = torch.sqrt(power)
    return torch.stack((amplitude * torch.cos(psi), amplitude * torch.sin(psi)), dim=-1)


def _pattern_field_component_power_moments(
    pattern: tuple[str, float, float, float, float, float] | None,
    orientation_rad: tuple[float, float, float],
    azimuth_deg: float,
    zenith_deg: float,
    azimuth_spread_deg: float,
    zenith_spread_deg: float,
) -> Any:
    """Average each global field component over independent ray permutations."""
    offsets = torch.as_tensor(_RAY_ANGLE_OFFSETS, dtype=torch.float64)
    azimuth = torch.as_tensor(azimuth_deg, dtype=torch.float64) + azimuth_spread_deg * offsets
    zenith = torch.as_tensor(zenith_deg, dtype=torch.float64) + zenith_spread_deg * offsets
    field = _element_field_gcs_components(
        zenith[:, None].expand(20, 20),
        azimuth[None, :].expand(20, 20),
        orientation_rad=orientation_rad,
        pattern=pattern,
    )
    return field.square().mean(dim=(0, 1))


def _mean_random_phase_polarization_power(rx: Any, tx: Any, xpr_linear: float) -> Any:
    """Average squared field contraction over the four independent phases."""
    cross_polar_power = 1.0 / xpr_linear
    return rx[0] * tx[0] + rx[1] * tx[1] + cross_polar_power * (rx[0] * tx[1] + rx[1] * tx[0])


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
    assert set(tap_profiles) == set(scalar_profiles) == set("ABCDE")
    assert tap_profiles[model]["los"] == scalar_profiles[model]["los"]
    assert tap_profiles[model]["xpr_db"] == scalar_profiles[model]["xpr_db"]
    assert (
        tap_profiles[model]["cluster_count"] == scalar_profiles[model]["output_delay_cluster_count"]
    )
    assert tap_profiles[model]["cluster_spreads_deg"] == scalar_profiles[model]["angle_spreads_deg"]
    return tap_profiles[model], scalar_profiles[model]


def _independent_profile_cluster_power(
    model: str,
    pattern: tuple[str, float, float, float, float, float],
    orientation_rad: tuple[float, float, float],
) -> tuple[Any, str]:
    """Derive UT-pattern-weighted cluster powers from pinned independent ETSI rows."""
    profile, scalar = _independent_profile_references(model)
    is_los = bool(profile["los"])
    k_factor = 10.0 ** (float(scalar["los_k_factor_db"]) / 10.0) if is_los else 0.0
    xpr_linear = 10.0 ** (float(profile["xpr_db"]) / 10.0)
    diffuse_components = sorted(
        (component for component in profile["components"] if component["component"] == "diffuse"),
        key=lambda component: float(component["normalized_delay"]),
    )
    raw_diffuse_powers = [
        10.0 ** (float(component["power_db"]) / 10.0) for component in diffuse_components
    ]
    diffuse_power_sum = math.fsum(raw_diffuse_powers)
    omni_tx_moments = torch.as_tensor((1.0, 0.0), dtype=torch.float64)
    expected: list[Any] = []

    for index, (component, raw_power) in enumerate(
        zip(diffuse_components, raw_diffuse_powers, strict=True)
    ):
        angles = component["angles_deg"]
        spreads = profile["cluster_spreads_deg"]
        rx_component_power = _pattern_field_component_power_moments(
            pattern,
            orientation_rad,
            float(angles["aoa"]),
            float(angles["zoa"]),
            float(spreads["asa"]),
            float(spreads["zsa"]),
        )
        diffuse_field_power = _mean_random_phase_polarization_power(
            rx_component_power, omni_tx_moments, xpr_linear
        )
        diffuse_power = raw_power / diffuse_power_sum / (k_factor + 1.0) * diffuse_field_power
        if is_los and index == 0:
            specular = next(
                item for item in profile["components"] if item["component"] == "specular"
            )
            specular_angles = specular["angles_deg"]
            los_rx_field = _element_field_gcs_components(
                torch.as_tensor([float(specular_angles["zoa"])], dtype=torch.float64),
                torch.as_tensor([float(specular_angles["aoa"])], dtype=torch.float64),
                orientation_rad=orientation_rad,
                pattern=pattern,
            )[0]
            # Pinned Sionna uses the deterministic LOS phase matrix diag(1,-1);
            # the omni, zero-oriented BS has global field [1,0].
            los_field_amplitude = los_rx_field[0]
            diffuse_power = (
                diffuse_power + k_factor / (k_factor + 1.0) * los_field_amplitude.square()
            )
        expected.append(diffuse_power)
    return torch.stack(expected), _CDL_TABLE_SHA256[model]


def _independent_both_endpoint_cluster_power(
    model: str,
    ut_pattern: tuple[str, float, float, float, float, float],
    bs_pattern: tuple[str, float, float, float, float, float],
    ut_orientation_rad: tuple[float, float, float],
    bs_orientation_rad: tuple[float, float, float],
) -> tuple[Any, str]:
    """Derive dual-end pattern powers from pinned independent ETSI cluster rows."""
    profile, scalar = _independent_profile_references(model)
    is_los = bool(profile["los"])
    k_factor = 10.0 ** (float(scalar["los_k_factor_db"]) / 10.0) if is_los else 0.0
    xpr_linear = 10.0 ** (float(profile["xpr_db"]) / 10.0)
    diffuse_components = sorted(
        (component for component in profile["components"] if component["component"] == "diffuse"),
        key=lambda component: float(component["normalized_delay"]),
    )
    raw_diffuse_powers = [
        10.0 ** (float(component["power_db"]) / 10.0) for component in diffuse_components
    ]
    diffuse_power_sum = math.fsum(raw_diffuse_powers)

    def pattern_mean(
        pattern: tuple[str, float, float, float, float, float],
        orientation: tuple[float, float, float],
        azimuth: float,
        zenith: float,
        azimuth_spread: float,
        zenith_spread: float,
    ) -> Any:
        return _pattern_field_component_power_moments(
            pattern,
            orientation_rad=orientation,
            azimuth_deg=azimuth,
            zenith_deg=zenith,
            azimuth_spread_deg=azimuth_spread,
            zenith_spread_deg=zenith_spread,
        )

    expected: list[Any] = []
    for index, (component, raw_power) in enumerate(
        zip(diffuse_components, raw_diffuse_powers, strict=True)
    ):
        angles = component["angles_deg"]
        spreads = profile["cluster_spreads_deg"]
        ut_gain = pattern_mean(
            ut_pattern,
            ut_orientation_rad,
            float(angles["aoa"]),
            float(angles["zoa"]),
            float(spreads["asa"]),
            float(spreads["zsa"]),
        )
        bs_gain = pattern_mean(
            bs_pattern,
            bs_orientation_rad,
            float(angles["aod"]),
            float(angles["zod"]),
            float(spreads["asd"]),
            float(spreads["zsd"]),
        )
        diffuse_field_power = _mean_random_phase_polarization_power(ut_gain, bs_gain, xpr_linear)
        cluster_power = raw_power / diffuse_power_sum / (k_factor + 1.0) * diffuse_field_power
        if is_los and index == 0:
            specular = next(
                item for item in profile["components"] if item["component"] == "specular"
            )
            specular_angles = specular["angles_deg"]

            def los_field(
                pattern: tuple[str, float, float, float, float, float],
                orientation: tuple[float, float, float],
                azimuth: float,
                zenith: float,
            ) -> Any:
                return _element_field_gcs_components(
                    torch.as_tensor([zenith], dtype=torch.float64),
                    torch.as_tensor([azimuth], dtype=torch.float64),
                    orientation_rad=orientation,
                    pattern=pattern,
                )[0]

            rx_los_field = los_field(
                ut_pattern,
                ut_orientation_rad,
                float(specular_angles["aoa"]),
                float(specular_angles["zoa"]),
            )
            tx_los_field = los_field(
                bs_pattern,
                bs_orientation_rad,
                float(specular_angles["aod"]),
                float(specular_angles["zod"]),
            )
            los_field_amplitude = (
                rx_los_field[0] * tx_los_field[0] - rx_los_field[1] * tx_los_field[1]
            )
            cluster_power = (
                cluster_power + k_factor / (k_factor + 1.0) * los_field_amplitude.square()
            )
        expected.append(cluster_power)
    return torch.stack(expected), _CDL_TABLE_SHA256[model]


def _pattern_orientation_seed(model: str, pattern_name: str, orientation_case: str) -> int:
    orientation_index = tuple(_ORIENTATIONS).index(orientation_case)
    pattern_index = tuple(item[0] for item in _PROFILE_ANTENNA_PATTERNS).index(pattern_name)
    case_index = 6 * tuple("ABCDE").index(model) + 3 * pattern_index + orientation_index
    return _ONE_END_VECTOR_FIELD_SEED_START + case_index


def _record_case_result(
    record_property: Any,
    *,
    cohort: str,
    model: str,
    seed: int,
    patterns: tuple[str, str],
    orientations: tuple[tuple[float, float, float], tuple[float, float, float]],
    source_digest: str,
    expected: Any,
    estimate: Any,
    half_width: Any,
) -> None:
    """Attach complete per-cluster summaries for the dedicated report runner."""
    claims = []
    for index, (reference, observed, width) in enumerate(
        zip(expected.tolist(), estimate.tolist(), half_width.tolist(), strict=True)
    ):
        claims.append(
            {
                "cluster_index": index,
                "reference": float(reference),
                "estimate": float(observed),
                "half_width": float(width),
                "interval": [float(observed - width), float(observed + width)],
                "passed": abs(observed - reference) <= width,
            }
        )
    record_property(
        "g4_cdl_directional_pattern_case",
        json.dumps(
            {
                "cohort": cohort,
                "profile": model,
                "seed": seed,
                "patterns": list(patterns),
                "orientations_rad": [list(row) for row in orientations],
                "parameter_table_sha256": source_digest,
                "sample_count": _BATCH_SIZE,
                "familywise_alpha": _FAMILYWISE_ALPHA,
                "family_claim_count": (
                    _CLUSTER_CLAIM_COUNT
                    if cohort == "one_end_zero"
                    else _ORIENTATION_CLAIM_COUNT
                    if cohort == "one_end_tilted"
                    else _BOTH_ENDS_CLAIM_COUNT
                ),
                "total_expected_power": float(expected.sum()),
                "maximum_half_width": float(half_width.max()),
                "maximum_allowed_half_width": _MAX_INTERVAL_FRACTION_OF_TOTAL_POWER
                * float(expected.sum()),
                "passed": bool(
                    (torch.abs(estimate - expected) <= half_width).all()
                    and half_width.max() <= _MAX_INTERVAL_FRACTION_OF_TOTAL_POWER * expected.sum()
                ),
                "claims": claims,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    )


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", tuple("ABCDE"))
@pytest.mark.parametrize("pattern", _PROFILE_ANTENNA_PATTERNS)
@pytest.mark.parametrize("orientation_case", tuple(_ORIENTATIONS))
def test_cdl_directional_element_cluster_power_matches_independent_reference(
    model: str,
    pattern: tuple[str, float, float, float, float, float],
    orientation_case: str,
    record_property: Any,
) -> None:
    """Check directional UT patterns against an independent per-cluster law."""
    pattern_name = pattern[0]
    orientation_rad = _ORIENTATIONS[orientation_case]
    expected, source_digest = _independent_profile_cluster_power(model, pattern, orientation_rad)
    seed = _pattern_orientation_seed(model, pattern_name, orientation_case)
    channel = generate_cdl_frequency_response(
        model=model,
        seed=seed,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_BATCH_SIZE,
        delay_spread_s=100e-9,
        ut_array=CDLArrayConfig(antenna_pattern=pattern_name),
        bs_array=CDLArrayConfig(antenna_pattern="omni"),
        ut_orientation_rad=orientation_rad,
        bs_orientation_rad=(0.0, 0.0, 0.0),
        spec_version="19.2",
        device="cpu",
    )
    samples = channel.h_path[:, 0, 0, 0, 0, :, 0]
    draw_level_power = samples.abs().square().to(torch.float64)
    estimate = draw_level_power.mean(dim=0)
    standard_error = draw_level_power.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)
    family_claim_count = (
        _CLUSTER_CLAIM_COUNT if orientation_case == "zero" else _ORIENTATION_CLAIM_COUNT
    )
    z_simultaneous = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * family_claim_count))
    half_width = z_simultaneous * standard_error
    _record_case_result(
        record_property,
        cohort="one_end_zero" if orientation_case == "zero" else "one_end_tilted",
        model=model,
        seed=seed,
        patterns=(pattern_name, "omni"),
        orientations=(orientation_rad, (0.0, 0.0, 0.0)),
        source_digest=source_digest,
        expected=expected,
        estimate=estimate,
        half_width=half_width,
    )

    assert tuple(draw_level_power.shape) == (_BATCH_SIZE, expected.numel())
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["sionna_version"] == "2.2.0"
    assert channel.metadata["spec_version"] == "19.2"
    assert channel.metadata["parameter_table_sha256"] == source_digest
    assert channel.metadata["ut_array"]["antenna_pattern"] == pattern_name
    assert channel.metadata["bs_array"]["antenna_pattern"] == "omni"
    assert bool((torch.abs(estimate - expected) <= half_width).all())
    assert float(half_width.max()) <= _MAX_INTERVAL_FRACTION_OF_TOTAL_POWER * float(expected.sum())


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", tuple("ABCDE"))
@pytest.mark.parametrize("pattern_pair", _BOTH_ENDS_PATTERN_PAIRS)
@pytest.mark.parametrize("orientation_case", tuple(_BOTH_ENDS_ORIENTATIONS))
def test_cdl_directional_patterns_at_both_link_ends_match_independent_reference(
    model: str,
    pattern_pair: tuple[
        tuple[str, float, float, float, float, float],
        tuple[str, float, float, float, float, float],
    ],
    orientation_case: str,
    record_property: Any,
) -> None:
    """Check independent UT/BS patterns and orientations under one fresh family."""
    ut_pattern, bs_pattern = pattern_pair
    ut_orientation, bs_orientation = _BOTH_ENDS_ORIENTATIONS[orientation_case]
    expected, source_digest = _independent_both_endpoint_cluster_power(
        model,
        ut_pattern,
        bs_pattern,
        ut_orientation,
        bs_orientation,
    )
    pattern_pair_index = _BOTH_ENDS_PATTERN_PAIRS.index(pattern_pair)
    orientation_index = tuple(_BOTH_ENDS_ORIENTATIONS).index(orientation_case)
    case_index = 8 * tuple("ABCDE").index(model) + 2 * pattern_pair_index + orientation_index
    seed = _BOTH_ENDS_VECTOR_FIELD_SEED_START + case_index
    channel = generate_cdl_frequency_response(
        model=model,
        seed=seed,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_BATCH_SIZE,
        delay_spread_s=100e-9,
        ut_array=CDLArrayConfig(antenna_pattern=ut_pattern[0]),
        bs_array=CDLArrayConfig(antenna_pattern=bs_pattern[0]),
        ut_orientation_rad=ut_orientation,
        bs_orientation_rad=bs_orientation,
        spec_version="19.2",
        device="cpu",
    )
    draw_level_power = channel.h_path[:, 0, 0, 0, 0, :, 0].abs().square().to(torch.float64)
    estimate = draw_level_power.mean(dim=0)
    standard_error = draw_level_power.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)
    z_simultaneous = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * _BOTH_ENDS_CLAIM_COUNT))
    half_width = z_simultaneous * standard_error
    _record_case_result(
        record_property,
        cohort="both_ends_zero" if orientation_case == "zero" else "both_ends_tilted",
        model=model,
        seed=seed,
        patterns=(ut_pattern[0], bs_pattern[0]),
        orientations=(ut_orientation, bs_orientation),
        source_digest=source_digest,
        expected=expected,
        estimate=estimate,
        half_width=half_width,
    )

    assert tuple(draw_level_power.shape) == (_BATCH_SIZE, expected.numel())
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["sionna_version"] == "2.2.0"
    assert channel.metadata["spec_version"] == "19.2"
    assert channel.metadata["parameter_table_sha256"] == source_digest
    assert channel.metadata["ut_array"]["antenna_pattern"] == ut_pattern[0]
    assert channel.metadata["bs_array"]["antenna_pattern"] == bs_pattern[0]
    assert bool((torch.abs(estimate - expected) <= half_width).all())
    assert float(half_width.max()) <= _MAX_INTERVAL_FRACTION_OF_TOTAL_POWER * float(expected.sum())
