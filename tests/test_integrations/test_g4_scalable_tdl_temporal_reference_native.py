"""ETSI-referenced temporal covariance checks for scalable TDL Rayleigh taps."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_PATH = _ROOT / "docs/research/G4_SCALABLE_TDL_POWER_DELAY_REFERENCE_2026-10-26.json"
_REFERENCE_SHA256 = "38b897029bd4cc6de49eece5096e69c75291eaf6dac45bebe04767ed24a4f0bd"
_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
_SEED_AUDIT_PATH = _ROOT / "docs/research/G4_SCALABLE_TDL_TEMPORAL_SEED_AUDIT_2026-10-27.json"
_SEED_AUDIT_SHA256 = "7c94438bc6380d99f0d6ffe66e83cb5eb1303df0ad32d7a9b705c42f75af9a1c"
_MODELS = tuple("ABCDE")
_SEED_START = 1_741_000
_SAMPLE_COUNT = 32_768
_CLAIM_COUNT = 570  # 95 reference Rayleigh taps x 3 lags x real/imaginary coordinates.
_FAMILYWISE_ALPHA = 0.01
_MAXIMUM_NORMALIZED_ERROR = 0.05
_NORMAL_DIST = NormalDist()


def _read_pinned_json(path: Path, expected_digest: str, *, label: str) -> dict[str, Any]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise ValueError(f"{label} does not match its frozen SHA-256")
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise ValueError(f"{label} root must be an object")
    return result


def _profile_rayleigh_taps(profile: dict[str, Any]) -> tuple[int, ...]:
    components = profile.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("profile must contain ETSI component rows")
    has_los = any(row.get("fading_distribution") == "LOS" for row in components)
    result: list[int] = []
    for expected_index, component in enumerate(components, start=1):
        if component.get("component_index") != expected_index:
            raise ValueError("ETSI component rows must use a contiguous one-based roster")
        if component.get("fading_distribution") != "Rayleigh":
            continue
        tap_index = expected_index - (2 if has_los else 1)
        if tap_index >= 0 and not (has_los and tap_index == 0):
            result.append(tap_index)
    return tuple(result)


def _load_protocol_inputs() -> tuple[dict[str, Any], list[int]]:
    reference = _read_pinned_json(
        _REFERENCE_PATH, _REFERENCE_SHA256, label="scalable TDL ETSI reference"
    )
    if (
        reference.get("schema") != "loslab.g4_scalable_tdl_power_delay_reference.v1"
        or reference.get("source_id") != "tr-138901-v19.2.0"
        or reference.get("source_sha256") != _SOURCE_SHA256
        or reference.get("status") != "PROFILE_TABLE_REFERENCE_ONLY_NOT_CONFORMANCE_QUALIFICATION"
    ):
        raise ValueError("scalable TDL reference has an invalid normative identity")
    profiles = reference.get("profiles")
    if not isinstance(profiles, list) or {row.get("profile_model") for row in profiles} != set(
        _MODELS
    ):
        raise ValueError("reference must cover exactly scalable TDL-A through TDL-E")
    profile_map = {row["profile_model"]: row for row in profiles}
    expected_counts = {"A": 23, "B": 23, "C": 24, "D": 12, "E": 13}
    if {model: len(_profile_rayleigh_taps(profile_map[model])) for model in _MODELS} != (
        expected_counts
    ):
        raise ValueError("ETSI Rayleigh tap roster differs from the frozen temporal family")

    seed_audit = _read_pinned_json(
        _SEED_AUDIT_PATH, _SEED_AUDIT_SHA256, label="scalable TDL temporal seed audit"
    )
    expected_seeds = [_SEED_START + index for index in range(len(_MODELS))]
    if (
        seed_audit.get("status") != "CLEAR_WITHIN_SCANNED_JSON_CORPUS"
        or seed_audit.get("reserved_seeds") != expected_seeds
        or seed_audit.get("incomplete_files")
        or seed_audit.get("collisions")
        or seed_audit.get("malformed_seed_fields")
    ):
        raise ValueError("scalable TDL temporal seed roster is not clear in its frozen audit")
    return profile_map, expected_seeds


def test_scalable_tdl_temporal_family_is_bound_to_etsi_taps_and_fresh_seeds() -> None:
    profiles, seeds = _load_protocol_inputs()
    assert seeds == [1_741_000, 1_741_001, 1_741_002, 1_741_003, 1_741_004]
    assert sum(len(_profile_rayleigh_taps(profiles[model])) for model in _MODELS) == 95
    assert _CLAIM_COUNT == 95 * 3 * 2
    assert _NORMAL_DIST.inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * _CLAIM_COUNT)) > 0.0


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
def test_scalable_tdl_rayleigh_tap_temporal_covariance_matches_jakes(model: str) -> None:
    """Bound all ETSI-labeled diffuse taps at three predeclared Jakes lags."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sionna.phy")
    from integrations.sionna.tdl import generate_tdl_frequency_response

    profiles, seeds = _load_protocol_inputs()
    tap_indices = _profile_rayleigh_taps(profiles[model])
    carrier_hz = 3.5e9
    speed_mps = 30.0
    maximum_doppler_hz = speed_mps * carrier_hz / 299_792_458.0
    base_lag_s = 1.0 / (8.0 * maximum_doppler_hz)
    channel = generate_tdl_frequency_response(
        model=model,
        seed=seeds[_MODELS.index(model)],
        carrier_frequency_hz=carrier_hz,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_SAMPLE_COUNT,
        delay_spread_s=100e-9,
        num_time_steps=5,
        sampling_frequency_hz=1.0 / base_lag_s,
        min_speed_mps=speed_mps,
        max_speed_mps=speed_mps,
        num_sinusoids=20,
        spec_version="19.2",
        device="cpu",
    )
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["maximum_doppler_hz"] == pytest.approx(maximum_doppler_hz)
    assert channel.metadata["num_sinusoids"] == 20
    assert tuple(channel.h_path.shape) == (
        _SAMPLE_COUNT,
        1,
        1,
        1,
        1,
        len(profiles[model]["components"])
        - int(any(row["fading_distribution"] == "LOS" for row in profiles[model]["components"])),
        5,
    )

    critical_z = _NORMAL_DIST.inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * _CLAIM_COUNT))
    lag_indices = (1, 2, 4)
    for tap_index in tap_indices:
        samples = channel.h_path[:, 0, 0, 0, 0, tap_index, :].to(torch.complex128)
        initial_power = samples[:, 0].abs().square()
        mean_power = initial_power.mean()
        for lag_index in lag_indices:
            lag_s = lag_index * base_lag_s
            expected = torch.special.bessel_j0(
                torch.tensor(2.0 * math.pi * maximum_doppler_hz * lag_s, dtype=torch.float64)
            )
            residual = samples[:, 0].conj() * samples[:, lag_index] - expected * initial_power
            normalized_error = residual.mean() / mean_power
            half_width_real = (
                critical_z
                * residual.real.std(unbiased=True)
                / math.sqrt(_SAMPLE_COUNT)
                / mean_power
            )
            half_width_imag = (
                critical_z
                * residual.imag.std(unbiased=True)
                / math.sqrt(_SAMPLE_COUNT)
                / mean_power
            )
            assert abs(normalized_error.real) + half_width_real <= _MAXIMUM_NORMALIZED_ERROR
            assert abs(normalized_error.imag) + half_width_imag <= _MAXIMUM_NORMALIZED_ERROR
