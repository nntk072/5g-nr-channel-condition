"""ETSI-referenced proper-complex checks for fixed TDL Rayleigh taps."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pytest
from integrations.sionna.tdl import generate_tdl_frequency_response

_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_PATH = _ROOT / "docs/research/G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json"
_REFERENCE_SHA256 = "fe22560a71f4e5683ea6780f73e84466155b26f5f97cc4b63e3250c102fafda3"
_SOURCE_SHA256 = "b3c6a73c7a40219e7270c08c7a57c84cb46f6757b303a64ec768ba04a6079e81"
_SEED_AUDIT_PATH = _ROOT / "docs/research/G4_FIXED_TDL_PROPERNESS_SEED_AUDIT_2026-10-27.json"
_SEED_AUDIT_SHA256 = "c9887cb5a39b858baacacbdd783b478638722fb657c512cad34fc83c285e3e56"
_MODELS = ("A10", "A30", "B100", "C60", "C300", "D10", "D30")
_SEED_START = 1_730_000
_SAMPLE_COUNT = 32_768
_FAMILYWISE_ALPHA = 0.01
_MAXIMUM_HALF_WIDTH = 0.05
_NORMAL_DIST = NormalDist()


def _load_reference() -> dict[str, Any]:
    payload = _REFERENCE_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _REFERENCE_SHA256:
        raise ValueError("fixed TDL tap reference does not match its frozen digest")
    reference = json.loads(payload)
    if (
        reference.get("schema") != "loslab.g4_fixed_tdl_tap_profile_reference.v1"
        or reference.get("source_id") != "ts-138101-4-v19.2.2"
        or reference.get("source_sha256") != _SOURCE_SHA256
        or reference.get("status") != "PER_TAP_REFERENCE_ONLY_NOT_CONFORMANCE_QUALIFICATION"
    ):
        raise ValueError("fixed TDL tap reference has an invalid normative identity")
    profiles = reference.get("profiles")
    if not isinstance(profiles, list) or {row.get("profile_model") for row in profiles} != set(
        _MODELS
    ):
        raise ValueError("fixed TDL tap reference does not cover the frozen profile roster")
    return reference


def _rayleigh_tap_indices(profile: dict[str, Any]) -> tuple[int, ...]:
    by_tap: dict[int, list[dict[str, Any]]] = {}
    for component in profile["components"]:
        by_tap.setdefault(int(component["tap_index"]), []).append(component)
    if sorted(by_tap) != list(range(1, int(profile["tap_count"]) + 1)):
        raise ValueError("fixed TDL component reference has a noncontiguous tap roster")
    return tuple(
        tap_index - 1
        for tap_index, components in sorted(by_tap.items())
        if all(component["fading_distribution"] == "Rayleigh" for component in components)
    )


def _rayleigh_component_count(reference: dict[str, Any]) -> int:
    return sum(
        len(_rayleigh_tap_indices(profile))
        for profile in reference["profiles"]
        if profile["profile_model"] in _MODELS
    )


def test_fixed_tdl_circularity_family_is_bound_to_the_expected_etsi_taps() -> None:
    reference = _load_reference()
    profiles = {row["profile_model"]: row for row in reference["profiles"]}
    assert {model: len(_rayleigh_tap_indices(profiles[model])) for model in _MODELS} == {
        "A10": 16,
        "A30": 12,
        "B100": 12,
        "C60": 12,
        "C300": 12,
        "D10": 9,
        "D30": 9,
    }
    assert _rayleigh_component_count(reference) == 82
    assert _rayleigh_tap_indices(profiles["D10"])[0] == 1
    assert _rayleigh_tap_indices(profiles["D30"])[0] == 1
    assert 0 not in _rayleigh_tap_indices(profiles["D10"])
    assert 0 not in _rayleigh_tap_indices(profiles["D30"])


def _verify_seed_audit() -> None:
    payload = _SEED_AUDIT_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _SEED_AUDIT_SHA256:
        raise ValueError("fixed TDL properness seed audit does not match its frozen digest")
    audit = json.loads(payload)
    expected_seeds = [_SEED_START + index for index in range(len(_MODELS))]
    if (
        audit.get("schema") != "loslab.seed_roster_collision_audit.v2"
        or audit.get("status") != "CLEAR_WITHIN_SCANNED_JSON_CORPUS"
        or audit.get("reserved_seeds") != expected_seeds
        or audit.get("scanned_file_count", 0) < 1
        or audit.get("collisions")
        or audit.get("malformed_seed_fields")
        or audit.get("incomplete_files")
    ):
        raise ValueError("fixed TDL properness seed roster is not clear in its frozen audit")


def test_fixed_tdl_circularity_seed_roster_has_a_complete_collision_audit() -> None:
    _verify_seed_audit()


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
def test_fixed_tdl_rayleigh_tap_mean_and_pseudocovariance_match_proper_complex_reference(
    model: str,
) -> None:
    """Check zero mean and properness for each independently drawn Rayleigh tap."""
    _verify_seed_audit()
    torch = pytest.importorskip("torch")
    pytest.importorskip("sionna.phy")
    reference = _load_reference()
    profile = next(row for row in reference["profiles"] if row["profile_model"] == model)
    rayleigh_taps = _rayleigh_tap_indices(profile)
    claim_count = 4 * _rayleigh_component_count(reference)
    simultaneous_z = _NORMAL_DIST.inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * claim_count))
    minimum_bandwidth_hz = profile["minimum_channel_bandwidth_hz_exclusive"]
    channel_bandwidth_hz = (
        float(minimum_bandwidth_hz) + 1.0 if minimum_bandwidth_hz is not None else 100e6
    )
    carrier_frequency_hz = 28e9 if model in {"A10", "C60", "D10"} else 3.5e9
    channel = generate_tdl_frequency_response(
        model=model,
        seed=_SEED_START + _MODELS.index(model),
        carrier_frequency_hz=carrier_frequency_hz,
        subcarrier_spacing_hz=30e3,
        num_subcarriers=24,
        channel_bandwidth_hz=channel_bandwidth_hz,
        batch_size=_SAMPLE_COUNT,
        spec_version="19.2",
        num_time_steps=1,
        device="cpu",
    )
    assert channel.metadata["sionna_version"] == "2.2.0"
    assert channel.metadata["profile_parameter_table_source"] == (
        f"TS 38.101-4:19.2.2:TDL-{model}.json"
    )
    assert channel.metadata["profile_los_component"] is (model in {"D10", "D30"})
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert tuple(channel.h_path.shape) == (_SAMPLE_COUNT, 1, 1, 1, 1, int(profile["tap_count"]), 1)

    for tap_index in rayleigh_taps:
        samples = channel.h_path[:, 0, 0, 0, 0, tap_index, 0].to(torch.complex128)
        power = samples.abs().square().mean()
        mean_scale = torch.sqrt(power)
        normalized_mean = samples.mean() / mean_scale
        mean_half_widths = (
            simultaneous_z
            * torch.stack((samples.real.std(unbiased=True), samples.imag.std(unbiased=True)))
            / (mean_scale * math.sqrt(_SAMPLE_COUNT))
        )
        assert bool(
            (
                torch.stack((normalized_mean.real, normalized_mean.imag)).abs() <= mean_half_widths
            ).all()
        ), (
            model,
            tap_index,
            normalized_mean.tolist(),
            mean_half_widths.tolist(),
        )

        pseudo_covariance = samples.square().mean() / power
        influence = samples.square() - pseudo_covariance * power
        pseudo_half_widths = (
            simultaneous_z
            * torch.stack((influence.real.std(unbiased=True), influence.imag.std(unbiased=True)))
            / (power * math.sqrt(_SAMPLE_COUNT))
        )
        assert bool(
            (
                torch.stack((pseudo_covariance.real, pseudo_covariance.imag)).abs()
                <= pseudo_half_widths
            ).all()
        ), (
            model,
            tap_index,
            pseudo_covariance.tolist(),
            pseudo_half_widths.tolist(),
        )
        assert float(torch.maximum(mean_half_widths.max(), pseudo_half_widths.max())) <= (
            _MAXIMUM_HALF_WIDTH
        )
