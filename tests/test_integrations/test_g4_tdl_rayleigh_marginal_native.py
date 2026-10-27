"""ETSI-referenced Rayleigh-power marginal checks for TDL profiles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from evaluation.g4_tdl_covariance_reference import (
    fixed_tdl_rayleigh_tap_power_fractions,
    scalable_tdl_rayleigh_tap_power_fractions,
)
from evaluation.g4_tdl_rayleigh_marginal import score_rayleigh_tap_marginal_family
from integrations.sionna.tdl import generate_tdl_frequency_response

_ROOT = Path(__file__).resolve().parents[2]
_SCALABLE_REFERENCE = _ROOT / "docs/research/G4_SCALABLE_TDL_POWER_DELAY_REFERENCE_2026-10-26.json"
_FIXED_REFERENCE = _ROOT / "docs/research/G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json"
_SCALABLE_REFERENCE_SHA256 = "38b897029bd4cc6de49eece5096e69c75291eaf6dac45bebe04767ed24a4f0bd"
_FIXED_REFERENCE_SHA256 = "fe22560a71f4e5683ea6780f73e84466155b26f5f97cc4b63e3250c102fafda3"
_SCALABLE_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
_FIXED_SOURCE_SHA256 = "b3c6a73c7a40219e7270c08c7a57c84cb46f6757b303a64ec768ba04a6079e81"
_MODELS = ("A", "B", "C", "D", "E", "A10", "A30", "B100", "C60", "C300", "D10", "D30")
_EXPECTED_TAP_COUNTS = {
    "A": 23,
    "B": 23,
    "C": 24,
    "D": 12,
    "E": 13,
    "A10": 16,
    "A30": 12,
    "B100": 12,
    "C60": 12,
    "C300": 12,
    "D10": 9,
    "D30": 9,
}
_SCALABLE_NATIVE_TAP_COUNTS = {"A": 23, "B": 23, "C": 24, "D": 12, "E": 13}
_SEED_START = 1_780_000
_DRAW_COUNT = 32_768
_FAMILY_TAP_COUNT = 177
_ALPHA = 0.01
_MAXIMUM_SIMULTANEOUS_CDF_ERROR = 0.05


def _load_reference(path: Path, expected_digest: str) -> dict[str, Any]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise ValueError(f"G4 TDL Rayleigh reference changed: {path.name}")
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise ValueError(f"G4 TDL Rayleigh reference must be an object: {path.name}")
    return result


def test_g4_tdl_rayleigh_marginal_contract_binds_normative_tables_and_rosters() -> None:
    scalable = _load_reference(_SCALABLE_REFERENCE, _SCALABLE_REFERENCE_SHA256)
    fixed = _load_reference(_FIXED_REFERENCE, _FIXED_REFERENCE_SHA256)
    assert scalable["source_sha256"] == _SCALABLE_SOURCE_SHA256
    assert fixed["source_sha256"] == _FIXED_SOURCE_SHA256
    assert scalable["status"] == "PROFILE_TABLE_REFERENCE_ONLY_NOT_CONFORMANCE_QUALIFICATION"
    assert fixed["status"] == "PER_TAP_REFERENCE_ONLY_NOT_CONFORMANCE_QUALIFICATION"
    counts = {
        model: len(
            scalable_tdl_rayleigh_tap_power_fractions(model)
            if model in {"A", "B", "C", "D", "E"}
            else fixed_tdl_rayleigh_tap_power_fractions(model)
        )
        for model in _MODELS
    }
    assert counts == _EXPECTED_TAP_COUNTS
    assert sum(counts.values()) == _FAMILY_TAP_COUNT


def _generation_config(model: str) -> dict[str, Any]:
    if model in {"A", "B", "C", "D", "E"}:
        return {
            "carrier_frequency_hz": 3.5e9,
            "channel_bandwidth_hz": 100e6,
            "delay_spread_s": 100e-9,
            "spec_version": "19.2",
        }
    reference = _load_reference(_FIXED_REFERENCE, _FIXED_REFERENCE_SHA256)
    profile = next(row for row in reference["profiles"] if row["profile_model"] == model)
    minimum_bandwidth = profile["minimum_channel_bandwidth_hz_exclusive"]
    return {
        "carrier_frequency_hz": 28e9 if model in {"A10", "C60", "D10"} else 3.5e9,
        "channel_bandwidth_hz": float(minimum_bandwidth) + 1.0
        if minimum_bandwidth is not None
        else 100e6,
        "spec_version": "19.2",
    }


def _expected_native_tap_count(model: str) -> int:
    if model in _SCALABLE_NATIVE_TAP_COUNTS:
        return _SCALABLE_NATIVE_TAP_COUNTS[model]
    reference = _load_reference(_FIXED_REFERENCE, _FIXED_REFERENCE_SHA256)
    profile = next(row for row in reference["profiles"] if row["profile_model"] == model)
    return int(profile["tap_count"])


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
def test_tdl_rayleigh_tap_power_cdfs_match_table_scaled_exponential_family(model: str) -> None:
    """Check each ETSI Rayleigh-only tap against its table-scaled Exp(1) power law."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sionna.phy")
    expected_powers = (
        scalable_tdl_rayleigh_tap_power_fractions(model)
        if model in {"A", "B", "C", "D", "E"}
        else fixed_tdl_rayleigh_tap_power_fractions(model)
    )
    channel = generate_tdl_frequency_response(
        model=model,
        seed=_SEED_START + _MODELS.index(model),
        carrier_frequency_hz=_generation_config(model)["carrier_frequency_hz"],
        subcarrier_spacing_hz=30e3,
        num_subcarriers=24,
        channel_bandwidth_hz=_generation_config(model)["channel_bandwidth_hz"],
        batch_size=_DRAW_COUNT,
        delay_spread_s=100e-9 if model in {"A", "B", "C", "D", "E"} else None,
        spec_version="19.2",
        num_time_steps=1,
        device="cpu",
    )
    expected_source = (
        f"TR 38.901:19.2:TDL-{model}.json"
        if model in {"A", "B", "C", "D", "E"}
        else f"TS 38.101-4:19.2.2:TDL-{model}.json"
    )
    expected_table_hash = channel.metadata[
        "profile_parameter_table_sha256"
    ]  # implementation provenance; ETSI bytes are the independent oracle
    assert channel.metadata["sionna_version"] == "2.2.0"
    assert channel.metadata["profile_parameter_table_source"] == expected_source
    assert len(expected_table_hash) == 64
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["num_profile_taps"] == _expected_native_tap_count(model)
    assert tuple(channel.h_path.shape) == (
        _DRAW_COUNT,
        1,
        1,
        1,
        1,
        _expected_native_tap_count(model),
        1,
    )

    path_samples = channel.h_path[:, 0, 0, 0, 0, :, 0].to(torch.complex128).cpu().numpy()
    results = score_rayleigh_tap_marginal_family(
        path_samples,
        expected_powers,
        alpha=_ALPHA,
        maximum_simultaneous_cdf_error=_MAXIMUM_SIMULTANEOUS_CDF_ERROR,
    )
    assert len(results) == _EXPECTED_TAP_COUNTS[model]
    assert all(result.passed for result in results), (model, results)
    assert all(result.simultaneous_upper_bound <= 0.05 for result in results)
