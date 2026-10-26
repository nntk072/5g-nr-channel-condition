"""Fixed TDL RMS-spread checks against ETSI Annex B scalar references."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_PATH = _ROOT / "docs/research/G4_FIXED_TDL_RMS_REFERENCE_2026-10-26.json"
_REFERENCE_SCHEMA_PATH = (
    _ROOT / "datasets/schema/g4-fixed-tdl-rms-reference.v1.schema.json"
)
_TAP_REFERENCE_PATH = _ROOT / "docs/research/G4_FIXED_TDL_TAP_SUMMARY_REFERENCE_2026-10-26.json"
_TAP_REFERENCE_SCHEMA_PATH = (
    _ROOT / "datasets/schema/g4-fixed-tdl-tap-summary-reference.v1.schema.json"
)
_TAP_PROFILE_REFERENCE_PATH = (
    _ROOT / "docs/research/G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json"
)
_TAP_PROFILE_REFERENCE_SCHEMA_PATH = (
    _ROOT / "datasets/schema/g4-fixed-tdl-tap-profile-reference.v1.schema.json"
)
_SOURCE_MANIFEST_PATH = _ROOT / "docs/research/G4_STANDARD_REFERENCE_MANIFEST_2026-10-26.json"


def _native_tdl_adapter():
    pytest.importorskip("torch")
    pytest.importorskip("sionna.phy.channel.tr38901")
    import integrations.sionna.tdl as tdl

    return tdl


def _load_reference() -> dict[str, Any]:
    return json.loads(_REFERENCE_PATH.read_text(encoding="utf-8"))


def _load_tap_reference() -> dict[str, Any]:
    return json.loads(_TAP_REFERENCE_PATH.read_text(encoding="utf-8"))


def _load_tap_profile_reference() -> dict[str, Any]:
    return json.loads(_TAP_PROFILE_REFERENCE_PATH.read_text(encoding="utf-8"))


def test_fixed_tdl_scalar_reference_is_bound_to_the_pinned_etsi_source() -> None:
    reference = _load_reference()
    source_manifest = json.loads(_SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    source = next(
        row for row in source_manifest["sources"] if row["source_id"] == reference["source_id"]
    )
    assert source["sha256"] == reference["source_sha256"]
    assert {
        row["profile_model"]: row["rms_delay_spread_ns"] for row in reference["profiles"]
    } == {
        "A10": 10.0,
        "A30": 30.0,
        "B100": 100.0,
        "C60": 60.0,
        "C300": 300.0,
        "D10": 10.0,
        "D30": 30.0,
    }
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(_REFERENCE_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(reference)


def test_fixed_tdl_tap_summary_reference_is_bound_to_the_pinned_etsi_source() -> None:
    reference = _load_tap_reference()
    source_manifest = json.loads(_SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    source = next(
        row for row in source_manifest["sources"] if row["source_id"] == reference["source_id"]
    )
    assert source["sha256"] == reference["source_sha256"]
    assert {
        row["profile_model"]: (
            row["has_los_component"],
            row["tap_count"],
            row["maximum_excess_delay_ns"],
            row["delay_resolution_ns"],
        )
        for row in reference["profiles"]
    } == {
        "A10": (False, 16, 96, 2),
        "A30": (False, 12, 290, 5),
        "B100": (False, 12, 480, 5),
        "C60": (False, 12, 520, 5),
        "C300": (False, 12, 2595, 5),
        "D10": (True, 10, 126, 2),
        "D30": (True, 10, 375, 5),
    }
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(_TAP_REFERENCE_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(reference)


def test_fixed_tdl_per_tap_reference_is_bound_to_the_pinned_etsi_source() -> None:
    reference = _load_tap_profile_reference()
    source_manifest = json.loads(_SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    source = next(
        row for row in source_manifest["sources"] if row["source_id"] == reference["source_id"]
    )
    assert source["sha256"] == reference["source_sha256"]
    profiles = {row["profile_model"]: row for row in reference["profiles"]}
    assert set(profiles) == {"A10", "A30", "B100", "C60", "C300", "D10", "D30"}
    for model, profile in profiles.items():
        components = profile["components"]
        assert [row["component_index"] for row in components] == list(
            range(1, len(components) + 1)
        )
        assert len({row["tap_index"] for row in components}) == profile["tap_count"]
        assert sorted({row["tap_index"] for row in components}) == list(
            range(1, profile["tap_count"] + 1)
        ), model
    for model in ("D10", "D30"):
        first_cluster = [
            row for row in profiles[model]["components"] if row["tap_index"] == 1
        ]
        assert [(row["delay_ns"], row["fading_distribution"]) for row in first_cluster] == [
            (0, "LOS" if model == "D10" else "LOS path"),
            (0, "Rayleigh"),
        ]
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(_TAP_PROFILE_REFERENCE_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(reference)


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("model", "carrier_frequency_hz", "channel_bandwidth_hz"),
    [
        ("A10", 28e9, 400e6),
        ("A30", 3.5e9, 100e6),
        ("B100", 3.5e9, 100e6),
        ("C60", 28e9, 100e6),
        ("C300", 3.5e9, 100e6),
        ("D10", 28e9, 400e6),
        ("D30", 3.5e9, 100e6),
    ],
)
def test_fixed_tdl_reported_rms_delay_spread_matches_etsi_scalar(
    model: str, carrier_frequency_hz: float, channel_bandwidth_hz: float
) -> None:
    tdl = _native_tdl_adapter()
    reference = _load_reference()
    expected_ns = next(
        row["rms_delay_spread_ns"]
        for row in reference["profiles"]
        if row["profile_model"] == model
    )
    channel = tdl.generate_tdl_frequency_response(
        model=model,
        seed=38_101,
        carrier_frequency_hz=carrier_frequency_hz,
        subcarrier_spacing_hz=30e3,
        num_subcarriers=24,
        channel_bandwidth_hz=channel_bandwidth_hz,
        spec_version="19.2",
        num_time_steps=1,
        device="cpu",
    )

    metadata = channel.metadata
    tap_reference = next(
        row for row in _load_tap_reference()["profiles"] if row["profile_model"] == model
    )
    minimum_bandwidth_hz = tap_reference["minimum_channel_bandwidth_hz_exclusive"]
    if minimum_bandwidth_hz is not None:
        assert channel_bandwidth_hz > minimum_bandwidth_hz
    nominal_delays_ns = [float(delay) * 1e9 for delay in metadata["nominal_tap_delays_s"]]
    mean_tap_powers = [float(power) for power in metadata["mean_tap_powers"]]
    resolution_ns = tap_reference["delay_resolution_ns"]
    # TDL-D's specular and diffuse first components share one zero-delay cluster.
    assert len(nominal_delays_ns) == tap_reference["tap_count"]
    assert len(mean_tap_powers) == len(nominal_delays_ns)
    assert metadata["num_profile_taps"] == tap_reference["tap_count"]
    assert metadata["profile_los_component"] is tap_reference["has_los_component"]
    assert math.isclose(
        max(nominal_delays_ns),
        tap_reference["maximum_excess_delay_ns"],
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert all(
        math.isclose(delay / resolution_ns, round(delay / resolution_ns), rel_tol=0.0, abs_tol=1e-7)
        for delay in nominal_delays_ns
    )
    assert math.gcd(*(round(delay) for delay in nominal_delays_ns)) == resolution_ns
    reported_ns = float(metadata["profile_rms_delay_spread_s"]) * 1e9
    total_mean_power = math.fsum(mean_tap_powers)
    weighted_mean_delay_s = math.fsum(
        power * delay_ns * 1e-9
        for power, delay_ns in zip(mean_tap_powers, nominal_delays_ns, strict=True)
    ) / total_mean_power
    recomputed_rms_s = math.sqrt(
        math.fsum(
            power * (delay_ns * 1e-9 - weighted_mean_delay_s) ** 2
            for power, delay_ns in zip(mean_tap_powers, nominal_delays_ns, strict=True)
        )
        / total_mean_power
    )
    # The standard tabulates tap powers to 0.1 dB; this is the resulting
    # multiplicative rounding bound for a power-weighted RMS statistic.
    rms_rounding_relative_tolerance = 10 ** (0.05 / 10) - 1
    assert math.isclose(
        recomputed_rms_s * 1e9,
        reported_ns,
        rel_tol=rms_rounding_relative_tolerance,
        abs_tol=1e-9,
    )
    assert metadata["profile_proxy"] == model
    assert metadata["fixed_profile_spec_version"] == reference["version"]
    assert metadata["profile_parameter_table_source"].startswith(
        f"{reference['standard'].removeprefix('3GPP ')}:{reference['version']}"
    )
    assert math.isclose(reported_ns, expected_ns, rel_tol=0.0, abs_tol=1e-9)


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", ["A10", "A30", "B100", "C60", "C300", "D10", "D30"])
def test_fixed_tdl_resolved_delay_clusters_match_each_etsi_tap(
    model: str,
) -> None:
    tdl = _native_tdl_adapter()
    profile = next(
        row
        for row in _load_tap_profile_reference()["profiles"]
        if row["profile_model"] == model
    )
    bandwidth_hz = (
        float(profile["minimum_channel_bandwidth_hz_exclusive"]) + 1.0
        if profile["minimum_channel_bandwidth_hz_exclusive"] is not None
        else 100e6
    )
    carrier_hz = 28e9 if model in {"A10", "C60", "D10"} else 3.5e9
    channel = tdl.generate_tdl_frequency_response(
        model=model,
        seed=38_101,
        carrier_frequency_hz=carrier_hz,
        subcarrier_spacing_hz=30e3,
        num_subcarriers=24,
        channel_bandwidth_hz=bandwidth_hz,
        spec_version="19.2",
        num_time_steps=1,
        device="cpu",
    )
    expected_by_tap: dict[int, list[dict[str, Any]]] = {}
    for component in profile["components"]:
        expected_by_tap.setdefault(component["tap_index"], []).append(component)
    expected_delays_ns = [
        float(expected_by_tap[tap][0]["delay_ns"])
        for tap in range(1, profile["tap_count"] + 1)
    ]
    actual_delays_ns = [float(delay) * 1e9 for delay in channel.metadata["nominal_tap_delays_s"]]
    assert actual_delays_ns == pytest.approx(expected_delays_ns, abs=1e-6, rel=0.0)

    # The ETSI table gives separate LOS and diffuse powers for TDL-D tap 1.
    # Sum those components in linear power before comparing to resolved clusters.
    expected_cluster_power_db = [
        10.0
        * math.log10(
            math.fsum(
                10.0 ** (float(component["power_db"]) / 10.0)
                for component in expected_by_tap[tap]
            )
        )
        for tap in range(1, profile["tap_count"] + 1)
    ]
    expected_peak_db = max(expected_cluster_power_db)
    expected_relative_power_db = [power_db - expected_peak_db for power_db in expected_cluster_power_db]
    actual_powers = [float(power) for power in channel.metadata["mean_tap_powers"]]
    actual_relative_power_db = [10.0 * math.log10(power / max(actual_powers)) for power in actual_powers]
    # Annex B prints component powers to 0.1 dB; this bound is only a table
    # rounding allowance and is not a statistical or conformance tolerance.
    assert actual_relative_power_db == pytest.approx(
        expected_relative_power_db, abs=0.11, rel=0.0
    )
