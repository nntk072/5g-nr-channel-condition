from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest
from evaluation.g4_tdl_profile_reference import (
    fixed_tdl_pdp_reference,
    scalable_tdl_pdp_reference,
)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = ROOT / "docs/research/G4_SCALABLE_TDL_POWER_DELAY_REFERENCE_2026-10-26.json"
REFERENCE_SHA256 = "38b897029bd4cc6de49eece5096e69c75291eaf6dac45bebe04767ed24a4f0bd"
SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
FIXED_REFERENCE_PATH = ROOT / "docs/research/G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json"
FIXED_REFERENCE_SHA256 = "fe22560a71f4e5683ea6780f73e84466155b26f5f97cc4b63e3250c102fafda3"
FIXED_SOURCE_SHA256 = "b3c6a73c7a40219e7270c08c7a57c84cb46f6757b303a64ec768ba04a6079e81"


def _reference() -> dict[str, object]:
    payload = REFERENCE_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == REFERENCE_SHA256
    result = json.loads(payload)
    assert isinstance(result, dict)
    return result


def _fixed_reference() -> dict[str, object]:
    payload = FIXED_REFERENCE_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == FIXED_REFERENCE_SHA256
    result = json.loads(payload)
    assert isinstance(result, dict)
    return result


def test_scalable_tdl_power_delay_reference_is_complete_and_source_bound() -> None:
    reference = _reference()
    manifest = json.loads(
        (ROOT / "docs/research/G4_STANDARD_REFERENCE_MANIFEST_2026-10-26.json").read_text(
            encoding="utf-8"
        )
    )
    source = next(row for row in manifest["sources"] if row["source_id"] == "tr-138901-v19.2.0")
    assert source["sha256"] == reference["source_sha256"] == SOURCE_SHA256
    assert [row["profile_model"] for row in reference["profiles"]] == list("ABCDE")
    expected_counts = {"A": 23, "B": 23, "C": 24, "D": 14, "E": 15}
    for profile in reference["profiles"]:
        rows = profile["components"]
        assert len(rows) == expected_counts[profile["profile_model"]]
        assert [row["component_index"] for row in rows] == list(range(1, len(rows) + 1))
        assert all(row["fading_distribution"] == "Rayleigh" for row in rows[1:])
    schema = json.loads(
        (ROOT / "datasets/schema/g4-scalable-tdl-power-delay-reference.v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(reference)


def test_fixed_tdl_power_delay_reference_is_complete_and_source_bound() -> None:
    reference = _fixed_reference()
    manifest = json.loads(
        (ROOT / "docs/research/G4_STANDARD_REFERENCE_MANIFEST_2026-10-26.json").read_text(
            encoding="utf-8"
        )
    )
    source = next(row for row in manifest["sources"] if row["source_id"] == "ts-138101-4-v19.2.2")
    assert source["sha256"] == reference["source_sha256"] == FIXED_SOURCE_SHA256
    assert [row["profile_model"] for row in reference["profiles"]] == [
        "A10",
        "A30",
        "B100",
        "C60",
        "C300",
        "D10",
        "D30",
    ]
    schema = json.loads(
        (ROOT / "datasets/schema/g4-fixed-tdl-tap-profile-reference.v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(reference)


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_scalable_tdl_reference_returns_normalized_power_and_scaled_delay(model: str) -> None:
    powers, delays = scalable_tdl_pdp_reference(model, delay_spread_s=100e-9)

    assert len(powers) == len(delays) == {"A": 23, "B": 23, "C": 24, "D": 13, "E": 14}[model]
    assert sum(powers) == pytest.approx(1.0, abs=1e-14)
    assert all(power > 0.0 for power in powers)
    assert all(delay >= 0.0 for delay in delays)
    assert delays[0] == 0.0
    source_component_index = 2 if model in {"D", "E"} else 1
    assert delays[1] == pytest.approx(
        _reference()["profiles"][ord(model) - ord("A")]["components"][source_component_index][
            "normalized_delay"
        ]
        * 100e-9
    )
    if model in {"D", "E"}:
        components = _reference()["profiles"][ord(model) - ord("A")]["components"]
        linear_powers = [10.0 ** (row["power_db"] / 10.0) for row in components]
        assert powers[0] == pytest.approx(sum(linear_powers[:2]) / sum(linear_powers))


@pytest.mark.parametrize("model", ("A10", "A30", "B100", "C60", "C300", "D10", "D30"))
def test_fixed_tdl_reference_merges_components_by_normative_tap(model: str) -> None:
    reference = _fixed_reference()
    profile = next(row for row in reference["profiles"] if row["profile_model"] == model)
    components = profile["components"]
    power_by_tap: dict[int, float] = {}
    delay_by_tap: dict[int, float] = {}
    for row in components:
        tap = row["tap_index"]
        power_by_tap[tap] = power_by_tap.get(tap, 0.0) + 10.0 ** (row["power_db"] / 10.0)
        delay_by_tap[tap] = row["delay_ns"] * 1e-9
    total = sum(power_by_tap.values())
    expected_powers, expected_delays = fixed_tdl_pdp_reference(model)

    assert expected_powers == pytest.approx(
        [power_by_tap[tap] / total for tap in sorted(power_by_tap)], abs=1e-14
    )
    assert expected_delays == pytest.approx(
        [delay_by_tap[tap] for tap in sorted(delay_by_tap)], abs=1e-20
    )
    assert len(expected_powers) == profile["tap_count"]
    assert sum(expected_powers) == pytest.approx(1.0, abs=1e-14)
    if model in {"D10", "D30"}:
        assert len(components) == profile["tap_count"] + 1
        assert expected_powers[0] == pytest.approx(
            sum(10.0 ** (row["power_db"] / 10.0) for row in components[:2]) / total
        )


def test_fixed_tdl_reference_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="fixed TDL model must be"):
        fixed_tdl_pdp_reference("A")


@pytest.mark.parametrize(
    ("model", "delay_spread_s", "message"),
    (("F", 100e-9, "model must be A, B, C, D, or E"), ("A", True, "finite and positive")),
)
def test_scalable_tdl_reference_rejects_unsupported_inputs(
    model: str, delay_spread_s: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        scalable_tdl_pdp_reference(model, delay_spread_s=delay_spread_s)
