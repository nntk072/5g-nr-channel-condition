from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import evaluation.covariance.tdl_profile_reference as reference_module
import jsonschema
import pytest
from evaluation.covariance.tdl_profile_reference import (
    fixed_tdl_pdp_reference,
    scalable_tdl_pdp_reference,
)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = ROOT / "config/reference/SCALABLE_TDL_POWER_DELAY_REFERENCE_2026-10-26.json"
REFERENCE_SHA256 = "38b897029bd4cc6de49eece5096e69c75291eaf6dac45bebe04767ed24a4f0bd"
SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
FIXED_REFERENCE_PATH = ROOT / "config/reference/FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json"
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
        (ROOT / "config/reference/STANDARD_REFERENCE_MANIFEST_2026-10-26.json").read_text(
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
        (ROOT / "datasets/schema/scalable-tdl-power-delay-reference.v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(reference)


def test_fixed_tdl_power_delay_reference_is_complete_and_source_bound() -> None:
    reference = _fixed_reference()
    manifest = json.loads(
        (ROOT / "config/reference/STANDARD_REFERENCE_MANIFEST_2026-10-26.json").read_text(
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
        (ROOT / "datasets/schema/fixed-tdl-tap-profile-reference.v1.schema.json").read_text(
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


def _scalable_fixture() -> dict[str, object]:
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def _fixed_fixture() -> dict[str, object]:
    return json.loads(FIXED_REFERENCE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("payload", "digest", "message"),
    [
        (b"{}", "wrong", "frozen digest"),
        (b"[]", hashlib.sha256(b"[]").hexdigest(), "root must be an object"),
    ],
)
def test_pinned_json_rejects_digest_or_root_mismatch(tmp_path, payload, digest, message):
    path = tmp_path / "reference.json"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match=message):
        reference_module._read_pinned_json(path, digest, label="fixture")


@pytest.mark.parametrize(
    ("fixture_factory", "field", "value", "message", "model"),
    [
        (_scalable_fixture, "schema", "wrong", "invalid normative identity", "A"),
        (_scalable_fixture, "profiles", [], "exactly one profile", "A"),
        (_fixed_fixture, "schema", "wrong", "invalid normative identity", "A10"),
        (_fixed_fixture, "profiles", None, "must contain a profile list", "A10"),
        (_fixed_fixture, "profiles", [], "exactly one profile", "A10"),
    ],
    ids=[
        "scalable-identity",
        "scalable-profile",
        "fixed-identity",
        "fixed-profiles-invalid",
        "fixed-profile-missing",
    ],
)
def test_pinned_reference_rejects_mutated_authority(
    fixture_factory, field, value, message, model, monkeypatch
):
    fixture = fixture_factory()
    fixture[field] = value
    monkeypatch.setattr(reference_module, "_read_pinned_json", lambda *_args, **_kwargs: fixture)

    with pytest.raises(ValueError, match=message):
        if fixture_factory is _scalable_fixture:
            scalable_tdl_pdp_reference(model, delay_spread_s=1e-6)
        else:
            fixed_tdl_pdp_reference(model)


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (None, "must contain component rows"),
        ({"components": ()}, "must contain component rows"),
        ({"components": []}, "cannot be empty"),
        ({"components": [{"component_index": 2}]}, "ordered contiguous index"),
        ({"components": [{"component_index": 1, "power_db": True}]}, "finite real dB"),
        ({"components": [{"component_index": 1, "power_db": "bad"}]}, "finite real dB"),
        ({"components": [{"component_index": 1, "power_db": math.inf}]}, "finite real dB"),
        (
            {"components": [{"component_index": 1, "power_db": -4000, "normalized_delay": 0}]},
            "not representable",
        ),
        (
            {"components": [{"component_index": 1, "power_db": 0, "normalized_delay": True}]},
            "finite and nonnegative",
        ),
        (
            {"components": [{"component_index": 1, "power_db": 0, "normalized_delay": -1}]},
            "finite and nonnegative",
        ),
        (
            {"components": [{"component_index": 1, "power_db": 0, "normalized_delay": math.inf}]},
            "finite and nonnegative",
        ),
    ],
)
def test_component_profile_rejects_malformed_rows(profile, message):
    with pytest.raises(ValueError, match=message):
        reference_module._component_profile(profile, name="fixture")


@pytest.mark.parametrize(
    "field,value",
    [
        ("tap_index", True),
        ("tap_index", 0),
        ("delay_ns", True),
        ("delay_ns", "zero"),
        ("delay_ns", math.inf),
        ("delay_ns", -1),
        ("power_db", True),
        ("power_db", "zero"),
        ("power_db", math.inf),
        ("fading_distribution", "unknown"),
    ],
)
def test_fixed_reference_rejects_invalid_component_fields(field, value, monkeypatch):
    fixture = _fixed_fixture()
    profile = next(row for row in fixture["profiles"] if row["profile_model"] == "A10")
    profile["components"][0][field] = value
    monkeypatch.setattr(reference_module, "_read_pinned_json", lambda *_args, **_kwargs: fixture)

    with pytest.raises(ValueError, match="component fields are invalid"):
        fixed_tdl_pdp_reference("A10")


def test_fixed_reference_rejects_unrepresentable_component_power(monkeypatch):
    fixture = _fixed_fixture()
    profile = next(row for row in fixture["profiles"] if row["profile_model"] == "A10")
    profile["components"][0]["power_db"] = -4000
    monkeypatch.setattr(reference_module, "_read_pinned_json", lambda *_args, **_kwargs: fixture)

    with pytest.raises(ValueError, match="power is not representable"):
        fixed_tdl_pdp_reference("A10")


@pytest.mark.parametrize(
    "components",
    [
        [
            {
                "component_index": 1,
                "power_db": 0,
                "normalized_delay": 0,
                "fading_distribution": "LOS",
            }
        ],
        [
            {
                "component_index": 1,
                "power_db": 0,
                "normalized_delay": 0,
                "fading_distribution": "LOS",
            },
            {
                "component_index": 2,
                "power_db": 0,
                "normalized_delay": 0,
                "fading_distribution": "Rician",
            },
        ],
        [
            {
                "component_index": 1,
                "power_db": 0,
                "normalized_delay": 0,
                "fading_distribution": "LOS",
            },
            {
                "component_index": 2,
                "power_db": 0,
                "normalized_delay": 1,
                "fading_distribution": "Rayleigh",
            },
        ],
    ],
    ids=["missing-partner", "wrong-distribution", "different-delay"],
)
def test_scalable_los_reference_requires_paired_first_diffuse_component(components, monkeypatch):
    fixture = _scalable_fixture()
    profile = next(row for row in fixture["profiles"] if row["profile_model"] == "A")
    profile["components"] = components
    monkeypatch.setattr(reference_module, "_read_pinned_json", lambda *_args, **_kwargs: fixture)

    with pytest.raises(ValueError, match="no paired first diffuse component"):
        scalable_tdl_pdp_reference("A", delay_spread_s=1e-6)


@pytest.mark.parametrize("components", [None, []], ids=["non-list", "empty"])
def test_fixed_reference_requires_component_rows(components, monkeypatch):
    fixture = _fixed_fixture()
    profile = next(row for row in fixture["profiles"] if row["profile_model"] == "A10")
    profile["components"] = components
    monkeypatch.setattr(reference_module, "_read_pinned_json", lambda *_args, **_kwargs: fixture)

    with pytest.raises(ValueError, match="must contain component rows"):
        fixed_tdl_pdp_reference("A10")


def test_fixed_reference_rejects_component_order_delay_and_roster_changes(monkeypatch):
    mutations = (
        (
            lambda profile: profile["components"][0].update(component_index=2),
            "ordered and contiguous",
        ),
        (
            lambda profile: profile["components"][1].update(tap_index=1, delay_ns=999),
            "different delays",
        ),
        (lambda profile: profile.update(tap_count=True), "roster is inconsistent"),
    )
    for mutate, message in mutations:
        fixture = _fixed_fixture()
        profile = next(row for row in fixture["profiles"] if row["profile_model"] == "A10")
        mutate(profile)
        monkeypatch.setattr(
            reference_module,
            "_read_pinned_json",
            lambda *_args, _fixture=fixture, **_kwargs: _fixture,
        )
        with pytest.raises(ValueError, match=message):
            fixed_tdl_pdp_reference("A10")
