"""Schema and source-identity contract for the compact CDL scalar reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE = _ROOT / "docs/research/G4_CDL_PROFILE_SCALAR_REFERENCE_2026-10-26.json"
_SCHEMA = _ROOT / "datasets/schema/g4-cdl-profile-scalar-reference.v1.schema.json"
_MANIFEST = _ROOT / "docs/research/G4_STANDARD_REFERENCE_MANIFEST_2026-10-26.json"
_TAP_ANGLE_REFERENCE = _ROOT / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
_TAP_ANGLE_SCHEMA = _ROOT / "datasets/schema/g4-cdl-tap-angle-reference.v1.schema.json"
_TAP_ANGLE_REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
_PROFILE_SCALAR_REFERENCE_SHA256 = (
    "3b5920e7c12b0e36c8e09d1c7f14025cfbb23b6d805e1044124bfcf147185620"
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def test_cdl_scalar_reference_schema_and_manifest_identity_are_bound() -> None:
    assert hashlib.sha256(_REFERENCE.read_bytes()).hexdigest() == _PROFILE_SCALAR_REFERENCE_SHA256
    reference = _read_json(_REFERENCE)
    schema = _read_json(_SCHEMA)
    manifest = _read_json(_MANIFEST)

    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator(schema).validate(reference)

    source = next(row for row in manifest["sources"] if row["source_id"] == reference["source_id"])
    assert reference["source_sha256"] == source["sha256"]
    assert reference["source_url"] == source["source_url"]
    assert reference["standard"] == source["standard"]
    assert reference["version"] == source["version"]
    assert {row["profile_model"] for row in reference["profiles"]} == set("ABCDE")

    for row in reference["profiles"]:
        assert row["source_component_count"] == row["output_delay_cluster_count"] + int(row["los"])
        assert (row["profile_model"] in "DE") is row["los"]


def test_cdl_scalar_reference_schema_rejects_unknown_fields() -> None:
    reference = _read_json(_REFERENCE)
    reference["unreviewed_claim"] = "must not be silently accepted"
    schema = _read_json(_SCHEMA)

    jsonschema = pytest.importorskip("jsonschema")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(reference)


def test_cdl_per_cluster_reference_schema_and_source_identity_are_bound() -> None:
    assert hashlib.sha256(_TAP_ANGLE_REFERENCE.read_bytes()).hexdigest() == (
        _TAP_ANGLE_REFERENCE_SHA256
    )
    reference = _read_json(_TAP_ANGLE_REFERENCE)
    schema = _read_json(_TAP_ANGLE_SCHEMA)
    manifest = _read_json(_MANIFEST)

    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator(schema).validate(reference)

    source = next(row for row in manifest["sources"] if row["source_id"] == reference["source_id"])
    assert reference["source_sha256"] == source["sha256"]
    assert reference["source_url"] == source["source_url"]
    assert reference["standard"] == source["standard"]
    assert reference["version"] == source["version"]
    profiles = {row["profile_model"]: row for row in reference["profiles"]}
    assert set(profiles) == set("ABCDE")
    expected_counts = {"A": 23, "B": 23, "C": 24, "D": 13, "E": 14}
    for model, profile in profiles.items():
        assert profile["cluster_count"] == expected_counts[model]
        assert {row["cluster_index"] for row in profile["components"]} == set(
            range(1, profile["cluster_count"] + 1)
        )
        assert (model in "DE") is profile["los"]
        assert (
            len([row for row in profile["components"] if row["component"] == "diffuse"])
            == (profile["cluster_count"])
        )
        assert len([row for row in profile["components"] if row["component"] == "specular"]) == int(
            profile["los"]
        )
    for model in "DE":
        first_cluster = [row for row in profiles[model]["components"] if row["cluster_index"] == 1]
        assert {row["component"] for row in first_cluster} == {"diffuse", "specular"}
        assert {row["normalized_delay"] for row in first_cluster} == {0}


def test_cdl_3d_temporal_reference_bytes_are_bound_to_pinned_etsi_source() -> None:
    reference_bytes = _TAP_ANGLE_REFERENCE.read_bytes()
    assert hashlib.sha256(reference_bytes).hexdigest() == _TAP_ANGLE_REFERENCE_SHA256
    reference = json.loads(reference_bytes)
    manifest = _read_json(_MANIFEST)
    source = next(row for row in manifest["sources"] if row["source_id"] == "tr-138901-v19.2.0")

    assert reference["source_id"] == source["source_id"]
    assert reference["source_sha256"] == source["sha256"]
    assert {row["profile_model"] for row in reference["profiles"]} == set("ABCDE")


def test_cdl_per_cluster_reference_schema_rejects_unknown_fields() -> None:
    reference = _read_json(_TAP_ANGLE_REFERENCE)
    reference["unreviewed_claim"] = "must not be silently accepted"
    schema = _read_json(_TAP_ANGLE_SCHEMA)

    jsonschema = pytest.importorskip("jsonschema")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(reference)
