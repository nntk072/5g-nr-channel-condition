"""Lineage status claims and evidence audits: canonical claims, byte budgets and no-follow file reads."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from datasets.lineage import (
    LineageStatusClaim,
    audit_lineage_evidence_files,
    audit_lineage_evidence_payloads,
    validate_lineage_claims,
)

DIGEST_A = "a" * 64
PAYLOAD = b"regression report bytes"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


def _synthetic_claim(**overrides) -> LineageStatusClaim:
    values = {
        "status": "synthetic_debug",
        "scope": "unit-test-scope",
        "evidence_sha256": {"regression_report_sha256": PAYLOAD_SHA},
    }
    values.update(overrides)
    return LineageStatusClaim(**values)


def test_claim_round_trips_through_its_canonical_schema() -> None:
    claim = _synthetic_claim()
    payload = claim.to_dict()
    assert payload == {
        "schema": "umcd.lineage_status_claim.v1",
        "status": "synthetic_debug",
        "scope": "unit-test-scope",
        "evidence_sha256": {"regression_report_sha256": PAYLOAD_SHA},
    }
    assert LineageStatusClaim.from_dict(payload) == claim


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"status": "scientific_release"}, "status must be one of", id="status"),
        pytest.param(
            {"scope": " padded "}, "scope must be a non-empty canonical string", id="scope"
        ),
        pytest.param({"scope": ""}, "scope must be a non-empty canonical string", id="empty-scope"),
        pytest.param({"evidence_sha256": ["x"]}, "string-keyed mapping", id="evidence-type"),
        pytest.param(
            {"evidence_sha256": {" ": PAYLOAD_SHA}}, "string-keyed mapping", id="blank-key"
        ),
        pytest.param(
            {"evidence_sha256": {"regression_report_sha256": "A" * 64}},
            "lowercase SHA256 digests",
            id="uppercase-digest",
        ),
        pytest.param(
            {
                "status": "native_verified",
                "evidence_sha256": {"native_run_manifest_sha256": DIGEST_A},
            },
            "missing required evidence hashes",
            id="missing-required",
        ),
    ],
)
def test_claim_constructor_rejects_malformed_claims(overrides, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _synthetic_claim(**overrides)


def test_claim_evidence_map_is_read_only_after_construction() -> None:
    claim = _synthetic_claim()
    with pytest.raises(TypeError):
        claim.evidence_sha256["regression_report_sha256"] = DIGEST_A  # type: ignore[index]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        pytest.param(
            {"schema": "umcd.lineage_status_claim.v1", "status": "proposed", "scope": "s"},
            "fields do not match schema v1",
            id="missing-field",
        ),
        pytest.param(
            {
                "schema": "umcd.lineage_status_claim.v0",
                "status": "proposed",
                "scope": "s",
                "evidence_sha256": {},
            },
            "unsupported lineage status claim schema",
            id="schema",
        ),
        pytest.param(
            {
                "schema": "umcd.lineage_status_claim.v1",
                "status": "proposed",
                "scope": "s",
                "evidence_sha256": {"regression_report_sha256": PAYLOAD_SHA},
                "extra": 1,
            },
            "fields do not match schema v1",
            id="extra-field",
        ),
    ],
)
def test_claim_parser_rejects_unknown_or_missing_fields(payload, message) -> None:
    with pytest.raises(ValueError, match=message):
        LineageStatusClaim.from_dict(payload)


def test_claim_roster_must_be_a_sequence_of_claims_with_unique_identities() -> None:
    with pytest.raises(TypeError, match="sequence of LineageStatusClaim values"):
        validate_lineage_claims("not-claims")
    with pytest.raises(TypeError, match="sequence of LineageStatusClaim values"):
        validate_lineage_claims([{"status": "proposed"}])
    with pytest.raises(ValueError, match="unique status/scope pairs"):
        validate_lineage_claims([_synthetic_claim(), _synthetic_claim()])
    assert len(validate_lineage_claims([_synthetic_claim(scope="other")])) == 1


def test_payload_audit_verifies_exact_bytes_and_budget() -> None:
    claims = [_synthetic_claim()]
    report = audit_lineage_evidence_payloads(
        claims, {PAYLOAD_SHA: PAYLOAD}, provenance_record_sha256=DIGEST_A
    )
    assert report["status"] == "EVIDENCE_BYTES_VERIFIED"
    assert report["scientific_status"] == "NOT_QUALIFIED"
    assert report["verified_evidence_sha256"] == [PAYLOAD_SHA]
    assert report["verified_evidence_bytes"] == len(PAYLOAD)
    assert report["provenance_record_sha256"] == DIGEST_A

    with pytest.raises(ValueError, match="exceed max_total_bytes"):
        audit_lineage_evidence_payloads(claims, {PAYLOAD_SHA: PAYLOAD}, max_total_bytes=3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"provenance_record_sha256": "XYZ"}, "must be lowercase SHA256", id="prov"),
        pytest.param({"max_total_bytes": True}, "max_total_bytes must be an integer", id="bool"),
        pytest.param({"max_total_bytes": 0}, "max_total_bytes must be positive", id="zero"),
    ],
)
def test_payload_audit_rejects_bad_options(kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        audit_lineage_evidence_payloads([_synthetic_claim()], {PAYLOAD_SHA: PAYLOAD}, **kwargs)


def test_payload_audit_rejects_rosters_types_and_digests() -> None:
    claims = [_synthetic_claim()]
    with pytest.raises(TypeError, match="digest-keyed mapping"):
        audit_lineage_evidence_payloads(claims, [PAYLOAD])
    with pytest.raises(ValueError, match="roster does not match"):
        audit_lineage_evidence_payloads(claims, {})
    with pytest.raises(ValueError, match="roster does not match"):
        audit_lineage_evidence_payloads(claims, {"A" * 64: PAYLOAD})
    with pytest.raises(TypeError, match="immutable bytes"):
        audit_lineage_evidence_payloads(claims, {PAYLOAD_SHA: bytearray(PAYLOAD)})
    with pytest.raises(ValueError, match="does not match digest"):
        audit_lineage_evidence_payloads(claims, {PAYLOAD_SHA: b"tampered"})


def test_empty_claim_roster_reports_no_evidence_required() -> None:
    report = audit_lineage_evidence_payloads([], {})
    assert report["status"] == "NO_EVIDENCE_REQUIRED"
    assert report["verified_evidence_sha256"] == []
    assert report["verified_evidence_bytes"] == 0
    assert len(report["audit_sha256"]) == 64


def _file_audit(root: Path, relative: str, **kwargs):
    evidence = {PAYLOAD_SHA: relative}
    return audit_lineage_evidence_files(
        [_synthetic_claim()], evidence, evidence_root=root, **kwargs
    )


def test_file_audit_streams_nested_evidence_beneath_the_root(tmp_path: Path) -> None:
    nested = tmp_path / "reports" / "2026"
    nested.mkdir(parents=True)
    (nested / "regression.json").write_bytes(PAYLOAD)
    report = _file_audit(tmp_path, "reports/2026/regression.json")
    assert report["status"] == "EVIDENCE_BYTES_VERIFIED"
    assert report["verified_evidence_bytes"] == len(PAYLOAD)


@pytest.mark.parametrize(
    "relative",
    ["../escape.json", "/absolute.json", "reports//double.json", "reports\\win.json", "./dot.json"],
    ids=["parent", "absolute", "empty-component", "backslash", "dot"],
)
def test_file_audit_refuses_noncanonical_or_escaping_paths(tmp_path: Path, relative) -> None:
    with pytest.raises(ValueError, match="canonical relative POSIX|stay beneath"):
        _file_audit(tmp_path, relative)


def test_file_audit_rejects_symlinked_evidence_and_directory_components(tmp_path: Path) -> None:
    target = tmp_path / "real.json"
    target.write_bytes(PAYLOAD)
    (tmp_path / "linked.json").symlink_to(target)
    with pytest.raises(OSError):
        _file_audit(tmp_path, "linked.json")

    (tmp_path / "folder").mkdir()
    (tmp_path / "folder" / "regression.json").write_bytes(PAYLOAD)
    (tmp_path / "linkdir").symlink_to(tmp_path / "folder", target_is_directory=True)
    with pytest.raises(OSError):
        _file_audit(tmp_path, "linkdir/regression.json")


def test_file_audit_requires_regular_single_link_files(tmp_path: Path) -> None:
    (tmp_path / "folder.json").mkdir()
    with pytest.raises(ValueError, match="stable single-link regular files"):
        _file_audit(tmp_path, "folder.json")

    original = tmp_path / "original.json"
    original.write_bytes(PAYLOAD)
    os.link(original, tmp_path / "alias.json")
    with pytest.raises(ValueError, match="stable single-link regular files"):
        _file_audit(tmp_path, "original.json")


def test_file_audit_enforces_content_digest_and_byte_budget(tmp_path: Path) -> None:
    (tmp_path / "tampered.json").write_bytes(b"not the reported bytes")
    with pytest.raises(ValueError, match="does not match digest"):
        _file_audit(tmp_path, "tampered.json")

    (tmp_path / "large.json").write_bytes(PAYLOAD)
    with pytest.raises(ValueError, match="exceed max_total_bytes"):
        _file_audit(tmp_path, "large.json", max_total_bytes=4)


def test_file_audit_requires_roster_root_and_option_agreement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="roster does not match"):
        audit_lineage_evidence_files([_synthetic_claim()], {}, evidence_root=tmp_path)
    with pytest.raises(TypeError, match="digest-keyed mapping"):
        audit_lineage_evidence_files([_synthetic_claim()], [PAYLOAD_SHA], evidence_root=tmp_path)
    with pytest.raises(ValueError, match="lowercase SHA256"):
        _file_audit(tmp_path, "x.json", provenance_record_sha256="bad")
    with pytest.raises(ValueError, match="max_total_bytes must be positive"):
        _file_audit(tmp_path, "x.json", max_total_bytes=0)
    with pytest.raises(TypeError, match="max_total_bytes must be an integer"):
        _file_audit(tmp_path, "x.json", max_total_bytes=1.0)
    empty = audit_lineage_evidence_files([], {}, evidence_root=tmp_path)
    assert empty["status"] == "NO_EVIDENCE_REQUIRED"


def test_file_audit_root_must_be_a_directory(tmp_path: Path) -> None:
    plain = tmp_path / "not-a-dir"
    plain.write_bytes(PAYLOAD)
    with pytest.raises(NotADirectoryError):
        _file_audit(plain, "anything.json")


def test_file_audit_requires_no_follow_platform_support(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "x.json").write_bytes(PAYLOAD)
    monkeypatch.delattr(os, "O_NOFOLLOW")
    with pytest.raises(ValueError, match="platform lacks safe no-follow"):
        _file_audit(tmp_path, "x.json")


def test_file_audit_walks_deeply_nested_evidence_directories(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "report.json").write_bytes(PAYLOAD)
    report = _file_audit(tmp_path, "a/b/c/report.json")
    assert report["verified_evidence_sha256"] == [PAYLOAD_SHA]


def test_file_audit_rejects_a_file_that_grows_beyond_its_budget_while_read(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "grows.json").write_bytes(PAYLOAD)
    real_read = os.read

    def oversized_read(descriptor, size):
        # Simulate a writer appending after the size check but before the read loop ends.
        return real_read(descriptor, size) + b"x" * len(PAYLOAD)

    monkeypatch.setattr(os, "read", oversized_read)
    with pytest.raises(ValueError, match="exceed max_total_bytes"):
        _file_audit(tmp_path, "grows.json", max_total_bytes=len(PAYLOAD))


def test_file_audit_refuses_a_file_whose_identity_changes_during_hashing(
    tmp_path: Path, monkeypatch
) -> None:
    from datasets import lineage

    (tmp_path / "moving.json").write_bytes(PAYLOAD)
    real_identity = lineage._file_identity
    calls = {"count": 0}

    def changing_identity(metadata):
        calls["count"] += 1
        identity = real_identity(metadata)
        # Calls 1-2 bind the pre-read identity; call 3 is the post-read descriptor identity.
        return (0, *identity[1:]) if calls["count"] == 3 else identity

    monkeypatch.setattr(lineage, "_file_identity", changing_identity)
    with pytest.raises(ValueError, match="changed while it was being read"):
        _file_audit(tmp_path, "moving.json")
