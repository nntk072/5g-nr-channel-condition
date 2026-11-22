from __future__ import annotations

import json

import jsonschema
import numpy as np
import pytest
from evaluation.cdl_four_element_covariance import (
    audit_cdl_spatial_covariance_claims,
    cdl_spatial_covariance_claims,
    cdl_spatial_covariance_reference,
)
from scripts.run_cdl_e_fresh_seed_replication import (
    SCHEMA,
    _report,
    _verify_preregistration,
    audit_replication_report,
)


def _balanced_fixture() -> np.ndarray:
    basis = np.eye(4, dtype=np.complex128)
    return np.tile(basis, (2048, 1))


def test_independent_cdl_reference_replays_pinned_etsi_fixture() -> None:
    reference, digest = cdl_spatial_covariance_reference("E")
    assert digest == "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
    assert reference.shape == (4, 4)
    np.testing.assert_allclose(reference, reference.conj().T, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.diag(reference).real, np.ones(4), rtol=0, atol=1e-12)
    assert np.linalg.eigvalsh(reference).min() >= -1e-12


def test_preregistered_seed_and_protocol_digests_are_current() -> None:
    audit, digests = _verify_preregistration()
    assert audit["status"] == "CLEAR_WITHIN_SCANNED_JSON_CORPUS"
    assert audit["reserved_seeds"] == [2_823_000]
    assert audit["scanned_file_count"] == 2302
    assert audit["global_exclusivity_established"] is False
    assert len(digests["protocol_sha256"]) == 64
    assert len(digests["seed_audit_sha256"]) == 64


def test_covariance_claim_report_recomputes_frozen_family_and_summary() -> None:
    samples = _balanced_fixture()
    empirical = np.einsum("ni,nj->ij", samples, samples.conj()) / len(samples)
    claims = cdl_spatial_covariance_claims(samples, empirical)
    summary = audit_cdl_spatial_covariance_claims(claims)

    assert len(claims) == 16
    assert summary["claims_passed"] == 16
    assert summary["claims_failed"] == 0
    assert summary["status"] == "PASS_WITHIN_BOUNDED_PROTOCOL_SCOPE"
    assert summary["maximum_half_width"] < 0.05

    audit, preregistration_digests = _verify_preregistration()
    report = _report(
        claims,
        audit=audit,
        preregistration_digests=preregistration_digests,
        source_digests={
            "profile_table_sha256": (
                "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480"
            ),
            "reference_sha256": (
                "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
            ),
            "evaluator_sha256": "a" * 64,
            "native_test_sha256": "b" * 64,
        },
        command=["uv", "run", "scripts.runner", "--output"],
        revision="c" * 40,
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(
        report
    )
    assert audit_replication_report(report)["status"] == "PASS_WITHIN_BOUNDED_PROTOCOL_SCOPE"

    incomplete = _report(
        [],
        audit=audit,
        preregistration_digests=preregistration_digests,
        source_digests={
            "profile_table_sha256": (
                "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480"
            ),
            "reference_sha256": (
                "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
            ),
            "evaluator_sha256": "a" * 64,
            "native_test_sha256": "b" * 64,
        },
        command=["uv", "run", "scripts.runner", "--output"],
        revision="c" * 40,
        error="RuntimeError: synthetic interrupted draw",
        profile_table_verified=False,
    )
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(
        incomplete
    )
    assert audit_replication_report(incomplete)["status"] == "INCOMPLETE"


def test_covariance_claim_audit_retains_a_miss_as_failure() -> None:
    samples = _balanced_fixture()
    empirical = np.einsum("ni,nj->ij", samples, samples.conj()) / len(samples)
    reference = empirical.copy()
    reference[0, 1] = 0.2 + 0.0j
    claims = cdl_spatial_covariance_claims(samples, reference)
    summary = audit_cdl_spatial_covariance_claims(claims)

    assert summary["claims_failed"] == 1
    assert summary["status"] == "FAIL"

    claims[0]["interval"][0] -= 1e-3
    with pytest.raises(ValueError, match="arithmetic is inconsistent"):
        audit_cdl_spatial_covariance_claims(claims)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("estimate", True, "must be finite and real"),
        ("reference", float("nan"), "must be finite and real"),
        ("standard_error", -1.0, "must be nonnegative"),
        ("half_width", float("inf"), "must be finite and real"),
        ("absolute_error", -1.0, "must be nonnegative"),
    ],
)
def test_covariance_claim_audit_rejects_malformed_numeric_claims(field, value, message):
    samples = _balanced_fixture()
    empirical = np.einsum("ni,nj->ij", samples, samples.conj()) / len(samples)
    claims = cdl_spatial_covariance_claims(samples, empirical)
    claims[0][field] = value
    with pytest.raises(ValueError, match=message):
        audit_cdl_spatial_covariance_claims(claims)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("familywise_alpha", True, "familywise_alpha must be a finite value"),
        ("familywise_alpha", 0.0, "familywise_alpha must be a finite value"),
        ("familywise_alpha", float("nan"), "familywise_alpha must be a finite value"),
        ("maximum_half_width", False, "maximum_half_width must be finite and positive"),
        ("maximum_half_width", 0.0, "maximum_half_width must be finite and positive"),
    ],
)
def test_covariance_claim_audit_rejects_invalid_criteria(field, value, message):
    samples = _balanced_fixture()
    empirical = np.einsum("ni,nj->ij", samples, samples.conj()) / len(samples)
    claims = cdl_spatial_covariance_claims(samples, empirical)
    with pytest.raises(ValueError, match=message):
        audit_cdl_spatial_covariance_claims(claims, **{field: value})


def test_replication_report_audit_preserves_incomplete_terminal_state() -> None:
    samples = _balanced_fixture()
    empirical = np.einsum("ni,nj->ij", samples, samples.conj()) / len(samples)
    claims = cdl_spatial_covariance_claims(samples, empirical)
    partial = claims[:5]
    report = {
        "status": "INCOMPLETE",
        "error": "RuntimeError: synthetic interrupted generation",
        "claims": partial,
        "summary": {
            "claims_passed": sum(claim["passed"] for claim in partial),
            "claims_failed": sum(not claim["passed"] for claim in partial),
        },
    }
    assert audit_replication_report(report) == {"status": "INCOMPLETE", "claim_count": 5}

    report["summary"]["claims_passed"] += 1
    with pytest.raises(ValueError, match="pass count"):
        audit_replication_report(report)
