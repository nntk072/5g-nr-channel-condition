"""Rejection contracts for signed G3 attestation envelopes, policies and package bindings.

The package-level consistency verifier has its own test module. Cases here that edit the
package bytes replace that verifier with a stub that reports a consistent proposal package,
so each case exercises the signature-envelope binding rules directly.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from evaluation.acceptance import signed_attestation
from evaluation.acceptance.signed_attestation import (
    SIGNATURE_NAMESPACE,
    main,
    verify_openssh_signature_for_bytes,
    verify_signed_attestation,
    verify_signed_attestation_set,
)
from tests.test_evaluation.test_signed_attestation import (
    _canonical_bytes,
    _make_signed_inputs,
    _make_signed_set,
)

SPLIT_ATTESTATION_ID = "attest.split.plan"
SPLIT_MANIFEST_FILE_ID = "file.split.manifest"
NON_SPLIT_TYPE = "rights_review"
Mutation = Callable[[dict[str, Any]], Any]


@pytest.fixture
def consistent_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the package consistency verifier with a stub reporting a consistent package."""
    monkeypatch.setattr(
        signed_attestation,
        "verify_acquisition_package",
        lambda path: {"status": "consistent_proposal_only"},
    )


def _rewrite_json(path: Path, mutate: Mutation) -> dict[str, Any]:
    document = json.loads(path.read_bytes())
    mutate(document)
    path.write_bytes(_canonical_bytes(document))
    return document


def _record(document: dict[str, Any], attestation_id: str = SPLIT_ATTESTATION_ID) -> dict[str, Any]:
    return next(
        item
        for item in document["external_attestations"]
        if item["attestation_id"] == attestation_id
    )


def _subject_entry(package_document: dict[str, Any], file_id: str) -> dict[str, Any]:
    entry = next(item for item in package_document["files"] if item["file_id"] == file_id)
    return {"file_id": file_id, "sha256": entry["sha256"], "size_bytes": entry["size_bytes"]}


def _edit_package(
    package: Path,
    envelope: Path,
    mutate_package: Mutation,
    mutate_envelope: Mutation = lambda document: None,
) -> None:
    """Edit the package and rebind the envelope to its new exact bytes."""
    _rewrite_json(package, mutate_package)

    def rebind(document: dict[str, Any]) -> None:
        document["package_sha256"] = hashlib.sha256(package.read_bytes()).hexdigest()
        mutate_envelope(document)

    _rewrite_json(envelope, rebind)


def _retype_to_non_split(package: Path, envelope: Path) -> None:
    def mutate_package(document: dict[str, Any]) -> None:
        _record(document)["attestation_type"] = NON_SPLIT_TYPE

    _edit_package(
        package,
        envelope,
        mutate_package,
        lambda document: document.update(attestation_type=NON_SPLIT_TYPE),
    )


ENVELOPE_CASES = [
    pytest.param(lambda d: d.update(extra="x"), "envelope fields do not match", id="extra-field"),
    pytest.param(lambda d: d.pop("signed_utc"), "envelope fields do not match", id="missing-field"),
    pytest.param(
        lambda d: d.update(schema="loslab.other.v1"),
        "unsupported signed-attestation schema",
        id="schema",
    ),
    pytest.param(
        lambda d: d.update(signature_namespace="other-namespace"),
        "attestation signature namespace is incorrect",
        id="namespace",
    ),
    pytest.param(
        lambda d: d.update(attestation_id="attest.missing"),
        "attestation ID must identify exactly one package record",
        id="unknown-attestation-id",
    ),
    pytest.param(
        lambda d: d.update(attestation_id=3),
        "package external_attestations or envelope ID is invalid",
        id="non-string-attestation-id",
    ),
    pytest.param(
        lambda d: d.update(claim_template="a different claim"),
        "attestation claim_template differs from its package record",
        id="claim-template-mismatch",
    ),
    pytest.param(
        lambda d: d.update(signer_role="other role"),
        "attestation signer_role differs from its package record",
        id="signer-role-mismatch",
    ),
    pytest.param(
        lambda d: d.update(signing_organization="   "),
        "attestation signing_organization must be a nonempty string",
        id="blank-organization",
    ),
    pytest.param(
        lambda d: d.update(signer_id="reviewer@example.org\nextra"),
        "signer_id must be a single-line OpenSSH principal",
        id="multiline-signer",
    ),
    pytest.param(
        lambda d: d.update(signed_utc="2025-12-31T09:30:00+00:00"),
        "signed_utc must be an ISO-8601 UTC timestamp ending in Z",
        id="offset-instead-of-z",
    ),
    pytest.param(
        lambda d: d.update(signed_utc=20251231),
        "signed_utc must be an ISO-8601 UTC timestamp ending in Z",
        id="non-string-time",
    ),
    pytest.param(
        lambda d: d.update(signed_utc="2025-13-31T09:30:00Z"),
        "signed_utc must be a valid ISO-8601 UTC timestamp",
        id="impossible-time",
    ),
    pytest.param(
        lambda d: d.update(subject_files="file.split.manifest"),
        "attestation subject files must be arrays",
        id="subjects-not-array",
    ),
    pytest.param(
        lambda d: d.update(subject_files=["file.split.manifest"]),
        "attestation subject_files entries must be objects",
        id="subject-not-object",
    ),
    pytest.param(
        lambda d: d.update(subject_files=[]),
        "attestation subject file IDs disagree with package record",
        id="subjects-dropped",
    ),
    pytest.param(
        lambda d: d.update(subject_files=d["subject_files"] + d["subject_files"]),
        "attestation subject file IDs disagree with package record",
        id="subjects-duplicated",
    ),
    pytest.param(
        lambda d: d.update(
            subject_files=[{**d["subject_files"][0], "extra": True}],
        ),
        "each attested subject must contain file_id, sha256, and size_bytes",
        id="subject-extra-key",
    ),
]


@pytest.mark.parametrize(("mutate", "message"), ENVELOPE_CASES)
def test_signed_attestation_rejects_envelope_contract_violation(
    tmp_path: Path, mutate: Mutation, message: str
) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    _rewrite_json(envelope, mutate)

    with pytest.raises(ValueError, match=message):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        pytest.param(b"\xff\n", "not valid strict UTF-8 JSON", id="invalid-utf8"),
        pytest.param(b'{"schema":"a","schema":"b"}\n', "duplicate JSON key", id="duplicate-key"),
        pytest.param(b'{"signed_utc":NaN}\n', "non-finite JSON value is forbidden", id="nan"),
        pytest.param(b"[]\n", "envelope fields do not match", id="root-array"),
    ],
)
def test_signed_attestation_rejects_malformed_envelope_bytes(
    tmp_path: Path, raw: bytes, message: str
) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    envelope.write_bytes(raw)

    with pytest.raises(ValueError, match=message):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


def test_signed_attestation_rejects_noncanonical_envelope_layout(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    document = json.loads(envelope.read_bytes())
    envelope.write_bytes(json.dumps(document, indent=2).encode())

    with pytest.raises(ValueError, match="attestation envelope must use canonical JSON bytes"):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


PACKAGE_CASES = [
    pytest.param(
        lambda d: d.update(proposal_status="accepted"),
        lambda e: None,
        "signed attestation verifier accepts proposal-only G3 packages",
        id="not-proposal-only",
    ),
    pytest.param(
        lambda d: d.update(schema_version="g3.03-acquisition-package.v9"),
        lambda e: None,
        "unsupported G3.03 package schema version",
        id="unsupported-package-schema",
    ),
    pytest.param(
        lambda d: d.update(external_attestations="attest.split.plan"),
        lambda e: None,
        "package external_attestations or envelope ID is invalid",
        id="attestations-not-array",
    ),
    pytest.param(
        lambda d: d["external_attestations"].append(dict(_record(d))),
        lambda e: None,
        "attestation ID must identify exactly one package record",
        id="duplicate-attestation-record",
    ),
    pytest.param(
        lambda d: d.update(files="file.split.manifest"),
        lambda e: None,
        "package file inventory must be an array",
        id="inventory-not-array",
    ),
    pytest.param(
        lambda d: d["files"].append(dict(d["files"][0])),
        lambda e: None,
        "package file inventory IDs must be unique strings",
        id="duplicate-inventory-id",
    ),
    pytest.param(
        lambda d: _record(d).update(claim_template=""),
        lambda e: e.update(claim_template=""),
        "attestation claim_template must be a nonempty string",
        id="blank-claim-template",
    ),
    pytest.param(
        lambda d: _record(d).update(subject_file_ids=[1, SPLIT_MANIFEST_FILE_ID]),
        lambda e: None,
        "package attestation subject_file_ids must be strings",
        id="non-string-subject-id",
    ),
    pytest.param(
        lambda d: _record(d).update(subject_file_ids=[SPLIT_MANIFEST_FILE_ID] * 2),
        lambda e: None,
        "attestation subject file IDs disagree with package record",
        id="repeated-subject-id",
    ),
    pytest.param(
        lambda d: _record(d).update(subject_file_ids="file.split.manifest"),
        lambda e: None,
        "must sign the evaluator-bound split manifest",
        id="subject-ids-not-array",
    ),
]


@pytest.mark.usefixtures("consistent_package")
@pytest.mark.parametrize(("mutate_package", "mutate_envelope", "message"), PACKAGE_CASES)
def test_signed_attestation_rejects_package_binding_violation(
    tmp_path: Path, mutate_package: Mutation, mutate_envelope: Mutation, message: str
) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    _edit_package(package, envelope, mutate_package, mutate_envelope)

    with pytest.raises(ValueError, match=message):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


@pytest.mark.usefixtures("consistent_package")
@pytest.mark.parametrize(
    ("size_value", "message"),
    [
        pytest.param("sha", "has an invalid SHA-256", id="invalid-digest"),
        pytest.param(True, "has an invalid byte size", id="boolean-size"),
        pytest.param(-1, "has an invalid byte size", id="negative-size"),
    ],
)
def test_signed_attestation_rejects_invalid_subject_digest_or_size(
    tmp_path: Path, size_value: Any, message: str
) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)

    def mutate_package(document: dict[str, Any]) -> None:
        entry = next(
            item for item in document["files"] if item["file_id"] == SPLIT_MANIFEST_FILE_ID
        )
        if size_value == "sha":
            entry["sha256"] = "G" * 64
        else:
            entry["size_bytes"] = size_value

    def mutate_envelope(document: dict[str, Any]) -> None:
        for item in document["subject_files"]:
            if item["file_id"] == SPLIT_MANIFEST_FILE_ID:
                if size_value == "sha":
                    item["sha256"] = "G" * 64
                else:
                    item["size_bytes"] = size_value

    _edit_package(package, envelope, mutate_package, mutate_envelope)

    with pytest.raises(ValueError, match=message):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


@pytest.mark.usefixtures("consistent_package")
def test_signed_attestation_requires_at_least_one_bound_subject(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)

    def mutate_package(document: dict[str, Any]) -> None:
        _record(document)["attestation_type"] = NON_SPLIT_TYPE
        _record(document)["subject_file_ids"] = []

    _edit_package(
        package,
        envelope,
        mutate_package,
        lambda document: document.update(attestation_type=NON_SPLIT_TYPE, subject_files=[]),
    )

    with pytest.raises(ValueError, match="at least one package file must be bound"):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


@pytest.mark.usefixtures("consistent_package")
def test_signed_attestation_subjects_must_be_sorted_by_file_id(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    package_document = json.loads(package.read_bytes())
    first, second = sorted(item["file_id"] for item in package_document["files"])[:2]

    def mutate_package(document: dict[str, Any]) -> None:
        _record(document)["attestation_type"] = NON_SPLIT_TYPE
        _record(document)["subject_file_ids"] = [first, second]

    _edit_package(
        package,
        envelope,
        mutate_package,
        lambda document: document.update(
            attestation_type=NON_SPLIT_TYPE,
            subject_files=[
                _subject_entry(package_document, second),
                _subject_entry(package_document, first),
            ],
        ),
    )

    with pytest.raises(ValueError, match="attestation subject_files must be sorted by file_id"):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


@pytest.mark.usefixtures("consistent_package")
def test_signed_attestation_rejects_unknown_attestation_type_on_both_sides(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    _edit_package(
        package,
        envelope,
        lambda d: _record(d).update(attestation_type="bogus_type"),
        lambda e: e.update(attestation_type="bogus_type"),
    )

    with pytest.raises(ValueError, match="unsupported G3.03 attestation type"):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


@pytest.mark.usefixtures("consistent_package")
@pytest.mark.parametrize(
    ("mutate_crosswalk", "message"),
    [
        pytest.param(
            None, "requires package samples and row_crosswalk arrays", id="samples-missing"
        ),
        pytest.param(
            "no-test-rows",
            "requires held-out acquisition sample crosswalk rows",
            id="no-held-out-rows",
        ),
        pytest.param(
            "non-string-id",
            "requires held-out acquisition sample crosswalk rows",
            id="non-string-held-out-id",
        ),
        pytest.param(
            "repeated-test-row",
            "held-out crosswalk repeats acquisition sample IDs",
            id="repeated-held-out-id",
        ),
        pytest.param(
            "absent-sample",
            "is absent from package samples",
            id="held-out-sample-absent",
        ),
        pytest.param(
            "bad-acquired-time",
            "acquired_utc must be an ISO-8601 UTC timestamp ending in Z",
            id="held-out-time-invalid",
        ),
        pytest.param(
            "too-early",
            "signed_utc must precede every declared held-out acquisition",
            id="signed-after-acquisition",
        ),
    ],
)
def test_split_preregistration_requires_held_out_crosswalk_and_timing(
    tmp_path: Path, mutate_crosswalk: str | None, message: str
) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    package_document = json.loads(package.read_bytes())
    held_out_row = next(row for row in package_document["row_crosswalk"] if row["split"] == "test")
    held_out_id = held_out_row["acquisition_sample_id"]
    sample = next(item for item in package_document["samples"] if item["sample_id"] == held_out_id)

    def mutate_package(document: dict[str, Any]) -> None:
        if mutate_crosswalk is None:
            document.pop("samples")
            return
        if mutate_crosswalk == "no-test-rows":
            for row in document["row_crosswalk"]:
                row["split"] = "train"
        elif mutate_crosswalk == "non-string-id":
            for row in document["row_crosswalk"]:
                if row["split"] == "test":
                    row["acquisition_sample_id"] = 5
        elif mutate_crosswalk == "repeated-test-row":
            document["row_crosswalk"].append(dict(held_out_row))
        elif mutate_crosswalk == "absent-sample":
            for row in document["row_crosswalk"]:
                if row["split"] == "test":
                    row["acquisition_sample_id"] = "sample.absent"
        elif mutate_crosswalk == "bad-acquired-time":
            for item in document["samples"]:
                if item["sample_id"] == sample["sample_id"]:
                    item["acquired_utc"] = "not-a-time"
        elif mutate_crosswalk == "too-early":
            for item in document["samples"]:
                if item["sample_id"] == sample["sample_id"]:
                    item["acquired_utc"] = "2025-01-01T00:00:00Z"

    _edit_package(package, envelope, mutate_package)

    with pytest.raises(ValueError, match=message):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


POLICY_CASES = [
    pytest.param(
        lambda d: d.update(extra=1), "trusted signer policy fields do not match", id="extra-field"
    ),
    pytest.param(
        lambda d: d.update(schema="loslab.other.v1"),
        "unsupported trusted signer policy schema",
        id="schema",
    ),
    pytest.param(
        lambda d: d.update(policy_id="   "),
        "trusted signer policy ID and signers are required",
        id="blank-policy-id",
    ),
    pytest.param(
        lambda d: d.update(signers="reviewer"),
        "trusted signer policy ID and signers are required",
        id="signers-not-array",
    ),
    pytest.param(
        lambda d: d.update(signers=["reviewer"]),
        "each trusted signer must match the v1 policy contract",
        id="signer-not-object",
    ),
    pytest.param(
        lambda d: d["signers"][0].update(extra=True),
        "each trusted signer must match the v1 policy contract",
        id="signer-extra-field",
    ),
    pytest.param(
        lambda d: d["signers"][0].update(principal="   "),
        "trusted signer principal and organization must be nonempty strings",
        id="blank-principal",
    ),
    pytest.param(
        lambda d: d["signers"][0].update(principal="reviewer@example.org\nother"),
        "trusted signer principal and organization must be nonempty strings",
        id="multiline-principal",
    ),
    pytest.param(
        lambda d: d["signers"][0].update(organization=""),
        "trusted signer principal and organization must be nonempty strings",
        id="blank-organization",
    ),
    pytest.param(
        lambda d: d["signers"][0].update(allowed_attestation_types=[]),
        "trusted signer allowed_attestation_types must be nonempty",
        id="no-attestation-types",
    ),
    pytest.param(
        lambda d: d["signers"][0].update(allowed_attestation_types=[1]),
        "trusted signer attestation types must be strings",
        id="non-string-attestation-type",
    ),
    pytest.param(
        lambda d: d["signers"][0].update(allowed_attestation_types=["bogus_type"]),
        "trusted signer has an unsupported attestation type",
        id="unknown-attestation-type",
    ),
    pytest.param(
        lambda d: d["signers"][0].update(
            allowed_attestation_types=["split_preregistration", "split_preregistration"]
        ),
        "trusted signer attestation types must be unique",
        id="duplicate-attestation-type",
    ),
    pytest.param(
        lambda d: d["signers"][0].update(allowed_signer_roles=[]),
        "trusted signer allowed_signer_roles must be unique nonempty strings",
        id="no-signer-roles",
    ),
    pytest.param(
        lambda d: d["signers"][0].update(allowed_signer_roles=["  "]),
        "trusted signer allowed_signer_roles must be unique nonempty strings",
        id="blank-signer-role",
    ),
    pytest.param(
        lambda d: d["signers"][0].update(allowed_signer_roles=["reviewer", "reviewer"]),
        "trusted signer allowed_signer_roles must be unique nonempty strings",
        id="duplicate-signer-role",
    ),
    pytest.param(
        lambda d: d["signers"].append(dict(d["signers"][0])),
        "trusted signer policy principals must be unique",
        id="duplicate-principal",
    ),
]


@pytest.mark.parametrize(("mutate", "message"), POLICY_CASES)
def test_trusted_signer_policy_rejects_contract_violation(
    tmp_path: Path, mutate: Mutation, message: str
) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    _rewrite_json(policy, mutate)

    with pytest.raises(ValueError, match=message):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


def test_trusted_signer_policy_must_be_canonical_object(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    document = json.loads(policy.read_bytes())
    policy.write_bytes(json.dumps(document, indent=2).encode())
    with pytest.raises(ValueError, match="trusted signer policy must use canonical JSON bytes"):
        verify_signed_attestation(package, envelope, signature, trusted, policy)

    policy.write_bytes(_canonical_bytes([]))
    with pytest.raises(ValueError, match="trusted signer policy fields do not match"):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


def test_signed_attestation_rejects_package_that_is_not_a_json_object(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    package.write_bytes(b"[]\n")
    with pytest.raises(ValueError, match="G3 package root must be an object"):
        verify_signed_attestation(package, envelope, signature, trusted, policy)

    package.write_bytes(b"\xff\n")
    with pytest.raises(ValueError, match="G3 package is not valid strict UTF-8 JSON"):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


def test_signed_attestation_rejects_inconsistent_package_verifier_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    monkeypatch.setattr(
        signed_attestation,
        "verify_acquisition_package",
        lambda path: {"status": "inconsistent"},
    )
    with pytest.raises(ValueError, match="fails the proposal-package consistency verifier"):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


def test_signed_attestation_rejects_package_changed_during_consistency_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)

    def rewrite_then_pass(path: str | Path) -> dict[str, str]:
        target = Path(path)
        target.write_bytes(target.read_bytes() + b" ")
        return {"status": "consistent_proposal_only"}

    monkeypatch.setattr(signed_attestation, "verify_acquisition_package", rewrite_then_pass)
    with pytest.raises(ValueError, match="changed while its consistency was being verified"):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


def test_signed_attestation_requires_ssh_keygen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    monkeypatch.setattr(signed_attestation.shutil, "which", lambda name: None)
    with pytest.raises(
        RuntimeError, match="OpenSSH ssh-keygen is required to verify G3 attestations"
    ):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


def test_signed_attestation_times_out_ssh_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)

    def expired(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="ssh-keygen", timeout=30)

    monkeypatch.setattr(signed_attestation.subprocess, "run", expired)
    with pytest.raises(RuntimeError, match="OpenSSH G3 attestation verification timed out"):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


@pytest.mark.parametrize(
    ("limit_name", "message"),
    [
        pytest.param("MAX_PACKAGE_BYTES", "G3 package exceeds 10 bytes", id="package"),
        pytest.param(
            "MAX_ATTESTATION_BYTES", "G3 attestation envelope exceeds 10 bytes", id="envelope"
        ),
        pytest.param(
            "MAX_SIGNER_POLICY_BYTES", "trusted signer role policy exceeds 10 bytes", id="policy"
        ),
        pytest.param(
            "MAX_ALLOWED_SIGNERS_BYTES",
            "trusted allowed_signers file exceeds 10 bytes",
            id="allowed-signers",
        ),
    ],
)
def test_signed_attestation_bounds_every_input_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit_name: str, message: str
) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    monkeypatch.setattr(signed_attestation, limit_name, 10)
    with pytest.raises(ValueError, match=message):
        verify_signed_attestation(package, envelope, signature, trusted, policy)


def test_signed_attestation_rejects_non_regular_inputs(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    directory = tmp_path / "package-directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="G3 package must be a single-link regular file"):
        verify_signed_attestation(directory, envelope, signature, trusted, policy)

    symbolic_signature = tmp_path / "signature-link.sig"
    symbolic_signature.symlink_to(signature)
    with pytest.raises(
        ValueError, match="G3 attestation signature must be a single-link regular file"
    ):
        verify_signed_attestation(package, envelope, symbolic_signature, trusted, policy)


@pytest.mark.parametrize(
    ("payload", "signer_id", "namespace", "message"),
    [
        pytest.param(
            "text",
            "reviewer@example.org",
            SIGNATURE_NAMESPACE,
            "signed payload must be bytes",
            id="not-bytes",
        ),
        pytest.param(
            b"x",
            "   ",
            SIGNATURE_NAMESPACE,
            "signer_id must be a nonempty single-line value",
            id="blank-signer",
        ),
        pytest.param(
            b"x",
            "a\nb",
            SIGNATURE_NAMESPACE,
            "signer_id must be a nonempty single-line value",
            id="multiline-signer",
        ),
        pytest.param(
            b"x",
            "reviewer@example.org",
            "",
            "namespace must be a nonempty single-line value",
            id="blank-namespace",
        ),
    ],
)
def test_openssh_bytes_verifier_rejects_invalid_arguments(
    tmp_path: Path, payload: Any, signer_id: str, namespace: str, message: str
) -> None:
    _, _, signature, trusted, _, _ = _make_signed_inputs(tmp_path)
    with pytest.raises(ValueError, match=message):
        verify_openssh_signature_for_bytes(
            payload, signature, trusted, signer_id=signer_id, namespace=namespace
        )


def _sign_payload(tmp_path: Path, key: Path, payload: bytes) -> Path:
    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(payload)
    subprocess.run(
        [
            shutil.which("ssh-keygen") or "ssh-keygen",
            "-Y",
            "sign",
            "-f",
            str(key),
            "-n",
            SIGNATURE_NAMESPACE,
            str(payload_path),
        ],
        check=True,
        capture_output=True,
    )
    return payload_path.with_name("payload.bin.sig")


def test_openssh_bytes_verifier_accepts_and_rejects_exact_payloads(tmp_path: Path) -> None:
    _, _, _, trusted, _, key = _make_signed_inputs(tmp_path)
    signature = _sign_payload(tmp_path, key, b"exact bytes")

    report = verify_openssh_signature_for_bytes(
        b"exact bytes",
        signature,
        trusted,
        signer_id="  reviewer@example.org  ",
        namespace=f"  {SIGNATURE_NAMESPACE}  ",
    )
    assert report["signer_id"] == "reviewer@example.org"
    assert report["namespace"] == SIGNATURE_NAMESPACE
    assert len(report["signature_sha256"]) == 64

    with pytest.raises(ValueError, match="signature is invalid or signer is not trusted"):
        verify_openssh_signature_for_bytes(
            b"other bytes",
            signature,
            trusted,
            signer_id="reviewer@example.org",
            namespace=SIGNATURE_NAMESPACE,
        )


def test_openssh_bytes_verifier_bounds_payload_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, signature, trusted, _, _ = _make_signed_inputs(tmp_path)
    monkeypatch.setattr(signed_attestation, "MAX_PACKAGE_BYTES", 3)
    with pytest.raises(ValueError, match="signed payload must be bytes no larger than 3"):
        verify_openssh_signature_for_bytes(
            b"abcd",
            signature,
            trusted,
            signer_id="reviewer@example.org",
            namespace=SIGNATURE_NAMESPACE,
        )


def test_openssh_bytes_verifier_requires_ssh_keygen_and_handles_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, signature, trusted, _, _ = _make_signed_inputs(tmp_path)
    monkeypatch.setattr(signed_attestation.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="required to verify signed G3 artifacts"):
        verify_openssh_signature_for_bytes(
            b"x",
            signature,
            trusted,
            signer_id="reviewer@example.org",
            namespace=SIGNATURE_NAMESPACE,
        )

    monkeypatch.undo()

    def expired(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="ssh-keygen", timeout=30)

    monkeypatch.setattr(signed_attestation.subprocess, "run", expired)
    with pytest.raises(RuntimeError, match="OpenSSH signature verification timed out"):
        verify_openssh_signature_for_bytes(
            b"x",
            signature,
            trusted,
            signer_id="reviewer@example.org",
            namespace=SIGNATURE_NAMESPACE,
        )


def test_signed_attestation_set_rejects_empty_and_duplicate_inputs(tmp_path: Path) -> None:
    package, envelopes, trusted, policy = _make_signed_set(tmp_path)
    with pytest.raises(ValueError, match="at least one attestation envelope is required"):
        verify_signed_attestation_set(package, [], trusted, policy)
    with pytest.raises(ValueError, match="attestation envelope paths must be unique"):
        verify_signed_attestation_set(package, [envelopes[0], envelopes[0]], trusted, policy)


@pytest.mark.usefixtures("consistent_package")
def test_signed_attestation_set_rejects_malformed_package_records(tmp_path: Path) -> None:
    package, envelopes, trusted, policy = _make_signed_set(tmp_path)

    _rewrite_json(package, lambda d: d.update(external_attestations="records"))
    with pytest.raises(ValueError, match="package external_attestations must be an array"):
        verify_signed_attestation_set(package, envelopes, trusted, policy)

    _rewrite_json(package, lambda d: d.update(external_attestations=[1]))
    with pytest.raises(ValueError, match="need string IDs and types"):
        verify_signed_attestation_set(package, envelopes, trusted, policy)


@pytest.mark.usefixtures("consistent_package")
def test_signed_attestation_set_rejects_duplicate_package_records(tmp_path: Path) -> None:
    package, envelopes, trusted, policy = _make_signed_set(tmp_path)
    _rewrite_json(
        package, lambda d: d["external_attestations"].append(dict(d["external_attestations"][0]))
    )
    with pytest.raises(ValueError, match="package external attestation IDs must be unique strings"):
        verify_signed_attestation_set(package, envelopes, trusted, policy)


@pytest.mark.usefixtures("consistent_package")
def test_signed_attestation_set_rejects_duplicate_verified_envelope_identity(
    tmp_path: Path,
) -> None:
    package, envelopes, trusted, policy = _make_signed_set(tmp_path)
    copied = tmp_path / "copied-attestation.json"
    shutil.copyfile(envelopes[0], copied)
    shutil.copyfile(
        envelopes[0].with_name(envelopes[0].name + ".sig"), copied.with_name(copied.name + ".sig")
    )
    # The copy shares the first record's attestation ID; the verifier must reject the repeat.
    with pytest.raises(
        ValueError, match="attestation set contains duplicate package attestation IDs"
    ):
        verify_signed_attestation_set(package, [*envelopes, copied], trusted, policy)


def test_signed_attestation_set_reports_missing_package_records(tmp_path: Path) -> None:
    package, envelopes, trusted, policy = _make_signed_set(tmp_path)
    with pytest.raises(ValueError, match="does not cover package records; missing="):
        verify_signed_attestation_set(package, envelopes[:-1], trusted, policy)


def test_signed_attestation_cli_reports_success_and_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    arguments = [
        str(package),
        str(envelope),
        str(signature),
        "--allowed-signers",
        str(trusted),
        "--signer-policy",
        str(policy),
    ]
    assert main(arguments) == 0
    assert '"signature_status": "verified"' in capsys.readouterr().out

    envelope.write_bytes(b"[]\n")
    with pytest.raises(SystemExit) as excinfo:
        main(arguments)
    assert excinfo.value.code == 2
    assert "error: attestation envelope fields do not match" in capsys.readouterr().err
