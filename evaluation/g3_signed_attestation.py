"""Verify an externally signed G3.03 attestation against an exact package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from evaluation.g3_acquisition_package import verify_g3_acquisition_package

SCHEMA = "loslab.g3_signed_attestation.v1"
SIGNATURE_NAMESPACE = "loslab-g3-acquisition-attestation-v1@github.com"
ATTESTATION_TYPES = {
    "split_preregistration",
    "physical_independence",
    "calibration_traceability",
    "rights_review",
    "scientific_review",
}
MAX_PACKAGE_BYTES = 16 * 1024 * 1024
MAX_ATTESTATION_BYTES = 1024 * 1024
MAX_SIGNATURE_BYTES = 64 * 1024
MAX_ALLOWED_SIGNERS_BYTES = 1024 * 1024
MAX_SIGNER_POLICY_BYTES = 1024 * 1024
TRUST_POLICY_SCHEMA = "loslab.g3_attestation_trust_policy.v1"
_ATTESTATION_FIELDS = {
    "schema",
    "package_sha256",
    "attestation_id",
    "attestation_type",
    "claim_template",
    "subject_files",
    "signer_id",
    "signer_role",
    "signing_organization",
    "signed_utc",
    "signature_namespace",
}


def _stat_key(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_single_link_file(path: Path, *, limit: int, label: str) -> bytes:
    """Read one bounded regular file while rejecting links and concurrent changes."""
    before_path = os.lstat(path)
    if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("platform lacks safe no-symlink file opening")
    descriptor = os.open(path, flags | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        path_after_open = os.lstat(path)
        if _stat_key(opened) != _stat_key(before_path) or _stat_key(path_after_open) != _stat_key(
            before_path
        ):
            raise ValueError(f"{label} changed while opening")
        if opened.st_size > limit:
            raise ValueError(f"{label} exceeds {limit} bytes")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                raise ValueError(f"{label} exceeds {limit} bytes")
        after = os.fstat(descriptor)
        path_after_read = os.lstat(path)
        if size != opened.st_size or _stat_key(after) != _stat_key(opened):
            raise ValueError(f"{label} changed while being read")
        if _stat_key(path_after_read) != _stat_key(opened):
            raise ValueError(f"{label} path changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _parse_json(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError(f"{label} is not valid strict UTF-8 JSON: {error}") from error


def _parse_declared_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be a valid ISO-8601 UTC timestamp") from error


def _validate_split_preregistration_order(
    package: dict[str, Any], signed_at: datetime
) -> dict[str, Any]:
    samples = package.get("samples")
    crosswalk = package.get("row_crosswalk")
    if not isinstance(samples, list) or not isinstance(crosswalk, list):
        raise ValueError("split_preregistration requires package samples and row_crosswalk arrays")
    samples_by_id = {
        item.get("sample_id"): item
        for item in samples
        if isinstance(item, dict) and isinstance(item.get("sample_id"), str)
    }
    held_out_ids = {
        row.get("acquisition_sample_id")
        for row in crosswalk
        if isinstance(row, dict) and row.get("split") == "test"
    }
    if not held_out_ids or any(not isinstance(sample_id, str) for sample_id in held_out_ids):
        raise ValueError(
            "split_preregistration requires held-out acquisition sample crosswalk rows"
        )
    if len(held_out_ids) != sum(
        1 for row in crosswalk if isinstance(row, dict) and row.get("split") == "test"
    ):
        raise ValueError("split_preregistration held-out crosswalk repeats acquisition sample IDs")
    acquired_times: list[tuple[str, datetime]] = []
    for sample_id in sorted(held_out_ids):
        sample = samples_by_id.get(sample_id)
        if sample is None:
            raise ValueError(
                f"split_preregistration held-out sample {sample_id!r} is absent from package samples"
            )
        acquired_times.append(
            (
                sample_id,
                _parse_declared_utc(
                    sample.get("acquired_utc"),
                    label=f"held-out sample {sample_id}.acquired_utc",
                ),
            )
        )
    latest_sample_id, latest_acquired_at = max(acquired_times, key=lambda item: item[1])
    if signed_at >= latest_acquired_at:
        raise ValueError(
            "split_preregistration signed_utc must precede every declared held-out acquisition"
        )
    return {
        "declared_temporal_order_status": "preregistration_precedes_declared_held_out_acquisitions",
        "held_out_acquisition_sample_count": len(acquired_times),
        "latest_bound_acquisition_sample_id": latest_sample_id,
        "latest_bound_acquisition_utc": latest_acquired_at.isoformat().replace("+00:00", "Z"),
    }


def _validate_attestation(
    package: dict[str, Any], package_bytes: bytes, envelope_bytes: bytes
) -> dict[str, Any]:
    envelope = _parse_json(envelope_bytes, label="attestation envelope")
    if not isinstance(envelope, dict) or set(envelope) != _ATTESTATION_FIELDS:
        raise ValueError("attestation envelope fields do not match the v1 contract")
    canonical = (
        json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    if canonical != envelope_bytes:
        raise ValueError("attestation envelope must use canonical JSON bytes")
    if envelope.get("schema") != SCHEMA:
        raise ValueError("unsupported signed-attestation schema")
    if envelope.get("signature_namespace") != SIGNATURE_NAMESPACE:
        raise ValueError("attestation signature namespace is incorrect")
    package_digest = hashlib.sha256(package_bytes).hexdigest()
    if envelope.get("package_sha256") != package_digest:
        raise ValueError("attestation package_sha256 does not match exact package bytes")
    if package.get("proposal_status") != "proposal_only":
        raise ValueError("signed attestation verifier accepts proposal-only G3 packages")
    package_schema = package.get("schema_version")
    if not isinstance(package_schema, str) or package_schema not in {
        "g3.03-acquisition-package.v0.2",
        "g3.03-acquisition-package.v0.3",
        "g3.03-acquisition-package.v0.4",
    }:
        raise ValueError("unsupported G3.03 package schema version")

    attestation_id = envelope.get("attestation_id")
    records = package.get("external_attestations")
    if not isinstance(attestation_id, str) or not isinstance(records, list):
        raise ValueError("package external_attestations or envelope ID is invalid")
    matches = [
        item
        for item in records
        if isinstance(item, dict) and item.get("attestation_id") == attestation_id
    ]
    if len(matches) != 1:
        raise ValueError("attestation ID must identify exactly one package record")
    planned = matches[0]
    for field in ("attestation_type", "claim_template", "signer_role"):
        if envelope.get(field) != planned.get(field):
            raise ValueError(f"attestation {field} differs from its package record")
    attestation_type = envelope.get("attestation_type")
    if not isinstance(attestation_type, str) or attestation_type not in ATTESTATION_TYPES:
        raise ValueError("unsupported G3.03 attestation type")
    if attestation_type == "split_preregistration":
        binding = package.get("evaluator_binding")
        split_reference = binding.get("split_manifest") if isinstance(binding, dict) else None
        split_file_id = (
            split_reference.get("file_id") if isinstance(split_reference, dict) else None
        )
        planned_subject_ids = planned.get("subject_file_ids")
        if (
            not isinstance(split_file_id, str)
            or not isinstance(planned_subject_ids, list)
            or split_file_id not in planned_subject_ids
        ):
            raise ValueError(
                "split_preregistration attestation must sign the evaluator-bound split manifest"
            )
    for field in ("claim_template", "signer_role", "signing_organization", "signer_id"):
        value = envelope.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"attestation {field} must be a nonempty string")
    signer_id = envelope["signer_id"]
    if any(character in signer_id for character in ("\n", "\r", "\x00")):
        raise ValueError("signer_id must be a single-line OpenSSH principal")
    signed_utc = envelope.get("signed_utc")
    if not isinstance(signed_utc, str) or not signed_utc.endswith("Z"):
        raise ValueError("signed_utc must be an ISO-8601 UTC timestamp ending in Z")
    signed_at = _parse_declared_utc(signed_utc, label="signed_utc")
    temporal_order = (
        _validate_split_preregistration_order(package, signed_at)
        if attestation_type == "split_preregistration"
        else {}
    )

    subject_ids = planned.get("subject_file_ids")
    subject_files = envelope.get("subject_files")
    file_entries = package.get("files")
    if not isinstance(subject_ids, list) or not isinstance(subject_files, list):
        raise ValueError("attestation subject files must be arrays")
    if not isinstance(file_entries, list):
        raise ValueError("package file inventory must be an array")
    file_index = {
        item.get("file_id"): item
        for item in file_entries
        if isinstance(item, dict) and isinstance(item.get("file_id"), str)
    }
    if len(file_index) != len(file_entries):
        raise ValueError("package file inventory IDs must be unique strings")
    if any(not isinstance(item, str) for item in subject_ids):
        raise ValueError("package attestation subject_file_ids must be strings")
    if any(not isinstance(item, dict) for item in subject_files):
        raise ValueError("attestation subject_files entries must be objects")
    subject_item_ids = [item.get("file_id") for item in subject_files]
    if (
        any(not isinstance(item, str) for item in subject_item_ids)
        or len(set(subject_ids)) != len(subject_ids)
        or len(set(subject_item_ids)) != len(subject_item_ids)
        or set(subject_ids) != set(subject_item_ids)
    ):
        raise ValueError("attestation subject file IDs disagree with package record")
    validated_subjects: list[dict[str, Any]] = []
    for item in subject_files:
        if not isinstance(item, dict) or set(item) != {"file_id", "sha256", "size_bytes"}:
            raise ValueError("each attested subject must contain file_id, sha256, and size_bytes")
        file_id = item["file_id"]
        record = file_index.get(file_id)
        if not isinstance(record, dict):
            raise ValueError(f"attested subject {file_id!r} is absent from package inventory")
        if item["sha256"] != record.get("sha256") or item["size_bytes"] != record.get("size_bytes"):
            raise ValueError(f"attested subject {file_id!r} differs from package inventory")
        digest = item["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or set(digest) - set("0123456789abcdef")
        ):
            raise ValueError(f"attested subject {file_id!r} has an invalid SHA-256")
        if (
            isinstance(item["size_bytes"], bool)
            or not isinstance(item["size_bytes"], int)
            or item["size_bytes"] < 0
        ):
            raise ValueError(f"attested subject {file_id!r} has an invalid byte size")
        validated_subjects.append(dict(item))
    if not validated_subjects:
        raise ValueError("at least one package file must be bound by the attestation")
    if validated_subjects != sorted(validated_subjects, key=lambda item: item["file_id"]):
        raise ValueError("attestation subject_files must be sorted by file_id")
    return {
        "package_sha256": package_digest,
        "package_id": package.get("package_id"),
        "attestation_id": attestation_id,
        "attestation_type": attestation_type,
        "subject_file_count": len(validated_subjects),
        "subject_file_ids": [item["file_id"] for item in validated_subjects],
        "subject_files": validated_subjects,
        "signer_id": signer_id,
        "signer_role": envelope["signer_role"],
        "signing_organization": envelope["signing_organization"],
        "signed_utc": signed_utc,
        **temporal_order,
    }


def _authorize_signer(
    envelope_identity: dict[str, Any], policy_bytes: bytes
) -> tuple[dict[str, Any], str, str]:
    policy = _parse_json(policy_bytes, label="trusted signer policy")
    if not isinstance(policy, dict) or set(policy) != {"schema", "policy_id", "signers"}:
        raise ValueError("trusted signer policy fields do not match the v1 contract")
    canonical = (
        json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    if canonical != policy_bytes:
        raise ValueError("trusted signer policy must use canonical JSON bytes")
    if policy.get("schema") != TRUST_POLICY_SCHEMA:
        raise ValueError("unsupported trusted signer policy schema")
    policy_id = policy.get("policy_id")
    signers = policy.get("signers")
    if not isinstance(policy_id, str) or not policy_id.strip() or not isinstance(signers, list):
        raise ValueError("trusted signer policy ID and signers are required")
    entries: dict[str, dict[str, Any]] = {}
    for signer in signers:
        if not isinstance(signer, dict) or set(signer) != {
            "principal",
            "organization",
            "allowed_attestation_types",
            "allowed_signer_roles",
        }:
            raise ValueError("each trusted signer must match the v1 policy contract")
        principal = signer.get("principal")
        organization = signer.get("organization")
        attestation_types = signer.get("allowed_attestation_types")
        signer_roles = signer.get("allowed_signer_roles")
        if (
            not isinstance(principal, str)
            or not principal.strip()
            or any(character in principal for character in ("\n", "\r", "\x00"))
            or not isinstance(organization, str)
            or not organization.strip()
        ):
            raise ValueError("trusted signer principal and organization must be nonempty strings")
        if not isinstance(attestation_types, list) or not attestation_types:
            raise ValueError("trusted signer allowed_attestation_types must be nonempty")
        if any(not isinstance(item, str) for item in attestation_types):
            raise ValueError("trusted signer attestation types must be strings")
        if any(item not in ATTESTATION_TYPES for item in attestation_types):
            raise ValueError("trusted signer has an unsupported attestation type")
        if len(set(attestation_types)) != len(attestation_types):
            raise ValueError("trusted signer attestation types must be unique")
        if (
            not isinstance(signer_roles, list)
            or not signer_roles
            or any(not isinstance(item, str) or not item.strip() for item in signer_roles)
            or len(set(signer_roles)) != len(signer_roles)
        ):
            raise ValueError("trusted signer allowed_signer_roles must be unique nonempty strings")
        if principal in entries:
            raise ValueError("trusted signer policy principals must be unique")
        entries[principal] = signer
    principal = envelope_identity["signer_id"]
    signer = entries.get(principal)
    if signer is None:
        raise ValueError("signer principal is absent from the trusted role policy")
    if envelope_identity["signing_organization"] != signer["organization"]:
        raise ValueError("signing organization is not authorized for this signer principal")
    if envelope_identity["attestation_type"] not in signer["allowed_attestation_types"]:
        raise ValueError("attestation type is not authorized for this signer principal")
    if envelope_identity["signer_role"] not in signer["allowed_signer_roles"]:
        raise ValueError("signer role is not authorized for this signer principal")
    return signer, hashlib.sha256(policy_bytes).hexdigest(), policy_id


@dataclass(frozen=True)
class _VerificationContext:
    package: dict[str, Any]
    package_bytes: bytes
    consistency_status: str
    allowed_signers_bytes: bytes
    signer_policy_bytes: bytes
    ssh_keygen: str


def _load_verification_context(
    package_path: str | Path,
    trusted_allowed_signers_path: str | Path,
    trusted_signer_policy_path: str | Path,
) -> _VerificationContext:
    package_bytes = _read_single_link_file(
        Path(package_path), limit=MAX_PACKAGE_BYTES, label="G3 package"
    )
    signers_bytes = _read_single_link_file(
        Path(trusted_allowed_signers_path),
        limit=MAX_ALLOWED_SIGNERS_BYTES,
        label="trusted allowed_signers file",
    )
    policy_bytes = _read_single_link_file(
        Path(trusted_signer_policy_path),
        limit=MAX_SIGNER_POLICY_BYTES,
        label="trusted signer role policy",
    )
    package = _parse_json(package_bytes, label="G3 package")
    if not isinstance(package, dict):
        raise ValueError("G3 package root must be an object")
    consistency = verify_g3_acquisition_package(package_path)
    if consistency.get("status") != "consistent_proposal_only":
        raise ValueError("G3 package fails the proposal-package consistency verifier")
    if (
        _read_single_link_file(Path(package_path), limit=MAX_PACKAGE_BYTES, label="G3 package")
        != package_bytes
    ):
        raise ValueError("G3 package changed while its consistency was being verified")
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise RuntimeError("OpenSSH ssh-keygen is required to verify G3 attestations")
    return _VerificationContext(
        package=package,
        package_bytes=package_bytes,
        consistency_status=consistency["status"],
        allowed_signers_bytes=signers_bytes,
        signer_policy_bytes=policy_bytes,
        ssh_keygen=executable,
    )


def _verify_one_attestation(
    context: _VerificationContext, attestation_path: str | Path, signature_path: str | Path
) -> dict[str, Any]:
    attestation_bytes = _read_single_link_file(
        Path(attestation_path), limit=MAX_ATTESTATION_BYTES, label="G3 attestation envelope"
    )
    signature_bytes = _read_single_link_file(
        Path(signature_path), limit=MAX_SIGNATURE_BYTES, label="G3 attestation signature"
    )
    identity = _validate_attestation(context.package, context.package_bytes, attestation_bytes)
    _, policy_digest, policy_id = _authorize_signer(identity, context.signer_policy_bytes)
    with tempfile.TemporaryDirectory(prefix="loslab-g3-attestation-") as temp_dir:
        snapshot = Path(temp_dir)
        signature_snapshot = snapshot / "attestation.json.sig"
        signers_snapshot = snapshot / "allowed_signers"
        signature_snapshot.write_bytes(signature_bytes)
        signers_snapshot.write_bytes(context.allowed_signers_bytes)
        try:
            verified = subprocess.run(
                [
                    context.ssh_keygen,
                    "-Y",
                    "verify",
                    "-f",
                    str(signers_snapshot),
                    "-I",
                    identity["signer_id"],
                    "-n",
                    SIGNATURE_NAMESPACE,
                    "-s",
                    str(signature_snapshot),
                ],
                input=attestation_bytes,
                capture_output=True,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("OpenSSH G3 attestation verification timed out") from error
    if verified.returncode != 0:
        raise ValueError("G3 attestation signature is invalid or signer is not trusted")
    return {
        "schema": "loslab.g3_signed_attestation_verification.v1",
        "signature_status": "verified",
        "package_consistency_status": context.consistency_status,
        "scientific_claim_status": "not_assessed",
        "qualification_status": "NOT_QUALIFIED",
        "signature_namespace": SIGNATURE_NAMESPACE,
        "attestation_envelope_sha256": hashlib.sha256(attestation_bytes).hexdigest(),
        "signature_file_sha256": hashlib.sha256(signature_bytes).hexdigest(),
        "trusted_allowed_signers_sha256": hashlib.sha256(context.allowed_signers_bytes).hexdigest(),
        "trusted_signer_policy_sha256": policy_digest,
        "trusted_signer_policy_id": policy_id,
        "verifier_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **identity,
        "limitations": [
            "A valid signature authenticates the exact envelope bytes and trusted signer principal; it does not establish that the signed statement is true.",
            "The preregistration ordering check compares declared timestamps only; OpenSSH signatures do not prove signing time, timestamp custody, or acquisition-time truth.",
            "Physical independence, calibration validity, rights sufficiency, and signer independence require separate review.",
            "The full package consistency verifier checks declared inventory hashes and joins; this signature tool does not independently authenticate the truth of those package records.",
            "G3 qualification and production RT authorization are not assessed.",
        ],
    }


def verify_openssh_signature_for_bytes(
    payload: bytes,
    signature_path: str | Path,
    trusted_allowed_signers_path: str | Path,
    *,
    signer_id: str,
    namespace: str,
) -> dict[str, str]:
    """Verify a detached OpenSSH signature over caller-captured exact bytes.

    The allowed-signers file is an external trust input. A successful result
    authenticates the principal for these bytes and namespace; it does not
    establish the trust file's custody or the truth of the signed content.
    """
    if not isinstance(payload, bytes) or len(payload) > MAX_PACKAGE_BYTES:
        raise ValueError(f"signed payload must be bytes no larger than {MAX_PACKAGE_BYTES}")
    for field_name, value in (("signer_id", signer_id), ("namespace", namespace)):
        if (
            not isinstance(value, str)
            or not value.strip()
            or any(character in value for character in ("\n", "\r", "\x00"))
        ):
            raise ValueError(f"{field_name} must be a nonempty single-line value")
    signer_id = signer_id.strip()
    namespace = namespace.strip()
    signature_bytes = _read_single_link_file(
        Path(signature_path), limit=MAX_SIGNATURE_BYTES, label="OpenSSH signature"
    )
    allowed_signers_bytes = _read_single_link_file(
        Path(trusted_allowed_signers_path),
        limit=MAX_ALLOWED_SIGNERS_BYTES,
        label="trusted allowed_signers file",
    )
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise RuntimeError("OpenSSH ssh-keygen is required to verify signed G3 artifacts")
    with tempfile.TemporaryDirectory(prefix="loslab-g3-signed-bytes-") as temp_dir:
        snapshot = Path(temp_dir)
        signature_snapshot = snapshot / "artifact.sig"
        signers_snapshot = snapshot / "allowed_signers"
        signature_snapshot.write_bytes(signature_bytes)
        signers_snapshot.write_bytes(allowed_signers_bytes)
        try:
            verified = subprocess.run(
                [
                    executable,
                    "-Y",
                    "verify",
                    "-f",
                    str(signers_snapshot),
                    "-I",
                    signer_id,
                    "-n",
                    namespace,
                    "-s",
                    str(signature_snapshot),
                ],
                input=payload,
                capture_output=True,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("OpenSSH signature verification timed out") from error
    if verified.returncode != 0:
        raise ValueError("signature is invalid or signer is not trusted")
    return {
        "signer_id": signer_id,
        "namespace": namespace,
        "signature_sha256": hashlib.sha256(signature_bytes).hexdigest(),
        "allowed_signers_sha256": hashlib.sha256(allowed_signers_bytes).hexdigest(),
    }


def verify_g3_signed_attestation(
    package_path: str | Path,
    attestation_path: str | Path,
    signature_path: str | Path,
    trusted_allowed_signers_path: str | Path,
    trusted_signer_policy_path: str | Path,
) -> dict[str, Any]:
    """Verify a detached OpenSSH signature over an envelope bound to exact package bytes.

    This authenticates the signer's statement and its named package files only. It does
    not prove the statement true, establish signer independence, or qualify G3 evidence.
    """
    context = _load_verification_context(
        package_path, trusted_allowed_signers_path, trusted_signer_policy_path
    )
    return _verify_one_attestation(context, attestation_path, signature_path)


def verify_g3_signed_attestation_set(
    package_path: str | Path,
    attestation_paths: Sequence[str | Path],
    trusted_allowed_signers_path: str | Path,
    trusted_signer_policy_path: str | Path,
    *,
    require_complete_qualification_attestations: bool = False,
) -> dict[str, Any]:
    """Verify package attestations and optionally require the complete G3 type roster.

    Complete type coverage is a prerequisite for independent scientific review,
    not a claim that the attestations are true or that G3 has qualified.
    """
    if type(require_complete_qualification_attestations) is not bool:
        raise TypeError("require_complete_qualification_attestations must be Boolean")
    if not attestation_paths:
        raise ValueError("at least one attestation envelope is required")
    envelope_paths = [Path(item) for item in attestation_paths]
    resolved_paths = [item.resolve(strict=True) for item in envelope_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("attestation envelope paths must be unique")
    context = _load_verification_context(
        package_path, trusted_allowed_signers_path, trusted_signer_policy_path
    )
    package_records = context.package.get("external_attestations")
    if not isinstance(package_records, list):
        raise ValueError("package external_attestations must be an array")
    expected_ids: set[str] = set()
    expected_types: set[str] = set()
    for item in package_records:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("attestation_id"), str)
            or not isinstance(item.get("attestation_type"), str)
        ):
            raise ValueError("package external attestation records need string IDs and types")
        expected_ids.add(item["attestation_id"])
        expected_types.add(item["attestation_type"])
    if len(expected_ids) != len(package_records):
        raise ValueError("package external attestation IDs must be unique strings")

    reports: list[dict[str, Any]] = []
    for envelope_path in envelope_paths:
        reports.append(
            _verify_one_attestation(
                context,
                envelope_path,
                Path(f"{envelope_path}.sig"),
            )
        )
    verified_ids = [item["attestation_id"] for item in reports]
    if len(set(verified_ids)) != len(verified_ids):
        raise ValueError("attestation set contains duplicate package attestation IDs")
    if set(verified_ids) != expected_ids:
        missing = sorted(expected_ids - set(verified_ids))
        extra = sorted(set(verified_ids) - expected_ids)
        raise ValueError(
            f"attestation set does not cover package records; missing={missing}, extra={extra}"
        )
    verified_types = {item["attestation_type"] for item in reports}
    if expected_types != verified_types:
        raise ValueError("attestation set types differ from package external attestation types")
    missing_qualification_types = sorted(ATTESTATION_TYPES - verified_types)
    if require_complete_qualification_attestations and missing_qualification_types:
        raise ValueError(
            "complete qualification attestation coverage is missing types: "
            + ", ".join(missing_qualification_types)
        )
    return {
        "schema": "loslab.g3_signed_attestation_set_verification.v1",
        "signature_status": "all_declared_attestations_verified",
        "package_consistency_status": context.consistency_status,
        "qualification_attestation_coverage_status": (
            "complete_type_roster_verified"
            if not missing_qualification_types
            else "incomplete_type_roster"
        ),
        "complete_qualification_attestation_types_required": (
            require_complete_qualification_attestations
        ),
        "missing_qualification_attestation_types": missing_qualification_types,
        "qualification_status": "NOT_QUALIFIED",
        "package_sha256": hashlib.sha256(context.package_bytes).hexdigest(),
        "attestation_count": len(reports),
        "attestation_ids": sorted(verified_ids),
        "attestation_types": sorted(verified_types),
        "unique_signer_count": len({item["signer_id"] for item in reports}),
        "attestations": reports,
        "limitations": [
            "All package-declared attestation records have valid signatures, but this does not establish that their claims are true.",
            "Distinct or independent reviewers are not inferred from signature completeness; signer independence requires external review.",
            "The package and all signatures remain proposal evidence and do not qualify G3 or authorize production RT.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("attestation", type=Path)
    parser.add_argument("signature", type=Path)
    parser.add_argument("--allowed-signers", required=True, type=Path)
    parser.add_argument("--signer-policy", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify_g3_signed_attestation(
            args.package,
            args.attestation,
            args.signature,
            args.allowed_signers,
            args.signer_policy,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
