"""Typed evidence-tier claims for multipath dataset artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

LINEAGE_STATUSES = (
    "proposed",
    "synthetic_debug",
    "native_verified",
    "scenario_qualified",
    "geometry_qualified",
    "receiver_qualified",
    "externally_calibrated",
    "external_tested",
)

_REQUIRED_EVIDENCE: dict[str, frozenset[str]] = {
    "proposed": frozenset(),
    "synthetic_debug": frozenset({"regression_report_sha256"}),
    "native_verified": frozenset(
        {"native_run_manifest_sha256", "numeric_shape_gauge_report_sha256"}
    ),
    "scenario_qualified": frozenset(
        {"native_run_manifest_sha256", "scenario_conformance_report_sha256"}
    ),
    "geometry_qualified": frozenset(
        {
            "geometry_material_audit_sha256",
            "visibility_audit_sha256",
            "convergence_audit_sha256",
        }
    ),
    "receiver_qualified": frozenset(
        {"receiver_contract_sha256", "receiver_qualification_report_sha256"}
    ),
    "externally_calibrated": frozenset(
        {
            "calibration_report_sha256",
            "measurement_package_sha256",
            "disjointness_audit_sha256",
        }
    ),
    "external_tested": frozenset(
        {
            "external_test_report_sha256",
            "measurement_package_sha256",
            "label_coverage_report_sha256",
        }
    ),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class LineageStatusClaim:
    """One artifact-scoped status label and its content-addressed evidence.

    The claim records which evidence a producer associates with a lineage tier.
    It does not validate the evidence contents, signer authority, or scientific
    acceptance; those checks belong to the corresponding gate auditor.
    """

    status: str
    scope: str
    evidence_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.status not in LINEAGE_STATUSES:
            raise ValueError(f"status must be one of {LINEAGE_STATUSES}")
        if (
            not isinstance(self.scope, str)
            or not self.scope.strip()
            or self.scope != self.scope.strip()
        ):
            raise ValueError("scope must be a non-empty canonical string")
        if not isinstance(self.evidence_sha256, Mapping) or any(
            not isinstance(key, str) or not key.strip() for key in self.evidence_sha256
        ):
            raise TypeError("evidence_sha256 must be a string-keyed mapping")
        evidence = dict(self.evidence_sha256)
        required = _REQUIRED_EVIDENCE[self.status]
        if not required.issubset(evidence):
            missing = sorted(required - set(evidence))
            raise ValueError(f"{self.status} claim is missing required evidence hashes: {missing}")
        if any(
            not isinstance(value, str) or not _SHA256.fullmatch(value)
            for value in evidence.values()
        ):
            raise ValueError("evidence_sha256 values must be lowercase SHA256 digests")
        object.__setattr__(self, "evidence_sha256", MappingProxyType(evidence))

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical schema-v1 claim payload."""
        return {
            "schema": "umcd.lineage_status_claim.v1",
            "status": self.status,
            "scope": self.scope,
            "evidence_sha256": dict(sorted(self.evidence_sha256.items())),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LineageStatusClaim:
        """Parse one claim, rejecting unknown fields and malformed values."""
        if set(value) != {"schema", "status", "scope", "evidence_sha256"}:
            raise ValueError("lineage status claim fields do not match schema v1")
        if value.get("schema") != "umcd.lineage_status_claim.v1":
            raise ValueError("unsupported lineage status claim schema")
        claim = cls(
            status=value["status"],
            scope=value["scope"],
            evidence_sha256=value["evidence_sha256"],
        )
        if claim.to_dict() != dict(value):
            raise ValueError("lineage status claim representation is not canonical")
        return claim


def validate_lineage_claims(claims: Any) -> tuple[LineageStatusClaim, ...]:
    """Validate and freeze a sequence of claims with unique status/scope pairs."""
    if not isinstance(claims, (tuple, list)) or any(
        not isinstance(item, LineageStatusClaim) for item in claims
    ):
        raise TypeError("lineage_claims must be a sequence of LineageStatusClaim values")
    frozen = tuple(claims)
    identities = [(item.status, item.scope) for item in frozen]
    if len(set(identities)) != len(identities):
        raise ValueError("lineage claims must have unique status/scope pairs")
    return frozen


def audit_lineage_evidence_payloads(
    claims: Any,
    evidence_payloads_by_sha256: Mapping[str, bytes],
    *,
    max_total_bytes: int = 128 * 1024 * 1024,
) -> dict[str, Any]:
    """Verify that supplied evidence payloads exactly match the claim roster.

    This checks byte identity only. It does not parse evidence, verify
    signatures, establish independence, or authorize the claimed tier.
    Callers are responsible for acquiring payloads through their trusted,
    stable-file reader before passing them here.
    """
    frozen_claims = validate_lineage_claims(claims)
    if isinstance(max_total_bytes, bool) or not isinstance(max_total_bytes, int):
        raise TypeError("max_total_bytes must be an integer")
    if max_total_bytes <= 0:
        raise ValueError("max_total_bytes must be positive")
    if not isinstance(evidence_payloads_by_sha256, Mapping):
        raise TypeError("evidence_payloads_by_sha256 must be a digest-keyed mapping")

    expected = {digest for claim in frozen_claims for digest in claim.evidence_sha256.values()}
    if set(evidence_payloads_by_sha256) != expected:
        raise ValueError("supplied evidence payload roster does not match lineage claims")
    total_bytes = 0
    for digest, payload in evidence_payloads_by_sha256.items():
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("evidence payload keys must be lowercase SHA256 digests")
        if not isinstance(payload, bytes):
            raise TypeError("evidence payload values must be immutable bytes")
        total_bytes += len(payload)
        if total_bytes > max_total_bytes:
            raise ValueError("lineage evidence payloads exceed max_total_bytes")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"lineage evidence payload does not match digest {digest}")

    claims_bytes = json.dumps(
        [claim.to_dict() for claim in frozen_claims],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    core = {
        "schema": "umcd.lineage_evidence_audit.v1",
        "status": "EVIDENCE_BYTES_VERIFIED" if expected else "NO_EVIDENCE_REQUIRED",
        "scientific_status": "NOT_QUALIFIED",
        "lineage_claims_sha256": hashlib.sha256(claims_bytes).hexdigest(),
        "verified_evidence_sha256": sorted(expected),
        "verified_evidence_bytes": total_bytes,
    }
    report_bytes = json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return {**core, "audit_sha256": hashlib.sha256(report_bytes).hexdigest()}
