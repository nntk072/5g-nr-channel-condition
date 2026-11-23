"""Signed runtime inventory, signer authorization and run verification report checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from evaluation.acquisition_package import (
    MAX_SCHEMA_BYTES,
    _json_bytes,
)
from evaluation.run_manifest.common import (
    _canonical_json_bytes,
)
from evaluation.run_manifest.constants import (
    SIGNER_POLICY_PURPOSE,
    SIGNER_POLICY_SCHEMA,
    VERIFICATION_REPORT_SCHEMA_RELATIVE_PATH,
)
from evaluation.run_manifest.sources import (
    _read_rooted_source_file,
)


def _verify_signed_runtime_inventory_bindings(
    runtime_inventory: dict[str, Any],
    runtime_identity: dict[str, Any],
    evaluator_sources: list[dict[str, Any]],
) -> None:
    """Join signed loaded-runtime evidence to the run's source and platform IDs."""
    source = runtime_inventory.get("source")
    python_identity = runtime_inventory.get("python")
    platform_identity = runtime_inventory.get("platform")
    loaded = runtime_inventory.get("loaded_native_libraries")
    if not all(
        isinstance(value, dict) for value in (source, python_identity, platform_identity, loaded)
    ):
        raise ValueError(
            "signed runtime inventory is missing source, platform, or loaded-library identity"
        )
    assert isinstance(source, dict)
    assert isinstance(python_identity, dict)
    assert isinstance(platform_identity, dict)
    assert isinstance(loaded, dict)
    if (
        runtime_inventory.get("schema") != "umcd.runtime_inventory.v3"
        or runtime_inventory.get("status") != "SELF_REPORTED_INTEGRITY_ONLY"
        or runtime_inventory.get("qualification") != "NOT_QUALIFIED"
        or loaded.get("status") != "COMPLETE"
        or loaded.get("unhashed") != []
    ):
        raise ValueError("strict G3 runtime inventory must be complete v3 integrity evidence")
    if source.get("tracked_worktree_clean") is not True:
        raise ValueError("strict G3 runtime inventory requires a clean tracked source tree")
    if source.get("git_commit") != runtime_identity.get("source_revision"):
        raise ValueError("runtime inventory commit differs from the run runtime identity")
    if (
        python_identity.get("version") != runtime_identity.get("python")
        or python_identity.get("implementation") != runtime_identity.get("python_implementation")
        or platform_identity.get("system") != runtime_identity.get("platform_system")
        or platform_identity.get("release") != runtime_identity.get("platform_release")
        or platform_identity.get("machine") != runtime_identity.get("machine")
    ):
        raise ValueError("runtime inventory platform differs from the run runtime identity")
    packages = runtime_identity.get("packages")
    distributions = runtime_inventory.get("distributions")
    if not isinstance(packages, dict) or not isinstance(distributions, list):
        raise ValueError("runtime inventory package identity is missing")
    distribution_by_name = {
        item.get("name", "").lower().replace("_", "-"): item
        for item in distributions
        if isinstance(item, dict)
    }
    if len(distribution_by_name) != len(distributions):
        raise ValueError("runtime inventory contains duplicate or malformed distribution names")
    package_distributions = {
        "numpy": ("numpy",),
        "scikit-learn": ("scikit-learn",),
        "torch": ("torch",),
        "sionna": ("sionna",),
        "sionna-rt": ("sionna-rt", "sionna-no-rt"),
        "sionna-phy": ("sionna-no-rt",),
    }
    for package_name, candidates in package_distributions.items():
        declared_version = packages.get(package_name)
        matching_distributions = [
            distribution_by_name[name] for name in candidates if name in distribution_by_name
        ]
        if not matching_distributions:
            raise ValueError(f"runtime inventory omits package identity: {package_name}")
        installed_matches = [
            distribution
            for distribution in matching_distributions
            if distribution.get("installed") is True
        ]
        if declared_version == "not installed":
            if installed_matches:
                raise ValueError(
                    f"runtime identity says {package_name} is absent but inventory records it installed"
                )
        elif not any(
            distribution.get("version") == declared_version for distribution in installed_matches
        ):
            raise ValueError(
                f"runtime inventory package version differs from runtime identity: {package_name}"
            )
    source_by_path = {item["path"]: item["sha256"] for item in evaluator_sources}
    for path in ("pyproject.toml", "uv.lock"):
        if source.get(path) != source_by_path.get(path):
            raise ValueError(
                f"runtime inventory {path} digest differs from evaluator source roster"
            )
    runtime_sources = source.get("runtime_inventory_sources")
    if not isinstance(runtime_sources, dict):
        raise ValueError("runtime inventory source-code digest roster is missing")
    for path in (
        "integrations/runtime_inventory.py",
        "integrations/runtime_compatibility.py",
        "scripts/report_runtime_inventory.py",
    ):
        if runtime_sources.get(path) != source_by_path.get(path):
            raise ValueError(
                f"runtime inventory source digest differs from evaluator roster: {path}"
            )


def _authorize_run_manifest_signer(signer_id: str, policy_bytes: bytes) -> dict[str, str]:
    """Resolve a cryptographically verified principal to an externally governed role."""
    policy = _json_bytes(policy_bytes, "G3 run-manifest signer policy")
    if not isinstance(policy, dict) or set(policy) != {"schema", "policy_id", "signers"}:
        raise ValueError("run-manifest signer policy fields do not match v1")
    if policy_bytes != _canonical_json_bytes(policy):
        raise ValueError("run-manifest signer policy must use canonical JSON bytes")
    if policy.get("schema") != SIGNER_POLICY_SCHEMA:
        raise ValueError("unsupported run-manifest signer policy schema")
    policy_id = policy.get("policy_id")
    signers = policy.get("signers")
    if (
        not isinstance(policy_id, str)
        or not policy_id.strip()
        or not isinstance(signers, list)
        or not signers
    ):
        raise ValueError("run-manifest signer policy needs a policy ID and signer list")
    principals: set[str] = set()
    authorized: dict[str, Any] | None = None
    for entry in signers:
        fields = {"principal", "organization", "roles", "allowed_purposes"}
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError("run-manifest signer entries must match the v1 contract")
        principal = entry.get("principal")
        organization = entry.get("organization")
        roles = entry.get("roles")
        purposes = entry.get("allowed_purposes")
        if (
            not isinstance(principal, str)
            or not principal.strip()
            or principal != principal.strip()
            or not isinstance(organization, str)
            or not organization.strip()
        ):
            raise ValueError("signer principal and organization must be nonempty strings")
        if principal in principals:
            raise ValueError("run-manifest signer principals must be unique")
        principals.add(principal)
        if (
            not isinstance(roles, list)
            or not roles
            or any(not isinstance(role, str) or not role.strip() for role in roles)
            or len(set(roles)) != len(roles)
        ):
            raise ValueError("signer roles must be unique nonempty strings")
        if (
            not isinstance(purposes, list)
            or not purposes
            or any(not isinstance(purpose, str) or not purpose.strip() for purpose in purposes)
            or len(set(purposes)) != len(purposes)
        ):
            raise ValueError("signer allowed_purposes must be unique nonempty strings")
        if any(purpose != SIGNER_POLICY_PURPOSE for purpose in purposes):
            raise ValueError("signer policy contains an unsupported run-manifest purpose")
        if principal == signer_id:
            authorized = entry
    if authorized is None:
        raise ValueError("run-manifest signer is absent from the trusted role policy")
    if SIGNER_POLICY_PURPOSE not in authorized["allowed_purposes"]:
        raise ValueError("run-manifest signer role policy does not authorize this purpose")
    return {
        "policy_id": policy_id,
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "signer_id": signer_id,
        "organization": authorized["organization"],
        "roles": ",".join(sorted(authorized["roles"])),
    }


def _require_independent_signer_policy_custody(
    manifest_signature: dict[str, str], policy_signature: dict[str, str]
) -> None:
    """Require a distinct principal and allowed-signers trust input for policy bytes."""
    if manifest_signature.get("signer_id") == policy_signature.get("signer_id"):
        raise ValueError("signer-policy custodian must differ from the run-manifest signer")
    if manifest_signature.get("allowed_signers_sha256") == policy_signature.get(
        "allowed_signers_sha256"
    ):
        raise ValueError("signer-policy trust keyring must differ from the run-manifest keyring")


def _validate_run_verification_report(report: dict[str, Any]) -> None:
    """Validate the public verifier result against its strict output contract."""
    source_dir = Path(__file__).resolve().parents[2]
    schema_bytes = _read_rooted_source_file(
        source_dir,
        VERIFICATION_REPORT_SCHEMA_RELATIVE_PATH,
        limit=MAX_SCHEMA_BYTES,
    )
    report["report_schema_sha256"] = hashlib.sha256(schema_bytes).hexdigest()
    schema = _json_bytes(schema_bytes, "G3 run verification report schema")
    import jsonschema

    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(report))
    if errors:
        raise RuntimeError(
            "G3 run verification report violates schema: "
            + "; ".join(error.message for error in errors[:10])
        )
