import json

import pytest
from evaluation.run_manifest.common import _canonical_json_bytes
from evaluation.run_manifest.constants import SIGNER_POLICY_PURPOSE, SIGNER_POLICY_SCHEMA
from evaluation.run_manifest.signatures import (
    _authorize_run_manifest_signer,
    _require_independent_signer_policy_custody,
    _validate_run_verification_report,
    _verify_signed_runtime_inventory_bindings,
)


def signer_policy(*, signers=None, policy_id="review-policy", schema=SIGNER_POLICY_SCHEMA):
    return {
        "schema": schema,
        "policy_id": policy_id,
        "signers": signers
        if signers is not None
        else [
            {
                "principal": "alice",
                "organization": "research-lab",
                "roles": ["reviewer", "release-manager"],
                "allowed_purposes": [SIGNER_POLICY_PURPOSE],
            }
        ],
    }


def test_signer_policy_resolves_authorized_principal_and_sorts_roles():
    raw = _canonical_json_bytes(signer_policy())

    result = _authorize_run_manifest_signer("alice", raw)

    assert result["policy_id"] == "review-policy"
    assert result["signer_id"] == "alice"
    assert result["organization"] == "research-lab"
    assert result["roles"] == "release-manager,reviewer"
    assert len(result["policy_sha256"]) == 64


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ({"schema": SIGNER_POLICY_SCHEMA}, "fields do not match"),
        (signer_policy(schema="unsupported"), "unsupported.*schema"),
        (signer_policy(policy_id=" "), "needs a policy ID"),
        (signer_policy(signers=[]), "needs a policy ID"),
        (signer_policy(signers=[{"principal": "alice"}]), "entries must match"),
        (
            signer_policy(
                signers=[
                    {
                        "principal": " alice ",
                        "organization": "lab",
                        "roles": ["reviewer"],
                        "allowed_purposes": [SIGNER_POLICY_PURPOSE],
                    }
                ]
            ),
            "principal and organization",
        ),
        (
            signer_policy(
                signers=[
                    {
                        "principal": "alice",
                        "organization": " ",
                        "roles": ["reviewer"],
                        "allowed_purposes": [SIGNER_POLICY_PURPOSE],
                    }
                ]
            ),
            "principal and organization",
        ),
        (
            signer_policy(
                signers=[
                    {
                        "principal": "alice",
                        "organization": "lab",
                        "roles": ["r"],
                        "allowed_purposes": [SIGNER_POLICY_PURPOSE],
                    },
                    {
                        "principal": "alice",
                        "organization": "lab",
                        "roles": ["r"],
                        "allowed_purposes": [SIGNER_POLICY_PURPOSE],
                    },
                ]
            ),
            "principals must be unique",
        ),
        (
            signer_policy(
                signers=[
                    {
                        "principal": "alice",
                        "organization": "lab",
                        "roles": ["reviewer", "reviewer"],
                        "allowed_purposes": [SIGNER_POLICY_PURPOSE],
                    }
                ]
            ),
            "roles must be unique",
        ),
        (
            signer_policy(
                signers=[
                    {
                        "principal": "alice",
                        "organization": "lab",
                        "roles": [" "],
                        "allowed_purposes": [SIGNER_POLICY_PURPOSE],
                    }
                ]
            ),
            "roles must be unique",
        ),
        (
            signer_policy(
                signers=[
                    {
                        "principal": "alice",
                        "organization": "lab",
                        "roles": ["reviewer"],
                        "allowed_purposes": [],
                    }
                ]
            ),
            "allowed_purposes must be unique",
        ),
        (
            signer_policy(
                signers=[
                    {
                        "principal": "alice",
                        "organization": "lab",
                        "roles": ["reviewer"],
                        "allowed_purposes": ["other-purpose"],
                    }
                ]
            ),
            "unsupported run-manifest purpose",
        ),
    ],
)
def test_signer_policy_rejects_invalid_authorization_contracts(policy, message):
    with pytest.raises(ValueError, match=message):
        _authorize_run_manifest_signer("alice", _canonical_json_bytes(policy))


def test_signer_policy_requires_canonical_json_and_a_trusted_signer():
    policy = signer_policy()
    with pytest.raises(ValueError, match="canonical JSON bytes"):
        _authorize_run_manifest_signer("alice", json.dumps(policy, indent=2).encode())
    with pytest.raises(ValueError, match="absent from the trusted role policy"):
        _authorize_run_manifest_signer("bob", _canonical_json_bytes(policy))


def test_signer_policy_requires_independent_custodian_and_trust_keyring():
    manifest_signature = {"signer_id": "alice", "allowed_signers_sha256": "a"}
    policy_signature = {"signer_id": "bob", "allowed_signers_sha256": "b"}
    _require_independent_signer_policy_custody(manifest_signature, policy_signature)

    with pytest.raises(ValueError, match="custodian must differ"):
        _require_independent_signer_policy_custody(
            manifest_signature, {**policy_signature, "signer_id": "alice"}
        )
    with pytest.raises(ValueError, match="keyring must differ"):
        _require_independent_signer_policy_custody(
            manifest_signature, {**policy_signature, "allowed_signers_sha256": "a"}
        )


def valid_runtime_bindings():
    source_paths = (
        "pyproject.toml",
        "uv.lock",
        "integrations/runtime_inventory.py",
        "integrations/runtime_compatibility.py",
        "scripts/report_runtime_inventory.py",
    )
    digests = {path: f"digest-{index}" for index, path in enumerate(source_paths)}
    versions = {
        "numpy": "1.26",
        "scikit-learn": "1.5",
        "torch": "2.5",
        "sionna": "1.2",
        "sionna-rt": "2.2",
        "sionna-phy": "1.2",
    }
    runtime_identity = {
        "source_revision": "revision-a",
        "python": "3.12",
        "python_implementation": "CPython",
        "platform_system": "Linux",
        "platform_release": "kernel-a",
        "machine": "x86_64",
        "packages": versions,
    }
    distributions = [
        {"name": name, "installed": True, "version": version} for name, version in versions.items()
    ]
    distributions.append({"name": "sionna-no-rt", "installed": True, "version": "1.2"})
    runtime_inventory = {
        "schema": "umcd.runtime_inventory.v3",
        "status": "SELF_REPORTED_INTEGRITY_ONLY",
        "qualification": "NOT_QUALIFIED",
        "source": {
            "tracked_worktree_clean": True,
            "git_commit": "revision-a",
            "pyproject.toml": digests["pyproject.toml"],
            "uv.lock": digests["uv.lock"],
            "runtime_inventory_sources": {
                path: digests[path]
                for path in source_paths
                if path.startswith(("integrations/", "scripts/"))
            },
        },
        "python": {"version": "3.12", "implementation": "CPython"},
        "platform": {"system": "Linux", "release": "kernel-a", "machine": "x86_64"},
        "loaded_native_libraries": {"status": "COMPLETE", "unhashed": []},
        "distributions": distributions,
    }
    evaluator_sources = [{"path": path, "sha256": digest} for path, digest in digests.items()]
    return runtime_inventory, runtime_identity, evaluator_sources


def test_signed_runtime_inventory_binds_versions_platform_and_evaluator_sources():
    args = valid_runtime_bindings()
    _verify_signed_runtime_inventory_bindings(*args)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("source", "tracked_worktree_clean"), False, "clean tracked source tree"),
        (("source", "git_commit"), "other-revision", "commit differs"),
        (("python", "version"), "other-version", "platform differs"),
        (("loaded_native_libraries", "unhashed"), ["lib.so"], "complete v3"),
    ],
)
def test_signed_runtime_inventory_rejects_identity_mismatch(path, value, message):
    runtime_inventory, runtime_identity, evaluator_sources = valid_runtime_bindings()
    target = runtime_inventory
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        _verify_signed_runtime_inventory_bindings(
            runtime_inventory, runtime_identity, evaluator_sources
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing-source", "missing source, platform, or loaded-library identity"),
        ("missing-packages", "package identity is missing"),
        ("duplicate-distribution", "duplicate or malformed distribution names"),
        ("missing-numpy", "omits package identity: numpy"),
        ("numpy-absent-but-installed", "says numpy is absent"),
        ("numpy-version", "package version differs.*numpy"),
        ("pyproject-digest", "pyproject.toml digest differs"),
        ("source-roster", "source-code digest roster is missing"),
        ("runtime-source-digest", "source digest differs from evaluator roster"),
    ],
)
def test_signed_runtime_inventory_rejects_package_and_source_roster_mismatches(change, message):
    runtime_inventory, runtime_identity, evaluator_sources = valid_runtime_bindings()
    if change == "missing-source":
        runtime_inventory["source"] = None
    elif change == "missing-packages":
        runtime_inventory["distributions"] = None
    elif change == "duplicate-distribution":
        runtime_inventory["distributions"].append(
            {"name": "numpy", "installed": True, "version": "1.26"}
        )
    elif change == "missing-numpy":
        runtime_inventory["distributions"] = [
            item for item in runtime_inventory["distributions"] if item["name"] != "numpy"
        ]
    elif change == "numpy-absent-but-installed":
        runtime_identity["packages"]["numpy"] = "not installed"
    elif change == "numpy-version":
        runtime_identity["packages"]["numpy"] = "different-version"
    elif change == "pyproject-digest":
        runtime_inventory["source"]["pyproject.toml"] = "changed-digest"
    elif change == "source-roster":
        runtime_inventory["source"]["runtime_inventory_sources"] = None
    else:
        runtime_inventory["source"]["runtime_inventory_sources"][
            "scripts/report_runtime_inventory.py"
        ] = "changed-digest"

    with pytest.raises(ValueError, match=message):
        _verify_signed_runtime_inventory_bindings(
            runtime_inventory, runtime_identity, evaluator_sources
        )


def test_run_verification_report_is_checked_against_repository_schema():
    report = {}
    with pytest.raises(RuntimeError, match="violates schema"):
        _validate_run_verification_report(report)
    assert len(report["report_schema_sha256"]) == 64
