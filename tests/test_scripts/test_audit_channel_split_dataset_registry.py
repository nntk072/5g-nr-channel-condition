from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from datasets.channel import ChannelDataset
from datasets.split_lineage import make_channel_dataset_split_registry
from datasets.split_policy import (
    SPLIT_POLICY_SIGNATURE_NAMESPACE,
    LeakageNamespace,
    SplitChildBinding,
    SplitPolicy,
    bind_split_children,
    freeze_split_policy_registry,
)
from scripts.audit_channel_split_lineage import (
    _CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
    audit_channel_datasets,
)
from tests.test_scripts.audit_channel_split_support import (
    _INVENTORY_NAMESPACE,
    _INVENTORY_SIGNER,
    _rt_subject_id,
    _sign_inventory,
    _write_child_roster,
    _write_dataset,
    _write_inventory,
)


def test_signed_split_registry_audit_rejects_provisional_rt_scene_identity(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datasets.split_lineage import make_channel_dataset_split_registry
    from scripts import audit_channel_split_lineage as audit_module

    paths = []
    for index in range(4):
        path = tmp_path / f"provisional-{index}.npz"
        _write_dataset(
            path,
            f"parent-{index}",
            float(index),
            f"{index + 1:064x}",
            complete_identity=False,
        )
        paths.append(path)
    registry, _ = make_channel_dataset_split_registry(
        [(path.stem, ChannelDataset.load(path)) for path in paths],
        spatial_buffer_m=0.0,
        policy=SplitPolicy.release_v2(seed="provisional-signed-lock"),
    )
    monkeypatch.setattr(
        audit_module.channel_audit,
        "verify_signed_split_policy_registry_attestation",
        lambda *_args, **_kwargs: (registry, object()),
    )

    with pytest.raises(ValueError, match="refuses provisional RT scene identities"):
        audit_channel_datasets(
            [(path.stem, path) for path in paths],
            spatial_buffer_m=0.0,
            split_registry_path=tmp_path / "signed-registry.json",
            split_registry_signature=tmp_path / "signed-registry.json.sig",
            trusted_split_allowed_signers=tmp_path / "allowed-signers",
            split_registry_signer_id="reviewer@example.org",
        )


def test_audit_binds_exact_input_bytes_and_returns_derived_subjects(tmp_path) -> None:
    scene_hash = "a" * 64
    first_path = tmp_path / "run-1.npz"
    second_path = tmp_path / "run-2.npz"
    _write_dataset(first_path, "parent-a", 0.0, scene_hash)
    _write_dataset(second_path, "parent-b", 0.5, scene_hash)

    report = audit_channel_datasets(
        [("run-2", second_path), ("run-1", first_path)], spatial_buffer_m=1.0
    )

    assert report["status"] == "NOT_QUALIFIED"
    assert report["lineage"]["spatially_linked_subjects"] == 2
    assert [item["name"] for item in report["inputs"]] == ["run-1", "run-2"]
    assert report["inputs"][0]["sha256"] == hashlib.sha256(first_path.read_bytes()).hexdigest()
    parents = {item["subject_id"]: item for item in report["subjects"]}
    first_id = _rt_subject_id(scene_hash, 0.0)
    second_id = _rt_subject_id(scene_hash, 0.5)
    assert set(parents) == {first_id, second_id}
    assert {
        relation["identity"]
        for relation in parents[first_id]["relations"]
        if relation["namespace"] == LeakageNamespace.SPATIAL_BUFFER.value
    } == {
        relation["identity"]
        for relation in parents[second_id]["relations"]
        if relation["namespace"] == LeakageNamespace.SPATIAL_BUFFER.value
    }
    json.dumps(report, sort_keys=True, allow_nan=False)


def test_saved_dataset_audit_rechecks_signed_v3_split_lock_and_footprints(tmp_path) -> None:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH is unavailable")
    paths = []
    for index in range(4):
        path = tmp_path / f"scene-{index}.npz"
        _write_dataset(path, f"parent-{index}", 0.0, f"{index + 1:064x}")
        paths.append(path)
    named_datasets = [(path.stem, ChannelDataset.load(path)) for path in paths]
    registry, _ = make_channel_dataset_split_registry(
        named_datasets,
        spatial_buffer_m=0.0,
        policy=SplitPolicy.release_v2(seed="signed-saved-lineage-audit"),
    )
    children = [
        SplitChildBinding(
            child_id=f"{path.stem}:row:0",
            parent_subject_id=str(ChannelDataset.load(path).metadata[0]["split_subject_id"]),
            child_kind="receiver_observation",
        )
        for path in paths
    ]
    registry = bind_split_children(registry, children)
    registry_path = tmp_path / "split-registry.json"
    freeze_split_policy_registry(registry_path, registry)
    child_roster_path = tmp_path / "receiver-observations.json"
    _write_child_roster(child_roster_path, "receiver_observation", children)

    signer_id = "split-lineage-reviewer@example.org"
    private_key = tmp_path / "split-lineage-review-key"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(private_key)],
        check=True,
        capture_output=True,
        text=True,
    )
    public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    allowed_signers = tmp_path / "split-lineage-allowed-signers"
    allowed_signers.write_text(
        f'{signer_id} namespaces="{SPLIT_POLICY_SIGNATURE_NAMESPACE},{_CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE}" {public_key}\n',
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-n",
            SPLIT_POLICY_SIGNATURE_NAMESPACE,
            "-f",
            str(private_key),
            str(registry_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    legacy_report = audit_channel_datasets(
        [(path.stem, path) for path in paths],
        spatial_buffer_m=0.0,
        split_registry_path=registry_path,
        split_registry_signature=Path(f"{registry_path}.sig"),
        trusted_split_allowed_signers=allowed_signers,
        split_registry_signer_id=signer_id,
    )
    assert legacy_report["schema"] == "loslab.channel_split_lineage_audit_bundle.v2"
    assert legacy_report["split_registry_verification"]["children_audited"] is False

    report = audit_channel_datasets(
        [(path.stem, path) for path in paths],
        spatial_buffer_m=0.0,
        split_registry_path=registry_path,
        split_registry_signature=Path(f"{registry_path}.sig"),
        trusted_split_allowed_signers=allowed_signers,
        split_registry_signer_id=signer_id,
        split_child_rosters=[("receiver_observation", child_roster_path)],
    )

    verification = report["split_registry_verification"]
    assert report["schema"] == "loslab.channel_split_lineage_audit_bundle.v3"
    assert verification["status"] == "PASS_SIGNED_PARENT_FOOTPRINT_AND_CHILD_BINDING_ROSTERS"
    assert verification["schema"] == "umcd.split_registry.v3"
    assert verification["parent_roster_audit"]["parent_count"] == 4
    assert all(
        len(verification[field]) == 64
        for field in (
            "registry_file_sha256",
            "signature_sha256",
            "allowed_signers_sha256",
        )
    )
    assert (
        verification["spatiotemporal_audit"]["footprints_sha256"]
        == registry.spatiotemporal_audit["footprints_sha256"]
    )
    assert verification["children_audited"] is False
    assert verification["child_bindings_verified"] is True
    assert verification["child_artifact_existence_verified"] is False
    assert verification["child_roster_audits"][0]["audit"]["child_count"] == 4
    assert verification["child_roster_audits"][0]["audit"]["status"] == "PASS_INPUT_SCOPED"

    artifact_root = tmp_path / "child-artifacts"
    artifact_root.mkdir()
    artifact_entries = []
    for index, binding in enumerate(children):
        relative_path = f"artifact-{index}.npz"
        artifact_path = artifact_root / relative_path
        ChannelDataset(
            h=np.ones((1, 1, 8), dtype=np.complex64),
            y=np.zeros(1, dtype=np.uint8),
            group=np.asarray([binding.parent_subject_id]),
            domain=np.asarray(["child_fixture"]),
            source=np.asarray(["synthetic"]),
            metadata=(
                {
                    "split_subject_id": binding.parent_subject_id,
                    "split_child_id": binding.child_id,
                    "split_child_kind": binding.child_kind,
                },
            ),
        ).save(artifact_path)
        artifact_bytes = artifact_path.read_bytes()
        artifact_entries.append(
            {
                "artifact_id": f"dataset-{index}",
                "artifact_type": "channel_dataset_npz",
                "child_id": binding.child_id,
                "path": relative_path,
                "bytes": len(artifact_bytes),
                "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            }
        )
    supplemental_path = artifact_root / "target-sidecar.json"
    supplemental_path.write_text('{"format":"opaque-fixture"}', encoding="utf-8")
    supplemental_bytes = supplemental_path.read_bytes()
    artifact_entries.append(
        {
            "artifact_id": "target-sidecar",
            "artifact_type": "target_sidecar_json",
            "child_id": children[0].child_id,
            "path": "target-sidecar.json",
            "bytes": len(supplemental_bytes),
            "sha256": hashlib.sha256(supplemental_bytes).hexdigest(),
        }
    )
    artifact_roster_path = tmp_path / "receiver-artifact-roster.json"
    artifact_roster_path.write_text(
        json.dumps(
            {
                "schema": "umcd.channel_dataset_child_artifact_roster.v2",
                "child_kind": "receiver_observation",
                "artifacts": [
                    {**entry, "path": f"child-artifacts/{entry['path']}"}
                    for entry in artifact_entries
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-n",
            _CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
            "-f",
            str(private_key),
            str(artifact_roster_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    artifacts_report = audit_channel_datasets(
        [(path.stem, path) for path in paths],
        spatial_buffer_m=0.0,
        split_registry_path=registry_path,
        split_registry_signature=Path(f"{registry_path}.sig"),
        trusted_split_allowed_signers=allowed_signers,
        split_registry_signer_id=signer_id,
        split_child_rosters=[("receiver_observation", child_roster_path)],
        split_child_artifact_rosters=[("receiver_observation", artifact_roster_path)],
    )
    artifact_verification = artifacts_report["split_registry_verification"]
    assert artifacts_report["schema"] == "loslab.channel_split_lineage_audit_bundle.v7"
    assert (
        artifact_verification["status"]
        == "PASS_SIGNED_PARENT_FOOTPRINT_CHILD_BINDINGS_AND_ARTIFACT_BYTES"
    )
    assert artifact_verification["child_artifact_existence_verified"] is True
    assert artifact_verification["child_artifact_bytes_verified"] is True
    assert artifact_verification["child_artifact_semantic_rows_verified"] is False
    assert artifact_verification["child_artifact_semantic_count"] == len(children)
    assert artifact_verification["child_artifact_byte_only_count"] == 1
    assert artifact_verification["child_artifact_trajectory_sample_count"] == 0
    assert artifact_verification["child_artifact_trajectory_count"] == 0
    assert artifact_verification["children_audited"] is True
    assert artifact_verification["child_artifact_roster_audits"][0]["audit"]["child_count"] == len(
        children
    )
    assert artifact_verification["child_artifact_roster_audits"][0]["audit"]["row_count"] == len(
        children
    )
    artifact_roster_metadata = artifact_verification["child_artifact_roster_audits"][0]
    assert artifact_roster_metadata["signature_namespace"] == (
        _CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE
    )
    assert artifact_roster_metadata["signer_id"] == signer_id
    assert all(
        len(artifact_roster_metadata[field]) == 64
        for field in ("sha256", "signature_sha256", "allowed_signers_sha256")
    )

    signed_roster_bytes = artifact_roster_path.read_bytes()
    artifact_roster_path.write_bytes(signed_roster_bytes + b" ")
    with pytest.raises(ValueError, match="signature is invalid"):
        audit_channel_datasets(
            [(path.stem, path) for path in paths],
            spatial_buffer_m=0.0,
            split_registry_path=registry_path,
            split_registry_signature=Path(f"{registry_path}.sig"),
            trusted_split_allowed_signers=allowed_signers,
            split_registry_signer_id=signer_id,
            split_child_rosters=[("receiver_observation", child_roster_path)],
            split_child_artifact_rosters=[("receiver_observation", artifact_roster_path)],
        )
    artifact_roster_path.write_bytes(signed_roster_bytes)

    tampered_artifact = artifact_root / artifact_entries[0]["path"]
    original_artifact_bytes = tampered_artifact.read_bytes()
    tampered_artifact.write_bytes(b"X" + original_artifact_bytes[1:])
    with pytest.raises(ValueError, match="SHA-256 differs from roster"):
        audit_channel_datasets(
            [(path.stem, path) for path in paths],
            spatial_buffer_m=0.0,
            split_registry_path=registry_path,
            split_registry_signature=Path(f"{registry_path}.sig"),
            trusted_split_allowed_signers=allowed_signers,
            split_registry_signer_id=signer_id,
            split_child_rosters=[("receiver_observation", child_roster_path)],
            split_child_artifact_rosters=[("receiver_observation", artifact_roster_path)],
        )

    _write_child_roster(child_roster_path, "receiver_observation", children[:-1])
    with pytest.raises(ValueError, match="do not cover exactly the frozen"):
        audit_channel_datasets(
            [(path.stem, path) for path in paths],
            spatial_buffer_m=0.0,
            split_registry_path=registry_path,
            split_registry_signature=Path(f"{registry_path}.sig"),
            trusted_split_allowed_signers=allowed_signers,
            split_registry_signer_id=signer_id,
            split_child_rosters=[("receiver_observation", child_roster_path)],
        )

    changed_dataset = ChannelDataset.load(paths[0])
    changed_metadata = dict(changed_dataset.metadata[0])
    changed_metadata["rx"] = [0.01, 0.0, 1.5]
    changed_metadata["split_positions_m"] = [0.01, 0.0, 1.5]
    ChannelDataset(
        h=changed_dataset.h,
        y=changed_dataset.y,
        group=changed_dataset.group,
        domain=changed_dataset.domain,
        source=changed_dataset.source,
        metadata=(changed_metadata,),
    ).save(paths[0])
    with pytest.raises(ValueError, match="footprint inventory differs from frozen registry"):
        audit_channel_datasets(
            [(path.stem, path) for path in paths],
            spatial_buffer_m=0.0,
            split_registry_path=registry_path,
            split_registry_signature=Path(f"{registry_path}.sig"),
            trusted_split_allowed_signers=allowed_signers,
            split_registry_signer_id=signer_id,
        )


def test_saved_dataset_audit_rejects_partial_split_signature_configuration(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires a registry, signature"):
        audit_channel_datasets(
            [("missing", tmp_path / "does-not-need-to-exist.npz")],
            spatial_buffer_m=0.0,
            split_registry_path=tmp_path / "registry.json",
        )


def test_audit_rejects_duplicate_names_and_missing_lineage(tmp_path) -> None:
    path = tmp_path / "run.npz"
    _write_dataset(path, "parent-a", 0.0, "a" * 64)

    with pytest.raises(ValueError, match="duplicate channel dataset name"):
        audit_channel_datasets([("same", path), ("same", path)], spatial_buffer_m=0.0)
    with pytest.raises(ValueError, match="dataset names must be canonical nonempty strings"):
        audit_channel_datasets([(" run ", path)], spatial_buffer_m=0.0)

    dataset = ChannelDataset.load(path)
    metadata = dict(dataset.metadata[0])
    del metadata["split_parent_type"]
    ChannelDataset(
        h=dataset.h,
        y=dataset.y,
        group=dataset.group,
        domain=dataset.domain,
        source=dataset.source,
        metadata=(metadata,),
    ).save(path)
    with pytest.raises(ValueError, match="invalid split_parent_type"):
        audit_channel_datasets([("legacy", path)], spatial_buffer_m=0.0)


def test_audit_checks_exact_frozen_inventory_and_preserves_unqualified_status(tmp_path) -> None:
    dataset_path = tmp_path / "run.npz"
    inventory_path = tmp_path / "inventory.json"
    _write_dataset(dataset_path, "parent-a", 0.0, "a" * 64)
    _write_inventory(inventory_path, dataset_path)

    report = audit_channel_datasets(
        [("pilot", dataset_path)],
        spatial_buffer_m=0.0,
        inventory_manifest=inventory_path,
    )

    assert report["input_completeness"]["status"] == "matched_frozen_inventory"
    assert report["input_completeness"]["matched_dataset_count"] == 1
    assert report["input_completeness"]["authority"].startswith("self-declared")
    assert report["status"] == "NOT_QUALIFIED"


def test_audit_verifies_signed_inventory_against_external_keyring(tmp_path) -> None:
    dataset_path = tmp_path / "run.npz"
    inventory_path = tmp_path / "inventory.json"
    _write_dataset(dataset_path, "parent-a", 0.0, "a" * 64)
    _write_inventory(inventory_path, dataset_path)
    signature_path, allowed_signers = _sign_inventory(inventory_path, tmp_path)

    report = audit_channel_datasets(
        [("pilot", dataset_path)],
        spatial_buffer_m=0.0,
        inventory_manifest=inventory_path,
        inventory_signature=signature_path,
        trusted_allowed_signers=allowed_signers,
        inventory_signer_id=_INVENTORY_SIGNER,
    )

    completeness = report["input_completeness"]
    assert completeness["status"] == "matched_signed_frozen_inventory"
    assert completeness["signature_namespace"] == _INVENTORY_NAMESPACE
    assert completeness["allowed_signers_sha256"]
    assert report["status"] == "NOT_QUALIFIED"
    with pytest.raises(ValueError, match="signature is invalid or signer is not trusted"):
        audit_channel_datasets(
            [("pilot", dataset_path)],
            spatial_buffer_m=0.0,
            inventory_manifest=inventory_path,
            inventory_signature=signature_path,
            trusted_allowed_signers=allowed_signers,
            inventory_signer_id="untrusted-inventory-reviewer",
        )


def test_audit_frozen_inventory_rejects_omitted_files_and_changed_bytes(tmp_path) -> None:
    first = tmp_path / "run-a.npz"
    second = tmp_path / "run-b.npz"
    inventory_path = tmp_path / "inventory.json"
    _write_dataset(first, "parent-a", 0.0, "a" * 64)
    _write_dataset(second, "parent-b", 0.0, "a" * 64)
    first_bytes = first.read_bytes()
    inventory_path.write_text(
        json.dumps(
            {
                "schema": "umcd.channel_dataset_inventory.v1",
                "frozen": True,
                "inventory_id": "fixture-v1",
                "datasets": [
                    {
                        "name": "run-a",
                        "path": first.name,
                        "sha256": hashlib.sha256(first_bytes).hexdigest(),
                        "bytes": len(first_bytes),
                        "rows": 1,
                    },
                    {
                        "name": "run-b",
                        "path": second.name,
                        "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
                        "bytes": second.stat().st_size,
                        "rows": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="do not match frozen inventory"):
        audit_channel_datasets(
            [("run-a", first)],
            spatial_buffer_m=0.0,
            inventory_manifest=inventory_path,
        )

    second.write_bytes(second.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        audit_channel_datasets(
            [("run-a", first), ("run-b", second)],
            spatial_buffer_m=0.0,
            inventory_manifest=inventory_path,
        )


@pytest.mark.parametrize("unsafe_path", ["../run.npz", "folder/..", "C:/run.npz"])
def test_audit_frozen_inventory_rejects_unsafe_paths(tmp_path, unsafe_path) -> None:
    dataset_path = tmp_path / "run.npz"
    inventory_path = tmp_path / "inventory.json"
    _write_dataset(dataset_path, "parent-a", 0.0, "a" * 64)
    _write_inventory(inventory_path, dataset_path)
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    payload["datasets"][0]["path"] = unsafe_path
    inventory_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe relative path"):
        audit_channel_datasets(
            [("pilot", dataset_path)],
            spatial_buffer_m=0.0,
            inventory_manifest=inventory_path,
        )
