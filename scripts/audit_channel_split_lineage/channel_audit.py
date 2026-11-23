"""Channel dataset audits and the command-line entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from datasets.channel import ChannelDataset
from datasets.legacy_rt_split import upgrade_legacy_rt_split_metadata
from datasets.split_lineage import derive_channel_dataset_split_evidence
from datasets.split_policy import (
    SPLIT_POLICY_SIGNATURE_NAMESPACE,
    audit_split_policy_child_roster,
    audit_split_policy_parent_roster,
    verify_openssh_signed_artifact,
    verify_registry_spatiotemporal_audit,
    verify_signed_split_policy_registry_attestation,
)

from .child_artifacts import (
    _CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
    _INVENTORY_SIGNATURE_NAMESPACE,
    _REPOSITORY_ROOT,
    _attach_reconstructed_scene_assets,
    _load_frozen_inventory,
    _load_legacy_asset_reconstruction,
    _load_split_child_roster,
    _parse_dataset,
    _parse_split_child_roster,
    _read_inventory_relative_file,
    _read_stable_regular_file,
    _validate_split_audit_report,
)
from .split_artifacts import (
    _load_split_child_artifact_roster,
)


def audit_channel_datasets(
    inputs: Sequence[tuple[str, Path]],
    *,
    spatial_buffer_m: float,
    inventory_manifest: Path | None = None,
    inventory_signature: Path | None = None,
    trusted_allowed_signers: Path | None = None,
    inventory_signer_id: str | None = None,
    split_registry_path: Path | None = None,
    split_registry_signature: Path | None = None,
    trusted_split_allowed_signers: Path | None = None,
    split_registry_signer_id: str | None = None,
    split_child_rosters: Sequence[tuple[str, Path]] = (),
    split_child_artifact_rosters: Sequence[tuple[str, Path]] = (),
    require_complete_split_audit: bool = False,
    derive_legacy_rt_split_identities: bool = False,
    legacy_scene_assets_manifest: Path | None = None,
) -> dict[str, Any]:
    """Audit named NPZ datasets and bind the result to the exact bytes read.

    Args:
        inputs: Named saved ChannelDataset files to audit.
        spatial_buffer_m: Minimum separation used for spatial leakage links.
        inventory_manifest: Optional frozen inventory covering every input.
        inventory_signature: Optional detached signature for the inventory.
        trusted_allowed_signers: External keyring used to verify that signature.
        inventory_signer_id: Expected signer principal for the inventory.
        split_registry_path: Optional signed immutable split-registry v3 lock.
        split_registry_signature: Detached OpenSSH signature for that registry.
        trusted_split_allowed_signers: External trust file for the registry signer.
        split_registry_signer_id: Expected principal for the split-registry signature.
        split_child_rosters: Optional complete JSON binding roster per frozen child
            kind. This verifies registry-to-roster correspondence, not artifact bytes.
        split_child_artifact_rosters: Optional complete per-kind roster of child
            ChannelDataset NPZ paths, byte counts, and hashes, relative to each
            roster file. Each row must bind the roster child ID/kind and frozen
            parent subject in metadata. Each roster requires a detached `.sig`
            OpenSSH signature from the configured split-registry signer under a
            separate namespace.
        require_complete_split_audit: Require signed corpus inventory and signed
            split registry, plus complete binding and semantic artifact rosters
            for every registered child kind. This checks release integrity inputs;
            it does not qualify the dataset scientifically.
        derive_legacy_rt_split_identities: Explicitly derive missing split metadata
            in memory from complete RT scene provenance; never rewrites inputs.
        legacy_scene_assets_manifest: Optional retrospective asset reconstruction
            bound to exact legacy dataset bytes and historical source provenance.

    Returns:
        Hash-bound, explicitly unqualified lineage report for the supplied inputs.

    Raises:
        ValueError: An input, inventory, or requested legacy identity is invalid.
    """
    if not inputs:
        raise ValueError("at least one named ChannelDataset input is required")
    if type(require_complete_split_audit) is not bool:
        raise TypeError("require_complete_split_audit must be a boolean")
    if require_complete_split_audit:
        if inventory_manifest is None or inventory_signature is None:
            raise ValueError("complete split audit requires a signed frozen corpus inventory")
        if split_registry_path is None:
            raise ValueError("complete split audit requires a signed split registry")
    split_signature_arguments = (
        split_registry_path,
        split_registry_signature,
        trusted_split_allowed_signers,
        split_registry_signer_id,
    )
    if any(value is not None for value in split_signature_arguments) and any(
        value is None for value in split_signature_arguments
    ):
        raise ValueError(
            "split-registry verification requires a registry, signature, external trust file, "
            "and signer ID"
        )
    if split_child_rosters and split_registry_path is None:
        raise ValueError("split child rosters require a signed split registry")
    if split_child_artifact_rosters and split_registry_path is None:
        raise ValueError("split child artifact rosters require a signed split registry")
    if split_child_artifact_rosters and not split_child_rosters:
        raise ValueError("split child artifact rosters require complete child binding rosters")
    if any(
        not isinstance(entry, (tuple, list)) or len(entry) != 2 for entry in split_child_rosters
    ):
        raise ValueError("split child rosters must be (child_kind, path) pairs")
    child_roster_kinds = [kind for kind, _ in split_child_rosters]
    if any(
        not isinstance(kind, str) or not kind or kind != kind.strip() for kind in child_roster_kinds
    ):
        raise ValueError("split child roster kinds must be canonical nonempty strings")
    if len(set(child_roster_kinds)) != len(child_roster_kinds):
        raise ValueError("split child roster kinds must be unique")
    if any(
        not isinstance(entry, (tuple, list)) or len(entry) != 2
        for entry in split_child_artifact_rosters
    ):
        raise ValueError("split child artifact rosters must be (child_kind, path) pairs")
    child_artifact_kinds = [kind for kind, _ in split_child_artifact_rosters]
    if any(
        not isinstance(kind, str) or not kind or kind != kind.strip()
        for kind in child_artifact_kinds
    ):
        raise ValueError("split child artifact roster kinds must be canonical nonempty strings")
    if len(set(child_artifact_kinds)) != len(child_artifact_kinds):
        raise ValueError("split child artifact roster kinds must be unique")
    if legacy_scene_assets_manifest is not None and not derive_legacy_rt_split_identities:
        raise ValueError(
            "legacy scene-asset reconstruction requires --derive-legacy-rt-split-identities"
        )

    declared_inventory: dict[str, dict[str, Any]] | None = None
    inventory_metadata: dict[str, Any] | None = None
    signature_arguments = (
        inventory_signature,
        trusted_allowed_signers,
        inventory_signer_id,
    )
    if any(value is not None for value in signature_arguments) and (
        inventory_manifest is None or any(value is None for value in signature_arguments)
    ):
        raise ValueError(
            "signed inventory verification requires a manifest, signature, allowed-signers file, and signer ID"
        )
    if inventory_manifest is not None:
        verified_artifact = None
        if inventory_signature is not None:
            assert trusted_allowed_signers is not None and inventory_signer_id is not None
            verified_artifact = verify_openssh_signed_artifact(
                inventory_manifest,
                inventory_signature,
                trusted_allowed_signers_path=trusted_allowed_signers,
                signer_id=inventory_signer_id,
                namespace=_INVENTORY_SIGNATURE_NAMESPACE,
            )
        declared_inventory, inventory_metadata = _load_frozen_inventory(
            inventory_manifest, verified_artifact
        )
        actual_names = {name for name, _ in inputs if isinstance(name, str)}
        if actual_names != set(declared_inventory):
            missing = sorted(set(declared_inventory) - actual_names)
            extra = sorted(actual_names - set(declared_inventory))
            raise ValueError(
                f"dataset inputs do not match frozen inventory (missing={missing}, extra={extra})"
            )

    reconstructed_assets: dict[str, dict[str, Any]] | None = None
    reconstruction_metadata: dict[str, Any] | None = None
    if legacy_scene_assets_manifest is not None:
        reconstructed_assets, reconstruction_metadata = _load_legacy_asset_reconstruction(
            legacy_scene_assets_manifest
        )
        actual_names = {name for name, _ in inputs if isinstance(name, str)}
        if actual_names != set(reconstructed_assets):
            missing = sorted(set(reconstructed_assets) - actual_names)
            extra = sorted(actual_names - set(reconstructed_assets))
            raise ValueError(
                "dataset inputs do not match legacy scene reconstruction "
                f"(missing={missing}, extra={extra})"
            )

    loaded: list[tuple[str, ChannelDataset]] = []
    inventory: list[dict[str, Any]] = []
    migrated_datasets: list[str] = []
    migrated_rows = 0
    seen_names: set[str] = set()
    for name, path in inputs:
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("dataset names must be canonical nonempty strings")
        normalized_name = name
        if normalized_name in seen_names:
            raise ValueError(f"duplicate channel dataset name: {normalized_name}")
        seen_names.add(normalized_name)
        dataset_path = Path(os.path.abspath(path))
        expected: dict[str, Any] | None = None
        if declared_inventory is not None:
            expected = declared_inventory[normalized_name]
            if dataset_path != expected["absolute_path"]:
                raise ValueError(f"dataset path differs from frozen inventory: {normalized_name}")
        if expected is None:
            payload = _read_stable_regular_file(
                dataset_path,
                label=f"ChannelDataset {normalized_name!r}",
            )
        else:
            assert inventory_manifest is not None
            payload = _read_inventory_relative_file(
                Path(os.path.abspath(inventory_manifest)).parent,
                expected["path"],
                label=f"ChannelDataset {normalized_name!r}",
            )
        if expected is not None:
            if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError(
                    f"dataset SHA-256 differs from frozen inventory: {normalized_name}"
                )
            if len(payload) != expected["bytes"]:
                raise ValueError(
                    f"dataset byte count differs from frozen inventory: {normalized_name}"
                )
        channel_dataset = ChannelDataset.load_bytes(payload)
        if expected is not None and len(channel_dataset.y) != expected["rows"]:
            raise ValueError(f"dataset row count differs from frozen inventory: {normalized_name}")
        if derive_legacy_rt_split_identities:
            if reconstructed_assets is not None:
                assert reconstruction_metadata is not None
                channel_dataset = _attach_reconstructed_scene_assets(
                    channel_dataset,
                    dataset_name=normalized_name,
                    entry=reconstructed_assets[normalized_name],
                    input_sha256=hashlib.sha256(payload).hexdigest(),
                    input_bytes=len(payload),
                    manifest_metadata=reconstruction_metadata,
                )
            channel_dataset, was_upgraded = upgrade_legacy_rt_split_metadata(
                channel_dataset, dataset_name=normalized_name
            )
            if was_upgraded:
                migrated_datasets.append(normalized_name)
                migrated_rows += len(channel_dataset.y)
        loaded.append((normalized_name, channel_dataset))
        try:
            display_path = dataset_path.relative_to(_REPOSITORY_ROOT).as_posix()
        except ValueError:
            display_path = str(dataset_path)
        inventory.append(
            {
                "name": normalized_name,
                "path": display_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "rows": len(loaded[-1][1].y),
            }
        )

    subjects, footprints, lineage_report = derive_channel_dataset_split_evidence(
        loaded, spatial_buffer_m=spatial_buffer_m
    )
    inventory.sort(key=lambda item: item["name"])
    identity_derivation: dict[str, Any] = {
        "mode": (
            "opt_in_historical_rt_scene_asset_reconstruction"
            if reconstructed_assets is not None
            else "opt_in_complete_rt_legacy_metadata"
            if derive_legacy_rt_split_identities
            else "disabled"
        ),
        "datasets_upgraded_in_memory": migrated_datasets,
        "rows_upgraded_in_memory": migrated_rows,
        "input_hashes_cover_original_bytes": True,
    }
    if reconstruction_metadata is not None:
        identity_derivation["scene_asset_reconstruction"] = reconstruction_metadata
    report = {
        "schema": "loslab.channel_split_lineage_audit_bundle.v1",
        "status": "NOT_QUALIFIED",
        "spatial_buffer_m": float(spatial_buffer_m),
        "identity_derivation": identity_derivation,
        "input_completeness": (
            {
                "status": (
                    "matched_signed_frozen_inventory"
                    if "signature_sha256" in inventory_metadata
                    else "matched_frozen_inventory"
                ),
                **inventory_metadata,
                "matched_dataset_count": len(inventory),
                "authority": (
                    "OpenSSH signature verified against the caller-supplied allowed-signers file"
                    if "signature_sha256" in inventory_metadata
                    else "self-declared inventory; signature and external trust are not verified"
                ),
            }
            if inventory_metadata is not None
            else {
                "status": "unverified_input_scope",
                "authority": "no frozen corpus inventory supplied",
            }
        ),
        "inputs": inventory,
        "lineage": lineage_report,
        "subjects": [
            {
                "subject_id": subject.subject_id,
                **subject.to_dict(),
            }
            for subject in subjects
        ],
        "interpretation": (
            "This report covers only the listed input bytes. It does not prove corpus "
            "completeness, acquisition provenance, missing relations, independent review, "
            "or release qualification."
        ),
    }
    if split_registry_path is not None:
        assert split_registry_signature is not None
        assert trusted_split_allowed_signers is not None
        assert split_registry_signer_id is not None
        registry, signed_artifact = verify_signed_split_policy_registry_attestation(
            split_registry_path,
            split_registry_signature,
            trusted_allowed_signers_path=trusted_split_allowed_signers,
            signer_id=split_registry_signer_id,
        )
        if require_complete_split_audit and not registry.child_bindings:
            raise ValueError("complete split audit requires at least one frozen child binding")
        if lineage_report.get("subjects_with_provisional_split_identity", 0) != 0:
            raise ValueError(
                "signed split-registry audit refuses provisional RT scene identities; "
                "resolve complete scene-asset lineage before release-lock comparison"
            )
        parent_roster_audit = audit_split_policy_parent_roster(registry, subjects)
        footprint_audit = verify_registry_spatiotemporal_audit(
            registry, footprints, spatial_buffer_m=spatial_buffer_m
        )
        registry_payload = registry.to_dict()
        child_roster_reports: list[dict[str, Any]] = []
        child_artifact_reports: list[dict[str, Any]] = []
        if split_child_rosters:
            expected_kinds = {binding.child_kind for binding in registry.child_bindings.values()}
            supplied_kinds = set(child_roster_kinds)
            if supplied_kinds != expected_kinds:
                raise ValueError(
                    "split child roster files must cover every frozen child kind "
                    f"(missing={sorted(expected_kinds - supplied_kinds)}, "
                    f"extra={sorted(supplied_kinds - expected_kinds)})"
                )
            for child_kind, roster_path in sorted(split_child_rosters, key=lambda item: item[0]):
                bindings, roster_metadata = _load_split_child_roster(
                    roster_path, expected_child_kind=child_kind
                )
                child_roster_reports.append(
                    {
                        **roster_metadata,
                        "audit": audit_split_policy_child_roster(
                            registry, bindings, child_kind=child_kind
                        ),
                    }
                )
        if split_child_artifact_rosters:
            expected_kinds = {binding.child_kind for binding in registry.child_bindings.values()}
            supplied_kinds = set(child_artifact_kinds)
            if supplied_kinds != expected_kinds:
                raise ValueError(
                    "split child artifact roster files must cover every frozen child kind "
                    f"(missing={sorted(expected_kinds - supplied_kinds)}, "
                    f"extra={sorted(supplied_kinds - expected_kinds)})"
                )
            for child_kind, artifact_roster_path in sorted(
                split_child_artifact_rosters, key=lambda item: item[0]
            ):
                assert trusted_split_allowed_signers is not None
                assert split_registry_signer_id is not None
                artifact_roster_signature = Path(f"{artifact_roster_path}.sig")
                verified_roster = verify_openssh_signed_artifact(
                    artifact_roster_path,
                    artifact_roster_signature,
                    trusted_allowed_signers_path=trusted_split_allowed_signers,
                    signer_id=split_registry_signer_id,
                    namespace=_CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
                )
                expected_child_bindings = {
                    child_id: binding
                    for child_id, binding in registry.child_bindings.items()
                    if binding.child_kind == child_kind
                }
                child_artifact_reports.append(
                    _load_split_child_artifact_roster(
                        artifact_roster_path,
                        expected_child_kind=child_kind,
                        expected_child_bindings=expected_child_bindings,
                        verified_artifact=verified_roster,
                        split_registry=registry,
                    )
                )
        byte_only_artifact_count = sum(
            roster["audit"]["byte_only_artifact_count"] for roster in child_artifact_reports
        )
        semantic_artifact_count = sum(
            roster["audit"]["semantic_artifact_count"] for roster in child_artifact_reports
        )
        trajectory_sample_count = sum(
            roster["audit"]["trajectory_sample_count"] for roster in child_artifact_reports
        )
        trajectory_count = sum(
            roster["audit"]["trajectory_count"] for roster in child_artifact_reports
        )
        episode_count = sum(
            roster["audit"].get("episode_count", 0) for roster in child_artifact_reports
        )
        scenario_gbsm_run_count = sum(
            roster["audit"].get("scenario_gbsm_run_count", 0) for roster in child_artifact_reports
        )
        scenario_gbsm_parent_count = sum(
            roster["audit"].get("scenario_gbsm_parent_count", 0)
            for roster in child_artifact_reports
        )
        expected_child_kinds = {binding.child_kind for binding in registry.child_bindings.values()}
        if require_complete_split_audit:
            supplied_binding_kinds = {kind for kind, _ in split_child_rosters}
            supplied_artifact_kinds = {kind for kind, _ in split_child_artifact_rosters}
            if supplied_binding_kinds != expected_child_kinds:
                raise ValueError(
                    "complete split audit requires binding rosters for every frozen child kind "
                    f"(missing={sorted(expected_child_kinds - supplied_binding_kinds)}, "
                    f"extra={sorted(supplied_binding_kinds - expected_child_kinds)})"
                )
            if supplied_artifact_kinds != expected_child_kinds:
                raise ValueError(
                    "complete split audit requires artifact rosters for every frozen child kind "
                    f"(missing={sorted(expected_child_kinds - supplied_artifact_kinds)}, "
                    f"extra={sorted(supplied_artifact_kinds - expected_child_kinds)})"
                )
            if byte_only_artifact_count:
                raise ValueError(
                    "complete split audit rejects child artifacts without semantic validators"
                )
        roster_schemas = {roster["audit"]["roster_schema"] for roster in child_artifact_reports}
        has_umcd_episode_artifacts = bool(
            roster_schemas
            & {
                "umcd.channel_dataset_child_artifact_roster.v4",
                "umcd.channel_dataset_child_artifact_roster.v5",
                "umcd.channel_dataset_child_artifact_roster.v6",
            }
        )
        report["schema"] = (
            "loslab.channel_split_lineage_audit_bundle.v10"
            if "umcd.channel_dataset_child_artifact_roster.v6" in roster_schemas
            else "loslab.channel_split_lineage_audit_bundle.v9"
            if "umcd.channel_dataset_child_artifact_roster.v5" in roster_schemas
            else "loslab.channel_split_lineage_audit_bundle.v8"
            if has_umcd_episode_artifacts
            else "loslab.channel_split_lineage_audit_bundle.v7"
            if split_child_artifact_rosters
            else "loslab.channel_split_lineage_audit_bundle.v3"
            if split_child_rosters
            else "loslab.channel_split_lineage_audit_bundle.v2"
        )
        report["split_registry_verification"] = {
            "status": (
                "PASS_SIGNED_COMPLETE_SPLIT_INTEGRITY"
                if require_complete_split_audit
                else (
                    "PASS_SIGNED_PARENT_FOOTPRINT_CHILD_BINDINGS_AND_ARTIFACT_BYTES"
                    if byte_only_artifact_count
                    else "PASS_SIGNED_PARENT_FOOTPRINT_CHILD_BINDINGS_AND_SEMANTIC_ARTIFACT_ROWS"
                )
                if split_child_artifact_rosters
                else "PASS_SIGNED_PARENT_FOOTPRINT_AND_CHILD_BINDING_ROSTERS"
                if split_child_rosters
                else "PASS_SIGNED_PARENT_AND_FOOTPRINT_AUDIT"
            ),
            "schema": registry_payload["schema"],
            "registry_sha256": registry_payload["registry_sha256"],
            "signer_id": split_registry_signer_id,
            "signature_namespace": SPLIT_POLICY_SIGNATURE_NAMESPACE,
            "registry_file_sha256": signed_artifact.artifact_sha256,
            "signature_sha256": signed_artifact.signature_sha256,
            "allowed_signers_sha256": signed_artifact.allowed_signers_sha256,
            "parent_roster_audit": parent_roster_audit,
            "spatiotemporal_audit": footprint_audit,
            "children_audited": bool(split_child_artifact_rosters),
            "child_bindings_verified": bool(split_child_rosters),
            "child_artifact_existence_verified": bool(split_child_artifact_rosters),
            "child_artifact_bytes_verified": bool(split_child_artifact_rosters),
            "child_artifact_semantic_rows_verified": (
                bool(split_child_artifact_rosters) and byte_only_artifact_count == 0
            ),
            "child_artifact_semantic_count": semantic_artifact_count,
            "child_artifact_byte_only_count": byte_only_artifact_count,
            "child_artifact_trajectory_sample_count": trajectory_sample_count,
            "child_artifact_trajectory_count": trajectory_count,
            "child_artifact_episode_count": episode_count,
            "child_artifact_scenario_gbsm_run_count": scenario_gbsm_run_count,
            "child_artifact_scenario_gbsm_parent_count": scenario_gbsm_parent_count,
            "complete_split_integrity_audit": require_complete_split_audit,
            "child_roster_audits": child_roster_reports,
            "child_artifact_roster_audits": child_artifact_reports,
        }
    _validate_split_audit_report(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset,
        required=True,
        metavar="NAME=PATH",
        help="saved ChannelDataset NPZ; repeat once per dataset/run",
    )
    parser.add_argument("--spatial-buffer-m", type=float, required=True)
    parser.add_argument(
        "--derive-legacy-rt-split-identities",
        action="store_true",
        help=(
            "opt in to derive missing RT split IDs in memory from complete saved scene closure, "
            "receiver positions, and matching block-size metadata"
        ),
    )
    parser.add_argument(
        "--legacy-scene-assets-manifest",
        type=Path,
        help=(
            "explicitly use a retrospective asset reconstruction bound to legacy input bytes; "
            "the report remains NOT_QUALIFIED"
        ),
    )
    parser.add_argument(
        "--inventory-manifest",
        type=Path,
        help="frozen JSON inventory that declares the complete dataset file set",
    )
    parser.add_argument("--inventory-signature", type=Path)
    parser.add_argument("--allowed-signers", type=Path)
    parser.add_argument("--inventory-signer-id")
    parser.add_argument("--split-registry", type=Path)
    parser.add_argument("--split-registry-signature", type=Path)
    parser.add_argument("--split-registry-allowed-signers", type=Path)
    parser.add_argument("--split-registry-signer-id")
    parser.add_argument(
        "--require-complete-split-audit",
        action="store_true",
        help=(
            "require signed corpus inventory and split registry plus complete, "
            "semantically validated child artifact rosters; does not qualify scientific evidence"
        ),
    )
    parser.add_argument(
        "--split-child-roster",
        action="append",
        type=_parse_split_child_roster,
        default=[],
        metavar="KIND=PATH",
        help="complete JSON roster for one frozen split-child kind; requires a signed registry",
    )
    parser.add_argument(
        "--split-child-artifact-roster",
        action="append",
        type=_parse_split_child_roster,
        default=[],
        metavar="KIND=PATH",
        help=(
            "complete JSON path/size/hash roster for one frozen child kind; "
            "requires PATH.sig signed by the split-registry signer under the child-artifact-roster namespace; "
            "artifact paths are relative to the roster directory and must identify ChannelDataset NPZ files "
            "whose row metadata matches the frozen child ID, kind, and parent"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    try:
        report = audit_channel_datasets(
            args.dataset,
            spatial_buffer_m=args.spatial_buffer_m,
            inventory_manifest=args.inventory_manifest,
            inventory_signature=args.inventory_signature,
            trusted_allowed_signers=args.allowed_signers,
            inventory_signer_id=args.inventory_signer_id,
            split_registry_path=args.split_registry,
            split_registry_signature=args.split_registry_signature,
            trusted_split_allowed_signers=args.split_registry_allowed_signers,
            split_registry_signer_id=args.split_registry_signer_id,
            split_child_rosters=args.split_child_roster,
            split_child_artifact_rosters=args.split_child_artifact_roster,
            require_complete_split_audit=args.require_complete_split_audit,
            derive_legacy_rt_split_identities=args.derive_legacy_rt_split_identities,
            legacy_scene_assets_manifest=args.legacy_scene_assets_manifest,
        )
    except (OSError, RuntimeError, TypeError, ValueError, RecursionError) as error:
        parser.exit(2, f"error: {error}\n")
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": report["status"],
                "dataset_count": report["lineage"]["dataset_count"],
                "unique_split_subjects": report["lineage"]["unique_split_subjects"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
