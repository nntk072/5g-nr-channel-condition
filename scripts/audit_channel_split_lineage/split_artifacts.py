"""Split roster loading for the channel-split lineage audit."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from datasets.split_policy import (
    SplitChildBinding,
    VerifiedOpenSSHArtifact,
)

from .child_artifacts import (
    _MAX_CHILD_ROSTER_BYTES,
    _SHA256_RE,
    _V3_SEMANTIC_CHILD_ARTIFACT_TYPES,
    _V4_SEMANTIC_CHILD_ARTIFACT_TYPES,
    _V6_MULTI_OWNER_ARTIFACT_TYPES,
    _V6_SEMANTIC_CHILD_ARTIFACT_TYPES,
    _enumerate_regular_files_beneath,
    _hash_inventory_relative_file,
    _read_channel_dataset_child_rows,
    _read_stable_regular_file,
    _require_exact_child_object_owners,
    _require_shared_content_owners_share_split,
    _unique_json_object,
)


def _load_split_child_artifact_roster(
    path: Path,
    *,
    expected_child_kind: str,
    expected_child_bindings: dict[str, SplitChildBinding],
    verified_artifact: VerifiedOpenSSHArtifact,
    split_registry: Any | None = None,
) -> dict[str, Any]:
    payload = verified_artifact.payload
    if len(payload) > _MAX_CHILD_ROSTER_BYTES:
        raise ValueError("split child artifact roster exceeds the 16 MiB limit")
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_json_object)
    if not isinstance(value, dict) or value.get("child_kind") != expected_child_kind:
        raise ValueError("split child artifact roster has an invalid schema or child kind")
    schema = value.get("schema")
    if schema == "umcd.channel_dataset_child_artifact_roster.v1":
        if set(value) != {"schema", "child_kind", "artifacts"}:
            raise ValueError("split child artifact v1 roster has unsupported fields")
        roster_version = 1
    elif schema == "umcd.channel_dataset_child_artifact_roster.v2":
        if set(value) != {"schema", "child_kind", "artifacts"}:
            raise ValueError("split child artifact v2 roster has unsupported fields")
        roster_version = 2
    elif schema == "umcd.channel_dataset_child_artifact_roster.v3":
        if set(value) != {"schema", "child_kind", "artifact_root", "artifacts"}:
            raise ValueError("split child artifact v3 roster has unsupported fields")
        roster_version = 3
    elif schema == "umcd.channel_dataset_child_artifact_roster.v4":
        if set(value) != {"schema", "child_kind", "artifact_root", "artifacts"}:
            raise ValueError("split child artifact v4 roster has unsupported fields")
        roster_version = 4
    elif schema == "umcd.channel_dataset_child_artifact_roster.v5":
        if set(value) != {"schema", "child_kind", "artifact_root", "artifacts"}:
            raise ValueError("split child artifact v5 roster has unsupported fields")
        roster_version = 5
    elif schema == "umcd.channel_dataset_child_artifact_roster.v6":
        if set(value) != {"schema", "child_kind", "artifact_root", "artifacts"}:
            raise ValueError("split child artifact v6 roster has unsupported fields")
        roster_version = 6
    else:
        raise ValueError("split child artifact roster has an unsupported schema")
    entries = value["artifacts"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("split child artifact roster artifacts must be a nonempty array")
    observed_child_ids: set[str] = set()
    observed_artifact_ids: set[str] = set()
    observed_paths: set[str] = set()
    observed_trajectory_ids: set[str] = set()
    observed_episode_ids: set[str] = set()
    observed_scenario_run_ids: set[str] = set()
    declared: list[dict[str, Any]] = []
    roster_directory = Path(os.path.abspath(path)).parent
    artifact_root_name: str | None = None
    if roster_version >= 3:
        artifact_root_name = value["artifact_root"]
        if (
            not isinstance(artifact_root_name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", artifact_root_name) is None
            or artifact_root_name in {".", ".."}
        ):
            raise ValueError("split child artifact v3 root must be one safe directory name")
    artifact_root = (
        roster_directory / artifact_root_name
        if artifact_root_name is not None
        else roster_directory
    )
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"split child artifact {index} has invalid fields")
        if roster_version == 1:
            expected_fields = {"child_id", "path", "bytes", "sha256"}
        elif roster_version in {5, 6}:
            artifact_type_hint = entry.get("artifact_type")
            owner_field = (
                "child_ids"
                if (
                    artifact_type_hint == "umcd_episode_store_content_npz"
                    if roster_version == 5
                    else artifact_type_hint in _V6_MULTI_OWNER_ARTIFACT_TYPES
                )
                else "child_id"
            )
            expected_fields = {
                "artifact_id",
                "artifact_type",
                owner_field,
                "path",
                "bytes",
                "sha256",
            }
        else:
            expected_fields = {
                "artifact_id",
                "artifact_type",
                "child_id",
                "path",
                "bytes",
                "sha256",
            }
        if set(entry) != expected_fields:
            raise ValueError(f"split child artifact {index} has invalid fields")
        relative_path = entry["path"]
        byte_count = entry["bytes"]
        digest = entry["sha256"]
        if roster_version == 1:
            child_id = entry["child_id"]
            if not isinstance(child_id, str) or not child_id or child_id.strip() != child_id:
                raise ValueError(f"split child artifact {index} has an invalid child ID")
            owner_child_ids = [child_id]
            observed_child_ids.add(child_id)
            artifact_id = child_id
            artifact_type = "channel_dataset_npz"
            if artifact_id in observed_artifact_ids:
                raise ValueError(f"split child artifact roster repeats child ID: {child_id}")
        else:
            artifact_id = entry["artifact_id"]
            artifact_type = entry["artifact_type"]
            if (roster_version == 5 and artifact_type == "umcd_episode_store_content_npz") or (
                roster_version == 6 and artifact_type in _V6_MULTI_OWNER_ARTIFACT_TYPES
            ):
                owner_child_ids = entry["child_ids"]
                if (
                    not isinstance(owner_child_ids, list)
                    or not owner_child_ids
                    or any(
                        not isinstance(owner, str) or not owner or owner.strip() != owner
                        for owner in owner_child_ids
                    )
                    or owner_child_ids != sorted(set(owner_child_ids))
                ):
                    raise ValueError(
                        f"split child artifact {index} has an invalid canonical child_ids roster"
                    )
                observed_child_ids.update(owner_child_ids)
            else:
                child_id = entry["child_id"]
                if not isinstance(child_id, str) or not child_id or child_id.strip() != child_id:
                    raise ValueError(f"split child artifact {index} has an invalid child ID")
                owner_child_ids = [child_id]
                observed_child_ids.add(child_id)
            if (
                not isinstance(artifact_id, str)
                or not artifact_id
                or artifact_id.strip() != artifact_id
            ):
                raise ValueError(f"split child artifact {index} has an invalid artifact ID")
            if (
                not isinstance(artifact_type, str)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", artifact_type) is None
            ):
                raise ValueError(f"split child artifact {index} has an invalid artifact type")
            semantic_types = (
                _V6_SEMANTIC_CHILD_ARTIFACT_TYPES
                if roster_version >= 6
                else _V4_SEMANTIC_CHILD_ARTIFACT_TYPES
                if roster_version >= 4
                else _V3_SEMANTIC_CHILD_ARTIFACT_TYPES
            )
            if roster_version >= 3 and artifact_type not in semantic_types:
                raise ValueError(
                    "split child artifact v3 requires a semantic validator for artifact type "
                    f"{artifact_type!r}"
                )
            if artifact_id in observed_artifact_ids:
                raise ValueError(f"split child artifact roster repeats artifact ID: {artifact_id}")
        observed_artifact_ids.add(artifact_id)
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or relative_path.startswith("/")
            or "\\" in relative_path
            or re.match(r"^[A-Za-z]:", relative_path) is not None
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
            or re.fullmatch(r"[A-Za-z0-9._/-]+", relative_path) is None
        ):
            raise ValueError(f"split child artifact {index} has an unsafe relative path")
        if relative_path in observed_paths:
            raise ValueError(f"split child artifact roster repeats path: {relative_path}")
        observed_paths.add(relative_path)
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError(f"split child artifact {index} has an invalid byte count")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"split child artifact {index} has an invalid SHA-256")
        if artifact_type == "channel_dataset_npz" and not relative_path.endswith(".npz"):
            raise ValueError("channel_dataset_npz artifacts must use an .npz path")
        if artifact_type == "umcd_episode_store_manifest_json" and not relative_path.endswith(
            ".json"
        ):
            raise ValueError("umcd_episode_store_manifest_json artifacts must use a .json path")
        if artifact_type == "umcd_episode_store_content_npz" and not relative_path.endswith(".npz"):
            raise ValueError("umcd_episode_store_content_npz artifacts must use a .npz path")
        declaration = {
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "path": relative_path,
            "bytes": byte_count,
            "sha256": digest,
        }
        if (roster_version == 5 and artifact_type == "umcd_episode_store_content_npz") or (
            roster_version == 6 and artifact_type in _V6_MULTI_OWNER_ARTIFACT_TYPES
        ):
            if any(owner not in expected_child_bindings for owner in owner_child_ids):
                raise ValueError(f"split child artifact {index} references an unknown child ID")
            if split_registry is not None:
                _require_shared_content_owners_share_split(
                    artifact_type,
                    owner_child_ids,
                    expected_child_bindings,
                    split_registry,
                )
            declaration["child_ids"] = list(owner_child_ids)
        else:
            declaration["child_id"] = owner_child_ids[0]
        declared.append(declaration)
    if roster_version >= 3:
        declared_paths = {entry["path"] for entry in declared}
        observed_paths_under_root = _enumerate_regular_files_beneath(artifact_root)
        if observed_paths_under_root != declared_paths:
            raise ValueError(
                "split child artifact v3 roster does not exactly cover its artifact_root "
                f"(missing={sorted(observed_paths_under_root - declared_paths)}, "
                f"unlisted={sorted(declared_paths - observed_paths_under_root)})"
            )
    scenario_content_owner_references: dict[str, list[tuple[str, str]]] = {}
    if roster_version >= 6:
        for candidate in declared:
            if candidate["artifact_type"] != "scenario_gbsm_run_manifest_json":
                continue
            manifest_bytes = _read_stable_regular_file(
                artifact_root / candidate["path"],
                label="scenario GBSM run manifest for owner inventory",
                max_bytes=64 * 1024 * 1024,
            )
            try:
                manifest_value = json.loads(
                    manifest_bytes.decode("utf-8"),
                    object_pairs_hook=_unique_json_object,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
                raise ValueError(
                    "scenario GBSM run manifest is not valid bounded UTF-8 JSON"
                ) from error
            if not isinstance(manifest_value, dict) or not isinstance(
                manifest_value.get("records"), list
            ):
                raise ValueError("scenario GBSM run manifest record roster is invalid")
            owner_parent_children: dict[str, list[str]] = {}
            for owner in candidate["child_ids"]:
                owner_parent_children.setdefault(
                    expected_child_bindings[owner].parent_subject_id, []
                ).append(owner)
            for reference in manifest_value["records"]:
                if not isinstance(reference, dict):
                    raise ValueError("scenario GBSM record reference is malformed")
                parent_id = reference.get("sample_id")
                object_relative_path = reference.get("relative_path")
                if (
                    not isinstance(parent_id, str)
                    or not parent_id
                    or parent_id.strip() != parent_id
                    or not isinstance(object_relative_path, str)
                    or not object_relative_path.startswith("scenario_gbsm/records/")
                    or not object_relative_path.endswith(".npz")
                ):
                    raise ValueError("scenario GBSM record reference has invalid identity or path")
                owners = owner_parent_children.get(parent_id)
                if not owners:
                    raise ValueError("scenario GBSM run manifest omits a record parent owner")
                scenario_content_owner_references.setdefault(object_relative_path, []).extend(
                    (owner, object_relative_path) for owner in owners
                )
    expected_child_ids = set(expected_child_bindings)
    if observed_child_ids != expected_child_ids:
        raise ValueError(
            "split child artifact roster does not cover exactly the frozen children "
            f"(missing={sorted(expected_child_ids - observed_child_ids)}, "
            f"extra={sorted(observed_child_ids - expected_child_ids)})"
        )
    semantic_child_ids = {
        entry["child_id"] for entry in declared if entry["artifact_type"] == "channel_dataset_npz"
    }
    if semantic_child_ids != expected_child_ids:
        raise ValueError(
            "split child artifact roster must include a ChannelDataset NPZ for every frozen child "
            f"(missing={sorted(expected_child_ids - semantic_child_ids)})"
        )
    verified: list[dict[str, Any]] = []
    for entry in declared:
        artifact_id = entry["artifact_id"]
        artifact_type = entry["artifact_type"]
        child_ids = entry.get("child_ids")
        child_id_value = entry.get("child_id", child_ids[0] if child_ids else None)
        if (
            not isinstance(child_id_value, str)
            or not child_id_value
            or child_id_value != child_id_value.strip()
        ):
            raise ValueError("split child artifact requires a canonical child_id")
        child_id = child_id_value
        relative_path = entry["path"]
        byte_count = entry["bytes"]
        digest = entry["sha256"]
        actual_bytes, actual_digest = _hash_inventory_relative_file(
            artifact_root, relative_path, label=f"split child artifact {child_id!r}"
        )
        if actual_bytes != byte_count:
            raise ValueError(f"split child artifact byte count differs from roster: {child_id}")
        if actual_digest != digest:
            raise ValueError(f"split child artifact SHA-256 differs from roster: {artifact_id}")
        if artifact_type == "channel_dataset_npz":
            semantic_rows = _read_channel_dataset_child_rows(
                artifact_root,
                relative_path,
                child_id=child_id,
                child_kind=expected_child_kind,
                parent_subject_id=expected_child_bindings[child_id].parent_subject_id,
                expected_bytes=actual_bytes,
                expected_sha256=actual_digest,
            )
            semantic_rows["semantic_validator"] = "channel_dataset_child_parent_rows_v1"
        elif artifact_type == "nr_trajectory_store_manifest_json":
            if not relative_path.startswith("manifests/trajectories/"):
                raise ValueError(
                    "nr_trajectory_store_manifest_json must be under manifests/trajectories/"
                )
            from datasets.nr_store import read_nr_trajectory

            trajectory = read_nr_trajectory(artifact_root, relative_path)
            if trajectory.trajectory_id in observed_trajectory_ids:
                raise ValueError(
                    f"split child artifact roster repeats NR trajectory ID: "
                    f"{trajectory.trajectory_id}"
                )
            observed_trajectory_ids.add(trajectory.trajectory_id)
            parent_subject_id = expected_child_bindings[child_id].parent_subject_id
            for sample_index, record in enumerate(trajectory.records):
                expected_lineage = {
                    "split_subject_id": parent_subject_id,
                    "split_child_id": child_id,
                    "split_child_kind": expected_child_kind,
                }
                for field, expected_value in expected_lineage.items():
                    if record.provenance.get(field) != expected_value:
                        raise ValueError(
                            f"NR trajectory child artifact {artifact_id!r} sample "
                            f"{sample_index} has a different {field}"
                        )
            sample_ids_payload = json.dumps(
                list(trajectory.sample_ids),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            semantic_rows = {
                "semantic_validator": "nr_trajectory_store_v1",
                "trajectory_id": trajectory.trajectory_id,
                "trajectory_sample_count": len(trajectory.sample_ids),
                "trajectory_sample_ids_sha256": hashlib.sha256(sample_ids_payload).hexdigest(),
            }
        elif artifact_type in {
            "nr_record_store_manifest_json",
            "nr_record_store_content_npz",
        }:
            from datasets.nr_store import read_nr_record

            if artifact_type == "nr_record_store_manifest_json":
                if not relative_path.startswith("manifests/") or relative_path.startswith(
                    "manifests/trajectories/"
                ):
                    raise ValueError(
                        "nr_record_store_manifest_json must be under manifests/ outside "
                        "manifests/trajectories/"
                    )
                manifest_relative_path = relative_path
            else:
                if not relative_path.startswith("records/") or not relative_path.endswith(".npz"):
                    raise ValueError(
                        "nr_record_store_content_npz must be beneath records/ with an .npz suffix"
                    )
                record_manifest_candidates: list[tuple[str, dict[str, Any], str]] = []
                for candidate in declared:
                    if candidate["artifact_type"] != "nr_record_store_manifest_json":
                        continue
                    manifest_path = candidate["path"]
                    manifest_bytes = _read_stable_regular_file(
                        artifact_root / manifest_path,
                        label="NR record manifest for content artifact",
                        max_bytes=1 * 1024 * 1024,
                    )
                    try:
                        manifest_value = json.loads(
                            manifest_bytes.decode("utf-8"),
                            object_pairs_hook=_unique_json_object,
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
                        raise ValueError(
                            "NR record manifest is not valid bounded UTF-8 JSON"
                        ) from error
                    if not isinstance(manifest_value, dict):
                        raise ValueError("NR record manifest must be a JSON object")
                    if manifest_value.get("relative_path") == relative_path:
                        record_manifest_candidates.append(
                            (manifest_path, manifest_value, candidate["child_id"])
                        )
                if len(record_manifest_candidates) != 1:
                    raise ValueError(
                        "nr_record_store_content_npz must be referenced by exactly one "
                        "rostered record manifest"
                    )
                manifest_relative_path, matched_manifest, manifest_child_id = (
                    record_manifest_candidates[0]
                )
                _require_exact_child_object_owners(
                    relative_path,
                    [child_id],
                    [(manifest_child_id, matched_manifest["relative_path"])],
                    label="NR record content object",
                )
                content_sha256 = matched_manifest.get("content_sha256")
                if (
                    not isinstance(content_sha256, str)
                    or _SHA256_RE.fullmatch(content_sha256) is None
                    or relative_path != f"records/{content_sha256[:2]}/{content_sha256}.npz"
                ):
                    raise ValueError(
                        "NR record content path does not match its canonical content hash"
                    )
            record = read_nr_record(artifact_root, manifest_relative_path)
            parent_subject_id = expected_child_bindings[child_id].parent_subject_id
            expected_lineage = {
                "split_subject_id": parent_subject_id,
                "split_child_id": child_id,
                "split_child_kind": expected_child_kind,
            }
            for field, expected_value in expected_lineage.items():
                if record.provenance.get(field) != expected_value:
                    raise ValueError(
                        f"NR record child artifact {artifact_id!r} has a different {field}"
                    )
            semantic_rows = {
                "semantic_validator": "nr_record_store_v1",
                "nr_sample_id": record.observation.sample_id,
                "nr_target_count": len(record.targets),
            }
        elif artifact_type == "scenario_gbsm_run_manifest_json":
            if roster_version < 6:
                raise ValueError("scenario GBSM run manifests require child artifact roster v6")
            if not relative_path.startswith("scenario_gbsm/manifests/"):
                raise ValueError(
                    "scenario_gbsm_run_manifest_json must be under scenario_gbsm/manifests/"
                )
            if split_registry is None:
                raise ValueError("scenario GBSM semantic admission requires the frozen registry")
            from datasets.scenario_gbsm import read_scenario_gbsm_run

            run = read_scenario_gbsm_run(artifact_root, relative_path)
            if run.run_id in observed_scenario_run_ids:
                raise ValueError(
                    f"split child artifact roster repeats scenario run ID: {run.run_id}"
                )
            observed_scenario_run_ids.add(run.run_id)
            parent_ids = {record.sample_id for record in run.records}
            parent_to_children: dict[str, list[str]] = {}
            for candidate_child_id, binding in expected_child_bindings.items():
                parent_to_children.setdefault(binding.parent_subject_id, []).append(
                    candidate_child_id
                )
            missing_parents = parent_ids - set(parent_to_children)
            if missing_parents:
                raise ValueError(
                    "scenario GBSM run has parents absent from the frozen child roster: "
                    f"{sorted(missing_parents)}"
                )
            expected_owners = sorted(
                child for parent in parent_ids for child in parent_to_children[parent]
            )
            if child_ids != expected_owners:
                raise ValueError(
                    "scenario GBSM run child_ids do not exactly cover its frozen parent rows"
                )
            split_audit = run.audit_split_assignment(split_registry)
            semantic_rows = {
                "semantic_validator": "scenario_gbsm_split_assignment_v1",
                "scenario_run_id": run.run_id,
                "scenario_parent_count": len(parent_ids),
                "scenario_record_count": len(run.records),
                "scenario_split_component_id": split_audit["component_id"],
                "scenario_split": split_audit["split"],
                "scenario_split_audit_sha256": split_audit["audit_sha256"],
            }
        elif artifact_type == "scenario_gbsm_record_content_npz":
            if roster_version < 6:
                raise ValueError("scenario GBSM record objects require child artifact roster v6")
            if not relative_path.startswith("scenario_gbsm/records/") or not relative_path.endswith(
                ".npz"
            ):
                raise ValueError(
                    "scenario_gbsm_record_content_npz must be beneath scenario_gbsm/records/"
                )
            scenario_references = scenario_content_owner_references.get(relative_path, [])
            if not scenario_references:
                raise ValueError(
                    "scenario GBSM record content must be referenced by a rostered run manifest"
                )
            _require_exact_child_object_owners(
                relative_path,
                child_ids if child_ids is not None else [child_id],
                scenario_references,
                label="scenario GBSM record content object",
            )
            semantic_rows = {"semantic_validator": "scenario_gbsm_record_content_v1"}
        elif artifact_type == "umcd_episode_store_manifest_json":
            if not relative_path.startswith("manifests/episodes/"):
                raise ValueError(
                    "umcd_episode_store_manifest_json must be under manifests/episodes/"
                )
            from datasets.episode_store import read_episode_bundle

            bundle = read_episode_bundle(
                artifact_root, relative_path, max_shard_bytes=512 * 1024 * 1024
            )
            episode_id = bundle.episode.episode_id
            if episode_id in observed_episode_ids:
                raise ValueError(
                    f"split child artifact roster repeats UMCD episode ID: {episode_id}"
                )
            observed_episode_ids.add(episode_id)
            parent_subject_id = expected_child_bindings[child_id].parent_subject_id
            if episode_id != parent_subject_id:
                raise ValueError(
                    f"UMCD episode child artifact {artifact_id!r} has a different "
                    "split parent episode ID"
                )
            semantic_rows = {
                "semantic_validator": "umcd_episode_store_v1",
                "episode_id": episode_id,
                "episode_link_count": len(bundle.episode.links),
                "episode_frame_count": len(bundle.episode.frame_times_s),
                "episode_target_count": len(bundle.targets),
            }
        elif artifact_type == "umcd_episode_store_content_npz":
            if not relative_path.startswith("objects/"):
                raise ValueError("umcd_episode_store_content_npz must be beneath objects/")
            episode_manifest_candidates: list[tuple[str, str, dict[str, Any]]] = []
            for candidate in declared:
                if candidate["artifact_type"] != "umcd_episode_store_manifest_json":
                    continue
                manifest_path = candidate["path"]
                manifest_bytes = _read_stable_regular_file(
                    artifact_root / manifest_path,
                    label="UMCD episode bundle manifest for content artifact",
                    max_bytes=1024 * 1024,
                )
                try:
                    manifest_value = json.loads(
                        manifest_bytes.decode("utf-8"),
                        object_pairs_hook=_unique_json_object,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
                    raise ValueError(
                        "UMCD episode bundle manifest is not valid bounded UTF-8 JSON"
                    ) from error
                if not isinstance(manifest_value, dict) or not isinstance(
                    manifest_value.get("layers"), list
                ):
                    raise ValueError("UMCD episode bundle manifest layers are invalid")
                for reference in manifest_value["layers"]:
                    if isinstance(reference, dict) and reference.get("path") == relative_path:
                        episode_manifest_candidates.append(
                            (manifest_path, candidate["child_id"], reference)
                        )
            if not episode_manifest_candidates:
                raise ValueError(
                    "umcd_episode_store_content_npz must be referenced by a rostered "
                    "episode manifest"
                )
            _require_exact_child_object_owners(
                relative_path,
                child_ids if child_ids is not None else [child_id],
                [
                    (manifest_child_id, reference["path"])
                    for _, manifest_child_id, reference in episode_manifest_candidates
                ],
                label="UMCD episode content object",
            )
            manifest_paths = sorted(
                {manifest_path for manifest_path, _, _ in episode_manifest_candidates}
            )
            _, _, reference = episode_manifest_candidates[0]
            layer = reference.get("layer")
            content_sha256 = reference.get("sha256")
            if (
                not isinstance(layer, str)
                or not layer
                or not isinstance(content_sha256, str)
                or _SHA256_RE.fullmatch(content_sha256) is None
                or relative_path != f"objects/{layer}/{content_sha256[:2]}/{content_sha256}.npz"
            ):
                raise ValueError("UMCD episode content path does not match its layer digest")
            semantic_rows = {
                "semantic_validator": "umcd_episode_store_content_v1",
                "episode_layer": layer,
            }
            if roster_version == 4:
                semantic_rows["episode_manifest_path"] = manifest_paths[0]
            else:
                semantic_rows["episode_manifest_paths"] = manifest_paths
        else:
            semantic_rows = {}
        verified_entry = {
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "path": relative_path,
            "bytes": actual_bytes,
            "sha256": actual_digest,
            **semantic_rows,
        }
        if child_ids is not None:
            verified_entry["child_ids"] = child_ids
        else:
            verified_entry["child_id"] = child_id
        verified.append(verified_entry)
    verified.sort(
        key=lambda item: (
            item.get("child_id", item.get("child_ids", [""])[0]),
            item["artifact_id"],
        )
    )
    byte_only_count = sum("semantic_validator" not in item for item in verified)
    semantic_artifact_count = len(verified) - byte_only_count
    authority = (
        "detached OpenSSH signature authenticates the typed child-to-artifact roster; all "
        "artifact bytes are verified; ChannelDataset rows are checked against frozen "
        "child/kind/parent bindings; recognized NR trajectory stores are fully traversed "
        "through their content and record hashes"
    )
    if roster_version == 1:
        authority += "; roster v1 admits only semantically checked ChannelDataset artifacts"
    elif roster_version == 2:
        authority += "; unrecognized supplemental types receive byte-integrity checks only"
    else:
        authority += "; artifact types without semantic validators are rejected"
    if roster_version >= 3:
        authority += (
            "; roster v3 and later require exact regular-file inventory coverage beneath "
            "the declared artifact_root"
        )
    if roster_version >= 4:
        authority += (
            f"; roster v{roster_version} also validates UMCD episode-store manifests, parent "
            "episode IDs, and exact content-object child ownership"
        )
    if roster_version >= 6:
        authority += (
            "; roster v6 also traverses scenario GBSM runs and checks every parent against the "
            "frozen registry's shared trajectory-session component and split assignment"
        )
    audit: dict[str, Any] = {
        "schema": (
            "umcd.split_child_artifact_audit.v7"
            if roster_version == 6
            else "umcd.split_child_artifact_audit.v6"
            if roster_version == 5
            else "umcd.split_child_artifact_audit.v5"
            if roster_version == 4
            else "umcd.split_child_artifact_audit.v4"
        ),
        "status": (
            "PASS_EXACT_FROZEN_CHILD_ARTIFACT_BYTES_WITH_PARTIAL_SEMANTICS"
            if byte_only_count
            else "PASS_EXACT_FROZEN_CHILD_BYTES_AND_ROWS"
        ),
        "child_kind": expected_child_kind,
        "roster_schema": schema,
        "child_count": len(observed_child_ids),
        "artifact_count": len(verified),
        "artifact_root": artifact_root_name,
        "artifact_directory_inventory_status": ("EXACT" if roster_version >= 3 else "NOT_DECLARED"),
        "semantic_artifact_count": semantic_artifact_count,
        "byte_only_artifact_count": byte_only_count,
        "artifact_types": sorted({item["artifact_type"] for item in verified}),
        "total_bytes": sum(item["bytes"] for item in verified),
        "authority": authority,
        "artifacts": verified,
        "row_count": sum(item.get("row_count", 0) for item in verified),
        "trajectory_sample_count": sum(item.get("trajectory_sample_count", 0) for item in verified),
        "trajectory_count": len(observed_trajectory_ids),
        "scenario_gbsm_run_count": len(observed_scenario_run_ids),
        "scenario_gbsm_parent_count": sum(
            item.get("scenario_parent_count", 0) for item in verified
        ),
    }
    if roster_version >= 4:
        audit["episode_count"] = len(observed_episode_ids)
    audit["audit_sha256"] = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {
        "path": str(Path(os.path.abspath(path))),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "child_kind": expected_child_kind,
        "signature_sha256": verified_artifact.signature_sha256,
        "allowed_signers_sha256": verified_artifact.allowed_signers_sha256,
        "signer_id": verified_artifact.signer_id,
        "signature_namespace": verified_artifact.namespace,
        "audit": audit,
    }
