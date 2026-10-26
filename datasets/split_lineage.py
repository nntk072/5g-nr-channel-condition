"""Derive pre-split lineage from generated channel datasets."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from numbers import Real
from pathlib import PurePosixPath
from typing import Any, Sequence

from datasets.channel import ChannelDataset
from datasets.rt_split_identity import rt_spatial_split_subject_id
from datasets.split_policy import (
    LeakageNamespace,
    LeakageRelation,
    SpatiotemporalFootprint,
    SplitParentType,
    SplitPolicy,
    SplitPolicyRegistry,
    SplitSubject,
    audit_spatiotemporal_leakage_relations,
    bind_spatiotemporal_audit_to_registry,
    derive_spatiotemporal_leakage_relations,
    derive_split_components,
    make_split_policy_registry,
)


def _require_canonical_identity(value: Any, *, name: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{context} {name} must be a canonical nonempty string")
    return value


def _scene_geometry_asset_relations_from_manifest(
    manifest: Any, *, context: str
) -> tuple[bool, str, frozenset[LeakageRelation]]:
    """Verify one scene manifest and derive exact mesh relations if complete."""
    if not isinstance(manifest, dict):
        raise ValueError(f"{context} scene_asset_manifest must be an object")
    complete = manifest.get("complete")
    if type(complete) is not bool:
        raise ValueError(f"{context} scene asset manifest has an invalid completeness flag")
    asset_set_digest = manifest.get("asset_set_sha256")
    if (
        not isinstance(asset_set_digest, str)
        or len(asset_set_digest) != 64
        or any(character not in "0123456789abcdef" for character in asset_set_digest)
    ):
        raise ValueError(f"{context} scene asset manifest has an invalid set digest")
    if not complete:
        return False, asset_set_digest, frozenset()
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError(f"{context} complete scene asset manifest must contain assets")
    normalized_assets: list[dict[str, Any]] = []
    geometry_relations: set[LeakageRelation] = set()
    asset_paths: list[str] = []
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict) or set(asset) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise ValueError(f"{context} scene asset {index} has an invalid field roster")
        path = asset["path"]
        byte_count = asset["bytes"]
        digest = asset["sha256"]
        if (
            not isinstance(path, str)
            or not path
            or path != PurePosixPath(path).as_posix()
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
        ):
            raise ValueError(f"{context} scene asset {index} has a noncanonical relative path")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError(f"{context} scene asset {index} has an invalid byte count")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{context} scene asset {index} has an invalid SHA-256")
        normalized_assets.append({"path": path, "bytes": byte_count, "sha256": digest})
        asset_paths.append(path)
        suffix = PurePosixPath(path).suffix.lower()
        if suffix in {".obj", ".ply"}:
            geometry_relations.add(
                LeakageRelation(
                    LeakageNamespace.ASSET_FAMILY,
                    f"geometry-asset:sha256:{digest}",
                )
            )
    if len(set(asset_paths)) != len(asset_paths):
        raise ValueError(f"{context} scene asset paths must be unique")
    canonical = json.dumps(
        normalized_assets, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    computed_digest = hashlib.sha256(canonical).hexdigest()
    if asset_set_digest != computed_digest:
        raise ValueError(f"{context} scene asset manifest digest does not match its entries")
    return True, computed_digest, frozenset(geometry_relations)


def _scene_geometry_asset_manifests(
    generation_report: Any, *, context: str
) -> dict[str, tuple[bool, frozenset[LeakageRelation]]]:
    """Find manifests in direct and derived dataset reports, rejecting conflicts."""
    if not isinstance(generation_report, dict):
        raise TypeError(f"{context} generation_report must be an object")
    result: dict[str, tuple[bool, frozenset[LeakageRelation]]] = {}
    pending = [generation_report]
    visited: set[int] = set()
    while pending:
        report = pending.pop()
        if id(report) in visited:
            continue
        visited.add(id(report))
        if not isinstance(report, dict):
            continue
        if "scene_asset_manifest" in report:
            complete, digest, relations = _scene_geometry_asset_relations_from_manifest(
                report["scene_asset_manifest"], context=context
            )
            record = (complete, relations)
            previous = result.setdefault(digest, record)
            if previous != record:
                raise ValueError(f"{context} contains conflicting scene manifests for {digest}")
        parent_report = report.get("parent_generation_report")
        if "parent_generation_report" in report and not isinstance(parent_report, dict):
            raise ValueError(f"{context} parent_generation_report must be an object")
        if isinstance(parent_report, dict):
            pending.append(parent_report)
        components = report.get("components")
        if "components" in report and not isinstance(components, list):
            raise ValueError(f"{context} generation-report components must be an array")
        if isinstance(components, list):
            if any(not isinstance(component, dict) for component in components):
                raise ValueError(f"{context} generation-report components must be objects")
            pending.extend(components)
    return result


def derive_channel_dataset_split_evidence(
    datasets: Sequence[tuple[str, ChannelDataset]], *, spatial_buffer_m: float
) -> tuple[tuple[SplitSubject, ...], tuple[SpatiotemporalFootprint, ...], dict[str, Any]]:
    """Reconstruct split parents and leakage links across supplied dataset runs.

    Adapters must supply ``split_subject_id``, ``split_parent_type``, and
    ``split_source_family``. Optional explicit relations and spatial/time
    footprints are merged by parent ID before connected leakage relations are
    derived. Conflicting identity metadata fails closed. The report describes
    only supplied inputs and is never a release qualification.
    """
    if not datasets:
        raise ValueError("at least one named channel dataset is required")

    dataset_names: set[str] = set()
    parent_records: dict[str, dict[str, Any]] = {}
    positions_by_parent: dict[str, list[tuple[float, float, float]]] = {}
    time_by_parent: dict[str, tuple[str, float, float]] = {}
    parent_run_counts: Counter[str] = Counter()
    input_rows = 0
    provisional_subjects: set[str] = set()
    geometry_asset_manifest_subjects: set[str] = set()

    for dataset_name, data in datasets:
        dataset_name = _require_canonical_identity(
            dataset_name, name="dataset name", context="split-lineage input"
        )
        if dataset_name in dataset_names:
            raise ValueError(f"duplicate channel dataset name: {dataset_name}")
        dataset_names.add(dataset_name)
        if not isinstance(data, ChannelDataset):
            raise TypeError("datasets must contain ChannelDataset values")
        scene_geometry_manifests = _scene_geometry_asset_manifests(
            data.generation_report, context=dataset_name
        )

        parents_in_dataset: set[str] = set()
        subjects_by_group: dict[str, set[str]] = {}
        for row_index, (group, metadata) in enumerate(zip(data.group, data.metadata, strict=True)):
            context = f"{dataset_name} row {row_index}"
            group_id = _require_canonical_identity(group, name="group ID", context=context)
            if not isinstance(metadata, dict):
                raise TypeError(f"{context} metadata must be an object")
            subject_id = _require_canonical_identity(
                metadata.get("split_subject_id"), name="split_subject_id", context=context
            )
            parent_type_value = metadata.get("split_parent_type")
            if not isinstance(parent_type_value, str):
                raise ValueError(f"{context} has invalid split_parent_type")
            try:
                parent_type = SplitParentType(parent_type_value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{context} has invalid split_parent_type") from error
            source_family = _require_canonical_identity(
                metadata.get("split_source_family"), name="split_source_family", context=context
            )
            partition_id = _require_canonical_identity(
                metadata.get("split_partition_id", subject_id),
                name="split_partition_id",
                context=context,
            )

            relations_value = metadata.get("split_leakage_relations", ())
            if not isinstance(relations_value, (list, tuple)):
                raise ValueError(f"{context} split_leakage_relations must be an array")
            relations: set[LeakageRelation] = set()
            for relation in relations_value:
                if not isinstance(relation, dict) or set(relation) != {"namespace", "identity"}:
                    raise ValueError(f"{context} has a malformed leakage relation")
                relation_identity = _require_canonical_identity(
                    relation["identity"], name="leakage relation identity", context=context
                )
                try:
                    relations.add(
                        LeakageRelation(LeakageNamespace(relation["namespace"]), relation_identity)
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{context} has an invalid leakage relation") from error
            scene_realization_id: str | None = None

            identity_fields = {
                key: metadata[key]
                for key in (
                    "scene_asset_set_sha256",
                    "split_identity_basis",
                    "split_identity_complete",
                    "split_block_size_m",
                    "split_coordinate_frame_id",
                    "scene_asset_manifest_complete",
                )
                if key in metadata
            }
            rt_identity_fields = {
                "scene_asset_set_sha256",
                "split_identity_basis",
                "split_identity_complete",
                "split_block_size_m",
                "scene_asset_manifest_complete",
            }
            manifest_record: tuple[bool, frozenset[LeakageRelation]] | None = None
            if rt_identity_fields.intersection(metadata):
                identity_complete = metadata.get("split_identity_complete")
                assets_complete = metadata.get("scene_asset_manifest_complete")
                asset_digest = metadata.get("scene_asset_set_sha256")
                identity_basis = metadata.get("split_identity_basis")
                block_size = metadata.get("split_block_size_m")
                coordinate_frame = metadata.get("split_coordinate_frame_id")
                if coordinate_frame is not None:
                    coordinate_frame = _require_canonical_identity(
                        coordinate_frame, name="coordinate-frame ID", context=context
                    )
                position_value = metadata.get("split_positions_m")
                if type(identity_complete) is not bool or type(assets_complete) is not bool:
                    raise ValueError(f"{context} has invalid RT identity completeness flags")
                if (
                    not isinstance(asset_digest, str)
                    or len(asset_digest) != 64
                    or any(character not in "0123456789abcdef" for character in asset_digest)
                ):
                    raise ValueError(f"{context} has an invalid RT scene asset digest")
                manifest_record = scene_geometry_manifests.get(asset_digest)
                if scene_geometry_manifests and manifest_record is None:
                    raise ValueError(
                        f"{context} scene asset digest is absent from its generation manifests"
                    )
                if manifest_record is not None and assets_complete is not manifest_record[0]:
                    raise ValueError(
                        f"{context} scene asset completeness disagrees with its generation manifest"
                    )
                if (
                    isinstance(block_size, bool)
                    or not isinstance(block_size, Real)
                    or not math.isfinite(float(block_size))
                    or block_size <= 0
                ):
                    raise ValueError(f"{context} has an invalid RT split block size")
                if identity_complete:
                    if (
                        not assets_complete
                        or identity_basis != "scene_asset_set_sha256+receiver_xy_grid"
                        or coordinate_frame != asset_digest
                        or position_value is None
                    ):
                        raise ValueError(
                            f"{context} has contradictory complete RT identity metadata"
                        )
                else:
                    if (
                        assets_complete
                        or identity_basis != "provisional_scene_name+receiver_xy_grid"
                        or coordinate_frame is not None
                        or position_value is not None
                        or not subject_id.startswith("rt:unclosed-")
                    ):
                        raise ValueError(
                            f"{context} has contradictory provisional RT identity metadata"
                        )

                receiver_position = metadata.get("rx")
                if (
                    not isinstance(receiver_position, (list, tuple))
                    or len(receiver_position) != 3
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, Real)
                        or not math.isfinite(float(value))
                        for value in receiver_position
                    )
                ):
                    raise ValueError(f"{context} has no finite RT receiver position for its ID")
                identity_position = position_value if identity_complete else receiver_position
                if (
                    not isinstance(identity_position, (list, tuple))
                    or len(identity_position) != 3
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, Real)
                        or not math.isfinite(float(value))
                        for value in identity_position
                    )
                ):
                    raise ValueError(f"{context} has no finite RT receiver position for its ID")
                if identity_complete and tuple(map(float, position_value)) != tuple(
                    map(float, receiver_position)
                ):
                    raise ValueError(
                        f"{context} split_positions_m disagrees with recorded rx position"
                    )
                scene_name = _require_canonical_identity(
                    metadata.get("scene"), name="scene name", context=context
                )
                scene_realization_id = (
                    f"asset-set:{asset_digest}"
                    if assets_complete
                    else f"provisional-name:{scene_name}"
                )
                identity_fields["scene_realization_id"] = scene_realization_id
                relations.add(
                    LeakageRelation(
                        LeakageNamespace.SCENE_REALIZATION,
                        scene_realization_id,
                    )
                )
                expected_subject_id, computed_complete = rt_spatial_split_subject_id(
                    scene_asset_set_sha256=asset_digest,
                    scene_asset_manifest_complete=assets_complete,
                    scene_name=scene_name,
                    rx_position_m=identity_position,
                    block_size_m=float(block_size),
                )
                if computed_complete is not identity_complete:
                    raise ValueError(f"{context} RT identity completeness is inconsistent")
                if subject_id != expected_subject_id:
                    raise ValueError(f"{context} RT split subject ID disagrees with its identity")
            signature = (
                parent_type,
                source_family,
                partition_id,
                frozenset(relations),
                identity_fields,
            )
            previous = parent_records.get(subject_id)
            if previous is None:
                parent_records[subject_id] = {
                    "parent_type": parent_type,
                    "source_family": source_family,
                    "partition_id": partition_id,
                    "input_relations": set(relations),
                    "relations": set(relations),
                    "derived_geometry_asset_relations": set(),
                    "identity_fields": identity_fields,
                }
            else:
                previous_signature = (
                    previous["parent_type"],
                    previous["source_family"],
                    previous["partition_id"],
                    frozenset(previous["input_relations"]),
                    previous["identity_fields"],
                )
                if signature != previous_signature:
                    raise ValueError(f"conflicting lineage metadata for split subject {subject_id}")

            if manifest_record is not None and manifest_record[0]:
                parent_records[subject_id]["derived_geometry_asset_relations"].update(
                    manifest_record[1]
                )
                geometry_asset_manifest_subjects.add(subject_id)

            parents_in_dataset.add(subject_id)
            subjects_by_group.setdefault(group_id, set()).add(subject_id)
            input_rows += 1
            if metadata.get("split_identity_complete") is False:
                provisional_subjects.add(subject_id)

            positions = metadata.get("split_positions_m")
            coordinate_frame_id = metadata.get("split_coordinate_frame_id")
            if positions is not None:
                if not isinstance(positions, (list, tuple)) or len(positions) != 3:
                    raise ValueError(f"{context} split_positions_m must be one 3D point")
                coordinate_frame_id = _require_canonical_identity(
                    coordinate_frame_id, name="coordinate frame ID", context=context
                )
                if any(
                    not isinstance(value, Real)
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    for value in positions
                ):
                    raise ValueError(f"{context} split_positions_m must contain finite numbers")
                point = (float(positions[0]), float(positions[1]), float(positions[2]))
                positions_by_parent.setdefault(subject_id, []).append(point)

            time_scope = metadata.get("split_time_scope_id")
            time_start = metadata.get("split_start_time_s")
            time_end = metadata.get("split_end_time_s")
            if any(value is not None for value in (time_scope, time_start, time_end)):
                time_scope = _require_canonical_identity(
                    time_scope, name="time-scope ID", context=context
                )
                if (
                    not isinstance(time_start, (int, float))
                    or isinstance(time_start, bool)
                    or not isinstance(time_end, (int, float))
                    or isinstance(time_end, bool)
                ):
                    raise ValueError(f"{context} has an incomplete split time window")
                observed_time_window = (time_scope, float(time_start), float(time_end))
                previous_time = time_by_parent.setdefault(subject_id, observed_time_window)
                if previous_time != observed_time_window:
                    raise ValueError(f"conflicting time windows for split subject {subject_id}")

        for subject_id in parents_in_dataset:
            parent_run_counts[subject_id] += 1
        for group_id, subject_ids in subjects_by_group.items():
            if len(subject_ids) > 1:
                relation = LeakageRelation(
                    LeakageNamespace.ASSET_FAMILY,
                    f"channel_group:{dataset_name}:{group_id}",
                )
                for subject_id in subject_ids:
                    parent_records[subject_id]["relations"].add(relation)

    footprints: list[SpatiotemporalFootprint] = []
    for subject_id in sorted(parent_records):
        positions = tuple(positions_by_parent.get(subject_id, ()))
        identity_fields = parent_records[subject_id]["identity_fields"]
        coordinate_frame_id = identity_fields.get("split_coordinate_frame_id")
        if positions and not isinstance(coordinate_frame_id, str):
            raise ValueError(f"spatial split subject {subject_id} lacks a scene asset identity")
        time_window: tuple[str, float, float] | None = time_by_parent.get(subject_id)
        footprints.append(
            SpatiotemporalFootprint(
                subject_id=subject_id,
                positions_m=positions,
                coordinate_frame_id=coordinate_frame_id if positions else None,
                time_scope_id=None if time_window is None else time_window[0],
                start_time_s=None if time_window is None else time_window[1],
                end_time_s=None if time_window is None else time_window[2],
            )
        )

    derived_relations = derive_spatiotemporal_leakage_relations(
        footprints, spatial_buffer_m=spatial_buffer_m
    )
    subjects: list[SplitSubject] = []
    for subject_id in sorted(parent_records):
        record = parent_records[subject_id]
        relations = set(record["relations"])
        relations.update(record["derived_geometry_asset_relations"])
        relations.update(derived_relations[subject_id])
        subjects.append(
            SplitSubject(
                subject_id=subject_id,
                parent_type=record["parent_type"],
                source_family=record["source_family"],
                relations=tuple(relations),
                partition_id=record["partition_id"],
            )
        )

    spatially_linked = {
        subject_id
        for subject_id, relations in derived_relations.items()
        if any(relation.namespace is LeakageNamespace.SPATIAL_BUFFER for relation in relations)
    }
    temporally_linked = {
        subject_id
        for subject_id, relations in derived_relations.items()
        if any(
            relation.namespace is LeakageNamespace.OVERLAPPING_TIME_WINDOW for relation in relations
        )
    }
    component_sizes = sorted(
        (len(component) for component in derive_split_components(subjects)), reverse=True
    )
    shared_geometry_relations: dict[str, set[str]] = {}
    for subject in subjects:
        for relation in subject.relations:
            if (
                relation.namespace is LeakageNamespace.ASSET_FAMILY
                and relation.identity.startswith("geometry-asset:")
            ):
                shared_geometry_relations.setdefault(relation.identity, set()).add(
                    subject.subject_id
                )
    duplicated_geometry_relations = {
        identity: members
        for identity, members in shared_geometry_relations.items()
        if len(members) > 1
    }
    spatial_footprint_subjects = set(positions_by_parent)
    temporal_footprint_subjects = set(time_by_parent)
    source_family_coverage: dict[str, dict[str, int]] = {}
    for subject_id, record in parent_records.items():
        source_family = record["source_family"]
        coverage = source_family_coverage.setdefault(
            source_family,
            {
                "subjects": 0,
                "with_spatial_footprint": 0,
                "with_temporal_footprint": 0,
                "with_both_footprints": 0,
                "with_either_footprint": 0,
                "without_either_footprint": 0,
            },
        )
        has_spatial = subject_id in spatial_footprint_subjects
        has_temporal = subject_id in temporal_footprint_subjects
        coverage["subjects"] += 1
        coverage["with_spatial_footprint"] += int(has_spatial)
        coverage["with_temporal_footprint"] += int(has_temporal)
        coverage["with_both_footprints"] += int(has_spatial and has_temporal)
        coverage["with_either_footprint"] += int(has_spatial or has_temporal)
        coverage["without_either_footprint"] += int(not has_spatial and not has_temporal)
    report = {
        "schema": "loslab.channel_split_lineage_audit.v3",
        "status": "LINEAGE_DERIVED_NOT_RELEASE_QUALIFIED",
        "qualification_status": "NOT_QUALIFIED",
        "dataset_count": len(dataset_names),
        "input_rows": input_rows,
        "unique_split_subjects": len(subjects),
        "leakage_component_count": len(component_sizes),
        "leakage_component_sizes_descending": component_sizes,
        "largest_leakage_component_subjects": component_sizes[0],
        "subjects_reused_across_inputs": sum(count > 1 for count in parent_run_counts.values()),
        "subjects_with_complete_geometry_asset_inventory": len(
            geometry_asset_manifest_subjects
        ),
        "subjects_without_complete_geometry_asset_inventory": len(parent_records)
        - len(geometry_asset_manifest_subjects),
        "subjects_linked_by_exact_shared_geometry_assets": len(
            set().union(*duplicated_geometry_relations.values())
            if duplicated_geometry_relations
            else set()
        ),
        "exact_shared_geometry_asset_identity_count": len(duplicated_geometry_relations),
        "spatially_linked_subjects": len(spatially_linked),
        "temporally_linked_subjects": len(temporally_linked),
        "subjects_with_provisional_split_identity": len(provisional_subjects),
        "subjects_without_spatial_or_time_footprints": sum(
            subject_id not in spatial_footprint_subjects
            and subject_id not in temporal_footprint_subjects
            for subject_id in parent_records
        ),
        "subjects_with_spatial_footprints": len(spatial_footprint_subjects),
        "subjects_with_temporal_footprints": len(temporal_footprint_subjects),
        "subjects_with_both_spatial_and_temporal_footprints": len(
            spatial_footprint_subjects & temporal_footprint_subjects
        ),
        "spatiotemporal_footprint_coverage_by_source_family": {
            family: source_family_coverage[family] for family in sorted(source_family_coverage)
        },
        "spatial_buffer_m": float(spatial_buffer_m),
        "spatiotemporal_derivation_audit": audit_spatiotemporal_leakage_relations(
            footprints, subjects, spatial_buffer_m=spatial_buffer_m
        ),
    }
    return tuple(subjects), tuple(footprints), report


def derive_channel_dataset_split_subjects(
    datasets: Sequence[tuple[str, ChannelDataset]], *, spatial_buffer_m: float
) -> tuple[tuple[SplitSubject, ...], dict[str, Any]]:
    """Compatibility wrapper returning subjects and the scoped audit report."""
    subjects, _, report = derive_channel_dataset_split_evidence(
        datasets, spatial_buffer_m=spatial_buffer_m
    )
    return subjects, report


def make_channel_dataset_split_registry(
    datasets: Sequence[tuple[str, ChannelDataset]],
    *,
    spatial_buffer_m: float,
    policy: SplitPolicy,
) -> tuple[SplitPolicyRegistry, dict[str, Any]]:
    """Build a registry whose v3 lock commits to saved-ChannelDataset footprints."""
    subjects, footprints, report = derive_channel_dataset_split_evidence(
        datasets, spatial_buffer_m=spatial_buffer_m
    )
    registry = make_split_policy_registry(subjects, policy=policy)
    registry = bind_spatiotemporal_audit_to_registry(
        registry, footprints, spatial_buffer_m=spatial_buffer_m
    )
    return registry, report
