from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from datasets.channel import ChannelDataset
from datasets.split_lineage import (
    derive_channel_dataset_split_evidence,
    derive_channel_dataset_split_subjects,
    make_channel_dataset_split_registry,
)
from datasets.split_policy import (
    LeakageNamespace,
    SplitPolicy,
    audit_whole_scene_holdout,
    make_split_policy_registry,
    verify_registry_spatiotemporal_audit,
)


def _rt_subject_id(asset_hash: str, x: float, *, block_size: float = 1.0) -> str:
    numerator, denominator = float(x).as_integer_ratio()
    size_numerator, size_denominator = float(block_size).as_integer_ratio()
    block_x = (numerator * size_denominator) // (denominator * size_numerator)
    return f"rt:{asset_hash}:block:{float(block_size).hex()}:{block_x}:0"


def _dataset(name: str, rows: list[tuple[str, float, str]]) -> tuple[str, ChannelDataset]:
    metadata = tuple(
        {
            "split_subject_id": _rt_subject_id(asset_hash, x),
            "split_parent_type": "network_episode",
            "split_source_family": "sionna_rt",
            "split_leakage_relations": [],
            "scene": name,
            "scene_asset_manifest_complete": True,
            "scene_asset_set_sha256": asset_hash,
            "split_identity_basis": "scene_asset_set_sha256+receiver_xy_grid",
            "split_identity_complete": True,
            "split_block_size_m": 1.0,
            "split_coordinate_frame_id": asset_hash,
            "split_positions_m": [x, 0.0, 1.5],
            "rx": [x, 0.0, 1.5],
        }
        for subject_id, x, asset_hash in rows
    )
    count = len(rows)
    data = ChannelDataset(
        h=np.ones((count, 1, 8), dtype=np.complex64),
        y=np.zeros(count, dtype=np.int64),
        group=np.asarray([_rt_subject_id(asset_hash, x) for _, x, asset_hash in rows]),
        domain=np.full(count, "rt_scene"),
        source=np.full(count, "sionna_rt"),
        metadata=metadata,
    )
    return name, data


def _with_metadata(name: str, data: ChannelDataset, metadata: dict) -> tuple[str, ChannelDataset]:
    return name, ChannelDataset(
        h=data.h,
        y=data.y,
        group=data.group,
        domain=data.domain,
        source=data.source,
        metadata=(metadata,),
    )


def _with_scene_asset_manifest(
    named_dataset: tuple[str, ChannelDataset], assets: list[dict[str, object]]
) -> tuple[str, ChannelDataset]:
    name, data = named_dataset
    canonical_assets = sorted(assets, key=lambda asset: str(asset["path"]))
    digest = hashlib.sha256(
        json.dumps(canonical_assets, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = tuple(
        {
            **row,
            "scene_asset_set_sha256": digest,
            "split_coordinate_frame_id": digest,
            "split_subject_id": _rt_subject_id(digest, float(row["rx"][0])),
        }
        for row in data.metadata
    )
    updated = ChannelDataset(
        h=data.h,
        y=data.y,
        group=np.asarray([row["split_subject_id"] for row in metadata]),
        domain=data.domain,
        source=data.source,
        metadata=metadata,
        generation_report={
            "scene_asset_manifest": {
                "complete": True,
                "assets": canonical_assets,
                "asset_set_sha256": digest,
            }
        },
    )
    return name, updated


def test_lineage_audit_merges_runs_and_derives_spatial_relations() -> None:
    scene_hash = "a" * 64
    inputs = (
        _dataset("run-1", [("parent-a", 0.0, scene_hash)]),
        _dataset("run-2", [("parent-a", 0.5, scene_hash), ("parent-b", 1.5, scene_hash)]),
    )

    subjects, report = derive_channel_dataset_split_subjects(inputs, spatial_buffer_m=1.0)

    by_id = {subject.subject_id: subject for subject in subjects}
    first_id = _rt_subject_id(scene_hash, 0.0)
    second_id = _rt_subject_id(scene_hash, 1.5)
    assert set(by_id) == {first_id, second_id}
    assert report["subjects_reused_across_inputs"] == 1
    assert report["spatially_linked_subjects"] == 2
    assert report["leakage_component_count"] == 1
    assert report["leakage_component_sizes_descending"] == [2]
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["schema"] == "loslab.channel_split_lineage_audit.v3"
    assert report["spatiotemporal_derivation_audit"]["status"] == "PASS_INPUT_SCOPED"
    assert len(report["spatiotemporal_derivation_audit"]["footprints_sha256"]) == 64
    assert {
        relation.identity
        for relation in by_id[first_id].relations
        if relation.namespace is LeakageNamespace.SPATIAL_BUFFER
    } == {
        relation.identity
        for relation in by_id[second_id].relations
        if relation.namespace is LeakageNamespace.SPATIAL_BUFFER
    }


@pytest.mark.parametrize(
    "group",
    (np.asarray([1]), np.asarray([" parent-a "])),
)
def test_channel_dataset_rejects_noncanonical_group_ids(group: np.ndarray) -> None:
    _, base = _dataset("run", [("parent-a", 0.0, "a" * 64)])
    with pytest.raises(ValueError, match="group IDs must be nonempty canonical strings"):
        ChannelDataset(
            h=base.h,
            y=base.y,
            group=group,
            domain=base.domain,
            source=base.source,
            metadata=base.metadata,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("split_subject_id", " padded-parent "),
        ("split_source_family", " sionna_rt "),
        ("split_partition_id", " padded-partition "),
        ("scene", " padded-scene "),
        ("split_coordinate_frame_id", " padded-coordinate-frame "),
        ("split_time_scope_id", " padded-clock "),
        (
            "split_leakage_relations",
            [{"namespace": "capture_session", "identity": " padded-session "}],
        ),
    ),
)
def test_lineage_builder_rejects_padded_identity_metadata(field: str, value: object) -> None:
    _, base = _dataset("run", [("parent-a", 0.0, "a" * 64)])
    metadata = dict(base.metadata[0])
    metadata[field] = value
    if field == "split_time_scope_id":
        metadata["split_start_time_s"] = 0.0
        metadata["split_end_time_s"] = 1.0
    malformed = ChannelDataset(
        h=base.h,
        y=base.y,
        group=base.group,
        domain=base.domain,
        source=base.source,
        metadata=(metadata,),
    )

    with pytest.raises(ValueError, match="canonical nonempty string"):
        derive_channel_dataset_split_evidence((("run", malformed),), spatial_buffer_m=1.0)


def test_lineage_builder_rejects_padded_dataset_names() -> None:
    _, base = _dataset("run", [("parent-a", 0.0, "a" * 64)])

    with pytest.raises(ValueError, match="dataset name must be a canonical nonempty string"):
        derive_channel_dataset_split_evidence(((" run ", base),), spatial_buffer_m=1.0)


def test_lineage_audit_keeps_distinct_scene_coordinate_frames_separate() -> None:
    inputs = (
        _dataset("scene-a", [("parent-a", 0.0, "a" * 64)]),
        _dataset("scene-b", [("parent-b", 0.0, "b" * 64)]),
    )

    subjects, report = derive_channel_dataset_split_subjects(inputs, spatial_buffer_m=5.0)

    assert all(
        relation.namespace is not LeakageNamespace.SPATIAL_BUFFER
        for subject in subjects
        for relation in subject.relations
    )
    assert report["spatially_linked_subjects"] == 0


def test_saved_channel_lineage_builds_footprint_bound_v3_registry() -> None:
    inputs = tuple(
        _dataset(
            f"scene-{index}",
            [(f"parent-{index}", float(index), f"{index + 1:064x}")],
        )
        for index in range(4)
    )
    subjects, footprints, evidence_report = derive_channel_dataset_split_evidence(
        inputs, spatial_buffer_m=0.0
    )
    registry, registry_report = make_channel_dataset_split_registry(
        inputs,
        spatial_buffer_m=0.0,
        policy=SplitPolicy.release_v2(seed="saved-channel-footprint-lock"),
    )

    assert len(subjects) == 4
    assert len(footprints) == 4
    assert registry.to_dict()["schema"] == "umcd.split_registry.v3"
    assert registry_report["spatiotemporal_derivation_audit"] == evidence_report[
        "spatiotemporal_derivation_audit"
    ]
    assert verify_registry_spatiotemporal_audit(
        registry, footprints, spatial_buffer_m=0.0
    ) == dict(registry.spatiotemporal_audit)


def test_lineage_report_counts_spatial_and_temporal_coverage_separately() -> None:
    rows = (
        ("spatial-only", "family-a", {"split_positions_m": [0.0, 0.0, 0.0], "split_coordinate_frame_id": "map"}),
        ("temporal-only", "family-a", {"split_time_scope_id": "clock", "split_start_time_s": 0.0, "split_end_time_s": 1.0}),
        (
            "both",
            "family-a",
            {
                "split_positions_m": [2.0, 0.0, 0.0],
                "split_coordinate_frame_id": "map",
                "split_time_scope_id": "clock",
                "split_start_time_s": 2.0,
                "split_end_time_s": 3.0,
            },
        ),
        ("neither", "family-b", {}),
    )
    metadata = tuple(
        {
            "split_subject_id": subject_id,
            "split_parent_type": "network_episode",
            "split_source_family": source_family,
            "split_leakage_relations": [],
            **footprint,
        }
        for subject_id, source_family, footprint in rows
    )
    data = ChannelDataset(
        h=np.ones((len(rows), 1, 8), dtype=np.complex64),
        y=np.zeros(len(rows), dtype=np.int64),
        group=np.asarray([row[0] for row in rows]),
        domain=np.full(len(rows), "synthetic"),
        source=np.full(len(rows), "unit-test"),
        metadata=metadata,
    )

    _, _, report = derive_channel_dataset_split_evidence(
        (("coverage", data),), spatial_buffer_m=0.0
    )

    assert report["subjects_with_spatial_footprints"] == 2
    assert report["subjects_with_temporal_footprints"] == 2
    assert report["subjects_with_both_spatial_and_temporal_footprints"] == 1
    assert report["subjects_without_spatial_or_time_footprints"] == 1
    assert report["spatiotemporal_footprint_coverage_by_source_family"] == {
        "family-a": {
            "subjects": 3,
            "with_spatial_footprint": 2,
            "with_temporal_footprint": 2,
            "with_both_footprints": 1,
            "with_either_footprint": 3,
            "without_either_footprint": 0,
        },
        "family-b": {
            "subjects": 1,
            "with_spatial_footprint": 0,
            "with_temporal_footprint": 0,
            "with_both_footprints": 0,
            "with_either_footprint": 0,
            "without_either_footprint": 1,
        },
    }


def test_lineage_audit_joins_complete_scene_assets_across_runs() -> None:
    scene_hash = "c" * 64
    inputs = (
        _dataset("run-left", [("parent-left", 0.0, scene_hash)]),
        _dataset("run-right", [("parent-right", 50.0, scene_hash)]),
    )

    subjects, report = derive_channel_dataset_split_subjects(inputs, spatial_buffer_m=1.0)

    scene_relations = [
        relation.identity
        for subject in subjects
        for relation in subject.relations
        if relation.namespace is LeakageNamespace.SCENE_REALIZATION
    ]
    assert scene_relations == [f"asset-set:{scene_hash}", f"asset-set:{scene_hash}"]
    assert report["spatially_linked_subjects"] == 0
    assert report["leakage_component_count"] == 1


def test_lineage_audit_joins_exact_shared_geometry_assets_across_scene_sets() -> None:
    left = _with_scene_asset_manifest(
        _dataset("left", [("parent-left", 0.0, "a" * 64)]),
        [
            {"path": "scene-a.xml", "bytes": 10, "sha256": "1" * 64},
            {"path": "shared.obj", "bytes": 20, "sha256": "f" * 64},
        ],
    )
    right = _with_scene_asset_manifest(
        _dataset("right", [("parent-right", 50.0, "b" * 64)]),
        [
            {"path": "scene-b.xml", "bytes": 10, "sha256": "2" * 64},
            {"path": "shared.obj", "bytes": 20, "sha256": "f" * 64},
        ],
    )

    subjects, report = derive_channel_dataset_split_subjects(
        (left, right), spatial_buffer_m=1.0
    )

    assert report["spatially_linked_subjects"] == 0
    assert report["leakage_component_count"] == 1
    assert report["subjects_with_complete_geometry_asset_inventory"] == 2
    assert report["subjects_linked_by_exact_shared_geometry_assets"] == 2
    assert report["exact_shared_geometry_asset_identity_count"] == 1
    assert all(
        any(
            relation.identity == f"geometry-asset:sha256:{'f' * 64}"
            for relation in subject.relations
        )
        for subject in subjects
    )


def test_lineage_audit_finds_scene_manifests_in_derived_dataset_reports() -> None:
    left_name, left_data = _with_scene_asset_manifest(
        _dataset("left", [("parent-left", 0.0, "a" * 64)]),
        [{"path": "left.obj", "bytes": 20, "sha256": "f" * 64}],
    )
    right = _with_scene_asset_manifest(
        _dataset("right", [("parent-right", 50.0, "b" * 64)]),
        [{"path": "right.obj", "bytes": 20, "sha256": "f" * 64}],
    )
    selected = left_data.select(np.asarray([0], dtype=np.int64))

    subjects, report = derive_channel_dataset_split_subjects(
        ((left_name, selected), right), spatial_buffer_m=1.0
    )

    assert report["subjects_with_complete_geometry_asset_inventory"] == 2
    assert report["exact_shared_geometry_asset_identity_count"] == 1
    assert report["leakage_component_count"] == 1
    assert len(subjects) == 2


def test_lineage_audit_does_not_join_scenes_only_for_shared_texture_assets() -> None:
    left = _with_scene_asset_manifest(
        _dataset("left", [("parent-left", 0.0, "a" * 64)]),
        [
            {"path": "scene-a.xml", "bytes": 10, "sha256": "1" * 64},
            {"path": "shared.png", "bytes": 20, "sha256": "f" * 64},
        ],
    )
    right = _with_scene_asset_manifest(
        _dataset("right", [("parent-right", 50.0, "b" * 64)]),
        [
            {"path": "scene-b.xml", "bytes": 10, "sha256": "2" * 64},
            {"path": "shared.png", "bytes": 20, "sha256": "f" * 64},
        ],
    )

    _, report = derive_channel_dataset_split_subjects((left, right), spatial_buffer_m=1.0)

    assert report["leakage_component_count"] == 2
    assert report["exact_shared_geometry_asset_identity_count"] == 0


def test_lineage_audit_rejects_tampered_scene_asset_manifest_digest() -> None:
    named = _with_scene_asset_manifest(
        _dataset("run", [("parent", 0.0, "a" * 64)]),
        [{"path": "mesh.obj", "bytes": 10, "sha256": "f" * 64}],
    )
    name, data = named
    report = dict(data.generation_report)
    manifest = dict(report["scene_asset_manifest"])
    manifest["asset_set_sha256"] = "0" * 64
    report["scene_asset_manifest"] = manifest
    tampered = ChannelDataset(
        h=data.h,
        y=data.y,
        group=data.group,
        domain=data.domain,
        source=data.source,
        metadata=data.metadata,
        generation_report=report,
    )

    with pytest.raises(ValueError, match="manifest digest does not match"):
        derive_channel_dataset_split_evidence(((name, tampered),), spatial_buffer_m=1.0)


def test_derived_scene_relations_support_whole_scene_holdout_audit() -> None:
    scene_hashes = {scene: scene * 64 for scene in "abcd"}
    inputs = tuple(
        _dataset(
            f"run-{scene}",
            [
                (f"parent-{scene}-a", 0.0, scene_hash),
                (f"parent-{scene}-b", 50.0, scene_hash),
            ],
        )
        for scene, scene_hash in scene_hashes.items()
    )
    subjects, _ = derive_channel_dataset_split_subjects(inputs, spatial_buffer_m=1.0)
    registry = make_split_policy_registry(
        subjects, policy=SplitPolicy.release_v2(seed="whole-scene-lineage-integration")
    )
    scene_subjects = {
        f"asset-set:{scene_hash}": tuple(
            subject.subject_id
            for subject in subjects
            if any(
                relation.namespace is LeakageNamespace.SCENE_REALIZATION
                and relation.identity == f"asset-set:{scene_hash}"
                for relation in subject.relations
            )
        )
        for scene_hash in scene_hashes.values()
    }

    report = audit_whole_scene_holdout(
        registry,
        scene_subjects,
        expected_scene_ids=tuple(scene_subjects),
        expected_subject_ids=tuple(subject.subject_id for subject in subjects),
    )

    assert report["status"] == "PASS_INPUT_SCOPED"
    assert report["scene_count"] == 4
    assert report["parent_subject_count"] == 8


def test_lineage_audit_rejects_subject_id_not_matching_rt_identity() -> None:
    name, data = _dataset("run", [("parent-a", 0.0, "a" * 64)])
    metadata = dict(data.metadata[0])
    metadata["split_positions_m"] = [1.0, 0.0, 1.5]
    metadata["rx"] = [1.0, 0.0, 1.5]

    with pytest.raises(ValueError, match="ID disagrees with its identity"):
        derive_channel_dataset_split_subjects(
            (_with_metadata(name, data, metadata),), spatial_buffer_m=1.0
        )


def test_lineage_audit_rejects_split_position_that_disagrees_with_recorded_rx() -> None:
    name, data = _dataset("run", [("parent-a", 0.0, "a" * 64)])
    metadata = dict(data.metadata[0])
    metadata["rx"] = [10.0, 0.0, 1.5]

    with pytest.raises(ValueError, match="disagrees with recorded rx position"):
        derive_channel_dataset_split_subjects(
            (_with_metadata(name, data, metadata),), spatial_buffer_m=1.0
        )


@pytest.mark.parametrize(
    "receiver_position",
    [None, [0.0, True, 1.5], [0.0, float("nan"), 1.5], [0.0, "0", 1.5]],
)
def test_lineage_audit_rejects_missing_or_malformed_recorded_rx(receiver_position) -> None:
    name, data = _dataset("run", [("parent-a", 0.0, "a" * 64)])
    metadata = dict(data.metadata[0])
    if receiver_position is None:
        del metadata["rx"]
    else:
        metadata["rx"] = receiver_position

    with pytest.raises(ValueError, match="finite RT receiver position"):
        derive_channel_dataset_split_subjects(
            (_with_metadata(name, data, metadata),), spatial_buffer_m=1.0
        )


def test_lineage_audit_requires_explicit_parent_semantics() -> None:
    name, data = _dataset("legacy", [("parent-a", 0.0, "a" * 64)])
    legacy_metadata = dict(data.metadata[0])
    del legacy_metadata["split_parent_type"]
    legacy = ChannelDataset(
        h=data.h,
        y=data.y,
        group=data.group,
        domain=data.domain,
        source=data.source,
        metadata=(legacy_metadata,),
    )

    with pytest.raises(ValueError, match="invalid split_parent_type"):
        derive_channel_dataset_split_subjects(((name, legacy),), spatial_buffer_m=0.0)


def test_lineage_audit_rejects_non_boolean_rt_identity_claim() -> None:
    name, data = _dataset("run", [("parent-a", 0.0, "a" * 64)])
    metadata = dict(data.metadata[0])
    metadata["split_identity_complete"] = "true"

    with pytest.raises(ValueError, match="completeness flags"):
        derive_channel_dataset_split_subjects(
            (_with_metadata(name, data, metadata),), spatial_buffer_m=1.0
        )


def test_lineage_audit_rejects_complete_rt_identity_with_wrong_coordinate_frame() -> None:
    name, data = _dataset("run", [("parent-a", 0.0, "a" * 64)])
    metadata = dict(data.metadata[0])
    metadata["split_coordinate_frame_id"] = "b" * 64

    with pytest.raises(ValueError, match="contradictory complete RT identity"):
        derive_channel_dataset_split_subjects(
            (_with_metadata(name, data, metadata),), spatial_buffer_m=1.0
        )


def test_lineage_audit_counts_only_consistent_provisional_rt_identities() -> None:
    name, data = _dataset("run", [("parent-a", 0.0, "a" * 64)])
    metadata = dict(data.metadata[0])
    metadata.update(
        split_subject_id="rt:unclosed-scene-a:block:0x1.0000000000000p+0:0:0",
        scene="scene-a",
        rx=[0.0, 0.0, 1.5],
        split_identity_basis="provisional_scene_name+receiver_xy_grid",
        split_identity_complete=False,
        split_coordinate_frame_id=None,
        split_positions_m=None,
        scene_asset_manifest_complete=False,
    )

    subjects, report = derive_channel_dataset_split_subjects(
        (_with_metadata(name, data, metadata),), spatial_buffer_m=1.0
    )

    assert subjects[0].subject_id == metadata["split_subject_id"]
    assert report["subjects_with_provisional_split_identity"] == 1
    assert report["qualification_status"] == "NOT_QUALIFIED"


def test_provisional_rt_identity_rejects_padded_scene_name() -> None:
    name, data = _dataset("run", [("parent-a", 0.0, "a" * 64)])
    metadata = dict(data.metadata[0])
    metadata.update(
        split_subject_id="rt:unclosed-scene-a:block:0x1.0000000000000p+0:0:0",
        scene=" scene-a ",
        split_identity_basis="provisional_scene_name+receiver_xy_grid",
        split_identity_complete=False,
        split_coordinate_frame_id=None,
        split_positions_m=None,
        scene_asset_manifest_complete=False,
        rx=[0.0, 0.0, 1.5],
    )

    with pytest.raises(ValueError, match="scene name must be a canonical nonempty string"):
        derive_channel_dataset_split_subjects(
            (_with_metadata(name, data, metadata),), spatial_buffer_m=1.0
        )
