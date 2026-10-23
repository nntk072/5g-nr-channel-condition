"""Sparse, leakage-safe UMCD Release Split Policy v2 contracts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from datasets.network import (
    AntennaArraySpec,
    DirectedLink,
    LinkDirection,
    LinkRole,
    NetworkEpisode,
    NetworkNode,
    NodeRole,
)
from datasets.split_policy import (
    LeakageNamespace,
    LeakageRelation,
    SpatiotemporalFootprint,
    SplitChildBinding,
    SplitParentType,
    SplitPolicy,
    SplitPolicyRegistry,
    SplitSubject,
    audit_split_policy_registry,
    bind_split_children,
    derive_spatiotemporal_leakage_relations,
    freeze_split_policy_registry,
    make_split_policy_registry,
    network_episode_subject,
    read_split_policy_registry,
    standardized_draw_subject,
)
from datasets.standardized import StandardizedChannelRecord, StandardizedFamily
from evaluation.independent_splits import make_independent_split


def _subject(
    subject_id: str,
    *relations: tuple[LeakageNamespace, str],
    family: str = "family-a",
) -> SplitSubject:
    return SplitSubject(
        subject_id=subject_id,
        parent_type=SplitParentType.NETWORK_EPISODE,
        source_family=family,
        relations=tuple(LeakageRelation(namespace, identity) for namespace, identity in relations),
    )


def _unlinked_subjects(count: int = 20) -> list[SplitSubject]:
    return [
        _subject(f"subject-{index:02d}", family=f"family-{index % 2}") for index in range(count)
    ]


def _network_episode() -> NetworkEpisode:
    array = AntennaArraySpec(
        "array",
        np.zeros((1, 3)),
        ("V",),
        np.eye(3),
        ("port",),
        np.ones((1, 1), dtype=np.complex128),
    )
    nodes = (
        NetworkNode("tx", NodeRole.GNB, "array", (0.0, 0.0, 10.0), 1),
        NetworkNode("rx", NodeRole.UE, "array", (10.0, 0.0, 1.5), 1),
    )
    link = DirectedLink("tx-rx", "tx", "rx", (0,), (0,), LinkDirection.DOWNLINK, LinkRole.SERVING)
    return NetworkEpisode(
        "network-parent",
        "mesh-family-1",
        "F03-RT-STATIC",
        3.5e9,
        (array,),
        nodes,
        (link,),
        (0.0,),
        np.asarray([0.0]),
    )


def _standardized_record() -> StandardizedChannelRecord:
    offsets = (np.arange(8) - 4) * 15_000.0
    h_path = np.ones((1, 1, 1, 1, 1, 1), dtype=np.complex64)
    tau_s = np.zeros((1, 1, 1), dtype=np.float64)
    cfr = np.broadcast_to(h_path[..., 0, None], (1, 1, 1, 1, 1, 8)).copy()
    return StandardizedChannelRecord(
        sample_id="draw-0",
        run_id="generator-batch-9",
        sample_index=0,
        family=StandardizedFamily.TDL,
        profile_model="A",
        profile_los_component=False,
        h_path=h_path,
        tau_s=tau_s,
        cfr=cfr,
        subcarrier_offsets_hz=offsets,
        metadata={
            "profile_model": "A",
            "profile_los_component": False,
            "delay_profile_semantics": "scalable_tr_38_901_profile",
            "delay_reference": "profile_excess_delay",
            "seed": 19,
            "spec_version": "19.2",
            "carrier_frequency_hz": 3.5e9,
            "subcarrier_spacing_hz": 15_000.0,
            "num_subcarriers": 8,
            "sampled_bandwidth_hz": 120_000.0,
            "num_time_steps": 1,
            "rx_antenna_count": 1,
            "tx_antenna_count": 1,
            "rx_corr_mat_real": None,
            "rx_corr_mat_imag": None,
        },
    )


def _four_way_registry(subjects: list[SplitSubject] | None = None):
    policy = SplitPolicy.release_v2(seed="release-v2-test")
    registry = make_split_policy_registry(subjects or _unlinked_subjects(), policy=policy)
    return policy, registry


def test_sparse_relations_allow_heterogeneous_subjects_without_sentinel_joining():
    subjects = [
        _subject("rt-a", (LeakageNamespace.SITE, "site-1")),
        _subject("standardized-a", family="standardized.tdl"),
        _subject("measured-a", family="measured.external"),
        *_unlinked_subjects(8),
    ]
    _, registry = _four_way_registry(subjects)
    assert (
        registry.subject_to_component["standardized-a"]
        != registry.subject_to_component["measured-a"]
    )
    assert registry.subject_to_component["rt-a"] != registry.subject_to_component["standardized-a"]
    assert registry.subjects["standardized-a"].relations == ()
    assert all(
        relation.identity != "N/A"
        for subject in registry.subjects.values()
        for relation in subject.relations
    )


def test_namespace_is_part_of_relation_identity():
    subjects = [
        _subject("site-parent", (LeakageNamespace.SITE, "x")),
        _subject("trajectory-parent", (LeakageNamespace.TRAJECTORY_SESSION, "x")),
        *_unlinked_subjects(5),
    ]
    _, registry = _four_way_registry(subjects)
    assert (
        registry.subject_to_component["site-parent"]
        != registry.subject_to_component["trajectory-parent"]
    )


@pytest.mark.parametrize("namespace", tuple(LeakageNamespace))
def test_each_supported_shared_identity_is_confined_to_one_component(namespace):
    subjects = [
        _subject("left", (namespace, "shared-value")),
        _subject("right", (namespace, "shared-value")),
        *_unlinked_subjects(5),
    ]
    _, registry = _four_way_registry(subjects)
    assert registry.subject_to_component["left"] == registry.subject_to_component["right"]
    audit_split_policy_registry(registry, subjects)


def test_connected_components_are_transitive_across_relation_types():
    subjects = [
        _subject("a", (LeakageNamespace.SITE, "site-a")),
        _subject(
            "b",
            (LeakageNamespace.SITE, "site-a"),
            (LeakageNamespace.TRAJECTORY_SESSION, "route-b"),
        ),
        _subject("c", (LeakageNamespace.TRAJECTORY_SESSION, "route-b")),
        *_unlinked_subjects(4),
    ]
    _, registry = _four_way_registry(subjects)
    assert len({registry.subject_to_component[item] for item in ("a", "b", "c")}) == 1


def test_spatial_buffer_relations_link_nearby_subjects_within_coordinate_frame():
    footprints = (
        SpatiotemporalFootprint("a", ((0.0, 0.0, 0.0),), "local-map"),
        SpatiotemporalFootprint("b", ((1.0, 0.0, 0.0),), "local-map"),
        SpatiotemporalFootprint("c", ((2.1, 0.0, 0.0),), "local-map"),
        SpatiotemporalFootprint("other-frame", ((0.2, 0.0, 0.0),), "other-map"),
    )
    relations = derive_spatiotemporal_leakage_relations(footprints, spatial_buffer_m=1.0)

    assert relations["a"] == relations["b"]
    assert relations["a"][0].namespace is LeakageNamespace.SPATIAL_BUFFER
    assert relations["c"] == ()
    assert relations["other-frame"] == ()

    subjects = [
        _subject(
            footprint.subject_id,
            *(
                (relation.namespace, relation.identity)
                for relation in relations[footprint.subject_id]
            ),
        )
        for footprint in footprints
    ]
    subjects.extend(_unlinked_subjects())
    _, registry = _four_way_registry(subjects)
    assert registry.split_for_parent("a") == registry.split_for_parent("b")


def test_spatial_buffer_comparison_does_not_underflow_and_create_false_leakage():
    footprints = (
        SpatiotemporalFootprint("origin", ((0.0, 0.0, 0.0),), "tiny-scale-map"),
        SpatiotemporalFootprint("inside", ((2e-201, 0.0, 0.0),), "tiny-scale-map"),
        SpatiotemporalFootprint("outside", ((1e-200, 0.0, 0.0),), "tiny-scale-map"),
    )

    relations = derive_spatiotemporal_leakage_relations(footprints, spatial_buffer_m=5e-201)

    assert relations["origin"] == relations["inside"]
    assert relations["outside"] == ()


def test_spatial_buffer_cell_lookup_handles_finite_coordinate_to_radius_overflow():
    footprints = (
        SpatiotemporalFootprint("left", ((1e100, 0.0, 0.0),), "large-scale-map"),
        SpatiotemporalFootprint("right", ((1e100, 0.0, 0.0),), "large-scale-map"),
    )

    relations = derive_spatiotemporal_leakage_relations(footprints, spatial_buffer_m=1e-200)

    assert relations["left"] == relations["right"]
    assert relations["left"][0].namespace is LeakageNamespace.SPATIAL_BUFFER


def test_overlapping_time_windows_are_half_open_and_scope_qualified():
    footprints = (
        SpatiotemporalFootprint("a", time_scope_id="clock-1", start_time_s=0, end_time_s=2),
        SpatiotemporalFootprint("b", time_scope_id="clock-1", start_time_s=1.5, end_time_s=3),
        SpatiotemporalFootprint("c", time_scope_id="clock-1", start_time_s=2.5, end_time_s=4),
        SpatiotemporalFootprint("touching", time_scope_id="clock-1", start_time_s=4, end_time_s=5),
        SpatiotemporalFootprint(
            "other-clock", time_scope_id="clock-2", start_time_s=1, end_time_s=3
        ),
    )
    relations = derive_spatiotemporal_leakage_relations(footprints, spatial_buffer_m=0)

    assert relations["a"] == relations["b"] == relations["c"]
    assert relations["a"][0].namespace is LeakageNamespace.OVERLAPPING_TIME_WINDOW
    assert relations["touching"] == ()
    assert relations["other-clock"] == ()


def test_spatiotemporal_footprints_reject_invalid_geometry():
    with pytest.raises(ValueError):
        SpatiotemporalFootprint("bad-coordinate", ((float("nan"), 0, 0),), "map")
    with pytest.raises(ValueError, match="coordinate_frame_id is required"):
        SpatiotemporalFootprint("missing-frame", ((0, 0, 0),))
    with pytest.raises(TypeError, match="coordinates must be real numbers"):
        SpatiotemporalFootprint("string-coordinate", (("0", 0, 0),), "map")


def test_spatiotemporal_footprints_require_valid_half_open_time_windows():
    with pytest.raises(ValueError, match="start_time_s < end_time_s"):
        SpatiotemporalFootprint("empty", time_scope_id="clock", start_time_s=2, end_time_s=2)
    with pytest.raises(ValueError, match="both start_time_s and end_time_s"):
        SpatiotemporalFootprint("partial", time_scope_id="clock", start_time_s=1)
    with pytest.raises(TypeError, match="endpoints must be real numbers"):
        SpatiotemporalFootprint(
            "string-time", time_scope_id="clock", start_time_s="0", end_time_s=1
        )


def test_registry_rejects_manually_split_shared_relation():
    subjects = [
        _subject("left", (LeakageNamespace.SITE, "shared")),
        _subject("right", (LeakageNamespace.SITE, "shared")),
        *_unlinked_subjects(4),
    ]
    policy = SplitPolicy.release_v2(seed="manual-registry")
    assignments = {item.subject_id: f"component-{index}" for index, item in enumerate(subjects)}
    component_splits = {
        component: ("train", "validation", "calibration", "test")[index % 4]
        for index, component in enumerate(assignments.values())
    }
    with pytest.raises(ValueError, match="canonical relation reconstruction"):
        SplitPolicyRegistry(
            policy=policy,
            subject_to_component=assignments,
            component_to_split=component_splits,
            subjects={item.subject_id: item for item in subjects},
        )


def test_relations_and_serialized_assignments_are_input_order_invariant():
    subjects = [
        _subject("a", (LeakageNamespace.ASSET_FAMILY, "mesh-x")),
        _subject("b", (LeakageNamespace.SITE, "site-y")),
        _subject("c", (LeakageNamespace.ASSET_FAMILY, "mesh-x"), (LeakageNamespace.SITE, "site-y")),
        *_unlinked_subjects(8),
    ]
    policy = SplitPolicy.release_v2(seed="canonical-v2")
    first = make_split_policy_registry(subjects, policy=policy)
    second = make_split_policy_registry(list(reversed(subjects)), policy=policy)
    assert first.to_dict() == second.to_dict()


def test_four_way_split_uses_independent_components_and_reports_effective_counts():
    _, registry = _four_way_registry()
    assert set(registry.component_to_split.values()) == {
        "train",
        "validation",
        "calibration",
        "test",
    }
    report = registry.report()
    assert sum(registry.independent_component_counts.values()) == 20
    assert all(
        report["splits"][name]["independent_components"] >= 1
        for name in ("train", "validation", "calibration", "test")
    )
    assert "parent_subjects" in report["splits"]["train"]
    assert report["splits"]["train"]["by_source_family"]


def test_three_way_legacy_assignment_and_serialization_remain_unchanged():
    records = [
        {"sample_id": f"sample-{index}", "scene_group_id": f"scene-{index}"} for index in range(12)
    ]
    legacy = make_independent_split(records, seed="legacy-v1")
    reversed_legacy = make_independent_split(list(reversed(records)), seed="legacy-v1")
    assert legacy.to_dict() == reversed_legacy.to_dict()
    assert set(legacy.group_to_split.values()) == {"train", "validation", "test"}
    assert "calibration" not in legacy.to_dict()["sample_to_split"].values()


def test_four_way_split_requires_at_least_four_independent_components():
    with pytest.raises(ValueError, match="at least 4 independent groups"):
        make_independent_split(
            [{"sample_id": str(i), "scene_group_id": str(i)} for i in range(3)],
            seed="small",
            fractions={"train": 0.7, "validation": 0.1, "calibration": 0.1, "test": 0.1},
        )


def test_network_episode_children_inherit_parent_split_and_asset_lineage():
    episode = _network_episode()
    parent = network_episode_subject(
        episode,
        relations=(LeakageRelation(LeakageNamespace.SITE, "held-site"),),
    )
    subjects = [parent, *_unlinked_subjects(8)]
    policy = SplitPolicy.release_v2(seed="network-children")
    registry = make_split_policy_registry(subjects, policy=policy)
    assert parent.subject_id == episode.episode_id
    assert registry.subjects[parent.subject_id].source_family == episode.family_id
    assert (
        LeakageRelation(LeakageNamespace.ASSET_FAMILY, episode.split_group_id) in parent.relations
    )
    parent_split = registry.split_for_parent(episode.episode_id)
    descendants = ["link", "frame", "path-oracle", "channel", "receiver", "noise", "target"]
    bindings = tuple(
        SplitChildBinding(f"{episode.episode_id}:{kind}:0", episode.episode_id, kind)
        for kind in descendants
    )
    registry = bind_split_children(registry, bindings)
    assert {registry.split_for_child(binding.child_id) for binding in bindings} == {parent_split}
    audit_split_policy_registry(registry, subjects, bindings)


def test_standardized_draw_not_generator_batch_is_parent_for_all_views():
    record = _standardized_record()
    parent = standardized_draw_subject(record)
    subjects = [parent, *_unlinked_subjects(8)]
    policy = SplitPolicy.release_v2(seed="standardized-children")
    registry = make_split_policy_registry(subjects, policy=policy)
    assert parent.subject_id == record.sample_id
    assert parent.parent_type is SplitParentType.STANDARDIZED_CHANNEL_DRAW
    assert parent.source_family == "standardized.tdl"
    bindings = tuple(
        SplitChildBinding(f"{record.sample_id}:{kind}:0", record.sample_id, kind)
        for kind in ("receiver_observation", "noise_realization", "pusch_view")
    )
    registry = bind_split_children(registry, bindings)
    assert {registry.split_for_child(binding.child_id) for binding in bindings} == {
        registry.split_for_parent(record.sample_id)
    }
    audit_split_policy_registry(registry, subjects, bindings)
    assert parent.subject_id != record.run_id
    assert parent.relations == ()


def test_counterfactual_descendants_and_repeated_receiver_variants_stay_with_parent():
    subjects = [
        _subject(
            "base-episode",
            (LeakageNamespace.COUNTERFACTUAL_FAMILY, "intervention-pair-1"),
            (LeakageNamespace.TRAJECTORY_SESSION, "session-8"),
        ),
        _subject(
            "counterfactual-twin", (LeakageNamespace.COUNTERFACTUAL_FAMILY, "intervention-pair-1")
        ),
        _subject("same-route-window", (LeakageNamespace.TRAJECTORY_SESSION, "session-8")),
        *_unlinked_subjects(6),
    ]
    policy = SplitPolicy.release_v2(seed="counterfactual-children")
    registry = make_split_policy_registry(subjects, policy=policy)
    base = registry.split_for_parent("base-episode")
    assert registry.split_for_parent("counterfactual-twin") == base
    assert registry.split_for_parent("same-route-window") == base
    # Noise, bandwidth, receiver, and PUSCH variants are children, never new parents.
    bindings = tuple(
        SplitChildBinding(f"view-{index}", "base-episode", kind)
        for index, kind in enumerate(
            ("noise_realization", "bandwidth_resampled_view", "receiver_observation", "pusch_view")
            * 3
        )
    )
    registry = bind_split_children(registry, bindings)
    assert {registry.split_for_child(binding.child_id) for binding in bindings} == {base}


def test_child_bindings_reject_unknown_parent_conflicts_and_changed_audit():
    subjects = [_subject("parent-a"), *_unlinked_subjects(8)]
    policy = SplitPolicy.release_v2(seed="binding-validation")
    registry = make_split_policy_registry(subjects, policy=policy)
    with pytest.raises(ValueError, match="unknown split parent"):
        bind_split_children(registry, [SplitChildBinding("orphan", "missing", "frame")])
    with pytest.raises(ValueError, match="conflicting parent or kind"):
        bind_split_children(
            registry,
            [
                SplitChildBinding("child-1", "parent-a", "frame"),
                SplitChildBinding("child-1", "subject-00", "frame"),
            ],
        )
    binding = SplitChildBinding("child-1", "parent-a", "frame")
    frozen = bind_split_children(registry, [binding])
    with pytest.raises(ValueError, match="frozen parent binding changed"):
        audit_split_policy_registry(
            frozen,
            subjects,
            [SplitChildBinding("child-1", "subject-00", "frame")],
        )
    with pytest.raises(KeyError, match="child is absent"):
        frozen.split_for_child("parent-a")
    with pytest.raises(ValueError, match="exactly the frozen child set"):
        audit_split_policy_registry(frozen, subjects)


def test_standardized_profile_and_family_metadata_do_not_change_opaque_partition():
    record_a = _standardized_record()
    record_d = replace(
        record_a,
        family=StandardizedFamily.CDL,
        profile_model="D",
        profile_los_component=True,
        metadata={
            **record_a.metadata,
            "profile_model": "D",
            "profile_proxy": "D",
            "profile_los_component": True,
            "direction": "downlink",
            "ut_array": {
                "num_rows_per_panel": 1,
                "num_cols_per_panel": 1,
                "num_rows": 1,
                "num_cols": 1,
                "polarization": "single",
                "polarization_type": "V",
                "antenna_pattern": "omni",
            },
            "bs_array": {
                "num_rows_per_panel": 1,
                "num_cols_per_panel": 1,
                "num_rows": 1,
                "num_cols": 1,
                "polarization": "single",
                "polarization_type": "V",
                "antenna_pattern": "omni",
            },
        },
    )
    subject_a = standardized_draw_subject(record_a)
    subject_d = standardized_draw_subject(record_d)
    assert subject_a.partition_id == subject_d.partition_id
    policy = SplitPolicy.release_v2(seed="profile-invariance")
    registry_a = make_split_policy_registry([subject_a, *_unlinked_subjects(8)], policy=policy)
    registry_d = make_split_policy_registry([subject_d, *_unlinked_subjects(8)], policy=policy)
    assert (
        registry_a.subject_to_component[subject_a.subject_id]
        == registry_d.subject_to_component[subject_d.subject_id]
    )
    assert registry_a.split_for_parent(subject_a.subject_id) == registry_d.split_for_parent(
        subject_d.subject_id
    )


def test_standardized_parent_requires_opaque_identity_when_seed_is_missing():
    record = _standardized_record()
    without_seed = replace(
        record,
        metadata={key: value for key, value in record.metadata.items() if key != "seed"},
    )
    with pytest.raises(ValueError, match="generation seed metadata or an explicit partition_id"):
        standardized_draw_subject(without_seed)
    explicit = standardized_draw_subject(without_seed, partition_id="opaque-parent-17")
    assert explicit.partition_id == "opaque-parent-17"
    duplicate = _subject("second-parent")
    collision = replace(duplicate, partition_id=explicit.partition_id)
    with pytest.raises(ValueError, match="uniquely identify parent subjects"):
        make_split_policy_registry(
            [explicit, collision, *_unlinked_subjects(4)],
            policy=SplitPolicy.release_v2(seed="partition-collision"),
        )


def test_seed_index_collision_fails_closed_and_explicit_ids_separate_generator_runs():
    record_a = _standardized_record()
    record_b = replace(
        record_a,
        sample_id="draw-from-independent-run",
        run_id="independent-generator-run",
        family=StandardizedFamily.CDL,
        profile_model="D",
        profile_los_component=True,
        metadata={
            **record_a.metadata,
            "profile_model": "D",
            "profile_los_component": True,
            "direction": "downlink",
            "ut_array": {
                "num_rows_per_panel": 1,
                "num_cols_per_panel": 1,
                "num_rows": 1,
                "num_cols": 1,
                "polarization": "single",
                "polarization_type": "V",
                "antenna_pattern": "omni",
            },
            "bs_array": {
                "num_rows_per_panel": 1,
                "num_cols_per_panel": 1,
                "num_rows": 1,
                "num_cols": 1,
                "polarization": "single",
                "polarization_type": "V",
                "antenna_pattern": "omni",
            },
        },
    )
    subject_a = standardized_draw_subject(record_a)
    subject_b = standardized_draw_subject(record_b)
    assert subject_a.partition_id == subject_b.partition_id

    policy = SplitPolicy.release_v2(seed="same-seed-independent-runs")
    with pytest.raises(ValueError, match="partition_id values must uniquely identify"):
        make_split_policy_registry([subject_a, subject_b, *_unlinked_subjects(8)], policy=policy)

    subject_a = standardized_draw_subject(record_a, partition_id="opaque-tdl-parent-0")
    subject_b = standardized_draw_subject(record_b, partition_id="opaque-cdl-parent-0")
    registry = make_split_policy_registry(
        [subject_a, subject_b, *_unlinked_subjects(8)], policy=policy
    )
    assert (
        registry.subject_to_component[subject_a.subject_id]
        != registry.subject_to_component[subject_b.subject_id]
    )


def test_audit_rejects_any_frozen_lineage_change():
    subjects = [_subject("a", (LeakageNamespace.SITE, "site-a")), *_unlinked_subjects(8)]
    _, registry = _four_way_registry(subjects)
    changed = list(subjects)
    changed[0] = _subject("a", (LeakageNamespace.SITE, "site-b"))
    with pytest.raises(ValueError, match="lineage changed"):
        audit_split_policy_registry(registry, changed)


def test_policy_hash_changes_with_relation_contract_and_detects_tampering():
    first = SplitPolicy.release_v2(seed="hash-check")
    second = SplitPolicy(
        seed="hash-check",
        split_fractions=first.split_fractions,
        relation_namespaces=tuple(
            item for item in LeakageNamespace if item is not LeakageNamespace.EQUIPMENT_FAMILY
        ),
    )
    assert first.to_dict()["policy_sha256"] != second.to_dict()["policy_sha256"]
    tampered = first.to_dict()
    tampered["split_fractions"]["test"] = 0.2
    with pytest.raises(ValueError, match="policy hash is invalid"):
        SplitPolicy.from_dict(tampered)


def test_registry_lock_is_canonical_immutable_and_detects_registry_or_policy_tampering(tmp_path):
    subjects = _unlinked_subjects()
    policy, registry = _four_way_registry(subjects)
    registry = bind_split_children(
        registry,
        [SplitChildBinding("receiver-view-0", subjects[0].subject_id, "receiver_observation")],
    )
    path = tmp_path / "release-v2.json"
    digest = freeze_split_policy_registry(path, registry)
    assert freeze_split_policy_registry(path, registry) == digest
    restored = read_split_policy_registry(path)
    assert restored.to_dict() == registry.to_dict()
    assert restored.split_for_child("receiver-view-0") == restored.split_for_parent(
        subjects[0].subject_id
    )

    payload = json.loads(path.read_text())
    payload["policy"]["seed"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="registry hash is invalid"):
        read_split_policy_registry(path)

    other = make_split_policy_registry(
        subjects, policy=SplitPolicy.release_v2(seed=policy.seed + "-other")
    )
    with pytest.raises(FileExistsError, match="refusing to modify"):
        freeze_split_policy_registry(path, other)


def test_registry_lock_rejects_duplicate_json_keys(tmp_path):
    _, registry = _four_way_registry(_unlinked_subjects())
    path = tmp_path / "duplicate-key.json"
    freeze_split_policy_registry(path, registry)
    encoded = path.read_text(encoding="utf-8")
    encoded = encoded.replace(
        '"schema": "umcd.split_registry.v2"',
        '"schema": "ambiguous",\n  "schema": "umcd.split_registry.v2"',
        1,
    )
    path.write_text(encoded, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key: schema"):
        read_split_policy_registry(path)


def test_registry_lock_rejects_symlink_alias_for_freeze_and_read(tmp_path):
    _, registry = _four_way_registry(_unlinked_subjects())
    target = tmp_path / "canonical.json"
    freeze_split_policy_registry(target, registry)
    alias = tmp_path / "symlink.json"
    alias.symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        freeze_split_policy_registry(alias, registry)
    with pytest.raises(ValueError, match="must not be a symlink"):
        read_split_policy_registry(alias)


def test_registry_lock_rejects_symlink_when_nofollow_is_unavailable(tmp_path, monkeypatch):
    _, registry = _four_way_registry(_unlinked_subjects())
    target = tmp_path / "canonical.json"
    freeze_split_policy_registry(target, registry)
    alias = tmp_path / "symlink.json"
    alias.symlink_to(target)
    original_open = os.open
    open_called = False

    def track_open(path, flags, *args, **kwargs):
        nonlocal open_called
        open_called = True
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        return original_open(path, flags & ~nofollow, *args, **kwargs)

    monkeypatch.setattr(os, "open", track_open)
    with pytest.raises(ValueError, match="must not be a symlink"):
        read_split_policy_registry(alias)
    assert not open_called


def test_registry_lock_rejects_leaf_symlink_swap_during_open(tmp_path, monkeypatch):
    _, registry = _four_way_registry(_unlinked_subjects())
    target = tmp_path / "canonical.json"
    freeze_split_policy_registry(target, registry)
    alias = tmp_path / "replace-me.json"
    alias.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    original_open = os.open
    swapped = False

    def swap_then_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if Path(path) == alias and not swapped:
            swapped = True
            alias.unlink()
            alias.symlink_to(target)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        return original_open(path, flags & ~nofollow, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap_then_open)
    with pytest.raises(ValueError, match="must not be a symlink"):
        read_split_policy_registry(alias)
    assert swapped


def test_registry_lock_rejects_hardlink_alias_for_freeze_and_read(tmp_path):
    _, registry = _four_way_registry(_unlinked_subjects())
    target = tmp_path / "canonical.json"
    freeze_split_policy_registry(target, registry)
    alias = tmp_path / "hardlink.json"
    os.link(target, alias)

    with pytest.raises(ValueError, match="single-link regular file"):
        freeze_split_policy_registry(alias, registry)
    with pytest.raises(ValueError, match="single-link regular file"):
        read_split_policy_registry(alias)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO files are unavailable")
def test_registry_lock_rejects_fifo_without_blocking(tmp_path):
    _, registry = _four_way_registry(_unlinked_subjects())
    path = tmp_path / "fifo.json"
    os.mkfifo(path)

    with pytest.raises(ValueError, match="single-link regular file"):
        freeze_split_policy_registry(path, registry)
    with pytest.raises(ValueError, match="single-link regular file"):
        read_split_policy_registry(path)


def _rehash_registry_payload(payload: dict) -> dict:
    core = {key: value for key, value in payload.items() if key != "registry_sha256"}
    payload["registry_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return payload


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("component_id", "canonical relation reconstruction"),
        ("assignment", "deterministic policy reconstruction"),
        ("overmerge", "canonical relation reconstruction"),
        ("missing_split", "deterministic policy reconstruction"),
        ("report", "representation is not canonical"),
    ],
)
def test_rehashed_noncanonical_registry_is_rejected(tmp_path, mutation: str, message: str):
    _, registry = _four_way_registry(_unlinked_subjects())
    payload = registry.to_dict()
    component_ids = sorted(payload["components"])

    if mutation == "component_id":
        old = component_ids[0]
        new = "component-not-derived-from-parent-identities"
        payload["components"][new] = payload["components"].pop(old)
        for subject in payload["subjects"].values():
            if subject["component_id"] == old:
                subject["component_id"] = new
    elif mutation == "assignment":
        component = component_ids[0]
        old_split = payload["components"][component]["split"]
        new_split = next(name for name in payload["policy"]["split_fractions"] if name != old_split)
        payload["components"][component]["split"] = new_split
        for subject in payload["subjects"].values():
            if subject["component_id"] == component:
                subject["split"] = new_split
    elif mutation == "overmerge":
        left, right = component_ids[:2]
        split = payload["components"][left]["split"]
        payload["components"].pop(right)
        for subject in payload["subjects"].values():
            if subject["component_id"] == right:
                subject["component_id"] = left
                subject["split"] = split
    elif mutation == "missing_split":
        for component, value in payload["components"].items():
            if value["split"] == "calibration":
                value["split"] = "train"
                for subject in payload["subjects"].values():
                    if subject["component_id"] == component:
                        subject["split"] = "train"
    else:
        payload["report"]["splits"]["train"]["independent_components"] += 1

    path = tmp_path / f"{mutation}.json"
    path.write_text(json.dumps(_rehash_registry_payload(payload)))
    with pytest.raises(ValueError, match=message):
        read_split_policy_registry(path)


def test_source_family_and_labels_are_not_implicit_leakage_relations():
    left = _subject("tdl-a", family="standardized.tdl")
    right = _subject("tdl-b", family="standardized.tdl")
    other = _subject("cdl-a", family="standardized.cdl")
    subjects = [left, right, other, *_unlinked_subjects(8)]
    _, registry = _four_way_registry(subjects)
    assert (
        registry.subject_to_component[left.subject_id]
        != registry.subject_to_component[right.subject_id]
    )
    assert (
        registry.subject_to_component[left.subject_id]
        != registry.subject_to_component[other.subject_id]
    )
