"""Sparse, leakage-safe UMCD Release Split Policy v2 contracts."""

from __future__ import annotations

import json

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
    SplitParentType,
    SplitPolicy,
    SplitPolicyRegistry,
    SplitSubject,
    audit_split_policy_registry,
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
            "delay_reference": "profile_excess_delay",
            "subcarrier_spacing_hz": 15_000.0,
            "num_subcarriers": 8,
            "num_time_steps": 1,
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
    with pytest.raises(ValueError, match="same component"):
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
    _, registry = _four_way_registry(subjects)
    assert parent.subject_id == episode.episode_id
    assert registry.subjects[parent.subject_id].source_family == episode.family_id
    assert (
        LeakageRelation(LeakageNamespace.ASSET_FAMILY, episode.split_group_id) in parent.relations
    )
    parent_split = registry.split_for_parent(episode.episode_id)
    descendants = ["link", "frame", "path-oracle", "channel", "receiver", "noise", "target"]
    assert {registry.split_for_child(episode.episode_id) for _child in descendants} == {
        parent_split
    }


def test_standardized_draw_not_generator_batch_is_parent_for_all_views():
    record = _standardized_record()
    parent = standardized_draw_subject(record)
    subjects = [parent, *_unlinked_subjects(8)]
    _, registry = _four_way_registry(subjects)
    assert parent.subject_id == record.sample_id
    assert parent.parent_type is SplitParentType.STANDARDIZED_CHANNEL_DRAW
    assert parent.source_family == "standardized.tdl"
    assert registry.split_for_parent(record.sample_id) == registry.split_for_child(record.sample_id)
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
    _, registry = _four_way_registry(subjects)
    base = registry.split_for_parent("base-episode")
    assert registry.split_for_parent("counterfactual-twin") == base
    assert registry.split_for_parent("same-route-window") == base
    # Noise, bandwidth, receiver, and PUSCH variants are children, never new parents.
    assert {registry.split_for_child("base-episode") for _ in range(12)} == {base}


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
    path = tmp_path / "release-v2.json"
    digest = freeze_split_policy_registry(path, registry)
    assert freeze_split_policy_registry(path, registry) == digest
    restored = read_split_policy_registry(path)
    assert restored.to_dict() == registry.to_dict()

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
