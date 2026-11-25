"""Split registry: roster, binding, audit and whole-scene holdout rules."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import pytest
from datasets.split_policy import SplitPolicy
from datasets.split_policy.registry import (
    audit_split_policy_child_roster,
    audit_split_policy_parent_roster,
    audit_whole_scene_holdout,
    make_split_policy_registry,
)
from tests.test_datasets.test_split_policy_v2 import (
    _four_way_registry,
    _subject,
    _unlinked_subjects,
)


def _registry(count: int = 12):
    return _four_way_registry(_unlinked_subjects(count))


def test_registry_rejects_non_policy_and_non_subject_inputs() -> None:
    with pytest.raises(TypeError, match="policy must be a SplitPolicy"):
        make_split_policy_registry(_unlinked_subjects(12), policy="not-a-policy")
    with pytest.raises(TypeError, match="subjects must contain SplitSubject values"):
        make_split_policy_registry(["not-a-subject"], policy=SplitPolicy.release_v2(seed="s"))
    with pytest.raises(ValueError, match="duplicate split subject ID"):
        make_split_policy_registry(
            [_subject("same"), _subject("same")], policy=SplitPolicy.release_v2(seed="s")
        )


def test_child_roster_requires_frozen_children_of_the_declared_kind() -> None:
    _, registry = _registry()
    with pytest.raises(ValueError, match="no frozen children of kind"):
        audit_split_policy_child_roster(registry, [], child_kind="run")


def test_parent_roster_must_cover_exactly_the_frozen_parents() -> None:
    _, registry = _registry()
    with pytest.raises(ValueError, match="absent from frozen split registry"):
        audit_split_policy_parent_roster(registry, [_subject("unknown-subject")])


def test_whole_scene_holdout_must_keep_each_scene_in_one_partition() -> None:
    _, registry = _registry()
    with pytest.raises(TypeError, match="registry must be a SplitPolicyRegistry"):
        audit_whole_scene_holdout(
            "registry", {"scene": ["a"]}, expected_scene_ids=["scene"], expected_subject_ids=["a"]
        )
    with pytest.raises(TypeError, match="scene_subjects must be a mapping"):
        audit_whole_scene_holdout(
            registry, ["not-a-mapping"], expected_scene_ids=["scene"], expected_subject_ids=["a"]
        )
    with pytest.raises(ValueError, match="must not contain duplicate IDs"):
        audit_whole_scene_holdout(
            registry,
            {"scene": ["a"]},
            expected_scene_ids=["scene", "scene"],
            expected_subject_ids=["a"],
        )
    with pytest.raises(ValueError, match="must be non-empty"):
        audit_whole_scene_holdout(registry, {}, expected_scene_ids=[], expected_subject_ids=["a"])


def test_whole_scene_holdout_validates_rosters_and_frozen_membership() -> None:
    from datasets.split_policy import LeakageNamespace

    scene_parent = _subject("scene-parent", (LeakageNamespace.SCENE_REALIZATION, "scene-1"))
    subjects = [scene_parent, *_unlinked_subjects(11)]
    registry = make_split_policy_registry(
        subjects, policy=SplitPolicy.release_v2(seed="scene-roster")
    )

    with pytest.raises(TypeError, match="must be a sequence of IDs"):
        audit_whole_scene_holdout(
            registry,
            {"scene-1": ["scene-parent"]},
            expected_scene_ids="scene-1",
            expected_subject_ids=["scene-parent"],
        )
    with pytest.raises(ValueError, match="scene has no parent subjects"):
        audit_whole_scene_holdout(
            registry,
            {"scene-1": []},
            expected_scene_ids=["scene-1"],
            expected_subject_ids=["scene-parent"],
        )
    with pytest.raises(ValueError, match="absent from frozen registry"):
        audit_whole_scene_holdout(
            registry,
            {"scene-1": ["ghost"]},
            expected_scene_ids=["scene-1"],
            expected_subject_ids=["ghost"],
        )
    with pytest.raises(ValueError, match="lacks its scene-realization relation"):
        audit_whole_scene_holdout(
            registry,
            {"scene-1": [subjects[1].subject_id]},
            expected_scene_ids=["scene-1"],
            expected_subject_ids=[subjects[1].subject_id],
        )

    report = audit_whole_scene_holdout(
        registry,
        {"scene-1": ["scene-parent"]},
        expected_scene_ids=["scene-1"],
        expected_subject_ids=["scene-parent"],
    )
    assert report["status"] == "PASS_INPUT_SCOPED"


def test_whole_scene_holdout_rejects_duplicate_scene_keys_and_cross_split_scene() -> None:
    from datasets.split_policy import LeakageNamespace

    class DuplicateSceneMapping(Mapping):
        def __getitem__(self, key):
            if key != "scene-1":
                raise KeyError(key)
            return ["scene-parent"]

        def __iter__(self):
            return iter(("scene-1", "scene-1"))

        def __len__(self):
            return 2

    one_scene = _subject("scene-parent", (LeakageNamespace.SCENE_REALIZATION, "scene-1"))
    subjects = [one_scene, *_unlinked_subjects(11)]
    registry = make_split_policy_registry(
        subjects, policy=SplitPolicy.release_v2(seed="scene-duplicate")
    )
    with pytest.raises(ValueError, match="duplicate scene ID"):
        audit_whole_scene_holdout(
            registry,
            DuplicateSceneMapping(),
            expected_scene_ids=["scene-1"],
            expected_subject_ids=["scene-parent"],
        )

    related = [
        _subject("scene-a", (LeakageNamespace.SCENE_REALIZATION, "shared-scene")),
        _subject("scene-b", (LeakageNamespace.SCENE_REALIZATION, "shared-scene")),
        *_unlinked_subjects(10),
    ]
    forged = make_split_policy_registry(related, policy=SplitPolicy.release_v2(seed="scene-splits"))
    component_map = dict(forged.subject_to_component)
    component_map["scene-a"] = "forged-a"
    component_map["scene-b"] = "forged-b"
    split_map = {**forged.component_to_split, "forged-a": "train", "forged-b": "test"}
    object.__setattr__(forged, "subject_to_component", MappingProxyType(component_map))
    object.__setattr__(forged, "component_to_split", MappingProxyType(split_map))
    with pytest.raises(ValueError, match="whole-scene holdout is violated"):
        audit_whole_scene_holdout(
            forged,
            {"shared-scene": ["scene-a", "scene-b"]},
            expected_scene_ids=["shared-scene"],
            expected_subject_ids=["scene-a", "scene-b"],
        )
