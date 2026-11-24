"""Split registry: roster, binding, audit and whole-scene holdout rules."""

from __future__ import annotations

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
