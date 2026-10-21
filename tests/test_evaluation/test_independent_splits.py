import json

import pytest
from evaluation.independent_splits import (
    audit_group_leakage,
    freeze_split_lock,
    make_independent_split,
)


def records():
    return [
        {"sample_id": f"{scene}-{link}", "scene_group_id": scene}
        for scene in ("scene-a", "scene-b", "scene-c", "scene-d", "scene-e", "scene-f")
        for link in range(3)
    ]


def test_split_is_deterministic_and_groups_never_cross_boundaries():
    samples = records()
    split = make_independent_split(samples, seed="paper-v1")
    repeated = make_independent_split(list(reversed(samples)), seed="paper-v1")

    assert split.to_dict() == repeated.to_dict()
    audit_group_leakage(split, samples)
    with pytest.raises(TypeError):
        split.sample_to_split["new-sample"] = "train"
    for scene in split.group_to_split:
        values = {
            split.sample_to_split[row["sample_id"]]
            for row in samples
            if row["scene_group_id"] == scene
        }
        assert values == {split.group_to_split[scene]}


def test_split_requires_group_identity_and_unique_sample_ids():
    with pytest.raises(ValueError, match="duplicate sample ID"):
        make_independent_split(
            [
                {"sample_id": "x", "scene_group_id": "a"},
                {"sample_id": "x", "scene_group_id": "b"},
                {"sample_id": "z", "scene_group_id": "c"},
            ],
            seed="v1",
        )
    with pytest.raises(ValueError, match="at least three independent groups"):
        make_independent_split(
            [
                {"sample_id": "a", "scene_group_id": "one"},
                {"sample_id": "b", "scene_group_id": "two"},
            ],
            seed="v1",
        )


def test_group_leakage_audit_rejects_changed_scene_group():
    samples = records()
    split = make_independent_split(samples, seed="paper-v1")
    samples[0] = {**samples[0], "scene_group_id": "different-scene"}

    with pytest.raises(ValueError, match="leakage or changed group identity"):
        audit_group_leakage(split, samples)


def test_group_leakage_audit_requires_exactly_the_locked_sample_set():
    samples = records()
    split = make_independent_split(samples, seed="paper-v1")

    with pytest.raises(ValueError, match="exactly the samples"):
        audit_group_leakage(split, samples[:-1])
    with pytest.raises(ValueError, match="duplicate sample ID"):
        audit_group_leakage(split, [*samples, samples[0]])


def test_split_lock_is_idempotent_but_cannot_be_rewritten(tmp_path):
    split = make_independent_split(records(), seed="paper-v1")
    lock = tmp_path / "paper_v1_split.json"

    digest = freeze_split_lock(lock, split)
    assert freeze_split_lock(lock, split) == digest
    assert json.loads(lock.read_text())["schema_version"] == 1

    changed = make_independent_split(records(), seed="paper-v2")
    with pytest.raises(FileExistsError, match="refusing to modify"):
        freeze_split_lock(lock, changed)
