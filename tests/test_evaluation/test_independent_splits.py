import json

import pytest
from evaluation.independent_splits import (
    IndependentSplit,
    audit_group_leakage,
    freeze_split_lock,
    make_independent_split,
)
from evaluation.spatial_audit import audit_spatial_buffer


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


def test_three_way_custom_fraction_mapping_keeps_historical_tie_break_order():
    samples = [
        {"sample_id": f"sample-{index}", "scene_group_id": f"scene-{index}"} for index in range(10)
    ]
    validation_first = make_independent_split(
        samples,
        seed="legacy-custom-order",
        fractions={"train": 0.6, "validation": 0.25, "test": 0.15},
    )
    test_first = make_independent_split(
        samples,
        seed="legacy-custom-order",
        fractions={"train": 0.6, "test": 0.15, "validation": 0.25},
    )
    assert sum(split == "validation" for split in validation_first.group_to_split.values()) == 3
    assert sum(split == "test" for split in validation_first.group_to_split.values()) == 1
    assert sum(split == "validation" for split in test_first.group_to_split.values()) == 2
    assert sum(split == "test" for split in test_first.group_to_split.values()) == 2


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


@pytest.mark.parametrize(
    "bad_record",
    [
        {"sample_id": None, "scene_group_id": "scene-a"},
        {"sample_id": 17, "scene_group_id": "scene-a"},
        {"sample_id": "sample-a", "scene_group_id": None},
        {"sample_id": "sample-a", "scene_group_id": 17},
        {"sample_id": "sample-a", "scene_group_id": "   "},
    ],
)
def test_split_rejects_missing_or_non_string_identity_fields(bad_record):
    with pytest.raises(ValueError, match="nonempty canonical string sample_id and scene_group_id"):
        make_independent_split(
            [bad_record, *records()[1:]],
            seed="strict-identities",
        )


def test_split_rejects_non_string_seed_and_identity_field_names():
    with pytest.raises(ValueError, match="seed must be canonical"):
        make_independent_split(records(), seed=7)
    with pytest.raises(ValueError, match="group_field and sample_id_field"):
        make_independent_split(records(), seed="strict-fields", group_field=None)


@pytest.mark.parametrize(
    ("sample_to_split", "group_to_split", "seed", "group_field", "message"),
    [
        ({None: "train"}, {"scene-a": "train"}, "seed", "scene_group_id", "sample assignment IDs"),
        ({"sample-a": "train"}, {17: "train"}, "seed", "scene_group_id", "group assignment IDs"),
        ({"sample-a": "unknown"}, {"scene-a": "train"}, "seed", "scene_group_id", "split names"),
        ({"sample-a": "train"}, {"scene-a": "train"}, None, "scene_group_id", "seed and group_field"),
    ],
)
def test_split_value_object_rejects_malformed_lock_fields(
    sample_to_split, group_to_split, seed, group_field, message
):
    with pytest.raises(ValueError, match=message):
        IndependentSplit(sample_to_split, group_to_split, seed, group_field)


def test_four_way_split_supports_spatial_buffer_audit():
    samples = [
        {
            "sample_id": f"sample-{index}",
            "scene_group_id": f"scene-{index}",
            "coordinates_m": [float(index * 100), 0.0],
        }
        for index in range(12)
    ]
    split = make_independent_split(
        samples,
        seed="four-way-spatial",
        fractions={"train": 0.55, "validation": 0.15, "calibration": 0.15, "test": 0.15},
    )
    assert set(split.sample_to_split.values()) == {"train", "validation", "calibration", "test"}
    assert audit_spatial_buffer(samples, split, buffer_m=1.0).is_clean


def test_group_leakage_audit_rejects_changed_scene_group():
    samples = records()
    split = make_independent_split(samples, seed="paper-v1")
    samples[0] = {**samples[0], "scene_group_id": "different-scene"}

    with pytest.raises(ValueError, match="leakage or changed group identity"):
        audit_group_leakage(split, samples)


def test_group_leakage_audit_rejects_non_string_identity_fields():
    samples = records()
    split = make_independent_split(samples, seed="strict-audit-identities")
    samples[0] = {**samples[0], "scene_group_id": None}

    with pytest.raises(ValueError, match="nonempty canonical sample and group IDs"):
        audit_group_leakage(split, samples)


@pytest.mark.parametrize(
    ("bad_record", "seed", "group_field"),
    [
        ({"sample_id": " sample-a", "scene_group_id": "scene-a"}, "canonical", "scene_group_id"),
        ({"sample_id": "sample-a", "scene_group_id": "scene-a "}, "canonical", "scene_group_id"),
        ({"sample_id": "sample-a", "scene_group_id": "scene-a"}, " padded", "scene_group_id"),
        ({"sample_id": "sample-a", "scene_group_id": "scene-a"}, "canonical", " scene_group_id"),
    ],
)
def test_split_rejects_padded_identity_values(bad_record, seed, group_field):
    with pytest.raises(ValueError):
        make_independent_split(
            [bad_record, *records()[1:]],
            seed=seed,
            group_field=group_field,
        )


def test_split_lock_value_and_audit_reject_padded_identity_keys():
    with pytest.raises(ValueError, match="canonical strings"):
        IndependentSplit(
            {" sample-a": "train"},
            {"scene-a": "train"},
            "seed",
            "scene_group_id",
        )

    samples = records()
    split = make_independent_split(samples, seed="canonical-audit")
    samples[0] = {**samples[0], "sample_id": f" {samples[0]['sample_id']}"}
    with pytest.raises(ValueError, match="canonical sample and group IDs"):
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
