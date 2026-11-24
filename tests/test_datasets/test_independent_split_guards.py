"""Independent scene-grouped splits: typed assignments, fraction domains, record roster audits and immutable locks."""

from __future__ import annotations

import pytest
from datasets.independent_splits import (
    IndependentSplit,
    audit_group_leakage,
    freeze_split_lock,
    make_independent_split,
)


def _records():
    return [
        {"sample_id": f"{scene}-{link}", "scene_group_id": scene}
        for scene in ("scene-a", "scene-b", "scene-c", "scene-d", "scene-e", "scene-f")
        for link in range(3)
    ]


def _split() -> IndependentSplit:
    return make_independent_split(_records(), seed="guards-v1")


def test_split_assignments_must_be_mappings_and_nonempty() -> None:
    with pytest.raises(TypeError, match="split assignments must be mappings"):
        IndependentSplit(
            sample_to_split=["not", "mapping"],
            group_to_split={},
            seed="s",
            group_field="scene_group_id",
        )
    with pytest.raises(ValueError, match="must not be empty"):
        IndependentSplit(
            sample_to_split={}, group_to_split={}, seed="s", group_field="scene_group_id"
        )


@pytest.mark.parametrize(
    "fractions",
    [
        {"train": 0.5, "test": 0.5},
        {"train": 0.5, "validation": 0.2, "calibration": 0.2, "test": 0.1, "extra": 0.0},
    ],
    ids=["two-way", "unexpected-key"],
)
def test_fraction_domain_is_fixed(fractions) -> None:
    with pytest.raises(ValueError, match="fractions must define"):
        make_independent_split(_records(), seed="s", fractions=fractions)


def test_fractions_must_be_positive_and_sum_to_one() -> None:
    with pytest.raises(ValueError, match="positive and sum to one"):
        make_independent_split(
            _records(), seed="s", fractions={"train": 0.5, "validation": 0.3, "test": 0.3}
        )


def test_seed_and_records_are_required() -> None:
    with pytest.raises(ValueError):
        make_independent_split(_records(), seed=" padded ")
    with pytest.raises(ValueError):
        make_independent_split([], seed="s")


def test_each_record_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="each split record must be a mapping"):
        make_independent_split(["not-a-record"], seed="s")


def test_leakage_audit_rejects_unknown_duplicate_and_partial_rosters() -> None:
    split = _split()
    with pytest.raises(TypeError, match="audit records must be mappings"):
        audit_group_leakage(split, ["not-a-record"])
    unknown = _records() + [{"sample_id": "ghost", "scene_group_id": "scene-a"}]
    with pytest.raises(ValueError, match="absent from frozen split"):
        audit_group_leakage(split, unknown)
    with pytest.raises(ValueError, match="do not cover exactly"):
        audit_group_leakage(split, _records()[:-1])


def test_leakage_audit_accepts_the_exact_roster() -> None:
    assert audit_group_leakage(_split(), _records()) is None


def test_lock_is_created_once_and_identical_reruns_are_accepted(tmp_path) -> None:
    split = _split()
    lock = tmp_path / "split.json"
    digest = freeze_split_lock(lock, split)
    assert freeze_split_lock(lock, split) == digest
    other = make_independent_split(_records(), seed="different-seed")
    with pytest.raises(FileExistsError, match="refusing to modify existing split lock"):
        freeze_split_lock(lock, other)
