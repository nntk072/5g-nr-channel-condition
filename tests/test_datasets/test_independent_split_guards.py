"""Independent scene-grouped splits: typed assignments, fraction domains, record roster audits and immutable locks."""

from __future__ import annotations

from pathlib import Path

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


@pytest.mark.parametrize(
    ("record_count", "fractions", "message"),
    [
        (2, None, "at least three independent groups"),
        (
            3,
            {"train": 0.4, "validation": 0.2, "calibration": 0.2, "test": 0.2},
            "at least 4 independent groups",
        ),
    ],
    ids=["three-way-minimum", "four-way-minimum"],
)
def test_split_requires_one_group_per_partition(record_count, fractions, message) -> None:
    records = [
        {"sample_id": f"sample-{index}", "scene_group_id": f"scene-{index}"}
        for index in range(record_count)
    ]
    with pytest.raises(ValueError, match=message):
        make_independent_split(records, seed="s", fractions=fractions)


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
    incomplete_groups = [row for row in _records() if row["scene_group_id"] != "scene-f"]
    with pytest.raises(ValueError, match="do not cover exactly the groups"):
        audit_group_leakage(split, incomplete_groups)


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


def test_split_lock_accepts_identical_concurrent_publication(tmp_path, monkeypatch) -> None:
    import datasets.independent_splits as split_module

    split = _split()
    destination = tmp_path / "split.json"

    def competing_same_writer(source, target):
        destination.write_bytes(Path(source).read_bytes())
        raise FileExistsError("another writer published identical bytes")

    monkeypatch.setattr(split_module.os, "link", competing_same_writer)

    digest = freeze_split_lock(destination, split)

    assert len(digest) == 64
    assert destination.exists()
    assert list(tmp_path.iterdir()) == [destination]


def test_split_lock_rejects_different_concurrent_publication(tmp_path, monkeypatch) -> None:
    import datasets.independent_splits as split_module

    destination = tmp_path / "split.json"

    def competing_different_writer(_source, target):
        Path(target).write_text("different lock", encoding="utf-8")
        raise FileExistsError("another writer published conflicting bytes")

    monkeypatch.setattr(split_module.os, "link", competing_different_writer)
    with pytest.raises(FileExistsError, match="refusing to modify existing split lock"):
        freeze_split_lock(destination, _split())

    assert destination.read_text(encoding="utf-8") == "different lock"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["split.json"]


def test_split_lock_cleans_up_if_temporary_file_creation_fails(tmp_path, monkeypatch) -> None:
    import datasets.independent_splits as split_module

    def fail_temporary_file(**_kwargs):
        raise OSError("temporary storage unavailable")

    monkeypatch.setattr(split_module.tempfile, "NamedTemporaryFile", fail_temporary_file)
    with pytest.raises(OSError, match="temporary storage unavailable"):
        freeze_split_lock(tmp_path / "split.json", _split())

    assert list(tmp_path.iterdir()) == []


def test_split_lock_accepts_identical_writer_before_temp_creation(tmp_path, monkeypatch) -> None:
    import datasets.independent_splits as split_module

    split = _split()
    canonical = freeze_split_lock(tmp_path / "canonical.json", split)
    destination = tmp_path / "split.json"

    def competing_writer(**_kwargs):
        destination.write_bytes((tmp_path / "canonical.json").read_bytes())
        raise FileExistsError("another writer published before temp creation")

    monkeypatch.setattr(split_module.tempfile, "NamedTemporaryFile", competing_writer)

    assert freeze_split_lock(destination, split) == canonical
    assert destination.read_bytes() == (tmp_path / "canonical.json").read_bytes()
