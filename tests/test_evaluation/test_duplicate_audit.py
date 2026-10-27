import numpy as np
import pytest
from evaluation.duplicate_audit import (
    audit_cfr_near_duplicates,
    find_cross_split_temporal_overlaps,
)
from evaluation.independent_splits import make_independent_split


def make_records():
    records = [
        {"sample_id": f"sample-{index}", "scene_group_id": f"scene-{index}"} for index in range(6)
    ]
    split = make_independent_split(records, seed="duplicate-audit")
    return records, split


def test_cfr_fingerprint_flags_phase_and_gain_scaled_cross_split_duplicates():
    records, split = make_records()
    ids = [row["sample_id"] for row in records]
    train_index = next(
        i for i, sample in enumerate(ids) if split.sample_to_split[sample] == "train"
    )
    test_index = next(i for i, sample in enumerate(ids) if split.sample_to_split[sample] == "test")
    cfr = np.eye(len(ids), dtype=np.complex128)
    cfr[test_index] = 7j * cfr[train_index]

    audit = audit_cfr_near_duplicates(ids, cfr, split, similarity_threshold=0.9999, block_size=2)

    assert len(audit.pairs) == 1
    assert audit.pairs[0].similarity == pytest.approx(1.0)
    assert {audit.pairs[0].sample_id_a, audit.pairs[0].sample_id_b} == {
        ids[train_index],
        ids[test_index],
    }


def test_cfr_fingerprint_flags_identical_zero_power_outages():
    records, split = make_records()
    ids = [row["sample_id"] for row in records]
    train_id = next(sample for sample, label in split.sample_to_split.items() if label == "train")
    test_id = next(sample for sample, label in split.sample_to_split.items() if label == "test")
    cfr = np.eye(len(ids), dtype=np.complex64)
    cfr[ids.index(train_id)] = 0
    cfr[ids.index(test_id)] = 0

    audit = audit_cfr_near_duplicates(ids, cfr, split, similarity_threshold=1.0)

    assert len(audit.pairs) == 1
    assert audit.pairs[0].similarity == 1.0


@pytest.mark.parametrize("sample_id", [7, " sample-0", "sample-0 "])
def test_cfr_fingerprint_rejects_noncanonical_sample_ids(sample_id):
    records, split = make_records()
    ids = [row["sample_id"] for row in records]
    ids[0] = sample_id
    with pytest.raises(ValueError, match="canonical strings"):
        audit_cfr_near_duplicates(ids, np.ones((len(ids), 2), dtype=np.complex64), split)


def test_temporal_audit_flags_same_trajectory_windows_across_splits():
    records, split = make_records()
    train_id = next(sample for sample, label in split.sample_to_split.items() if label == "train")
    validation_id = next(
        sample for sample, label in split.sample_to_split.items() if label == "validation"
    )
    temporal = [
        {
            **row,
            "trajectory_id": row["sample_id"],
            "window_start_s": float(index * 10),
            "window_end_s": float(index * 10 + 1),
        }
        for index, row in enumerate(records)
    ]
    for row in temporal:
        if row["sample_id"] == train_id:
            row.update(trajectory_id="route-1", window_start_s=0.0, window_end_s=2.0)
        if row["sample_id"] == validation_id:
            row.update(trajectory_id="route-1", window_start_s=1.0, window_end_s=3.0)

    overlaps = find_cross_split_temporal_overlaps(temporal, split)

    assert len(overlaps) == 1
    assert overlaps[0].overlap_duration_s == pytest.approx(1.0)


def test_temporal_audit_requires_exact_sample_coverage():
    records, split = make_records()
    temporal_records = [
        {
            **row,
            "trajectory_id": row["sample_id"],
            "window_start_s": 0.0,
            "window_end_s": 1.0,
        }
        for row in records
    ]
    with pytest.raises(ValueError, match="exactly the samples"):
        find_cross_split_temporal_overlaps(temporal_records[:-1], split)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_id", 4, "sample_id.*canonical string"),
        ("sample_id", " sample-0", "sample_id.*canonical string"),
        ("trajectory_id", 4, "trajectory_id.*canonical string"),
        ("window_start_s", True, "finite real seconds"),
        ("window_end_s", "1.0", "finite real seconds"),
    ],
)
def test_temporal_audit_rejects_coerced_ids_and_time_values(field, value, message):
    records, split = make_records()
    temporal_records = [
        {
            **row,
            "trajectory_id": row["sample_id"],
            "window_start_s": 0.0,
            "window_end_s": 1.0,
        }
        for row in records
    ]
    temporal_records[0][field] = value

    with pytest.raises(ValueError, match=message):
        find_cross_split_temporal_overlaps(temporal_records, split)
