"""CFR and temporal duplicate audits: input shape, identity, block-size and window rules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from evaluation.audits.duplicate_audit import (
    audit_cfr_near_duplicates,
    find_cross_split_temporal_overlaps,
)
from tests.test_evaluation.test_duplicate_audit import make_records


def _ids_and_split():
    records, split = make_records()
    return [row["sample_id"] for row in records], split


def test_clean_audit_reports_is_clean() -> None:
    ids, split = _ids_and_split()
    audit = audit_cfr_near_duplicates(ids, np.eye(len(ids), dtype=np.complex128), split)
    assert audit.is_clean


@pytest.mark.parametrize(
    ("cfr_factory", "message"),
    [
        pytest.param(
            lambda n: np.ones((n, 2), dtype=np.float64), "finite complex array", id="real"
        ),
        pytest.param(
            lambda n: np.ones((n, 2), dtype=np.complex128) * np.nan,
            "finite complex array",
            id="nan",
        ),
        pytest.param(
            lambda n: np.ones((n - 1, 2), dtype=np.complex128), "identify every CFR row", id="rows"
        ),
    ],
)
def test_cfr_array_must_identify_every_sample(cfr_factory, message) -> None:
    ids, split = _ids_and_split()
    with pytest.raises(ValueError, match=message):
        audit_cfr_near_duplicates(ids, cfr_factory(len(ids)), split)


def test_sample_ids_must_be_unique_and_cover_the_split() -> None:
    ids, split = _ids_and_split()
    cfr = np.eye(len(ids), dtype=np.complex128)
    with pytest.raises(ValueError, match="must be unique"):
        audit_cfr_near_duplicates([ids[0]] * len(ids), cfr, split)
    with pytest.raises(ValueError, match="exactly the frozen split"):
        audit_cfr_near_duplicates([*ids[:-1], "absent-sample"], cfr, split)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param(
            {"similarity_threshold": 0.0}, "similarity_threshold must be in", id="threshold-zero"
        ),
        pytest.param(
            {"similarity_threshold": 1.5}, "similarity_threshold must be in", id="threshold-high"
        ),
        pytest.param({"block_size": 0}, "block_size must be a positive integer", id="block-zero"),
    ],
)
def test_similarity_and_block_parameters_are_checked(kwargs, message) -> None:
    ids, split = _ids_and_split()
    with pytest.raises(ValueError, match=message):
        audit_cfr_near_duplicates(ids, np.eye(len(ids), dtype=np.complex128), split, **kwargs)


def _window(
    sample_id: str, trajectory: str = "t", start: float = 0.0, end: float = 1.0
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "trajectory_id": trajectory,
        "window_start_s": start,
        "window_end_s": end,
    }


def test_temporal_records_must_be_mappings_with_canonical_identity() -> None:
    ids, split = _ids_and_split()
    with pytest.raises(TypeError, match="must be a mapping"):
        find_cross_split_temporal_overlaps(["not-a-mapping"], split)
    records = [_window(sample) for sample in ids]
    records[0]["sample_id"] = " padded"
    with pytest.raises(ValueError, match="sample_id must be a non-empty canonical string"):
        find_cross_split_temporal_overlaps(records, split)
    records = [_window(sample) for sample in ids]
    records[0]["trajectory_id"] = ""
    with pytest.raises(ValueError, match="trajectory_id must be a non-empty canonical string"):
        find_cross_split_temporal_overlaps(records, split)


def test_temporal_windows_must_be_finite_and_ordered() -> None:
    ids, split = _ids_and_split()
    records = [_window(sample) for sample in ids]
    records[0]["window_end_s"] = -1.0
    with pytest.raises(ValueError, match="finite ordered time window"):
        find_cross_split_temporal_overlaps(records, split)
    records = [_window(sample) for sample in ids]
    records[0]["window_start_s"] = "0"
    with pytest.raises(ValueError, match="window boundaries must be finite real seconds"):
        find_cross_split_temporal_overlaps(records, split)


def test_temporal_records_must_cover_exactly_the_split() -> None:
    ids, split = _ids_and_split()
    records = [_window(sample) for sample in ids]
    with pytest.raises(ValueError, match="exactly the samples in the frozen split"):
        find_cross_split_temporal_overlaps(records[:-1], split)
    records = [_window(ids[0]), _window(ids[0])] + [_window(sample) for sample in ids[1:]]
    with pytest.raises(ValueError, match="non-empty and unique"):
        find_cross_split_temporal_overlaps(records, split)
