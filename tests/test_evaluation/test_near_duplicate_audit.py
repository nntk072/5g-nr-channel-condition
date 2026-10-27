from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from evaluation.near_duplicate_audit import CFRArray, ChannelRecord, audit_near_duplicates


def _record(
    sample_id: str,
    split_id: str,
    values: tuple[complex, ...],
    geometry_hash: str | None = None,
    context_id: str | None = None,
) -> ChannelRecord:
    return ChannelRecord(
        sample_id=sample_id,
        split_id=split_id,
        family_site_geometry_hash=geometry_hash,
        cfr=np.asarray(values, dtype=np.complex128),
        comparison_context_id=context_id,
    )


def test_flags_exact_cfr_copy_and_global_phase_gain_scaled_copy() -> None:
    baseline = np.asarray([1 + 2j, -3 + 0.5j, 2 - 1j], dtype=np.complex128)
    scaled = (4.2 * np.exp(1.37j)) * baseline
    report = audit_near_duplicates(
        (
            ChannelRecord("copy", "test", None, baseline.copy()),
            ChannelRecord("original", "train", None, baseline),
            ChannelRecord("scaled", "validation", None, scaled),
        ),
        normalized_inner_product_threshold=0.999999,
    )

    assert [(pair.sample_id_a, pair.sample_id_b) for pair in report.candidates] == [
        ("copy", "original"),
        ("copy", "scaled"),
        ("original", "scaled"),
    ]
    assert all(
        pair.normalized_complex_inner_product_magnitude == pytest.approx(1.0)
        for pair in report.candidates
    )
    assert all(not pair.geometry_hash_match for pair in report.candidates)


def test_flags_matching_geometry_hash_even_when_cfr_is_independent() -> None:
    report = audit_near_duplicates(
        (
            _record("a", "train", (1 + 0j, 0j), "geometry-123"),
            _record("b", "test", (0j, 1 + 0j), "geometry-123"),
        ),
        normalized_inner_product_threshold=0.99,
    )

    assert len(report.candidates) == 1
    assert report.candidates[0].geometry_hash_match
    assert report.candidates[0].normalized_complex_inner_product_magnitude == pytest.approx(0.0)


def test_independent_cross_split_pair_is_not_flagged() -> None:
    report = audit_near_duplicates(
        (
            _record("a", "train", (1 + 0j, 0j)),
            _record("b", "test", (0j, 1 + 0j)),
        ),
        normalized_inner_product_threshold=0.9,
    )

    assert report.candidates == ()
    assert report.skipped_comparisons == ()


def test_same_split_duplicates_are_ignored_and_output_is_sorted() -> None:
    records = (
        _record("z", "train", (1 + 0j, 0j)),
        _record("b", "test", (1 + 0j, 0j)),
        _record("a", "train", (1 + 0j, 0j)),
    )
    report = audit_near_duplicates(records, normalized_inner_product_threshold=0.8)

    assert [(pair.sample_id_a, pair.sample_id_b) for pair in report.candidates] == [
        ("a", "b"),
        ("b", "z"),
    ]
    assert all(pair.sample_id_a != "a" or pair.sample_id_b != "z" for pair in report.candidates)


def test_zero_norm_pair_is_reported_as_skipped() -> None:
    report = audit_near_duplicates(
        (
            _record("zero", "train", (0j, 0j)),
            _record("nonzero", "test", (1 + 0j, 2j)),
        ),
        normalized_inner_product_threshold=0.5,
    )

    assert report.candidates == ()
    assert [
        (pair.sample_id_a, pair.sample_id_b, pair.reason) for pair in report.skipped_comparisons
    ] == [("nonzero", "zero", "zero_norm_cfr")]


def test_zero_norm_pair_can_still_be_flagged_by_geometry() -> None:
    report = audit_near_duplicates(
        (
            _record("a", "train", (0j,), "same"),
            _record("b", "test", (0j,), "same"),
        ),
        normalized_inner_product_threshold=0.5,
    )

    assert report.candidates[0].geometry_hash_match
    assert report.candidates[0].normalized_complex_inner_product_magnitude is None
    assert report.skipped_comparisons[0].reason == "zero_norm_cfr"


def test_incompatible_channel_contexts_are_skipped_instead_of_scored() -> None:
    report = audit_near_duplicates(
        (
            _record("a", "train", (1 + 0j, 0j), context_id="frequency-grid-a"),
            _record("b", "test", (1 + 0j, 0j), context_id="frequency-grid-b"),
        ),
        normalized_inner_product_threshold=0.99,
    )

    assert report.candidates == ()
    assert [(item.sample_id_a, item.sample_id_b, item.reason) for item in report.skipped_comparisons] == [
        ("a", "b", "incompatible_channel_context")
    ]


def test_matching_geometry_remains_flagged_across_incompatible_channel_contexts() -> None:
    report = audit_near_duplicates(
        (
            _record("a", "train", (1 + 0j,), "same-geometry", "context-a"),
            _record("b", "test", (1 + 0j, 2j), "same-geometry", "context-b"),
        ),
        normalized_inner_product_threshold=0.99,
    )

    assert len(report.candidates) == 1
    assert report.candidates[0].geometry_hash_match
    assert report.candidates[0].normalized_complex_inner_product_magnitude is None
    assert report.skipped_comparisons[0].reason == "incompatible_channel_context"


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_invalid_threshold_fails_closed(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        audit_near_duplicates(
            (_record("a", "train", (1 + 0j,)),), normalized_inner_product_threshold=threshold
        )


def test_duplicate_sample_ids_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate sample_id"):
        audit_near_duplicates(
            (_record("same", "train", (1 + 0j,)), _record("same", "test", (1 + 0j,))),
            normalized_inner_product_threshold=0.9,
        )


@pytest.mark.parametrize(
    ("record", "field"),
    [
        (_record(" padded", "train", (1 + 0j,)), "sample_id"),
        (_record("sample", "train ", (1 + 0j,)), "split_id"),
        (_record("sample", "train", (1 + 0j,), " geometry-hash"), "family_site_geometry_hash"),
    ],
)
def test_padded_split_audit_identities_fail_closed(record: ChannelRecord, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        audit_near_duplicates((record,), normalized_inner_product_threshold=0.9)


def test_inconsistent_shapes_fail_closed() -> None:
    with pytest.raises(ValueError, match="inconsistent CFR shapes"):
        audit_near_duplicates(
            (_record("a", "train", (1 + 0j,)), _record("b", "test", (1 + 0j, 2j))),
            normalized_inner_product_threshold=0.9,
        )


def test_nonfinite_cfr_fails_closed() -> None:
    with pytest.raises(ValueError, match="finite"):
        audit_near_duplicates(
            (ChannelRecord("a", "train", None, np.asarray([complex(float("nan"), 0)])),),
            normalized_inner_product_threshold=0.9,
        )


def test_noncomplex_cfr_fails_closed() -> None:
    invalid_cfr = cast(CFRArray, np.asarray([1.0, 2.0]))
    record = replace(_record("a", "train", (1 + 0j, 2 + 0j)), cfr=invalid_cfr)
    with pytest.raises(ValueError, match="complex dtype"):
        audit_near_duplicates((record,), normalized_inner_product_threshold=0.9)


def test_finite_extreme_complex_values_do_not_create_false_duplicate_match() -> None:
    large = 1.7e308 + 1.7e308j
    report = audit_near_duplicates(
        (_record("a", "train", (large, 0j)), _record("b", "test", (0j, large))),
        normalized_inner_product_threshold=0.1,
    )

    assert report.candidates == ()
    assert report.skipped_comparisons == ()
