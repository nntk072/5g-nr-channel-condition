"""Near-duplicate audit: identity, geometry-fingerprint and CFR admission rules."""

from __future__ import annotations

import numpy as np
import pytest
from evaluation.audits import near_duplicate_audit as audit_module
from evaluation.audits.near_duplicate_audit import (
    ChannelRecord,
    GeometryRecord,
    audit_geometry_duplicates,
    audit_near_duplicates,
)


def _cfr(value: float = 1.0) -> np.ndarray:
    return np.full((2, 4), value + 0.5j, dtype=np.complex128)


@pytest.mark.parametrize(
    ("records", "message"),
    [
        pytest.param([], "nonempty tuple", id="empty"),
        pytest.param(
            (GeometryRecord(" a", "train", "h"),),
            "sample_id must be a canonical",
            id="sample-padded",
        ),
        pytest.param(
            (GeometryRecord("a", "", "h"),), "split_id must be a canonical", id="split-empty"
        ),
        pytest.param(
            (GeometryRecord("a", "train", " h"),),
            "family_site_geometry_hash must be None",
            id="hash-padded",
        ),
        pytest.param(
            (GeometryRecord("a", "train", "h"), GeometryRecord("a", "test", "h")),
            "duplicate sample_id",
            id="sample-duplicate",
        ),
    ],
)
def test_geometry_audit_rejects_malformed_records(records, message) -> None:
    with pytest.raises(ValueError, match=message):
        audit_geometry_duplicates(records)


def test_geometry_audit_flags_only_cross_split_matches() -> None:
    candidates = audit_geometry_duplicates(
        (
            GeometryRecord("a", "train", "same"),
            GeometryRecord("b", "test", "same"),
            GeometryRecord("c", "test", "same"),
            GeometryRecord("d", "train", None),
        )
    )
    assert [(item.sample_id_a, item.sample_id_b) for item in candidates] == [
        ("a", "b"),
        ("a", "c"),
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"sample_id": " a"}, "sample_id must be a canonical", id="sample"),
        pytest.param(
            {"family_site_geometry_hash": " h"}, "family_site_geometry_hash must be None", id="hash"
        ),
        pytest.param(
            {"comparison_context_id": ""}, "comparison_context_id must be None", id="context"
        ),
        pytest.param({"cfr": [1.0, 2.0]}, "must be a NumPy array", id="cfr-type"),
        pytest.param(
            {"cfr": np.zeros((0, 2), dtype=np.complex128)}, "must be nonempty", id="cfr-empty"
        ),
        pytest.param({"cfr": np.ones((2, 2), dtype=np.float64)}, "complex dtype", id="cfr-real"),
        pytest.param({"cfr": np.full((2, 2), np.inf + 0j)}, "only finite values", id="cfr-inf"),
    ],
)
def test_channel_records_are_validated_before_comparison(overrides, message) -> None:
    base = {"sample_id": "a", "split_id": "train", "family_site_geometry_hash": None, "cfr": _cfr()}
    base.update(overrides)
    records = (
        ChannelRecord(**base),
        ChannelRecord(
            sample_id="b", split_id="test", family_site_geometry_hash=None, cfr=_cfr(2.0)
        ),
    )
    with pytest.raises(ValueError, match=message):
        audit_near_duplicates(records, normalized_inner_product_threshold=0.9)


def test_channel_audit_requires_nonempty_records_and_unique_identities() -> None:
    with pytest.raises(ValueError, match="nonempty tuple"):
        audit_near_duplicates((), normalized_inner_product_threshold=0.9)
    first = ChannelRecord("a", "train", None, _cfr())
    with pytest.raises(ValueError, match="duplicate sample_id"):
        audit_near_duplicates(
            (first, ChannelRecord("a", "test", None, _cfr())),
            normalized_inner_product_threshold=0.9,
        )


def test_identical_cfrs_across_splits_are_flagged() -> None:
    report = audit_near_duplicates(
        (ChannelRecord("a", "train", None, _cfr()), ChannelRecord("b", "test", None, _cfr())),
        normalized_inner_product_threshold=0.9,
    )
    assert report is not None


def test_nonfinite_normalized_cfr_similarity_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit_module.np, "vdot", lambda *_args: complex(float("nan"), 0.0))

    with pytest.raises(ValueError, match="could not be computed as a finite value"):
        audit_module._normalized_inner_product_magnitude(_cfr(), _cfr())
