"""Class-support audit: inputs must be aligned one-dimensional rosters with unique, hashable classes and strata."""

from __future__ import annotations

import pytest
from evaluation.scoring.class_support import audit_class_support

STRATA = [("train", "family-a")]


def _audit(**overrides):
    kwargs = {
        "labels": [0, 1],
        "independent_group_ids": ["g0", "g1"],
        "split_ids": ["train", "train"],
        "family_ids": ["family-a", "family-a"],
        "expected_strata": STRATA,
        "classes": [0, 1],
    }
    kwargs.update(overrides)
    return audit_class_support(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"labels": [[0, 1]]}, "must be one-dimensional", id="labels-2d"),
        pytest.param({"classes": []}, "classes must be nonempty", id="no-classes"),
        pytest.param({"classes": [0, 0]}, "classes must be unique", id="duplicate-classes"),
        pytest.param({"classes": [[0], [1]]}, "hashable scalars", id="unhashable-classes"),
        pytest.param({"expected_strata": []}, "expected_strata must be nonempty", id="no-strata"),
        pytest.param(
            {"expected_strata": ["train"]},
            r"each expected stratum must be a \(split_id, family_id\) tuple",
            id="stratum-shape",
        ),
    ],
)
def test_class_support_rejects_malformed_rosters(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _audit(**overrides)


def test_class_support_accepts_a_well_formed_roster() -> None:
    report = _audit()
    assert report is not None
