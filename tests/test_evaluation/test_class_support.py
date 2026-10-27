from __future__ import annotations

import numpy as np
import pytest
from evaluation.class_support import audit_class_support


def test_reports_each_declared_class_and_independent_group_by_stratum():
    report = audit_class_support(
        labels=np.array([0, 0, 1, 1, 0]),
        independent_group_ids=np.array(["g1", "g1", "g2", "g3", "g4"]),
        split_ids=np.array(["train", "train", "train", "train", "test"]),
        family_ids=np.array(["urban"] * 5),
        expected_strata=(("train", "urban"), ("test", "urban")),
        classes=(0, 1),
    )

    assert [
        (
            cell.split_id,
            cell.class_label,
            cell.record_count,
            cell.independent_group_count,
            cell.status,
        )
        for cell in report.cells
    ] == [
        ("train", 0, 2, 1, "present"),
        ("train", 1, 2, 2, "present"),
        ("test", 0, 1, 1, "present"),
        ("test", 1, 0, 0, "class_absent"),
    ]


def test_empty_declared_stratum_is_retained():
    report = audit_class_support(
        labels=[0, 1],
        independent_group_ids=["g0", "g1"],
        split_ids=["train", "train"],
        family_ids=["urban", "urban"],
        expected_strata=(("train", "urban"), ("test", "rural")),
        classes=(0, 1),
    )

    assert [cell.status for cell in report.cells[-2:]] == ["no_records", "no_records"]
    assert [cell.independent_group_count for cell in report.cells[-2:]] == [0, 0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"independent_group_ids": []}, "equal length"),
        ({"independent_group_ids": [""]}, "canonical strings"),
        ({"independent_group_ids": [" g "]}, "canonical strings"),
        ({"split_ids": ["unregistered"]}, "undeclared"),
        ({"labels": [2]}, "outside declared classes"),
    ],
)
def test_rejects_invalid_support_inputs(kwargs, message):
    values = {
        "labels": [0],
        "independent_group_ids": ["g0"],
        "split_ids": ["test"],
        "family_ids": ["urban"],
        "expected_strata": (("test", "urban"),),
        "classes": (0, 1),
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        audit_class_support(**values)
