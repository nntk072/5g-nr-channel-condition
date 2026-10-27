"""Class availability and independent-group support for evaluation splits.

The report counts rows and caller-declared independent groups separately. It
does not infer independence from record identity or pool support across frozen
split/family strata.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ClassSupportCell:
    """Observed support for one declared split/family/class cell."""

    split_id: str
    family_id: str
    class_label: Any
    record_count: int
    independent_group_count: int
    status: str


@dataclass(frozen=True, slots=True)
class ClassSupportReport:
    """Stable support rows in declared stratum and class order."""

    cells: tuple[ClassSupportCell, ...]


def audit_class_support(
    labels: Sequence[Any] | np.ndarray,
    independent_group_ids: Sequence[str] | np.ndarray,
    split_ids: Sequence[str] | np.ndarray,
    family_ids: Sequence[str] | np.ndarray,
    *,
    expected_strata: Sequence[tuple[str, str]],
    classes: Sequence[Any],
) -> ClassSupportReport:
    """Report rows and distinct independent groups for every frozen class cell.

    Empty declared split/family strata and absent classes remain in the report
    with zero counts and explicit statuses. Every observed stratum must belong
    to ``expected_strata``. Group IDs are required and are only counted within
    their class and stratum; the caller remains responsible for their validity.
    """

    label_array = np.asarray(labels, dtype=object)
    group_array = np.asarray(independent_group_ids, dtype=object)
    split_array = np.asarray(split_ids, dtype=object)
    family_array = np.asarray(family_ids, dtype=object)
    arrays = (label_array, group_array, split_array, family_array)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("labels, group IDs, split IDs, and family IDs must be one-dimensional")
    if len({array.size for array in arrays}) != 1:
        raise ValueError("labels, group IDs, split IDs, and family IDs must have equal length")
    class_values = tuple(classes)
    if not class_values:
        raise ValueError("classes must be nonempty")
    try:
        if len(set(class_values)) != len(class_values):
            raise ValueError("classes must be unique")
    except TypeError as exc:
        raise ValueError("class labels must be hashable scalars") from exc
    strata = tuple(expected_strata)
    if not strata:
        raise ValueError("expected_strata must be nonempty")
    if any(not isinstance(stratum, tuple) or len(stratum) != 2 for stratum in strata):
        raise ValueError("each expected stratum must be a (split_id, family_id) tuple")
    if len(set(strata)) != len(strata):
        raise ValueError("expected_strata must be unique")
    for split_id, family_id in strata:
        if not isinstance(split_id, str) or not split_id or split_id != split_id.strip():
            raise ValueError("expected split IDs must be nonempty canonical strings")
        if not isinstance(family_id, str) or not family_id or family_id != family_id.strip():
            raise ValueError("expected family IDs must be nonempty canonical strings")

    for field_name, array in (
        ("group IDs", group_array),
        ("split IDs", split_array),
        ("family IDs", family_array),
    ):
        for value in array.tolist():
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{field_name} must contain nonempty canonical strings")
    allowed_strata = set(strata)
    observed_classes = set()
    for label, split_id, family_id in zip(
        label_array.tolist(), split_array.tolist(), family_array.tolist(), strict=True
    ):
        try:
            observed_classes.add(label)
        except TypeError as exc:
            raise ValueError("observed class labels must be hashable scalars") from exc
        if (split_id, family_id) not in allowed_strata:
            raise ValueError(f"observed undeclared split/family stratum {(split_id, family_id)!r}")
    unknown_classes = observed_classes - set(class_values)
    if unknown_classes:
        raise ValueError(f"observed labels are outside declared classes: {unknown_classes!r}")

    cells: list[ClassSupportCell] = []
    for split_id, family_id in strata:
        in_stratum = (split_array == split_id) & (family_array == family_id)
        has_records = bool(in_stratum.any())
        for class_label in class_values:
            selected = in_stratum & (label_array == class_label)
            record_count = int(selected.sum())
            group_count = len(set(group_array[selected].tolist()))
            status = (
                "no_records" if not has_records else "present" if record_count else "class_absent"
            )
            cells.append(
                ClassSupportCell(
                    split_id=split_id,
                    family_id=family_id,
                    class_label=class_label,
                    record_count=record_count,
                    independent_group_count=group_count,
                    status=status,
                )
            )
    return ClassSupportReport(cells=tuple(cells))
