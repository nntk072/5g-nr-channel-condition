"""Deterministic coverage accounting for explicitly inventoried targets.

Unavailable values are represented by ``None`` and a reason code. This module
never imputes values and deliberately performs no file or plotting operations.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from datasets.targets import (
    TargetDefinition as DatasetTargetDefinition,
)
from datasets.targets import (
    TargetValidity,
    TargetValue,
)

NumericValue: TypeAlias = int | float | None

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True, slots=True)
class TargetDefinition:
    """Definition and fixed shape for one named target."""

    target_id: str
    units: str
    definition_sha256: str
    element_count: int


@dataclass(frozen=True, slots=True)
class TargetObservation:
    """Per-element values, availability mask, and unavailable reason codes."""

    target_id: str
    units: str
    definition_sha256: str
    values: tuple[NumericValue, ...]
    valid_mask: tuple[bool, ...]
    reason_codes: tuple[str | None, ...]


@dataclass(frozen=True, slots=True)
class TargetRecord:
    """A uniquely identified record assigned to an explicit split and family."""

    record_id: str
    split_id: str
    family_id: str
    targets: tuple[TargetObservation, ...]


@dataclass(frozen=True, slots=True)
class ExpectedInventoryCell:
    """One required split × family × target coverage cell."""

    split_id: str
    family_id: str
    target_id: str


@dataclass(frozen=True, slots=True)
class ReasonCount:
    reason_code: str
    element_count: int


@dataclass(frozen=True, slots=True)
class CoverageCellResult:
    split_id: str
    family_id: str
    target_id: str
    valid_element_count: int
    unavailable_element_count: int
    unavailable_by_reason: tuple[ReasonCount, ...]
    coverage_fraction: float


@dataclass(frozen=True, slots=True)
class TargetCoverageReport:
    """Stable, inventory-ordered coverage results."""

    cells: tuple[CoverageCellResult, ...]


def convert_target_map(
    target_map: Mapping[str, TargetValue],
) -> tuple[tuple[TargetDefinition, ...], tuple[TargetObservation, ...]]:
    """Convert validated UMCD target-map values to evaluator definitions/observations.

    Target-map keys become evaluator target IDs. Definitions are revalidated
    against their canonical digest, including reference/gauge and aggregation
    fields. Arrays are flattened in NumPy's default row-major order. Dataset
    zero-storage for unavailable elements is converted to ``None`` and the
    corresponding :class:`TargetValidity` value is retained as its reason code.
    """
    definitions: list[TargetDefinition] = []
    observations: list[TargetObservation] = []
    for target_id in sorted(target_map):
        _require_nonempty(target_id, "target map key")
        target_value = target_map[target_id]
        if not isinstance(target_value, TargetValue):
            raise TypeError(f"target map value for {target_id!r} must be a TargetValue")
        dataset_definition: DatasetTargetDefinition = target_value.definition
        dataset_definition.require_versioned()

        flat_values = target_value.values.reshape(-1)
        flat_mask = target_value.valid_mask.reshape(-1)
        if flat_values.size == 0:
            raise ValueError(f"target value for {target_id!r} must contain at least one element")
        definitions.append(
            TargetDefinition(
                target_id=target_id,
                units=dataset_definition.units,
                definition_sha256=dataset_definition.definition_sha256,
                element_count=int(flat_values.size),
            )
        )
        observations.append(
            TargetObservation(
                target_id=target_id,
                units=dataset_definition.units,
                definition_sha256=dataset_definition.definition_sha256,
                values=tuple(
                    float(value) if bool(is_valid) else None
                    for value, is_valid in zip(flat_values, flat_mask, strict=True)
                ),
                valid_mask=tuple(bool(is_valid) for is_valid in flat_mask),
                reason_codes=tuple(
                    None if reason is TargetValidity.VALID else reason.value
                    for reason in target_value.reason_codes
                ),
            )
        )
    if not target_map:
        raise ValueError("target_map must be nonempty")
    return tuple(definitions), tuple(observations)


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")


def _cell_key(cell: ExpectedInventoryCell) -> tuple[str, str, str]:
    return cell.split_id, cell.family_id, cell.target_id


def _validate_definition(definition: TargetDefinition) -> None:
    _require_nonempty(definition.target_id, "target_id")
    _require_nonempty(definition.units, f"units for target {definition.target_id!r}")
    if not isinstance(definition.definition_sha256, str) or not _SHA256_RE.fullmatch(
        definition.definition_sha256
    ):
        raise ValueError(
            f"definition_sha256 for {definition.target_id!r} must be a SHA-256 hex digest"
        )
    if (
        isinstance(definition.element_count, bool)
        or not isinstance(definition.element_count, int)
        or definition.element_count <= 0
    ):
        raise ValueError(f"element_count for {definition.target_id!r} must be a positive integer")


def audit_target_coverage(
    records: tuple[TargetRecord, ...],
    target_definitions: tuple[TargetDefinition, ...],
    expected_inventory: tuple[ExpectedInventoryCell, ...],
) -> TargetCoverageReport:
    """Validate and summarize coverage for a complete declared inventory.

    Every record must provide exactly the targets inventoried for its
    split/family. Each target must contain exactly its declared number of
    elements. Valid elements have finite numeric values and no reason code;
    unavailable elements have ``None`` values and a nonempty reason code.
    """
    if not records:
        raise ValueError("records must be nonempty")
    if not target_definitions:
        raise ValueError("target_definitions must be nonempty")
    if not expected_inventory:
        raise ValueError("expected_inventory must be nonempty")

    definitions: dict[str, TargetDefinition] = {}
    for definition in target_definitions:
        _validate_definition(definition)
        if definition.target_id in definitions:
            raise ValueError(f"duplicate target definition {definition.target_id!r}")
        definitions[definition.target_id] = definition

    inventory: dict[tuple[str, str, str], ExpectedInventoryCell] = {}
    targets_by_assignment: dict[tuple[str, str], set[str]] = {}
    for cell in expected_inventory:
        _require_nonempty(cell.split_id, "inventory split_id")
        _require_nonempty(cell.family_id, "inventory family_id")
        _require_nonempty(cell.target_id, "inventory target_id")
        if cell.target_id not in definitions:
            raise ValueError(f"inventory references unknown target {cell.target_id!r}")
        key = _cell_key(cell)
        if key in inventory:
            raise ValueError(f"duplicate expected inventory cell {key!r}")
        inventory[key] = cell
        targets_by_assignment.setdefault((cell.split_id, cell.family_id), set()).add(cell.target_id)

    aggregates: dict[tuple[str, str, str], tuple[int, int, Counter[str]]] = {
        key: (0, 0, Counter()) for key in inventory
    }
    seen_record_ids: set[str] = set()
    for record in records:
        _require_nonempty(record.record_id, "record_id")
        _require_nonempty(record.split_id, f"split_id for record {record.record_id!r}")
        _require_nonempty(record.family_id, f"family_id for record {record.record_id!r}")
        if record.record_id in seen_record_ids:
            raise ValueError(f"duplicate record_id {record.record_id!r}")
        seen_record_ids.add(record.record_id)

        assignment = (record.split_id, record.family_id)
        expected_targets = targets_by_assignment.get(assignment)
        if expected_targets is None:
            raise ValueError(f"record {record.record_id!r} has no declared inventory cells")
        observations: dict[str, TargetObservation] = {}
        for observation in record.targets:
            _require_nonempty(observation.target_id, f"target_id in record {record.record_id!r}")
            if observation.target_id not in definitions:
                raise ValueError(f"unknown target {observation.target_id!r}")
            if observation.target_id in observations:
                raise ValueError(
                    f"duplicate target {observation.target_id!r} in record {record.record_id!r}"
                )
            if observation.target_id not in expected_targets:
                raise ValueError(
                    f"target {observation.target_id!r} is not inventoried for {assignment!r}"
                )
            definition = definitions[observation.target_id]
            if observation.units != definition.units:
                raise ValueError(f"inconsistent units for target {observation.target_id!r}")
            if observation.definition_sha256 != definition.definition_sha256:
                raise ValueError(
                    f"inconsistent definition hash for target {observation.target_id!r}"
                )
            if not (
                len(observation.values)
                == len(observation.valid_mask)
                == len(observation.reason_codes)
                == definition.element_count
            ):
                raise ValueError(f"shape mismatch for target {observation.target_id!r}")

            observations[observation.target_id] = observation

        missing_targets = expected_targets - observations.keys()
        if missing_targets:
            raise ValueError(
                f"record {record.record_id!r} is missing inventoried targets "
                f"{sorted(missing_targets)!r}"
            )

        for target_id, observation in observations.items():
            key = (record.split_id, record.family_id, target_id)
            valid_count, unavailable_count, reasons = aggregates[key]
            for value, valid, reason in zip(
                observation.values,
                observation.valid_mask,
                observation.reason_codes,
                strict=True,
            ):
                if not isinstance(valid, bool):
                    raise ValueError(f"valid_mask for target {target_id!r} must contain booleans")
                if valid:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError(
                            f"valid element in target {target_id!r} must have a numeric value"
                        )
                    if not math.isfinite(float(value)):
                        raise ValueError(f"valid element in target {target_id!r} must be finite")
                    if reason is not None:
                        raise ValueError(
                            f"valid element in target {target_id!r} cannot have a reason code"
                        )
                    valid_count += 1
                else:
                    if value is not None:
                        raise ValueError(
                            f"unavailable element in target {target_id!r} must use None; "
                            "numeric imputation is forbidden"
                        )
                    if not isinstance(reason, str) or not reason.strip():
                        raise ValueError(
                            f"unavailable reason code for target {target_id!r} "
                            "must be a nonempty string"
                        )
                    unavailable_count += 1
                    reasons[reason] += 1
            aggregates[key] = valid_count, unavailable_count, reasons

    empty_cells = [key for key, counts in aggregates.items() if counts[0] + counts[1] == 0]
    if empty_cells:
        raise ValueError(f"expected inventory cells have no records: {sorted(empty_cells)!r}")

    results: list[CoverageCellResult] = []
    for key in sorted(inventory):
        split_id, family_id, target_id = key
        valid_count, unavailable_count, reasons = aggregates[key]
        total = valid_count + unavailable_count
        results.append(
            CoverageCellResult(
                split_id=split_id,
                family_id=family_id,
                target_id=target_id,
                valid_element_count=valid_count,
                unavailable_element_count=unavailable_count,
                unavailable_by_reason=tuple(
                    ReasonCount(reason_code=reason, element_count=reasons[reason])
                    for reason in sorted(reasons)
                ),
                coverage_fraction=valid_count / total,
            )
        )
    return TargetCoverageReport(cells=tuple(results))
