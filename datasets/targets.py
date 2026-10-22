"""Versioned target definitions and element-wise availability masks."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class TargetValidity(StrEnum):
    """Why a target element can or cannot be used in a loss or metric."""

    VALID = "valid"
    NOT_APPLICABLE = "not_applicable"
    NOT_OBSERVABLE = "not_observable"
    REFERENCE_MISSING = "reference_missing"
    LOW_CONFIDENCE_GT = "low_confidence_ground_truth"
    SOLVER_FAILURE = "solver_failure"
    CENSORED = "censored"
    OUTSIDE_CALIBRATED_DOMAIN = "outside_calibrated_domain"
    SENSOR_DROPOUT = "sensor_dropout"
    SATURATED = "saturated"
    TIMESTAMP_GAP = "timestamp_gap"
    LICENSE_RESTRICTED = "license_restricted"


@dataclass(frozen=True, slots=True)
class TargetDefinition:
    """Machine-readable semantic, unit, reference and source contract."""

    semantic: str
    units: str
    source: str
    reference: str = "unspecified_legacy"
    definition_sha256: str = "unversioned_legacy"

    def __post_init__(self) -> None:
        for name in ("semantic", "units", "source", "reference", "definition_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"target definition {name} must be a non-empty string")

    def require_versioned(self) -> None:
        """Reject legacy definitions when used in a canonical UMCD target."""
        digest = self.definition_sha256
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("canonical target definitions need a lowercase SHA-256 digest")
        if self.reference == "unspecified_legacy":
            raise ValueError("canonical target definitions need an explicit physical reference")


@dataclass(frozen=True, slots=True, eq=False)
class TargetValue:
    """One target field with a reason code for every valid or unavailable value."""

    entity_id: str
    definition: TargetDefinition
    values: np.ndarray
    valid_mask: np.ndarray
    reason_codes: tuple[TargetValidity, ...]
    confidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str) or not self.entity_id.strip():
            raise ValueError("entity_id must be a non-empty string")
        if not isinstance(self.definition, TargetDefinition):
            raise TypeError("definition must be a TargetDefinition")
        self.definition.require_versioned()
        if not isinstance(self.confidence, str) or not self.confidence.strip():
            raise ValueError("confidence must state the target provenance confidence")
        values = np.array(self.values, copy=True)
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("target values must be finite real numeric values")
        raw_valid = np.asarray(self.valid_mask)
        if raw_valid.dtype.kind != "b":
            raise TypeError("valid_mask must use boolean dtype")
        valid = np.array(raw_valid, dtype=bool, copy=True)
        if valid.shape != values.shape:
            raise ValueError("valid_mask must have exactly the target value shape")
        reasons = tuple(self.reason_codes)
        if len(reasons) != values.size or any(
            not isinstance(reason, TargetValidity) for reason in reasons
        ):
            raise ValueError("reason_codes must contain one TargetValidity per target element")
        flat_values = values.reshape(-1)
        flat_valid = valid.reshape(-1)
        for index, (is_valid, reason, value) in enumerate(zip(flat_valid, reasons, flat_values)):
            if is_valid != (reason is TargetValidity.VALID):
                raise ValueError("valid_mask and reason_codes disagree")
            if not is_valid and value != 0:
                raise ValueError(
                    f"masked target element {index} must use zero storage plus its reason code"
                )
            if is_valid and not math.isfinite(float(value)):
                raise ValueError("valid target values must be finite")
        values.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "reason_codes", reasons)

    @property
    def valid_count(self) -> int:
        """Number of supported target elements."""
        return int(np.count_nonzero(self.valid_mask))

    @property
    def unavailable_counts(self) -> dict[TargetValidity, int]:
        """Count unavailable elements by explicit reason, excluding valid entries."""
        counts: dict[TargetValidity, int] = {}
        for reason in self.reason_codes:
            if reason is not TargetValidity.VALID:
                counts[reason] = counts.get(reason, 0) + 1
        return counts


def definition_digest(canonical_definition: str) -> str:
    """Hash canonical UTF-8 target-definition text for stable provenance."""
    if not isinstance(canonical_definition, str) or not canonical_definition.strip():
        raise ValueError("canonical_definition must be a non-empty string")
    return hashlib.sha256(canonical_definition.encode("utf-8")).hexdigest()
