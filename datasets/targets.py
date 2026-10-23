"""Versioned target definitions and element-wise availability masks."""

from __future__ import annotations

import hashlib
import json
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
    aggregation_rule: str = "unspecified_legacy"
    definition_version: str = "unversioned_legacy"

    def __post_init__(self) -> None:
        for name in (
            "semantic",
            "units",
            "source",
            "reference",
            "definition_sha256",
            "aggregation_rule",
            "definition_version",
        ):
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
        if self.aggregation_rule == "unspecified_legacy":
            raise ValueError("canonical target definitions need an explicit aggregation rule")
        if self.definition_version == "unversioned_legacy":
            raise ValueError("canonical target definitions need an explicit version")
        expected_digest = target_definition_digest(
            semantic=self.semantic,
            units=self.units,
            reference=self.reference,
            aggregation_rule=self.aggregation_rule,
            source=self.source,
            definition_version=self.definition_version,
        )
        if digest != expected_digest:
            raise ValueError("target definition SHA-256 does not match its canonical fields")


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


def target_definition_digest(
    *,
    semantic: str,
    units: str,
    reference: str,
    aggregation_rule: str,
    source: str,
    definition_version: str,
) -> str:
    """Hash every field in a versioned target-definition contract."""
    fields = {
        "aggregation_rule": aggregation_rule,
        "definition_version": definition_version,
        "reference": reference,
        "semantic": semantic,
        "source": source,
        "units": units,
    }
    if any(not isinstance(value, str) or not value.strip() for value in fields.values()):
        raise ValueError("canonical target-definition fields must be non-empty strings")
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def migrate_legacy_target_definition_v1(payload: dict[str, object]) -> TargetDefinition:
    """Upgrade a v1 definition while marking fields absent from that schema unknown."""
    semantic_value = payload.get("semantic")
    units_value = payload.get("units")
    source_value = payload.get("source")
    required_values = (semantic_value, units_value, source_value)
    if any(not isinstance(value, str) or not value.strip() for value in required_values):
        raise ValueError("legacy v1 target definitions require semantic, units, and source strings")
    semantic, units, source = str(semantic_value), str(units_value), str(source_value)
    reference_value = payload.get("reference", "")
    reference = (
        str(reference_value)
        if isinstance(reference_value, str) and reference_value.strip()
        else "not recorded by legacy target schema v1"
    )
    if reference == "unspecified_legacy":
        reference = "not recorded by legacy target schema v1"
    aggregation_rule = "not recorded by legacy target schema v1"
    definition_version = "migrated_from_legacy_target_v1"
    return TargetDefinition(
        semantic=semantic,
        units=units,
        source=source,
        reference=reference,
        aggregation_rule=aggregation_rule,
        definition_version=definition_version,
        definition_sha256=target_definition_digest(
            semantic=semantic,
            units=units,
            reference=reference,
            aggregation_rule=aggregation_rule,
            source=source,
            definition_version=definition_version,
        ),
    )
