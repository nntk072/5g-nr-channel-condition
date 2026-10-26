"""Versioned target definitions and element-wise availability masks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

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
    LEGACY_REASON_UNRECORDED = "legacy_reason_unrecorded"


_CANONICAL_TARGET_CONTRACTS: dict[str, tuple[str, str, str, bool]] = {
    "umcd.standardized.profile-los-proxy.v1": (
        "profile_los_proxy",
        "standardized_profile_los_category_proxy",
        "binary_class",
        True,
    ),
    "profile_los_proxy.v2": (
        "profile_los_proxy",
        "standardized_profile_los_category_proxy",
        "binary_class",
        True,
    ),
    "umcd.standardized.effective-cfr.v1": (
        "channel_cfr",
        "perfect_synchronized_effective_subcarrier_cfr",
        "complex channel gain (dimensionless)",
        False,
    ),
    "umcd.tdl-pusch.effective-cfr.v1": (
        "channel_cfr",
        "perfect synchronized effective-subcarrier CFR",
        "complex channel gain (dimensionless)",
        False,
    ),
    "umcd.rt-trajectory.absolute-cfr.v1": (
        "channel_cfr",
        "absolute-delay full-path complex CFR on effective PUSCH subcarriers",
        "complex channel gain (dimensionless)",
        False,
    ),
    "umcd.rt-trajectory.symbol-cfr.v1": (
        "channel_cfr_ofdm_symbols",
        (
            "absolute-delay full-path CFR with path-Doppler phase evolution "
            "sampled at PUSCH OFDM symbols"
        ),
        "complex channel gain (dimensionless)",
        False,
    ),
    "umcd.rt-trajectory.geometric-los.v1": (
        "geometric_los",
        "independent Mitsuba segment-intersection geometry oracle",
        "boolean",
        True,
    ),
    "umcd.f03_rt_static.direct_visibility.v1": (
        "geometric_los",
        "geometric direct-path visibility",
        "boolean",
        True,
    ),
    "umcd.f04_rt_dynamic.direct_visibility.v2": (
        "geometric_los",
        "geometric direct-path visibility",
        "boolean",
        True,
    ),
}

# These fields describe what a recognized target version actually means. The
# definition digest makes declarations immutable, but does not make an
# arbitrary declaration true; pin the producer and aggregation semantics at
# the storage boundary as well.
_CANONICAL_TARGET_PROVENANCE: dict[str, tuple[str, frozenset[str], str]] = {
    "profile_los_proxy.v2": (
        "Sionna standardized TDL/CDL profile model",
        frozenset(
            {"profile_los_component false maps to 0 and true maps to 1; not geometric LOS truth"}
        ),
        "single standardized profile category; false=0, true=1",
    ),
    "umcd.standardized.effective-cfr.v1": (
        "standardized Sionna TDL/CDL CFR adapter",
        frozenset(
            {
                "profile-relative excess-delay path truth sinc-discretized for this NR sample rate; "
                "the static per-subcarrier target does not encode time-domain ISI/ICI",
                "profile-relative excess-delay CFR resampled to this resource grid",
            }
        ),
        "retain each effective subcarrier coefficient in receiver FFT-bin order; "
        "no antenna, symbol, or power averaging",
    ),
    "umcd.tdl-pusch.effective-cfr.v1": (
        "sionna_phy_ofdmchannel_return_channel",
        frozenset(
            {
                "OFDMChannel return_channel with the configured TDL profile, carrier, "
                "sample rate, and effective FFT-bin mapping"
            }
        ),
        "retain each effective subcarrier coefficient; no antenna, symbol, or power averaging",
    ),
    "umcd.rt-trajectory.absolute-cfr.v1": (
        "sionna_rt_pathsolver",
        frozenset(
            {
                "Sionna RT full solver-valid path set evaluated with absolute propagation "
                "delays and configured carrier/subcarrier offsets"
            }
        ),
        "coherent sum of all retained solver-valid paths per effective subcarrier and frame; "
        "no path-count or power normalization",
    ),
    "umcd.rt-trajectory.symbol-cfr.v1": (
        "sionna_rt_paths_cfr_doppler",
        frozenset(
            {
                "absolute-delay RT path channel evaluated at the declared NR OFDM symbol "
                "start timestamps"
            }
        ),
        "coherent sum of retained solver-valid paths per effective subcarrier and OFDM symbol; "
        "no path or symbol averaging",
    ),
    "umcd.rt-trajectory.geometric-los.v1": (
        "mitsuba_segment_intersection_oracle",
        frozenset(
            {
                "independent finite-segment visibility query for the recorded TX/RX poses "
                "and scene geometry"
            }
        ),
        "one independent direct-path visibility value per frame",
    ),
    "umcd.f03_rt_static.direct_visibility.v1": (
        "independent_segment_visibility_oracle",
        frozenset({"finite scene-segment visibility cross-checked by depth-zero PathSolver"}),
        "one direct-path visibility label per link; no temporal or port pooling",
    ),
    "umcd.f04_rt_dynamic.direct_visibility.v2": (
        "mitsuba_segment_intersection_oracle",
        frozenset({"independent finite-segment Mitsuba geometry query for the RT scene and poses"}),
        "one visibility value per trajectory frame; no spatial or temporal pooling",
    ),
}


def validate_canonical_target_contract(
    values: object,
    definition: TargetDefinition,
    *,
    name: str | None = None,
    valid_mask: object | None = None,
) -> None:
    """Validate value-domain rules for recognized, versioned project targets.

    Unknown definition versions remain application-defined. Recognized versions
    are pinned to their target name, semantic, units, declared provenance and
    aggregation, and real/complex value domain; scalar categorical targets are
    constrained to one binary value.
    """
    contract = _CANONICAL_TARGET_CONTRACTS.get(definition.definition_version)
    if contract is None:
        return
    expected_name, expected_semantic, expected_units, binary_scalar = contract
    if name is not None and name != expected_name:
        raise ValueError(
            f"target definition {definition.definition_version!r} is only valid for "
            f"target {expected_name!r}"
        )
    target_name = expected_name
    if definition.semantic != expected_semantic or definition.units != expected_units:
        raise ValueError(
            f"target {target_name!r} semantic or units differ from definition "
            f"{definition.definition_version!r}"
        )
    provenance_contract = _CANONICAL_TARGET_PROVENANCE.get(definition.definition_version)
    if provenance_contract is not None:
        expected_source, allowed_references, expected_aggregation = provenance_contract
        if definition.source != expected_source:
            raise ValueError(
                f"canonical target {target_name!r} source differs from definition "
                f"{definition.definition_version!r}"
            )
        if definition.reference not in allowed_references:
            raise ValueError(
                f"canonical target {target_name!r} reference differs from definition "
                f"{definition.definition_version!r}"
            )
        if definition.aggregation_rule != expected_aggregation:
            raise ValueError(
                f"canonical target {target_name!r} aggregation differs from definition "
                f"{definition.definition_version!r}"
            )
    array_input: Any = values
    detach = getattr(array_input, "detach", None)
    if callable(detach):
        array_input = detach().cpu().numpy()
    array = np.asarray(array_input)
    if binary_scalar:
        if array.size != 1:
            raise ValueError(f"canonical target {target_name!r} must contain exactly one value")
        if array.dtype.kind not in "biuf":
            raise TypeError(
                f"canonical target {target_name!r} must use a real boolean or numeric dtype"
            )
        if valid_mask is None:
            valid = np.ones(array.shape, dtype=bool)
        else:
            mask_input: Any = valid_mask
            mask_detach = getattr(mask_input, "detach", None)
            if callable(mask_detach):
                mask_input = mask_detach().cpu().numpy()
            raw_mask = np.asarray(mask_input)
            if raw_mask.dtype.kind != "b":
                raise TypeError(f"canonical target mask for {target_name!r} must be boolean")
            try:
                valid = np.broadcast_to(raw_mask, array.shape)
            except ValueError as error:
                raise ValueError(
                    f"canonical target mask for {target_name!r} has the wrong shape"
                ) from error
        selected = array[valid]
        if np.any((selected != 0) & (selected != 1)):
            raise ValueError(f"canonical target {target_name!r} values must be binary (0 or 1)")
    elif array.dtype.kind != "c":
        raise TypeError(f"canonical complex target {target_name!r} must use a complex dtype")


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

    @classmethod
    def versioned(
        cls,
        *,
        semantic: str,
        units: str,
        source: str,
        reference: str,
        aggregation_rule: str,
        definition_version: str,
    ) -> TargetDefinition:
        """Create a canonical target definition with every field digest-bound."""
        digest = target_definition_digest(
            semantic=semantic,
            units=units,
            reference=reference,
            aggregation_rule=aggregation_rule,
            source=source,
            definition_version=definition_version,
        )
        return cls(
            semantic=semantic,
            units=units,
            source=source,
            reference=reference,
            aggregation_rule=aggregation_rule,
            definition_version=definition_version,
            definition_sha256=digest,
        )

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
        for name in (
            "semantic",
            "units",
            "source",
            "reference",
            "aggregation_rule",
            "definition_version",
        ):
            value = getattr(self, name)
            if value != value.strip():
                raise ValueError(
                    f"canonical target definition {name} must not have surrounding whitespace"
                )
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
        if (
            not isinstance(self.entity_id, str)
            or not self.entity_id
            or self.entity_id != self.entity_id.strip()
        ):
            raise ValueError("entity_id must be a non-empty canonical string")
        if not isinstance(self.definition, TargetDefinition):
            raise TypeError("definition must be a TargetDefinition")
        self.definition.require_versioned()
        if (
            not isinstance(self.confidence, str)
            or not self.confidence
            or self.confidence != self.confidence.strip()
        ):
            raise ValueError("confidence must be a non-empty canonical string")
        values = np.array(self.values, copy=True)
        if values.dtype.kind not in "iufc" or not np.isfinite(values).all():
            raise ValueError("target values must be finite real or complex numeric values")
        if values.size == 0:
            raise ValueError(
                "target values must contain at least one element; represent unavailable truth "
                "with a masked element and an explicit reason code"
            )
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
            if is_valid and not bool(np.isfinite(value)):
                raise ValueError("valid target values must be finite")
        values.setflags(write=False)
        valid.setflags(write=False)
        validate_canonical_target_contract(
            values,
            self.definition,
            valid_mask=valid,
        )
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
