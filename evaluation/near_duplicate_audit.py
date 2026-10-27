"""Local audit for possible cross-split channel-response duplicates.

Matches are review candidates only. Similar channel responses can arise for
legitimate physical reasons and are not, by themselves, proof of leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

CFRArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]


@dataclass(frozen=True, slots=True)
class ChannelRecord:
    """One sample's split, optional geometry fingerprint, and complex CFR."""

    sample_id: str
    split_id: str
    family_site_geometry_hash: str | None
    cfr: CFRArray
    comparison_context_id: str | None = None


@dataclass(frozen=True, slots=True)
class NearDuplicateCandidate:
    """A cross-split pair that merits review, with the signals that matched."""

    sample_id_a: str
    sample_id_b: str
    split_id_a: str
    split_id_b: str
    geometry_hash_match: bool
    normalized_complex_inner_product_magnitude: float | None
    comparison_context_id_a: str | None
    comparison_context_id_b: str | None


@dataclass(frozen=True, slots=True)
class SkippedComparison:
    """A cross-split CFR comparison that could not be normalized."""

    sample_id_a: str
    sample_id_b: str
    reason: str


@dataclass(frozen=True, slots=True)
class NearDuplicateAuditReport:
    """Sorted review candidates and explicitly skipped comparisons."""

    threshold: float
    candidates: tuple[NearDuplicateCandidate, ...]
    skipped_comparisons: tuple[SkippedComparison, ...]


def _validate_record(record: ChannelRecord) -> None:
    for value, name in ((record.sample_id, "sample_id"), (record.split_id, "split_id")):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{name} must be a canonical nonempty string")
    if record.family_site_geometry_hash is not None and (
        not isinstance(record.family_site_geometry_hash, str)
        or not record.family_site_geometry_hash
        or record.family_site_geometry_hash != record.family_site_geometry_hash.strip()
    ):
        raise ValueError("family_site_geometry_hash must be None or a canonical nonempty string")
    if record.comparison_context_id is not None and (
        not isinstance(record.comparison_context_id, str)
        or not record.comparison_context_id
        or record.comparison_context_id != record.comparison_context_id.strip()
    ):
        raise ValueError("comparison_context_id must be None or a canonical nonempty string")
    if not isinstance(record.cfr, np.ndarray):
        raise ValueError(f"CFR for sample {record.sample_id!r} must be a NumPy array")
    if record.cfr.size == 0:
        raise ValueError(f"CFR for sample {record.sample_id!r} must be nonempty")
    if not np.issubdtype(record.cfr.dtype, np.complexfloating):
        raise ValueError(f"CFR for sample {record.sample_id!r} must have a complex dtype")
    if not np.isfinite(record.cfr).all():
        raise ValueError(f"CFR for sample {record.sample_id!r} must contain only finite values")


def _normalized_inner_product_magnitude(first: CFRArray, second: CFRArray) -> float | None:
    """Return a scale-safe normalized inner product, or None for a zero norm."""
    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    # ``abs(complex)`` can overflow for finite components near float64's limit.
    # Scaling by the largest real or imaginary component preserves the cosine
    # while keeping the subsequent norm and inner product near unity.
    first_scale = float(np.max(np.maximum(np.abs(first_flat.real), np.abs(first_flat.imag))))
    second_scale = float(np.max(np.maximum(np.abs(second_flat.real), np.abs(second_flat.imag))))
    if first_scale == 0.0 or second_scale == 0.0:
        return None

    first_scaled = first_flat / first_scale
    second_scaled = second_flat / second_scale
    first_norm = float(np.linalg.norm(first_scaled))
    second_norm = float(np.linalg.norm(second_scaled))
    normalized = float(abs(np.vdot(first_scaled, second_scaled)) / (first_norm * second_norm))
    if not np.isfinite(normalized):
        raise ValueError("CFR similarity could not be computed as a finite value")
    # Roundoff can put a cosine a few ulps above its mathematical upper bound.
    return min(1.0, normalized)


def audit_near_duplicates(
    records: tuple[ChannelRecord, ...],
    *,
    normalized_inner_product_threshold: float,
) -> NearDuplicateAuditReport:
    """Find flagged pairs across different splits using O(n²) comparisons.

    A pair is flagged when both records have the same non-None geometry hash,
    or when the magnitude of their normalized complex inner product is strictly
    greater than the caller-supplied threshold. CFRs are compared only when both
    records carry the same nonempty comparison-context ID; absent or different
    contexts are retained in ``skipped_comparisons``. Geometry and CFR flags are
    review cues, not proof of dataset leakage. Zero-norm CFR pairs are also
    skipped; an exact geometry match can still flag a pair.
    """
    threshold = normalized_inner_product_threshold
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("normalized_inner_product_threshold must be a finite number in [0, 1]")
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("normalized_inner_product_threshold must be a finite number in [0, 1]")
    if not isinstance(records, tuple) or not records:
        raise ValueError("records must be a nonempty tuple")

    seen_ids: set[str] = set()
    expected_shape_by_context: dict[str, tuple[int, ...]] = {}
    for record in records:
        _validate_record(record)
        if record.sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id {record.sample_id!r}")
        seen_ids.add(record.sample_id)
        if record.comparison_context_id is not None:
            expected_shape = expected_shape_by_context.setdefault(
                record.comparison_context_id, record.cfr.shape
            )
            if record.cfr.shape != expected_shape:
                raise ValueError(
                    "inconsistent CFR shapes for comparison context: "
                    f"expected {expected_shape}, got {record.cfr.shape} "
                    f"for sample {record.sample_id!r}"
                )

    ordered = sorted(records, key=lambda item: item.sample_id)
    candidates: list[NearDuplicateCandidate] = []
    skipped: list[SkippedComparison] = []
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if first.split_id == second.split_id:
                continue

            geometry_match = (
                first.family_site_geometry_hash is not None
                and first.family_site_geometry_hash == second.family_site_geometry_hash
            )
            has_context = (
                first.comparison_context_id is not None
                and second.comparison_context_id is not None
            )
            compatible_context = (
                has_context
                and first.comparison_context_id == second.comparison_context_id
            )
            similarity = (
                _normalized_inner_product_magnitude(first.cfr, second.cfr)
                if compatible_context
                else None
            )
            if similarity is None:
                skipped.append(
                    SkippedComparison(
                        sample_id_a=first.sample_id,
                        sample_id_b=second.sample_id,
                        reason=(
                            "zero_norm_cfr"
                            if compatible_context
                            else "missing_channel_context"
                            if not has_context
                            else "incompatible_channel_context"
                        ),
                    )
                )
            if geometry_match or (similarity is not None and similarity > threshold):
                candidates.append(
                    NearDuplicateCandidate(
                        sample_id_a=first.sample_id,
                        sample_id_b=second.sample_id,
                        split_id_a=first.split_id,
                        split_id_b=second.split_id,
                        geometry_hash_match=geometry_match,
                        normalized_complex_inner_product_magnitude=similarity,
                        comparison_context_id_a=first.comparison_context_id,
                        comparison_context_id_b=second.comparison_context_id,
                    )
                )

    return NearDuplicateAuditReport(
        threshold=threshold,
        candidates=tuple(candidates),
        skipped_comparisons=tuple(skipped),
    )
