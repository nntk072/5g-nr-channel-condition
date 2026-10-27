"""Pilot-scale cross-split channel fingerprint and temporal duplicate audits."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from .independent_splits import IndependentSplit


@dataclass(frozen=True, slots=True)
class CFRDuplicatePair:
    """Cross-split CFRs whose normalized complex correlation is high."""

    sample_id_a: str
    sample_id_b: str
    split_a: str
    split_b: str
    similarity: float


@dataclass(frozen=True, slots=True)
class CFRDuplicateAudit:
    """Result of phase- and common-gain-invariant CFR fingerprint QA."""

    similarity_threshold: float
    pairs: tuple[CFRDuplicatePair, ...]

    @property
    def is_clean(self) -> bool:
        return not self.pairs


def audit_cfr_near_duplicates(
    sample_ids: Sequence[str],
    cfr: np.ndarray,
    split: IndependentSplit,
    *,
    similarity_threshold: float = 0.999,
    block_size: int = 256,
) -> CFRDuplicateAudit:
    """Find cross-split CFRs with high normalized complex correlation.

    Similarity is ``abs(vdot(x, y))/(norm(x)*norm(y))`` and is invariant to a
    per-sample common complex gain. The pairwise computation is O(N²F), so this
    is intended for pilot-shard QA; use a separately validated approximate
    index for large corpora. A flagged pair is a review candidate, not by
    itself proof of data leakage.
    """
    values = np.asarray(cfr)
    if (
        values.ndim < 2
        or any(size == 0 for size in values.shape)
        or not np.iscomplexobj(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("cfr must be a finite complex array with sample and feature axes")
    if len(sample_ids) != values.shape[0] or len(sample_ids) == 0:
        raise ValueError("sample_ids must identify every CFR row")
    if any(not isinstance(x, str) or not x or x != x.strip() for x in sample_ids):
        raise ValueError("sample_ids must be non-empty canonical strings")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")
    if set(sample_ids) != set(split.sample_to_split):
        raise ValueError("sample_ids must cover exactly the frozen split sample set")
    if not np.isfinite(similarity_threshold) or not 0 < similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in (0, 1]")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")

    matrix = values.reshape(values.shape[0], -1).astype(np.complex128, copy=False)
    norms = np.linalg.norm(matrix, axis=1)
    normalized = np.zeros_like(matrix)
    nonzero = norms > 0
    normalized[nonzero] = matrix[nonzero] / norms[nonzero, None]
    labels = [split.sample_to_split[sample_id] for sample_id in sample_ids]
    pairs: list[CFRDuplicatePair] = []
    count = len(sample_ids)
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        similarity = np.abs(normalized[start:stop] @ normalized.conj().T)
        for local_index, row in enumerate(similarity):
            index = start + local_index
            for other in range(index + 1, count):
                if labels[index] == labels[other]:
                    continue
                score = float(row[other])
                # Two all-zero channel fingerprints carry no distinguishing
                # observation and are reported as identical outage rows.
                if not nonzero[index] and not nonzero[other]:
                    score = 1.0
                if score >= similarity_threshold:
                    pairs.append(
                        CFRDuplicatePair(
                            sample_id_a=sample_ids[index],
                            sample_id_b=sample_ids[other],
                            split_a=labels[index],
                            split_b=labels[other],
                            similarity=score,
                        )
                    )
    pairs.sort(key=lambda pair: (pair.split_a, pair.split_b, pair.sample_id_a, pair.sample_id_b))
    return CFRDuplicateAudit(similarity_threshold, tuple(pairs))


@dataclass(frozen=True, slots=True)
class TemporalOverlap:
    """Overlapping windows from one trajectory assigned to different splits."""

    sample_id_a: str
    sample_id_b: str
    split_a: str
    split_b: str
    overlap_duration_s: float


def find_cross_split_temporal_overlaps(
    records: Sequence[Mapping[str, Any]],
    split: IndependentSplit,
) -> tuple[TemporalOverlap, ...]:
    """Find overlapping time windows of the same trajectory across splits."""
    observed: dict[str, tuple[str, str, float, float]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each temporal audit record must be a mapping")
        sample_id = record.get("sample_id")
        trajectory_id = record.get("trajectory_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id != sample_id.strip():
            raise ValueError("sample_id must be a non-empty canonical string")
        if (
            not isinstance(trajectory_id, str)
            or not trajectory_id
            or trajectory_id != trajectory_id.strip()
        ):
            raise ValueError("trajectory_id must be a non-empty canonical string")
        if not sample_id or sample_id in observed:
            raise ValueError(f"sample IDs must be non-empty and unique: {sample_id!r}")
        if sample_id not in split.sample_to_split:
            raise ValueError(f"sample is absent from frozen split: {sample_id}")
        raw_start = record.get("window_start_s")
        raw_end = record.get("window_end_s")
        if (
            isinstance(raw_start, (bool, np.bool_))
            or not isinstance(raw_start, Real)
            or isinstance(raw_end, (bool, np.bool_))
            or not isinstance(raw_end, Real)
        ):
            raise ValueError("window boundaries must be finite real seconds")
        start, end = float(raw_start), float(raw_end)
        if not np.isfinite([start, end]).all() or end < start:
            raise ValueError("each record needs a trajectory and finite ordered time window")
        observed[sample_id] = (trajectory_id, split.sample_to_split[sample_id], start, end)
    if set(observed) != set(split.sample_to_split):
        raise ValueError("records do not cover exactly the samples in the frozen split")

    overlaps: list[TemporalOverlap] = []
    items = sorted(observed.items())
    for index, (sample_a, (trajectory_a, split_a, start_a, end_a)) in enumerate(items):
        for sample_b, (trajectory_b, split_b, start_b, end_b) in items[index + 1 :]:
            if trajectory_a != trajectory_b or split_a == split_b:
                continue
            overlap = min(end_a, end_b) - max(start_a, start_b)
            if overlap > 0:
                overlaps.append(
                    TemporalOverlap(sample_a, sample_b, split_a, split_b, float(overlap))
                )
    return tuple(overlaps)
