"""Geographic proximity checks for already-frozen independent splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .independent_splits import IndependentSplit


@dataclass(frozen=True, slots=True)
class SpatialLeakagePair:
    """Two samples in different splits separated by less than the buffer."""

    sample_id_a: str
    sample_id_b: str
    split_a: str
    split_b: str
    distance_m: float


@dataclass(frozen=True, slots=True)
class SpatialLeakageAudit:
    """Proximity-audit result; an empty pair list means clean at this buffer."""

    buffer_m: float
    pairs: tuple[SpatialLeakagePair, ...]

    @property
    def is_clean(self) -> bool:
        return not self.pairs


def audit_spatial_buffer(
    records: Sequence[Mapping[str, Any]],
    split: IndependentSplit,
    *,
    buffer_m: float,
    coordinate_field: str = "coordinates_m",
    coordinate_frame_field: str = "coordinate_frame_id",
) -> SpatialLeakageAudit:
    """Report cross-split samples closer than ``buffer_m`` in projected space.

    Coordinates must be finite 2D/3D points in metres and share one explicit,
    canonical projected coordinate-frame identity. They are QA-only metadata
    and must not be model inputs.
    Every sample in the frozen split must be provided exactly once.
    """
    if isinstance(buffer_m, (bool, np.bool_)) or not np.isfinite(buffer_m) or buffer_m <= 0:
        raise ValueError("buffer_m must be finite and positive")
    if (
        not isinstance(coordinate_field, str)
        or not coordinate_field
        or coordinate_field != coordinate_field.strip()
        or not isinstance(coordinate_frame_field, str)
        or not coordinate_frame_field
        or coordinate_frame_field != coordinate_frame_field.strip()
        or coordinate_field == coordinate_frame_field
    ):
        raise ValueError("coordinate field names must be distinct nonempty canonical strings")
    if not records:
        raise ValueError("records must not be empty")

    sample_ids: list[str] = []
    coordinates: list[np.ndarray] = []
    labels: list[str] = []
    seen: set[str] = set()
    expected_dims: int | None = None
    coordinate_frame_id: str | None = None
    for record in records:
        sample_value = record.get("sample_id")
        if (
            not isinstance(sample_value, str)
            or not sample_value
            or sample_value != sample_value.strip()
        ):
            raise ValueError("sample IDs must be non-empty canonical strings")
        sample_id = sample_value
        if sample_id in seen:
            raise ValueError(f"sample IDs must be non-empty and unique: {sample_id!r}")
        seen.add(sample_id)
        if sample_id not in split.sample_to_split:
            raise ValueError(f"sample is absent from frozen split: {sample_id}")
        frame_value = record.get(coordinate_frame_field)
        if (
            not isinstance(frame_value, str)
            or not frame_value
            or frame_value != frame_value.strip()
        ):
            raise ValueError(f"{coordinate_frame_field} must be a non-empty canonical string")
        if coordinate_frame_id is None:
            coordinate_frame_id = frame_value
        elif frame_value != coordinate_frame_id:
            raise ValueError("all coordinates must use the same coordinate frame")
        point = np.asarray(record.get(coordinate_field), dtype=np.float64)
        if point.ndim != 1 or point.size not in {2, 3} or not np.isfinite(point).all():
            raise ValueError(f"{coordinate_field} must be a finite 2D/3D point in metres")
        if expected_dims is not None and point.size != expected_dims:
            raise ValueError("all coordinates must use the same dimension")
        expected_dims = int(point.size)
        sample_ids.append(sample_id)
        coordinates.append(point)
        labels.append(split.sample_to_split[sample_id])

    if seen != set(split.sample_to_split):
        raise ValueError("records do not cover exactly the samples in the frozen split")

    points = np.stack(coordinates)
    labels_array = np.asarray(labels)
    pairs: list[SpatialLeakagePair] = []
    split_names = tuple(
        name
        for name in ("train", "validation", "calibration", "test")
        if name in set(split.sample_to_split.values())
    )
    for index, first_split in enumerate(split_names):
        first_indices = np.flatnonzero(labels_array == first_split)
        if first_indices.size == 0:
            continue
        first_tree = cKDTree(points[first_indices])
        for second_split in split_names[index + 1 :]:
            second_indices = np.flatnonzero(labels_array == second_split)
            if second_indices.size == 0:
                continue
            second_tree = cKDTree(points[second_indices])
            for local_i, neighbors in enumerate(
                first_tree.query_ball_tree(second_tree, r=buffer_m)
            ):
                global_i = int(first_indices[local_i])
                for local_j in neighbors:
                    global_j = int(second_indices[local_j])
                    distance = float(np.linalg.norm(points[global_i] - points[global_j]))
                    pairs.append(
                        SpatialLeakagePair(
                            sample_id_a=sample_ids[global_i],
                            sample_id_b=sample_ids[global_j],
                            split_a=first_split,
                            split_b=second_split,
                            distance_m=distance,
                        )
                    )
    pairs.sort(key=lambda pair: (pair.split_a, pair.split_b, pair.sample_id_a, pair.sample_id_b))
    return SpatialLeakageAudit(buffer_m=float(buffer_m), pairs=tuple(pairs))
