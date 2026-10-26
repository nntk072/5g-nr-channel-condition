"""Independent segment-versus-triangle visibility checks.

This small geometric oracle is deliberately separate from Sionna RT. It
answers whether a straight segment intersects supplied opaque triangle
surfaces; it does not infer visibility from solver paths or model material
transmission, diffraction, or reflection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Visibility = Literal["visible", "blocked", "ambiguous"]


@dataclass(frozen=True, slots=True)
class SegmentVisibility:
    """Result for one finite segment against an explicit triangle inventory."""

    outcome: Visibility
    intersected_triangle_indices: tuple[int, ...]
    ambiguous_triangle_indices: tuple[int, ...]
    segment_length_m: float
    tolerance_m: float


def segment_mesh_visibility(
    start: object,
    end: object,
    triangles: object,
    *,
    meters_per_unit: float = 1.0,
    transform_to_world: object | None = None,
    tolerance_m: float = 1e-8,
) -> SegmentVisibility:
    """Classify a segment using independent triangle intersection geometry.

    ``triangles`` is ``[n, 3, 3]`` in the same source coordinate units as the
    endpoints. ``transform_to_world``, when supplied, is a finite affine 4x4
    matrix applied to vertices before the common unit conversion. Triangle
    interiors crossing the open segment are blocked; edge/vertex contact,
    endpoint contact, and coplanar overlap are ambiguous. An empty triangle
    roster is visible. Every supplied face is treated as an opaque surface.

    This conservative primitive does not check mesh watertightness, object
    identity, wall thickness, or material transmission. Those require a
    separately reviewed mesh-inventory adapter and material policy.
    """
    a = _point(start, "start")
    b = _point(end, "end")
    # Unit scaling and transform application below mutate the working copy;
    # callers may intentionally pass immutable inventory arrays.
    mesh = np.array(triangles, dtype=np.float64, copy=True)
    if mesh.ndim != 3 or mesh.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape [n, 3, 3]")
    if not np.isfinite(mesh).all():
        raise ValueError("triangles must contain only finite coordinates")
    if (
        isinstance(meters_per_unit, bool)
        or not np.isfinite(meters_per_unit)
        or meters_per_unit <= 0
    ):
        raise ValueError("meters_per_unit must be finite and positive")
    if isinstance(tolerance_m, bool) or not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("tolerance_m must be finite and positive")

    if transform_to_world is not None:
        transform = np.asarray(transform_to_world, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("transform_to_world must be a finite 4x4 matrix")
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-12):
            raise ValueError("transform_to_world must be affine")
        if abs(float(np.linalg.det(transform[:3, :3]))) <= np.finfo(np.float64).eps:
            raise ValueError("transform_to_world must have an invertible linear part")
        homogeneous = np.concatenate(
            (mesh.reshape(-1, 3), np.ones((mesh.size // 3, 1), dtype=np.float64)), axis=1
        )
        mesh = (homogeneous @ transform.T)[:, :3].reshape(mesh.shape)

    scale = float(meters_per_unit)
    a *= scale
    b *= scale
    mesh *= scale
    direction = b - a
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length <= tolerance_m:
        raise ValueError("segment endpoints must be separated by more than tolerance_m")

    hit_indices: list[int] = []
    ambiguous_indices: list[int] = []
    for index, triangle in enumerate(mesh):
        classification = _intersect_open_segment_triangle(
            a, direction, triangle, tolerance_m=tolerance_m
        )
        if classification == "interior":
            hit_indices.append(index)
        elif classification == "ambiguous":
            ambiguous_indices.append(index)

    outcome: Visibility
    if hit_indices:
        outcome = "blocked"
    elif ambiguous_indices:
        outcome = "ambiguous"
    else:
        outcome = "visible"
    return SegmentVisibility(
        outcome=outcome,
        intersected_triangle_indices=tuple(hit_indices),
        ambiguous_triangle_indices=tuple(ambiguous_indices),
        segment_length_m=length,
        tolerance_m=float(tolerance_m),
    )


def _point(value: object, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must be a finite three-dimensional point")
    return point.copy()


def _intersect_open_segment_triangle(
    start: np.ndarray,
    direction: np.ndarray,
    triangle: np.ndarray,
    *,
    tolerance_m: float,
) -> Literal["none", "interior", "ambiguous"]:
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    normal = np.cross(edge1, edge2)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= tolerance_m * tolerance_m:
        return "ambiguous"

    direction_length = float(np.linalg.norm(direction))
    unit_normal = normal / normal_length
    start_distance = float(np.dot(start - triangle[0], unit_normal))
    end_distance = float(np.dot(start + direction - triangle[0], unit_normal))
    plane_tolerance = tolerance_m
    if abs(start_distance) <= plane_tolerance and abs(end_distance) <= plane_tolerance:
        if _coplanar_segment_overlaps_triangle(start, direction, triangle, tolerance_m):
            return "ambiguous"
        return "none"

    denominator = float(np.dot(normal, direction))
    if abs(denominator) <= tolerance_m * normal_length * direction_length:
        # Nearly tangent intersections are numerically fragile. Report an
        # ambiguity when the extended plane crossing falls on the face.
        if abs(denominator) > np.finfo(np.float64).tiny:
            near_t = float(np.dot(normal, triangle[0] - start) / denominator)
            if -tolerance_m / direction_length <= near_t <= 1.0 + tolerance_m / direction_length:
                near_point = start + near_t * direction
                if _point_in_triangle(near_point, triangle, tolerance_m):
                    return "ambiguous"
        return "none"
    t = float(np.dot(normal, triangle[0] - start) / denominator)
    endpoint_parameter_tolerance = tolerance_m / direction_length
    if t <= endpoint_parameter_tolerance or t >= 1.0 - endpoint_parameter_tolerance:
        if -endpoint_parameter_tolerance <= t <= 1.0 + endpoint_parameter_tolerance:
            point = start + t * direction
            return "ambiguous" if _point_in_triangle(point, triangle, tolerance_m) else "none"
        return "none"

    point = start + t * direction
    barycentric = _barycentric(point, triangle)
    if barycentric is None:
        return "ambiguous"
    # Convert the length tolerance into a conservative dimensionless margin.
    face_scale = max(float(np.sqrt(normal_length)), tolerance_m)
    margin = tolerance_m / face_scale
    if np.min(barycentric) < -margin:
        return "none"
    if np.min(barycentric) <= margin:
        return "ambiguous"
    return "interior"


def _barycentric(point: np.ndarray, triangle: np.ndarray) -> np.ndarray | None:
    v0 = triangle[1] - triangle[0]
    v1 = triangle[2] - triangle[0]
    v2 = point - triangle[0]
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denominator = d00 * d11 - d01 * d01
    if denominator <= np.finfo(np.float64).eps * max(d00 * d11, 1.0):
        return None
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    return np.asarray((1.0 - v - w, v, w), dtype=np.float64)


def _point_in_triangle(point: np.ndarray, triangle: np.ndarray, tolerance_m: float) -> bool:
    barycentric = _barycentric(point, triangle)
    if barycentric is None:
        return False
    cross_norm = np.linalg.norm(
        np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    )
    face_scale = max(float(np.sqrt(cross_norm)), tolerance_m)
    return bool(np.min(barycentric) >= -tolerance_m / face_scale)


def _coplanar_segment_overlaps_triangle(
    start: np.ndarray, direction: np.ndarray, triangle: np.ndarray, tolerance_m: float
) -> bool:
    if _point_in_triangle(start, triangle, tolerance_m) or _point_in_triangle(
        start + direction, triangle, tolerance_m
    ):
        return True
    # A coplanar segment can cross a triangle without either endpoint inside.
    direction_length = float(np.linalg.norm(direction))
    for index in range(3):
        edge_start = triangle[index]
        edge = triangle[(index + 1) % 3] - edge_start
        edge_length = float(np.linalg.norm(edge))
        cross = np.cross(direction, edge)
        cross_norm = float(np.linalg.norm(cross))
        parallel_tolerance = np.finfo(np.float64).eps * direction_length * edge_length
        if cross_norm <= parallel_tolerance:
            if edge_length <= tolerance_m:
                continue
            distance_to_line = (
                float(np.linalg.norm(np.cross(edge_start - start, direction))) / direction_length
            )
            edge_end = edge_start + edge
            end_distance_to_line = (
                float(np.linalg.norm(np.cross(edge_end - start, direction))) / direction_length
            )
            if max(distance_to_line, end_distance_to_line) <= tolerance_m:
                length_squared = direction_length * direction_length
                start_t = float(np.dot(edge_start - start, direction) / length_squared)
                end_t = float(np.dot(edge_end - start, direction) / length_squared)
                if (
                    max(min(start_t, end_t), 0.0)
                    <= min(max(start_t, end_t), 1.0) + tolerance_m / direction_length
                ):
                    return True
            continue
        delta = edge_start - start
        cross_squared = cross_norm * cross_norm
        t = float(np.dot(np.cross(delta, edge), cross) / cross_squared)
        u = float(np.dot(np.cross(delta, direction), cross) / cross_squared)
        if (
            -tolerance_m / direction_length <= t <= 1.0 + tolerance_m / direction_length
            and -tolerance_m / max(float(np.linalg.norm(edge)), tolerance_m)
            <= u
            <= 1.0 + tolerance_m / max(float(np.linalg.norm(edge)), tolerance_m)
        ):
            return True
    return False
