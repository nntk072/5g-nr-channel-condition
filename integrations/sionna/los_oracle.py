"""Scene-geometry LOS labels independent of Sionna's path search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

import numpy as np
from propagation.scene_visibility import (
    SceneSegmentVisibility,
    SceneTriangleInventory,
    extract_scene_triangle_inventory,
    query_scene_segment_visibility,
)


class SegmentVisibilityState(StrEnum):
    """Outcome of a finite transmitter-to-receiver segment query."""

    VISIBLE = "visible"
    BLOCKED = "blocked"
    AMBIGUOUS = "ambiguous"
    COINCIDENT_ENDPOINTS = "coincident_endpoints"


@dataclass(frozen=True, slots=True)
class SegmentVisibilityResult:
    """Geometry-only result; it makes no statement about RF detectability."""

    state: SegmentVisibilityState
    segment_length_m: float
    first_intersection_distance_m: float | None
    incidence_cosine: float | None
    boundary_probe_blocked: tuple[bool, ...]

    @property
    def visible(self) -> bool | None:
        """Return visibility only when the geometric query is unambiguous."""
        if self.state is SegmentVisibilityState.VISIBLE:
            return True
        if self.state is SegmentVisibilityState.BLOCKED:
            return False
        return None


def _position(value: Sequence[float], *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite scene-coordinate meters")
    return result


def segment_visibility(
    scene: object,
    source_position_m: Sequence[float],
    target_position_m: Sequence[float],
    *,
    grazing_cosine_threshold: float = 1e-3,
    boundary_probe_m: float = 1e-3,
) -> SegmentVisibilityResult:
    """Trace a finite Mitsuba scene segment without invoking ``PathSolver``.

    Four parallel probes offset the segment by ``boundary_probe_m`` in two
    orthogonal directions. A disagreement among the center ray and probes is
    reported as ambiguous, as is a first surface hit whose incidence is within
    ``grazing_cosine_threshold`` of tangent. Endpoints use Sionna's pinned
    ``spawn_ray_to`` helper, which truncates the segment before the target.
    """
    if (
        isinstance(grazing_cosine_threshold, bool)
        or not np.isfinite(grazing_cosine_threshold)
        or not 0.0 <= grazing_cosine_threshold <= 1.0
    ):
        raise ValueError("grazing_cosine_threshold must be finite and in [0, 1]")
    if (
        isinstance(boundary_probe_m, bool)
        or not np.isfinite(boundary_probe_m)
        or boundary_probe_m < 0
    ):
        raise ValueError("boundary_probe_m must be finite and nonnegative")
    source = _position(source_position_m, name="source_position_m")
    target = _position(target_position_m, name="target_position_m")
    displacement = target - source
    distance = float(np.linalg.norm(displacement))
    if distance == 0.0:
        return SegmentVisibilityResult(
            SegmentVisibilityState.COINCIDENT_ENDPOINTS, 0.0, None, None, ()
        )

    mitsuba_scene = getattr(scene, "mi_scene", None)
    if mitsuba_scene is None:
        raise TypeError("scene must expose its Mitsuba scene through .mi_scene")
    try:
        import mitsuba as mi
        from sionna.rt.utils import spawn_ray_to
    except ImportError as error:
        raise RuntimeError(
            "native Mitsuba and Sionna RT are required for scene visibility"
        ) from error

    direction = displacement / distance
    # Mitsuba's scalar Point3f overload rejects Python/NumPy float64 scalars;
    # its vector overload accepts the documented xyz array form.
    ray = spawn_ray_to(mi.Point3f(source), mi.Point3f(target))
    intersection = mitsuba_scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=True)
    hit = bool(np.asarray(intersection.is_valid().numpy()).reshape(-1)[0])
    first_distance: float | None = None
    incidence_cosine: float | None = None
    if hit:
        first_distance = float(np.asarray(intersection.t.numpy()).reshape(-1)[0])
        normal = np.asarray(intersection.n.numpy(), dtype=np.float64).reshape(3)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm == 0.0 or not np.isfinite(normal_norm):
            incidence_cosine = 0.0
        else:
            incidence_cosine = abs(float(np.dot(normal / normal_norm, direction)))

    probe_blocked: tuple[bool, ...] = ()
    if boundary_probe_m > 0.0:
        reference = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(direction)))]
        tangent_u = np.cross(direction, reference)
        tangent_u /= np.linalg.norm(tangent_u)
        tangent_v = np.cross(direction, tangent_u)
        offsets = (
            tangent_u,
            -tangent_u,
            tangent_v,
            -tangent_v,
        )
        probe_blocked = tuple(
            bool(
                np.asarray(
                    mitsuba_scene.ray_test(
                        spawn_ray_to(
                            mi.Point3f(source + boundary_probe_m * offset),
                            mi.Point3f(target + boundary_probe_m * offset),
                        ),
                        coherent=True,
                    ).numpy()
                )
            )
            for offset in offsets
        )

    states = (hit, *probe_blocked)
    if any(state != states[0] for state in states[1:]):
        state = SegmentVisibilityState.AMBIGUOUS
    elif hit and incidence_cosine is not None and incidence_cosine <= grazing_cosine_threshold:
        state = SegmentVisibilityState.AMBIGUOUS
    else:
        state = SegmentVisibilityState.BLOCKED if hit else SegmentVisibilityState.VISIBLE
    return SegmentVisibilityResult(state, distance, first_distance, incidence_cosine, probe_blocked)


def independent_mesh_segment_visibility(
    scene: object,
    source_position_m: Sequence[float],
    target_position_m: Sequence[float],
    *,
    inventory: SceneTriangleInventory | None = None,
    tolerance_m: float = 1e-8,
) -> SceneSegmentVisibility:
    """Query loaded scene triangles with the independent NumPy geometry oracle.

    Unlike :func:`segment_visibility`, this route does not call Mitsuba's
    ``ray_intersect`` or ``ray_test``. It extracts the loaded mesh buffers and
    their object-to-world transforms, then applies the repository's explicit
    segment/triangle predicate. All triangle surfaces are treated as opaque;
    the result is an audit of geometric contact, not material-aware RF LOS.
    """
    if inventory is None:
        inventory = extract_scene_triangle_inventory(scene)
    return query_scene_segment_visibility(
        inventory, source_position_m, target_position_m, tolerance_m=tolerance_m
    )
