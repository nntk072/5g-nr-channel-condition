"""Independent segment visibility over a loaded scene's explicit triangle mesh."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .visibility import SegmentVisibility, segment_mesh_visibility


@dataclass(frozen=True, slots=True, eq=False)
class SceneTriangleInventory:
    """Immutable, digest-bound triangles extracted from loaded scene objects.

    The extraction API is duck-typed so the geometry oracle stays independent
    of Mitsuba/Sionna imports. Callers provide a loaded scene whose ``objects``
    mapping contains objects with ``mi_mesh.vertex_positions_buffer()`` and
    ``mi_mesh.faces_buffer()``. Mitsuba applies XML ``to_world`` transforms to
    these loaded mesh buffers; if a mesh exposes an additional transform, it is
    applied here as well.
    """

    object_names: tuple[str, ...]
    triangles_m: np.ndarray
    triangle_object_names: tuple[str, ...]
    scene_geometry_sha256: str
    meters_per_unit: float

    def __post_init__(self) -> None:
        names = tuple(self.object_names)
        owners = tuple(self.triangle_object_names)
        triangles = np.array(self.triangles_m, dtype=np.float64, copy=True)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("scene inventory requires named mesh objects")
        if len(set(names)) != len(names) or tuple(sorted(names)) != names:
            raise ValueError("scene inventory object names must be unique and sorted")
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
            raise ValueError("scene inventory triangles must have non-empty shape [n, 3, 3]")
        if not np.isfinite(triangles).all():
            raise ValueError("scene inventory triangles must be finite")
        if len(owners) != len(triangles) or any(owner not in names for owner in owners):
            raise ValueError("each triangle must identify one inventoried scene object")
        if (
            not isinstance(self.scene_geometry_sha256, str)
            or len(self.scene_geometry_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.scene_geometry_sha256)
        ):
            raise ValueError("scene_geometry_sha256 must be lowercase SHA-256 hex")
        if (
            isinstance(self.meters_per_unit, bool)
            or not isinstance(self.meters_per_unit, (int, float))
            or not math.isfinite(float(self.meters_per_unit))
            or self.meters_per_unit <= 0
        ):
            raise ValueError("meters_per_unit must be finite and positive")
        triangles.setflags(write=False)
        object.__setattr__(self, "object_names", names)
        object.__setattr__(self, "triangles_m", triangles)
        object.__setattr__(self, "triangle_object_names", owners)
        object.__setattr__(self, "meters_per_unit", float(self.meters_per_unit))


@dataclass(frozen=True, slots=True)
class SceneSegmentVisibility:
    """Independent geometry result bound to its exact scene triangle inventory."""

    geometry: SegmentVisibility
    scene_geometry_sha256: str
    object_names: tuple[str, ...]
    triangle_count: int
    surface_model: str = "all_scene_triangles_are_opaque_surfaces"


def _world_vertices(mesh: object, vertices: np.ndarray, *, name: str) -> np.ndarray:
    """Apply the mesh's Mitsuba object-to-world transform to local vertices."""
    transform_method = getattr(mesh, "to_world", None)
    if not callable(transform_method):
        # Mitsuba's loaded OBJ/PLY shapes expose transformed vertex buffers and
        # do not expose a separate `to_world` method on mi.Mesh.
        return vertices.copy()
    transform = transform_method()
    matrix_value = getattr(transform, "matrix", None)
    if callable(matrix_value):
        matrix_value = matrix_value()
    if matrix_value is None:
        raise ValueError(f"scene object {name!r} has an unsupported object-to-world transform")
    if callable(getattr(matrix_value, "numpy", None)):
        matrix_value = matrix_value.numpy()
    matrix = np.asarray(matrix_value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-12)
    ):
        raise ValueError(f"scene object {name!r} transform must be finite and affine")
    world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    if not np.isfinite(world).all():
        raise ValueError(f"scene object {name!r} transform overflows world coordinates")
    return world


def extract_scene_triangle_inventory(
    scene: object, *, meters_per_unit: float = 1.0
) -> SceneTriangleInventory:
    """Extract a deterministic triangle roster from loaded scene mesh buffers.

    The hash binds object names, exact float64 vertex coordinates, exact int64
    triangle indices, units, and roster lengths. It is an identity digest, not
    a signature or proof that the loaded assets are licensed or complete.
    Empty/unsupported scene objects are rejected instead of silently omitted.
    """
    if (
        isinstance(meters_per_unit, bool)
        or not isinstance(meters_per_unit, (int, float))
        or not math.isfinite(float(meters_per_unit))
        or meters_per_unit <= 0
    ):
        raise ValueError("meters_per_unit must be finite and positive")
    objects = getattr(scene, "objects", None)
    if not isinstance(objects, Mapping) or not objects:
        raise TypeError("scene must expose a non-empty objects mapping")

    digest = hashlib.sha256()
    digest.update(b"loslab.loaded_scene_triangle_inventory.v1\0")
    digest.update(np.asarray([meters_per_unit], dtype="<f8").tobytes())
    triangles_by_object: list[np.ndarray] = []
    owner_names: list[str] = []
    object_names: list[str] = []
    for name, scene_object in sorted(objects.items(), key=lambda pair: pair[0]):
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("scene object keys must be canonical non-empty strings")
        mesh = getattr(scene_object, "mi_mesh", None)
        if mesh is None:
            raise ValueError(f"scene object {name!r} has no Mitsuba triangle mesh")
        vertex_buffer = getattr(mesh, "vertex_positions_buffer", None)
        face_buffer = getattr(mesh, "faces_buffer", None)
        if not callable(vertex_buffer) or not callable(face_buffer):
            raise ValueError(f"scene object {name!r} does not expose supported mesh buffers")
        try:
            local_vertices = np.asarray(vertex_buffer().numpy(), dtype=np.float64).reshape(-1, 3)
            faces_raw = np.asarray(face_buffer().numpy())
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"scene object {name!r} mesh buffers are unreadable") from error
        if faces_raw.dtype.kind not in "iu" or faces_raw.size % 3:
            raise ValueError(f"scene object {name!r} faces must be integer triangles")
        faces = np.asarray(faces_raw, dtype=np.int64).reshape(-1, 3)
        if (
            local_vertices.ndim != 2
            or local_vertices.shape[1] != 3
            or not len(local_vertices)
            or not np.isfinite(local_vertices).all()
            or not len(faces)
            or np.any(faces < 0)
            or np.any(faces >= len(local_vertices))
        ):
            raise ValueError(f"scene object {name!r} has invalid vertex/face geometry")
        vertices = _world_vertices(mesh, local_vertices, name=name)
        triangles = vertices[faces] * float(meters_per_unit)
        if not np.isfinite(triangles).all():
            raise ValueError(f"scene object {name!r} geometry overflows metric coordinates")

        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        digest.update(len(vertices).to_bytes(8, "little"))
        digest.update(np.asarray(vertices, dtype="<f8").tobytes(order="C"))
        digest.update(len(faces).to_bytes(8, "little"))
        digest.update(np.asarray(faces, dtype="<i8").tobytes(order="C"))
        object_names.append(name)
        triangles_by_object.append(triangles)
        owner_names.extend([name] * len(triangles))

    triangles_m = np.concatenate(triangles_by_object, axis=0)
    return SceneTriangleInventory(
        object_names=tuple(object_names),
        triangles_m=triangles_m,
        triangle_object_names=tuple(owner_names),
        scene_geometry_sha256=digest.hexdigest(),
        meters_per_unit=float(meters_per_unit),
    )


def query_scene_segment_visibility(
    inventory: SceneTriangleInventory,
    start_m: object,
    end_m: object,
    *,
    tolerance_m: float = 1e-8,
) -> SceneSegmentVisibility:
    """Run the independent triangle oracle and bind its result to inventory.

    Every scene triangle is treated as an opaque geometric surface. The result
    does not model radio-material transmission, diffraction, or RF blockage.
    """
    if not isinstance(inventory, SceneTriangleInventory):
        raise TypeError("inventory must be a SceneTriangleInventory")
    geometry = segment_mesh_visibility(
        start_m,
        end_m,
        inventory.triangles_m,
        meters_per_unit=1.0,
        tolerance_m=tolerance_m,
    )
    return SceneSegmentVisibility(
        geometry=geometry,
        scene_geometry_sha256=inventory.scene_geometry_sha256,
        object_names=inventory.object_names,
        triangle_count=len(inventory.triangles_m),
    )
