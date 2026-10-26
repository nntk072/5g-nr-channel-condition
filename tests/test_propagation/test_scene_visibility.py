from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from propagation import extract_scene_triangle_inventory, query_scene_segment_visibility


class _Buffer:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def numpy(self) -> np.ndarray:
        return self._value


class _Mesh:
    def __init__(
        self, vertices: np.ndarray, faces: np.ndarray, transform: np.ndarray | None = None
    ) -> None:
        self._vertices = vertices
        self._faces = faces
        self._transform = np.eye(4) if transform is None else transform

    def vertex_positions_buffer(self) -> _Buffer:
        return _Buffer(self._vertices.reshape(-1))

    def faces_buffer(self) -> _Buffer:
        return _Buffer(self._faces.reshape(-1))

    def to_world(self) -> SimpleNamespace:
        return SimpleNamespace(matrix=_Buffer(self._transform))


def _triangle_scene() -> SimpleNamespace:
    vertices = np.asarray(
        [[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    face = np.asarray([[0, 1, 2]], dtype=np.uint32)
    screen = SimpleNamespace(mi_mesh=_Mesh(vertices, face))
    return SimpleNamespace(objects={"screen": screen})


def test_loaded_scene_inventory_binds_mesh_identity_and_independent_query() -> None:
    scene = _triangle_scene()
    inventory = extract_scene_triangle_inventory(scene)
    repeated = extract_scene_triangle_inventory(scene)

    assert inventory.scene_geometry_sha256 == repeated.scene_geometry_sha256
    assert inventory.object_names == ("screen",)
    assert inventory.triangles_m.shape == (1, 3, 3)
    assert inventory.triangles_m.dtype == np.float64
    assert inventory.triangles_m.flags.writeable is False
    blocked = query_scene_segment_visibility(inventory, (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    visible = query_scene_segment_visibility(inventory, (-1.0, 2.0, 0.0), (1.0, 2.0, 0.0))

    assert blocked.geometry.outcome == "blocked"
    assert blocked.scene_geometry_sha256 == inventory.scene_geometry_sha256
    assert blocked.object_names == ("screen",)
    assert blocked.triangle_count == 1
    assert visible.geometry.outcome == "visible"


def test_scene_inventory_hash_is_order_independent_and_records_metric_scale() -> None:
    base = _triangle_scene().objects["screen"]
    far_vertices = np.asarray(
        [[10.0, -1.0, -1.0], [10.0, 1.0, -1.0], [10.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    far = SimpleNamespace(mi_mesh=_Mesh(far_vertices, np.asarray([[0, 1, 2]], dtype=np.uint32)))
    inventory_a = extract_scene_triangle_inventory(
        SimpleNamespace(objects={"z-far": far, "a-near": base}), meters_per_unit=0.5
    )
    inventory_b = extract_scene_triangle_inventory(
        SimpleNamespace(objects={"a-near": base, "z-far": far}), meters_per_unit=0.5
    )
    inventory_other_scale = extract_scene_triangle_inventory(
        SimpleNamespace(objects={"a-near": base, "z-far": far}), meters_per_unit=1.0
    )

    assert inventory_a.object_names == ("a-near", "z-far")
    assert inventory_a.triangle_object_names == ("a-near", "z-far")
    assert inventory_a.scene_geometry_sha256 == inventory_b.scene_geometry_sha256
    assert inventory_a.scene_geometry_sha256 != inventory_other_scale.scene_geometry_sha256
    assert inventory_a.triangles_m[1, :, 0].mean() == pytest.approx(5.0)


def test_scene_inventory_applies_mesh_to_world_transform_before_unit_conversion() -> None:
    transform = np.eye(4)
    transform[0, 3] = 2.0
    mesh = _Mesh(
        np.asarray([[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]),
        np.asarray([[0, 1, 2]], dtype=np.uint32),
        transform,
    )
    inventory = extract_scene_triangle_inventory(
        SimpleNamespace(objects={"screen": SimpleNamespace(mi_mesh=mesh)}),
        meters_per_unit=0.5,
    )

    assert inventory.triangles_m[:, :, 0].mean() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "scene",
    (
        SimpleNamespace(objects={}),
        SimpleNamespace(objects={"screen": SimpleNamespace(mi_mesh=object())}),
        SimpleNamespace(
            objects={
                "screen": SimpleNamespace(
                    mi_mesh=_Mesh(
                        np.zeros((3, 3), dtype=np.float64),
                        np.asarray([[0, 1, 3]], dtype=np.uint32),
                    )
                )
            }
        ),
    ),
)
def test_scene_inventory_rejects_empty_unsupported_and_invalid_meshes(scene: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        extract_scene_triangle_inventory(scene)


def test_scene_inventory_rejects_ambiguous_names_and_invalid_scale() -> None:
    scene = SimpleNamespace(objects={" screen ": _triangle_scene().objects["screen"]})
    with pytest.raises(ValueError, match="canonical"):
        extract_scene_triangle_inventory(scene)
    with pytest.raises(ValueError, match="meters_per_unit"):
        extract_scene_triangle_inventory(_triangle_scene(), meters_per_unit=True)
