from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from propagation import extract_scene_triangle_inventory, query_scene_segment_visibility
from propagation.scene_visibility import SceneTriangleInventory


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
    transform[:3, :3] = np.asarray(
        [[2.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]], dtype=np.float64
    )
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

    np.testing.assert_allclose(
        inventory.triangles_m,
        np.asarray([[[0.75, -0.5, -0.25], [1.25, 0.5, -0.25], [1.0, 0.0, 0.25]]]),
    )


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("object_names", (), "requires named mesh objects"),
        ("object_names", ("",), "requires named mesh objects"),
        ("object_names", ("screen", "screen"), "unique and sorted"),
        ("object_names", ("z", "a"), "unique and sorted"),
        ("triangles_m", np.zeros((3, 3)), r"shape \[n, 3, 3\]"),
        ("triangles_m", np.full((1, 3, 3), np.nan), "must be finite"),
        ("triangle_object_names", (), "each triangle must identify"),
        ("triangle_object_names", ("missing",), "each triangle must identify"),
        ("scene_geometry_sha256", "A" * 64, "lowercase SHA-256"),
        ("scene_geometry_sha256", "abc", "lowercase SHA-256"),
        ("meters_per_unit", True, "finite and positive"),
        ("meters_per_unit", np.inf, "finite and positive"),
        ("meters_per_unit", 0, "finite and positive"),
    ],
)
def test_scene_triangle_inventory_record_rejects_invalid_fields(field, value, message):
    options = {
        "object_names": ("screen",),
        "triangles_m": np.zeros((1, 3, 3)),
        "triangle_object_names": ("screen",),
        "scene_geometry_sha256": "a" * 64,
        "meters_per_unit": 1.0,
    }
    options[field] = value
    with pytest.raises(ValueError, match=message):
        SceneTriangleInventory(**options)


class _BareMesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

    def vertex_positions_buffer(self):
        return _Buffer(self.vertices.reshape(-1))

    def faces_buffer(self):
        return _Buffer(self.faces.reshape(-1))


@pytest.mark.parametrize(
    ("scene", "message"),
    [
        (SimpleNamespace(objects=None), "non-empty objects mapping"),
        (SimpleNamespace(objects={"screen": SimpleNamespace()}), "no Mitsuba triangle mesh"),
        (
            SimpleNamespace(objects={"screen": SimpleNamespace(mi_mesh=object())}),
            "does not expose supported mesh buffers",
        ),
        (
            SimpleNamespace(
                objects={
                    "screen": SimpleNamespace(
                        mi_mesh=_BareMesh(np.zeros((3, 3)), np.asarray([[0.0, 1.0, 2.0]]))
                    )
                }
            ),
            "faces must be integer triangles",
        ),
        (
            SimpleNamespace(
                objects={
                    "screen": SimpleNamespace(
                        mi_mesh=_BareMesh(np.zeros((3, 3)), np.asarray([[0, 1]]))
                    )
                }
            ),
            "faces must be integer triangles",
        ),
    ],
)
def test_scene_inventory_rejects_missing_objects_and_unsupported_mesh_buffers(scene, message):
    with pytest.raises((TypeError, ValueError), match=message):
        extract_scene_triangle_inventory(scene)


def test_scene_inventory_supports_pretransformed_buffers_and_callable_matrix():
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    bare = SimpleNamespace(mi_mesh=_BareMesh(vertices, faces))
    without_transform = extract_scene_triangle_inventory(SimpleNamespace(objects={"bare": bare}))
    np.testing.assert_array_equal(without_transform.triangles_m[0], vertices)

    transform = np.eye(4)
    transform[0, 3] = 5
    mesh = _Mesh(vertices, faces)
    mesh.to_world = lambda: SimpleNamespace(matrix=lambda: _Buffer(transform))
    transformed = extract_scene_triangle_inventory(
        SimpleNamespace(objects={"transformed": SimpleNamespace(mi_mesh=mesh)})
    )
    np.testing.assert_array_equal(transformed.triangles_m[0, :, 0], [5, 6, 5])


def test_scene_inventory_reports_unreadable_mesh_buffers():
    class BrokenBuffer:
        def numpy(self):
            raise RuntimeError("native buffer unavailable")

    mesh = SimpleNamespace(
        vertex_positions_buffer=lambda: BrokenBuffer(),
        faces_buffer=lambda: _Buffer(np.asarray([[0, 1, 2]], dtype=np.uint32)),
    )
    with pytest.raises(ValueError, match="mesh buffers are unreadable"):
        extract_scene_triangle_inventory(
            SimpleNamespace(objects={"screen": SimpleNamespace(mi_mesh=mesh)})
        )


@pytest.mark.parametrize(
    "transform",
    [
        None,
        np.zeros((3, 3)),
        np.full((4, 4), np.nan),
        np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]),
    ],
)
def test_scene_inventory_rejects_invalid_object_to_world_transforms(transform):
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    mesh = _Mesh(vertices, np.asarray([[0, 1, 2]], dtype=np.int32))
    mesh.to_world = lambda: SimpleNamespace(matrix=transform)
    scene = SimpleNamespace(objects={"screen": SimpleNamespace(mi_mesh=mesh)})
    expected = "unsupported object-to-world transform" if transform is None else "finite and affine"
    with pytest.raises(ValueError, match=expected):
        extract_scene_triangle_inventory(scene)


def test_scene_inventory_rejects_transform_and_unit_overflow_and_bad_query_input():
    vertices = np.asarray([[0, 0, 0], [1e308, 0, 0], [0, 1, 0]], dtype=np.float64)
    mesh = _Mesh(vertices, np.asarray([[0, 1, 2]], dtype=np.int32))
    transform = np.eye(4)
    transform[0, 0] = 2
    mesh.to_world = lambda: SimpleNamespace(matrix=transform)
    scene = SimpleNamespace(objects={"screen": SimpleNamespace(mi_mesh=mesh)})
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="overflows world coordinates"):
        extract_scene_triangle_inventory(scene)

    mesh.to_world = lambda: SimpleNamespace(matrix=np.eye(4))
    with (
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="overflows metric coordinates"),
    ):
        extract_scene_triangle_inventory(scene, meters_per_unit=2)

    with pytest.raises(TypeError, match="SceneTriangleInventory"):
        query_scene_segment_visibility(object(), (0, 0, 0), (1, 0, 0))
