"""Dependency-isolated contracts for the scene-geometry visibility adapter."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import integrations.sionna.los_oracle as los_oracle
import numpy as np
import pytest


class _NativeValue:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


class _FakeIntersection:
    def __init__(self, *, hit: bool, distance: float, normal):
        self._hit = hit
        self.t = _NativeValue(distance)
        self.n = _NativeValue(normal)

    def is_valid(self):
        return _NativeValue([self._hit])


class _FakeScene:
    def __init__(self, *, hit=False, probe_results=(False, False, False, False), normal=(0, 1, 0)):
        self.hit = hit
        self.probe_results = iter(probe_results)
        self.normal = normal
        self.intersect_calls = 0
        self.probe_calls = 0

    def ray_intersect(self, _ray, **_kwargs):
        self.intersect_calls += 1
        return _FakeIntersection(hit=self.hit, distance=1.0, normal=self.normal)

    def ray_test(self, _ray, **_kwargs):
        self.probe_calls += 1
        return _NativeValue(next(self.probe_results))


def _install_fake_native_modules(monkeypatch):
    mitsuba = ModuleType("mitsuba")
    mitsuba.Point3f = lambda value: np.asarray(value, dtype=np.float32)
    mitsuba.RayFlags = SimpleNamespace(All=1)
    sionna = ModuleType("sionna")
    sionna.__path__ = []
    rt = ModuleType("sionna.rt")
    rt.__path__ = []
    utils = ModuleType("sionna.rt.utils")
    utils.spawn_ray_to = lambda source, target: (source, target)
    monkeypatch.setitem(sys.modules, "mitsuba", mitsuba)
    monkeypatch.setitem(sys.modules, "sionna", sionna)
    monkeypatch.setitem(sys.modules, "sionna.rt", rt)
    monkeypatch.setitem(sys.modules, "sionna.rt.utils", utils)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"grazing_cosine_threshold": True}, "grazing_cosine_threshold"),
        ({"grazing_cosine_threshold": float("nan")}, "grazing_cosine_threshold"),
        ({"grazing_cosine_threshold": 1.1}, "grazing_cosine_threshold"),
        ({"boundary_probe_m": True}, "boundary_probe_m"),
        ({"boundary_probe_m": float("inf")}, "boundary_probe_m"),
        ({"boundary_probe_m": -1.0}, "boundary_probe_m"),
    ],
)
def test_segment_visibility_rejects_invalid_tolerances(kwargs, message):
    with pytest.raises(ValueError, match=message):
        los_oracle.segment_visibility(object(), (0, 0, 0), (0, 1, 0), **kwargs)


@pytest.mark.parametrize("position", [(0, 1), (0, float("inf"), 1)])
def test_segment_visibility_rejects_malformed_endpoint_positions(position):
    with pytest.raises(ValueError, match="three finite"):
        los_oracle.segment_visibility(object(), position, (0, 1, 0))


def test_segment_visibility_returns_coincident_result_without_a_scene_or_native_import():
    result = los_oracle.segment_visibility(object(), (1, 2, 3), (1, 2, 3))
    assert result.state is los_oracle.SegmentVisibilityState.COINCIDENT_ENDPOINTS
    assert result.segment_length_m == 0.0
    assert result.visible is None


def test_segment_visibility_requires_a_mitsuba_scene():
    with pytest.raises(TypeError, match="through .mi_scene"):
        los_oracle.segment_visibility(object(), (0, 0, 0), (0, 1, 0))


def test_segment_visibility_reports_missing_native_dependencies(monkeypatch):
    import builtins

    scene = SimpleNamespace(mi_scene=_FakeScene())
    original_import = builtins.__import__

    def reject_mitsuba(name, *args, **kwargs):
        if name == "mitsuba":
            raise ImportError("fixture optional dependency missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_mitsuba)
    with pytest.raises(RuntimeError, match="native Mitsuba and Sionna RT are required"):
        los_oracle.segment_visibility(scene, (0, 0, 0), (0, 1, 0))


def test_segment_visibility_classifies_clear_and_blocked_segments(monkeypatch):
    _install_fake_native_modules(monkeypatch)
    clear_scene = SimpleNamespace(mi_scene=_FakeScene())
    clear = los_oracle.segment_visibility(clear_scene, (0, 0, 0), (0, 1, 0))
    assert clear.state is los_oracle.SegmentVisibilityState.VISIBLE
    assert clear.visible is True
    assert clear.first_intersection_distance_m is None
    assert clear.incidence_cosine is None
    assert len(clear.boundary_probe_blocked) == 4
    assert clear_scene.mi_scene.probe_calls == 4

    blocked_scene = SimpleNamespace(
        mi_scene=_FakeScene(hit=True, probe_results=(True, True, True, True))
    )
    blocked = los_oracle.segment_visibility(blocked_scene, (0, 0, 0), (0, 1, 0))
    assert blocked.state is los_oracle.SegmentVisibilityState.BLOCKED
    assert blocked.visible is False
    assert blocked.first_intersection_distance_m == 1.0
    assert blocked.incidence_cosine == pytest.approx(1.0)


def test_segment_visibility_can_skip_boundary_probes_and_detect_disagreement(monkeypatch):
    _install_fake_native_modules(monkeypatch)
    scene = SimpleNamespace(mi_scene=_FakeScene(hit=False))
    result = los_oracle.segment_visibility(scene, (0, 0, 0), (0, 1, 0), boundary_probe_m=0.0)
    assert result.state is los_oracle.SegmentVisibilityState.VISIBLE
    assert result.boundary_probe_blocked == ()
    assert scene.mi_scene.probe_calls == 0

    ambiguous_scene = SimpleNamespace(
        mi_scene=_FakeScene(probe_results=(False, True, False, False))
    )
    ambiguous = los_oracle.segment_visibility(ambiguous_scene, (0, 0, 0), (0, 1, 0))
    assert ambiguous.state is los_oracle.SegmentVisibilityState.AMBIGUOUS
    assert ambiguous.visible is None


@pytest.mark.parametrize("normal", [(0, 0, 0), (float("nan"), 0, 0)])
def test_segment_visibility_marks_invalid_or_grazing_surface_normals_ambiguous(monkeypatch, normal):
    _install_fake_native_modules(monkeypatch)
    scene = SimpleNamespace(mi_scene=_FakeScene(hit=True, probe_results=(True,) * 4, normal=normal))
    result = los_oracle.segment_visibility(scene, (0, 0, 0), (0, 1, 0))
    assert result.state is los_oracle.SegmentVisibilityState.AMBIGUOUS
    assert result.incidence_cosine == 0.0


def test_independent_mesh_visibility_extracts_or_uses_supplied_inventory(monkeypatch):
    inventory = object()
    visibility = object()
    calls = []
    monkeypatch.setattr(
        los_oracle,
        "extract_scene_triangle_inventory",
        lambda scene: calls.append(("extract", scene)) or inventory,
    )
    monkeypatch.setattr(
        los_oracle,
        "query_scene_segment_visibility",
        lambda *args, **kwargs: calls.append((args, kwargs)) or visibility,
    )
    scene = object()

    assert (
        los_oracle.independent_mesh_segment_visibility(
            scene, (0, 0, 0), (0, 1, 0), tolerance_m=0.01
        )
        is visibility
    )
    assert calls[0] == ("extract", scene)
    assert calls[1][0][0] is inventory
    assert calls[1][1] == {"tolerance_m": 0.01}

    calls.clear()
    assert (
        los_oracle.independent_mesh_segment_visibility(
            scene, (0, 0, 0), (0, 1, 0), inventory=inventory
        )
        is visibility
    )
    assert len(calls) == 1
