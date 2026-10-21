"""Mock the Sionna 2.x boundary; not a substitute for a native RT smoke test."""

import sys
import types

import numpy as np
from integrations.sionna.adapter import GridConfig, RayScene, generate_raytraced


def test_full_rt_oracle_is_separate_and_los_contains_reflections(monkeypatch):
    scene_module = types.ModuleType("sionna.rt")
    sionna = types.ModuleType("sionna")
    sionna.__path__ = []
    sionna.rt = scene_module
    scene_module.scene = types.SimpleNamespace()
    calls = []

    class FakeScene:
        def add(self, object):
            if object.name == "research_rx":
                self.receiver = object

    class FakeElement:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class FakePaths:
        def __init__(self, num_paths):
            self.valid = np.ones(num_paths, dtype=bool)

        def cfr(self, **kwargs):
            # Sionna 2.2 out_type='numpy' returns complex ndarray.
            n = len(kwargs["frequencies"])
            return np.ones((1, 1, 1, 1, 1, n), dtype=np.complex64) * (1 + 0.5j)

    class FakeSolver:
        def __init__(self, **kw):
            pass

        def __call__(self, scene, **kw):
            calls.append(kw)
            visible = scene.receiver.position[0] > 0
            if kw["max_depth"] == 0:
                return FakePaths(1 if visible else 0)
            return FakePaths(3 if visible else 2)

    scene_module.load_scene = lambda name: FakeScene()
    scene_module.PlanarArray = FakeElement
    scene_module.Transmitter = FakeElement
    scene_module.Receiver = FakeElement
    scene_module.PathSolver = FakeSolver
    mi = types.ModuleType("mitsuba")
    mi.Float = float
    dr = types.ModuleType("drjit")
    dr.arange = lambda dtype, n: np.arange(n, dtype=float)
    monkeypatch.setitem(sys.modules, "sionna", sionna)
    monkeypatch.setitem(sys.modules, "sionna.rt", scene_module)
    monkeypatch.setitem(sys.modules, "mitsuba", mi)
    monkeypatch.setitem(sys.modules, "drjit", dr)
    d = generate_raytraced(
        RayScene(
            name="mock",
            tx_position=(0.0, 0.0, 10.0),
            rx_center=(0.0, 0.0, 1.0),
            radius_m=5.0,
            per_class=3,
            max_attempts=200,
            samples_per_src=100,
        ),
        grid=GridConfig(n_subcarriers=32),
        seed=13,
    )
    assert len(d.y) == 6 and sum(d.y) == 3
    assert all(m["los_plus_indirect_paths"] for m, y in zip(d.metadata, d.y) if y == 1)
    assert all(m["valid_paths_full"] >= 2 for m in d.metadata)
    assert set(d.source) == {"sionna_rt"}
    assert all(kw["specular_reflection"] is False for kw in calls if kw["max_depth"] == 0)
    assert all(
        kw["specular_reflection"] is True and kw["los"] for kw in calls if kw["max_depth"] > 0
    )
