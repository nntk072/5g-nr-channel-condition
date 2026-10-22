"""Mock the Sionna 2.x boundary; not a substitute for a native RT smoke test."""

import sys
import types

import numpy as np
from integrations.sionna.adapter import GridConfig, RayScene, generate_raytraced


def test_full_rt_oracle_labels_and_los_contains_reflections(monkeypatch):
    scene_module = types.ModuleType("sionna.rt")
    sionna = types.ModuleType("sionna")
    sionna.__path__ = []
    sionna.rt = scene_module
    scene_module.scene = types.SimpleNamespace()
    calls = []
    full_solver_calls = 0
    empty_path_class: str | None = None
    force_solver_error = False
    force_oracle_error = False
    force_all_los = False

    class FakeScene:
        def add(self, object):
            if object.name == "research_rx":
                self.receiver = object

    class FakeElement:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class FakePaths:
        def __init__(self, num_paths):
            self.valid = np.ones((1, 1, 1, 1, num_paths), dtype=bool)
            self._a_real = np.ones(self.valid.shape, dtype=np.float32)
            self._a_imag = np.full(self.valid.shape, 0.5, dtype=np.float32)
            self.interactions = np.zeros((1, *self.valid.shape), dtype=np.uint32)
            if num_paths > 1:
                self.interactions[..., 1:] = 1

        @property
        def a(self):
            return self._a_real, self._a_imag

        def cfr(self, **kwargs):
            # Sionna 2.2 out_type='numpy' returns complex ndarray.
            n = len(kwargs["frequencies"])
            gain = self._a_real.sum() + 1j * self._a_imag.sum()
            return np.full((1, 1, 1, 1, 1, n), gain, dtype=np.complex64)

    class FakeSolver:
        def __init__(self, **kw):
            pass

        def __call__(self, scene, **kw):
            nonlocal empty_path_class, full_solver_calls
            calls.append(kw)
            visible = scene.receiver.position[0] > 0
            if kw["max_depth"] == 0:
                if force_oracle_error:
                    raise RuntimeError("mock oracle failure")
                if force_all_los:
                    return FakePaths(1)
                return FakePaths(1 if visible else 0)
            if force_solver_error:
                raise ValueError("mock full solver failure")
            full_solver_calls += 1
            if full_solver_calls == 1:
                empty_path_class = (
                    "geometric_los" if (visible or force_all_los) else "geometric_nlos"
                )
                return FakePaths(0)
            return FakePaths(3 if visible else 2)

    scene_module.load_scene = lambda name: FakeScene()
    scene_module.PlanarArray = FakeElement
    scene_module.Transmitter = FakeElement
    scene_module.Receiver = FakeElement
    scene_module.PathSolver = FakeSolver
    mi = types.ModuleType("mitsuba")
    mi.Float = float
    mi.TensorXb = lambda value: np.asarray(value, dtype=bool)
    mi.variants = lambda: ["llvm_ad_mono_polarized"]
    selected_variants = []
    mi.set_variant = selected_variants.append
    dr = types.ModuleType("drjit")
    dr.arange = lambda dtype, n: np.arange(n, dtype=float)
    dr.select = np.where
    monkeypatch.setitem(sys.modules, "sionna", sionna)
    monkeypatch.setitem(sys.modules, "sionna.rt", scene_module)
    monkeypatch.setitem(sys.modules, "mitsuba", mi)
    monkeypatch.setitem(sys.modules, "drjit", dr)
    from integrations.sionna import backend

    monkeypatch.setattr(
        backend,
        "probe_native_rt_variant",
        lambda variant: (True, f"mock RT probe passed: {variant}"),
    )
    d = generate_raytraced(
        RayScene(
            name="mock",
            transmitter_id="tx-a",
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
    report = d.generation_report
    assert report["schema"] == "loslab.raytraced_generation_report.v2"
    assert report["label_semantics"] == "geometric_direct_path_oracle"
    assert report["channel_semantics"] == {
        "schema": "loslab.rt_absolute_channel.v1",
        "delay_reference": "absolute_sionna_path_delay_s",
        "normalize_delays": False,
        "normalize_channel_power": False,
        "frequency_reference": "baseband_subcarrier_offsets_hz",
        "phase_convention": "sionna_paths_cfr_baseband_equivalent",
        "component_partition": "same_paths_valid_zero_interaction_mask",
        "time_steps": 1,
    }
    assert d.label_semantics == report["label_semantics"]
    assert report["environment"]["python"]
    assert len(report["package_source_sha256"]) == 64
    assert report["accepted_counts"] == {
        "total": 6,
        "geometric_nlos": 3,
        "geometric_los": 3,
    }
    assert report["attempt_counts"]["attempts_started"] == (
        report["attempt_counts"]["obstacle_rejections"]
        + report["attempt_counts"]["depth_zero_oracle_calls"]
    )
    assert report["attempt_counts"]["depth_zero_oracle_calls"] == (
        report["attempt_counts"]["candidate_geometric_los"]
        + report["attempt_counts"]["candidate_geometric_nlos"]
    )
    assert report["attempt_counts"]["full_solver_calls"] >= len(d.y)
    assert report["attempt_counts"]["full_solver_empty_paths"] == 1
    assert (
        sum(report["full_solver_outcomes"].values())
        == report["attempt_counts"]["full_solver_calls"]
    )
    by_class = report["full_solver_outcomes_by_geometric_class"]
    assert (
        sum(outcome["calls"] for outcome in by_class.values())
        == (report["attempt_counts"]["full_solver_calls"])
    )
    assert sum(outcome["accepted"] for outcome in by_class.values()) == len(d.y)
    assert sum(outcome["empty_paths"] for outcome in by_class.values()) == 1
    assert empty_path_class is not None
    assert by_class[empty_path_class]["empty_paths"] == 1
    assert "Empty full-solver path outcomes are counted" in report["selection_note"]
    assert all(m["los_plus_indirect_paths"] for m, y in zip(d.metadata, d.y) if y == 1)
    assert all(m["valid_paths_full"] >= 2 for m in d.metadata)
    assert all(
        m["valid_paths_direct_component"] == (1 if y else 0)
        for m, y in zip(d.metadata, d.y, strict=True)
    )
    assert set(d.source) == {"sionna_rt"}
    assert set(d.domain) == {"rt_mock_tx-a"}
    assert {row["transmitter_id"] for row in d.metadata} == {"tx-a"}
    other_tx = generate_raytraced(
        RayScene(
            name="mock",
            transmitter_id="tx-b",
            tx_position=(1.0, 0.0, 10.0),
            rx_center=(0.0, 0.0, 1.0),
            radius_m=5.0,
            per_class=3,
            max_attempts=200,
            samples_per_src=100,
        ),
        grid=GridConfig(n_subcarriers=32),
        seed=13,
    )
    assert set(other_tx.domain) == {"rt_mock_tx-b"}
    assert not set(d.group).intersection(other_tx.group)
    assert selected_variants == ["llvm_ad_mono_polarized"] * 2
    assert all(m["mitsuba_variant"] == "llvm_ad_mono_polarized" for m in d.metadata)
    assert all(m["scene_config"]["name"] == "mock" for m in d.metadata)
    assert all(m["grid_config"]["n_subcarriers"] == 32 for m in d.metadata)
    assert all(m["solver_config"]["deterministic"] for m in d.metadata)
    assert all(kw["specular_reflection"] is False for kw in calls if kw["max_depth"] == 0)
    assert all(
        kw["specular_reflection"] is True and kw["los"] for kw in calls if kw["max_depth"] > 0
    )

    from integrations.sionna import RaytraceGenerationError

    force_oracle_error = True
    try:
        generate_raytraced(
            RayScene(name="mock", per_class=1, max_attempts=2),
            grid=GridConfig(n_subcarriers=32),
            seed=20,
        )
    except RaytraceGenerationError as error:
        report = error.generation_report
        assert report["status"] == "solver_error"
        assert report["solver_failure"]["stage"] == "depth_zero_oracle"
        assert report["attempt_counts"]["depth_zero_oracle_attempts"] == 1
        assert report["attempt_counts"]["depth_zero_oracle_errors"] == 1
        assert report["attempt_counts"]["depth_zero_oracle_calls"] == 0
    else:
        raise AssertionError("oracle failure should carry an RT generation report")
    force_oracle_error = False

    force_solver_error = True
    try:
        generate_raytraced(
            RayScene(name="mock", per_class=1, max_attempts=2),
            grid=GridConfig(n_subcarriers=32),
            seed=21,
        )
    except RaytraceGenerationError as error:
        assert error.generation_report["status"] == "solver_error"
        assert error.generation_report["solver_failure"] == {
            "stage": "full_solver",
            "attempt_index": 0,
            "exception_type": "ValueError",
            "message": "mock full solver failure",
        }
        assert error.generation_report["attempt_counts"]["full_solver_attempts"] == 1
        assert error.generation_report["attempt_counts"]["full_solver_errors"] == 1
        assert error.generation_report["attempt_counts"]["full_solver_calls"] == 0
        assert (
            sum(
                outcome["errors"]
                for outcome in error.generation_report[
                    "full_solver_outcomes_by_geometric_class"
                ].values()
            )
            == 1
        )
    else:
        raise AssertionError("solver failure should carry an RT generation report")

    force_solver_error = False
    force_all_los = True
    try:
        generate_raytraced(
            RayScene(name="mock", per_class=1, max_attempts=2),
            grid=GridConfig(n_subcarriers=32),
            seed=22,
        )
    except RaytraceGenerationError as error:
        report = error.generation_report
        assert report["status"] == "insufficient_samples"
        assert report["requested_per_class"] == 1
        assert report["accepted_counts"] == {
            "total": 1,
            "geometric_nlos": 0,
            "geometric_los": 1,
        }
        assert report["attempt_counts"]["class_quota_skips"] == 1
        assert report["attempt_counts"]["attempts_started"] == 2
    else:
        raise AssertionError("insufficient class quota should carry an RT generation report")
