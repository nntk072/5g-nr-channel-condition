"""Mock the Sionna 2.x boundary; not a substitute for a native RT smoke test."""

import sys
import types

import numpy as np
import pytest
from integrations.sionna.adapter import (
    GridConfig,
    RayScene,
    _rt_spatial_split_subject_id,
    generate_raytraced,
)
from integrations.sionna.los_oracle import SegmentVisibilityResult, SegmentVisibilityState


def _generate_raytraced_pilot(*args, **kwargs):
    kwargs.setdefault("pilot", True)
    kwargs.setdefault("output_destination", "/tmp/loslab-static-mock-test.npz")
    return generate_raytraced(*args, **kwargs)


def test_rt_split_parent_id_is_stable_across_scene_names_and_transmitters() -> None:
    common_assets = {"asset_set_sha256": "a" * 64, "complete": True}
    first, first_complete = _rt_spatial_split_subject_id(
        scene_asset_set_sha256=common_assets["asset_set_sha256"],
        scene_asset_manifest_complete=common_assets["complete"],
        scene_name="scene-copy-a",
        rx_position_m=(30.0, -1.0, 1.5),
        block_size_m=10.0,
    )
    same_scene_other_run, second_complete = _rt_spatial_split_subject_id(
        scene_asset_set_sha256=common_assets["asset_set_sha256"],
        scene_asset_manifest_complete=common_assets["complete"],
        scene_name="renamed-scene-copy",
        rx_position_m=(30.5, -1.5, 8.0),
        block_size_m=10.0,
    )
    different_scene, _ = _rt_spatial_split_subject_id(
        scene_asset_set_sha256="b" * 64,
        scene_asset_manifest_complete=True,
        scene_name="scene-copy-a",
        rx_position_m=(30.0, -1.0, 1.5),
        block_size_m=10.0,
    )

    assert first_complete is second_complete is True
    assert first == same_scene_other_run
    assert first != different_scene
    with pytest.raises(ValueError, match="block_size_m must be finite and positive"):
        RayScene(block_size_m=float("inf"))
    with pytest.raises(ValueError, match="block_size_m must be finite and positive"):
        RayScene(block_size_m=True)


def test_rt_split_identity_marks_unclosed_scenes_as_provisional() -> None:
    first, complete = _rt_spatial_split_subject_id(
        scene_asset_set_sha256="a" * 64,
        scene_asset_manifest_complete=False,
        scene_name="unresolved-scene-a",
        rx_position_m=(0.0, 0.0, 1.5),
        block_size_m=10.0,
    )
    renamed, _ = _rt_spatial_split_subject_id(
        scene_asset_set_sha256="a" * 64,
        scene_asset_manifest_complete=False,
        scene_name="unresolved-scene-b",
        rx_position_m=(0.0, 0.0, 1.5),
        block_size_m=10.0,
    )

    assert complete is False
    assert first.startswith("rt:unclosed-unresolved-scene-a:")
    assert first != renamed
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _rt_spatial_split_subject_id(
            scene_asset_set_sha256="not-a-digest",
            scene_asset_manifest_complete=True,
            scene_name="scene",
            rx_position_m=(0.0, 0.0, 0.0),
            block_size_m=10.0,
        )


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
    force_geometry_error = False
    force_all_los = False
    force_visibility_disagreement = False
    force_candidate_cap = False

    class FakeScene:
        def __init__(self):
            self.radio_materials = {}
            self.objects = {}

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

    class FakeCandidateBuffer:
        def __init__(self, count):
            self.paths_counter = np.asarray([count])

    class FakeSolver:
        def __init__(self, **kw):
            self._candidate_generator = lambda **kwargs: FakeCandidateBuffer(
                kwargs["max_num_paths_per_src"] if force_candidate_cap else 3
            )

        def __call__(self, scene, **kw):
            nonlocal empty_path_class, full_solver_calls
            calls.append(kw)
            visible = scene.receiver.position[0] > 0
            if kw["max_depth"] == 0:
                if force_oracle_error:
                    raise RuntimeError("mock oracle failure")
                if force_visibility_disagreement:
                    visible = not visible
                if force_all_los:
                    return FakePaths(1)
                return FakePaths(1 if visible else 0)
            if force_solver_error:
                raise ValueError("mock full solver failure")
            self._candidate_generator(max_num_paths_per_src=kw["max_num_paths_per_src"])
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
    from integrations.sionna import adapter

    def fake_segment_visibility(scene, source_position_m, target_position_m):
        if force_geometry_error:
            raise RuntimeError("mock geometry failure")
        visible = force_all_los or target_position_m[0] > 0
        return SegmentVisibilityResult(
            SegmentVisibilityState.VISIBLE if visible else SegmentVisibilityState.BLOCKED,
            1.0,
            None,
            None,
            (),
        )

    monkeypatch.setattr(adapter, "segment_visibility", fake_segment_visibility)
    mi = types.ModuleType("mitsuba")
    mi.Float = float
    mi.TensorXb = lambda value: np.asarray(value, dtype=bool)
    mi.variants = lambda: ["llvm_ad_mono_polarized"]
    selected_variants = []
    mi.set_variant = selected_variants.append
    dr = types.ModuleType("drjit")
    dr.arange = lambda dtype, n: np.arange(n, dtype=float)
    dr.select = np.where
    dr.eval = lambda *args: None
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
    d = _generate_raytraced_pilot(
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
    assert all(
        row["split_subject_id"] == str(group)
        for row, group in zip(d.metadata, d.group, strict=True)
    )
    report = d.generation_report
    assert report["schema"] == "loslab.raytraced_generation_report.v3"
    assert report["label_semantics"] == "geometric_direct_path_oracle"
    assert report["label_oracle"] == {
        "primary": "independent_finite_scene_segment_visibility_query",
        "cross_check": "depth_zero_los_only_path_solver",
        "policy": "skip_ambiguous_or_disagreeing_candidates",
        "grazing_cosine_threshold": 1e-3,
        "boundary_probe_offset_m": 1e-3,
    }
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
        + report["attempt_counts"]["geometry_oracle_attempts"]
    )
    assert report["attempt_counts"]["geometry_oracle_attempts"] == (
        report["attempt_counts"]["geometry_oracle_calls"]
        + report["attempt_counts"]["geometry_oracle_errors"]
    )
    assert report["attempt_counts"]["depth_zero_oracle_calls"] == (
        report["attempt_counts"]["candidate_geometric_los"]
        + report["attempt_counts"]["candidate_geometric_nlos"]
        + report["attempt_counts"]["depth_zero_geometry_disagreements"]
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
    assert all(m["candidate_paths_before_validation"] == 3 for m in d.metadata)
    assert all(m["candidate_paths_per_source_before_validation"] == [3] for m in d.metadata)
    assert all(not m["candidate_path_cap_filled"] for m in d.metadata)
    assert all(m["geometry_oracle"]["state"] in {"visible", "blocked"} for m in d.metadata)
    assert all(len(m["geometry_oracle"]["boundary_probe_blocked"]) == 0 for m in d.metadata)
    assert all(
        m["valid_paths_direct_component"] == (1 if y else 0)
        for m, y in zip(d.metadata, d.y, strict=True)
    )
    assert set(d.source) == {"sionna_rt"}
    assert set(d.domain) == {"rt_mock_tx-a"}
    assert {row["transmitter_id"] for row in d.metadata} == {"tx-a"}
    other_tx = _generate_raytraced_pilot(
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
    assert set(d.group).intersection(other_tx.group)
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
        _generate_raytraced_pilot(
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

    force_geometry_error = True
    try:
        _generate_raytraced_pilot(
            RayScene(name="mock", per_class=1, max_attempts=2),
            grid=GridConfig(n_subcarriers=32),
            seed=19,
        )
    except RaytraceGenerationError as error:
        report = error.generation_report
        assert report["solver_failure"]["stage"] == "geometry_oracle"
        assert report["attempt_counts"]["geometry_oracle_attempts"] == 1
        assert report["attempt_counts"]["geometry_oracle_errors"] == 1
    else:
        raise AssertionError("geometry oracle failure should carry an RT generation report")
    force_geometry_error = False

    force_visibility_disagreement = True
    try:
        _generate_raytraced_pilot(
            RayScene(name="mock", per_class=1, max_attempts=2),
            grid=GridConfig(n_subcarriers=32),
            seed=18,
        )
    except RaytraceGenerationError as error:
        report = error.generation_report
        assert report["status"] == "insufficient_samples"
        assert report["attempt_counts"]["depth_zero_geometry_disagreements"] == 2
        assert report["attempt_counts"]["full_solver_attempts"] == 0
    else:
        raise AssertionError("a geometry/solver disagreement must not produce a sample")
    force_visibility_disagreement = False

    force_solver_error = True
    try:
        _generate_raytraced_pilot(
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
        _generate_raytraced_pilot(
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

    force_all_los = False
    force_candidate_cap = True
    try:
        _generate_raytraced_pilot(
            RayScene(
                name="mock",
                per_class=1,
                max_attempts=2,
                max_num_paths_per_src=3,
            ),
            grid=GridConfig(n_subcarriers=32),
            seed=23,
        )
    except RaytraceGenerationError as error:
        report = error.generation_report
        assert report["status"] == "insufficient_samples"
        assert report["attempt_counts"]["full_solver_candidate_cap_filled"] == 2
        assert report["attempt_counts"]["full_solver_candidate_cap_rejections"] == 2
        assert report["full_solver_outcomes"]["candidate_cap_filled_rejected"] == 2
        assert report["accepted_counts"]["total"] == 0
    else:
        raise AssertionError("cap-filled candidates must be rejected from generated datasets")
