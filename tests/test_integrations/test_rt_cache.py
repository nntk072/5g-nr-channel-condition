from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import types
import zipfile
from typing import Any

import numpy as np
import pytest
from datasets.channel import ChannelDataset
from integrations.sionna.rt_cache import (
    RTCacheConflictError,
    load_rt_dataset,
    store_rt_dataset,
)
from integrations.sionna.rt_identity import rt_run_identity, static_rt_solver_config


def _mock_sionna_scene():
    material = types.SimpleNamespace(
        name="wall",
        relative_permittivity=4.2,
        conductivity=0.01,
        thickness=0.1,
        scattering_coefficient=0.2,
        xpd_coefficient=0.0,
        scattering_pattern=types.SimpleNamespace(),
        frequency_update_callback=None,
    )
    scene = types.SimpleNamespace(
        radio_materials={"wall": material},
        objects={"wall-1": types.SimpleNamespace(radio_material=material)},
        frequency=0.0,
        bandwidth=0.0,
    )
    return scene, material


def _identity(*, complete: bool = True) -> dict:
    return rt_run_identity(
        scene_asset_manifest={
            "complete": complete,
            "asset_set_sha256": "scene-v1",
        },
        physical_config={"carrier_hz": 3.5e9},
        solver_config={"max_depth": 2},
        seed=9,
        backend_selection={
            "backend": "llvm_cpu",
            "variant": "llvm_ad_mono_polarized",
            "runtime_versions": {"sionna-rt": "2.2.0", "mitsuba": "3.9.1", "drjit": "1.5.0"},
        },
        runtime_versions={"python": "3.12.0", "numpy": "2.5.0"},
        package_source_sha256="source-v1",
    )


def _dataset(identity: dict, *, value: float = 1.0) -> ChannelDataset:
    return ChannelDataset(
        h=np.full((2, 1, 8), value + 0j, dtype=np.complex64),
        y=np.asarray([0, 1], dtype=np.uint8),
        group=np.asarray(["g0", "g1"]),
        domain=np.asarray(["scene", "scene"]),
        source=np.asarray(["rt", "rt"]),
        metadata=({}, {}),
        generation_report={"schema": "fixture", "rt_run_identity": identity},
    )


def test_rt_dataset_cache_round_trips_and_reuses_verified_payload(tmp_path):
    identity = _identity()
    dataset = _dataset(identity)

    payload_path = store_rt_dataset(tmp_path, dataset, identity)
    assert store_rt_dataset(tmp_path, dataset, identity) == payload_path
    loaded = load_rt_dataset(tmp_path, identity)

    assert payload_path.is_file()
    assert loaded is not None
    np.testing.assert_array_equal(loaded.h, dataset.h)
    np.testing.assert_array_equal(loaded.y, dataset.y)
    assert loaded.generation_report == dataset.generation_report


def test_rt_dataset_cache_detects_payload_tampering(tmp_path):
    identity = _identity()
    payload_path = store_rt_dataset(tmp_path, _dataset(identity), identity)
    payload = bytearray(payload_path.read_bytes())
    payload[-10] ^= 1
    payload_path.write_bytes(payload)

    with pytest.raises(ValueError, match="payload hash mismatch"):
        load_rt_dataset(tmp_path, identity)


def test_rt_dataset_cache_detects_manifest_metadata_tampering(tmp_path):
    identity = _identity()
    payload_path = store_rt_dataset(tmp_path, _dataset(identity), identity)
    manifest_path = payload_path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sample_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="shape/sample count mismatch"):
        load_rt_dataset(tmp_path, identity)


def test_rt_dataset_cache_load_parses_the_verified_bounded_payload(tmp_path):
    identity = _identity()
    payload_path = store_rt_dataset(tmp_path, _dataset(identity), identity)

    with pytest.raises(ValueError, match="configured byte limit"):
        load_rt_dataset(tmp_path, identity, max_payload_bytes=payload_path.stat().st_size - 1)
    with pytest.raises(TypeError, match="must be an integer"):
        load_rt_dataset(tmp_path, identity, max_payload_bytes=True)
    with pytest.raises(ValueError, match="must be positive"):
        load_rt_dataset(tmp_path, identity, max_payload_bytes=0)


def test_rt_dataset_cache_bounds_npz_expansion_before_array_loading(tmp_path):
    identity = _identity()
    payload_path = store_rt_dataset(tmp_path, _dataset(identity), identity)
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("oversized.npy", bytes(8192))
    payload = archive_bytes.getvalue()
    assert len(payload) < 4096
    payload_path.write_bytes(payload)

    manifest_path = payload_path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="uncompressed byte limit"):
        load_rt_dataset(tmp_path, identity, max_payload_bytes=4096)


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_rt_dataset_cache_rejects_aliased_payload_files(tmp_path, alias_kind):
    identity = _identity()
    payload_path = store_rt_dataset(tmp_path, _dataset(identity), identity)
    outside = tmp_path / "outside.npz"
    outside.write_bytes(payload_path.read_bytes())
    payload_path.unlink()
    if alias_kind == "symlink":
        payload_path.symlink_to(outside)
    else:
        os.link(outside, payload_path)

    with pytest.raises(ValueError, match="single-link regular file"):
        load_rt_dataset(tmp_path, identity)


def test_rt_dataset_cache_rejects_duplicate_manifest_keys(tmp_path):
    identity = _identity()
    payload_path = store_rt_dataset(tmp_path, _dataset(identity), identity)
    manifest_path = payload_path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest["schema"]
    manifest_path.write_text(
        f'{{"schema":"loslab.rt_dataset_cache.v1","schema":"{schema}"' + "}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid RT cache manifest") as error:
        load_rt_dataset(tmp_path, identity)
    assert "duplicate cache manifest key" in str(error.value.__cause__)


def test_rt_dataset_cache_rejects_conflicting_outputs_for_one_run_key(tmp_path):
    identity = _identity()
    store_rt_dataset(tmp_path, _dataset(identity), identity)

    with pytest.raises(RTCacheConflictError, match="conflicts"):
        store_rt_dataset(tmp_path, _dataset(identity, value=2.0), identity)


def test_rt_dataset_cache_fails_closed_for_incomplete_assets_and_partial_entries(tmp_path):
    incomplete = _identity(complete=False)
    with pytest.raises(ValueError, match="complete Sionna RT run identities"):
        store_rt_dataset(tmp_path, _dataset(incomplete), incomplete)

    identity = _identity()
    payload_path = store_rt_dataset(tmp_path, _dataset(identity), identity)
    payload_path.unlink()
    with pytest.raises(ValueError, match="incomplete RT cache entry"):
        load_rt_dataset(tmp_path, identity)


def test_rt_dataset_cache_rejects_identity_or_manifest_mismatch(tmp_path):
    identity = _identity()
    store_rt_dataset(tmp_path, _dataset(identity), identity)
    other = _identity()
    other["inputs"]["seed"] = 10
    other["cache_key_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="inputs do not match"):
        load_rt_dataset(tmp_path, other)


def test_cache_aware_rt_generator_reuses_dataset_without_retracing(tmp_path, monkeypatch):
    from integrations.sionna import adapter, backend
    from integrations.sionna.adapter import RayScene
    from integrations.sionna.scene_assets import scene_asset_manifest

    fake_inventory = types.SimpleNamespace(
        scene_geometry_sha256="b" * 64,
        object_names=("wall-1",),
        triangles_m=np.zeros((4, 3, 3), dtype=np.float64),
    )
    monkeypatch.setattr(adapter, "extract_scene_triangle_inventory", lambda _scene: fake_inventory)

    scene_xml = tmp_path / "mock_scene.xml"
    scene_xml.write_text("<scene/>", encoding="utf-8")
    sionna_module: Any = types.ModuleType("sionna")
    sionna_module.__path__ = []
    rt_module: Any = types.ModuleType("sionna.rt")
    rt_module.scene = types.SimpleNamespace(mock_scene=scene_xml)
    scene, material = _mock_sionna_scene()
    rt_module.load_scene = lambda _reference: scene
    sionna_module.rt = rt_module
    monkeypatch.setitem(sys.modules, "sionna", sionna_module)
    monkeypatch.setitem(sys.modules, "sionna.rt", rt_module)

    from integrations.sionna.backend import RTBackendSelection

    selection = RTBackendSelection(
        requested="auto",
        backend="llvm_cpu",
        variant="llvm_ad_mono_polarized",
        available_variants=("llvm_ad_mono_polarized",),
        runtime_versions={"sionna-rt": "2.2.0", "mitsuba": "3.9.1", "drjit": "1.5.0"},
        probe_passed=True,
        probe_summary="mock native trace passed",
    )
    monkeypatch.setattr(backend, "select_rt_backend", lambda requested: selection)
    monkeypatch.setattr(backend, "activate_rt_backend", lambda chosen: None)
    monkeypatch.setattr(adapter, "versions", lambda: {"python": "3.12.0", "git_sha": "fixture"})
    monkeypatch.setattr(adapter, "package_source_sha256", lambda: "source-v1")
    monkeypatch.setattr(
        adapter,
        "rt_runtime_fingerprint",
        lambda base: {**base, "platform_machine": "test-machine"},
    )

    calls = []

    def generate(
        config,
        *,
        grid,
        seed,
        _backend_selection,
        _runtime_fingerprint,
        _material_manifest_override,
        **_kwargs,
    ):
        calls.append(config)
        identity = rt_run_identity(
            scene_asset_manifest=scene_asset_manifest(scene_xml),
            physical_config={
                "scene": adapter.asdict(config),
                "grid": adapter.asdict(grid),
                "material_state": _material_manifest_override,
                "scene_geometry": {
                    "scene_geometry_sha256": fake_inventory.scene_geometry_sha256,
                    "scene_object_count": len(fake_inventory.object_names),
                    "scene_triangle_count": len(fake_inventory.triangles_m),
                },
            },
            solver_config=static_rt_solver_config(config.max_depth),
            seed=seed,
            backend_selection=_backend_selection.as_dict(),
            runtime_versions=_runtime_fingerprint,
            package_source_sha256="source-v1",
        )
        return _dataset(identity)

    monkeypatch.setattr(adapter, "generate_raytraced", generate)
    config = RayScene(name="mock_scene", per_class=1, max_attempts=2, samples_per_src=100)
    first = adapter.generate_raytraced_cached(
        config, cache_directory=tmp_path / "cache", seed=12, pilot=True
    )
    second = adapter.generate_raytraced_cached(
        config, cache_directory=tmp_path / "cache", seed=12, pilot=True
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(first.h, second.h)
    assert first.generation_report == second.generation_report

    material.scattering_coefficient = 0.5
    third = adapter.generate_raytraced_cached(
        config, cache_directory=tmp_path / "cache", seed=12, pilot=True
    )

    assert len(calls) == 2
    assert (
        third.generation_report["rt_run_identity"]["cache_key_sha256"]
        != (first.generation_report["rt_run_identity"]["cache_key_sha256"])
    )


def test_cache_aware_rt_generator_bypasses_unresolved_scene_assets(tmp_path, monkeypatch):
    from integrations.sionna import adapter, backend
    from integrations.sionna.adapter import RayScene
    from integrations.sionna.backend import RTBackendSelection

    fake_inventory = types.SimpleNamespace(
        scene_geometry_sha256="c" * 64,
        object_names=("wall-1",),
        triangles_m=np.zeros((4, 3, 3), dtype=np.float64),
    )
    monkeypatch.setattr(adapter, "extract_scene_triangle_inventory", lambda _scene: fake_inventory)

    scene_xml = tmp_path / "mock_scene.xml"
    scene_xml.write_text(
        '<scene><string name="filename" value="https://assets.invalid/mesh.obj"/></scene>',
        encoding="utf-8",
    )
    sionna_module: Any = types.ModuleType("sionna")
    sionna_module.__path__ = []
    rt_module: Any = types.ModuleType("sionna.rt")
    rt_module.scene = types.SimpleNamespace(mock_scene=scene_xml)
    scene, _material = _mock_sionna_scene()
    rt_module.load_scene = lambda _reference: scene
    sionna_module.rt = rt_module
    monkeypatch.setitem(sys.modules, "sionna", sionna_module)
    monkeypatch.setitem(sys.modules, "sionna.rt", rt_module)

    selection = RTBackendSelection(
        requested="llvm",
        backend="llvm_cpu",
        variant="llvm_ad_mono_polarized",
        available_variants=("llvm_ad_mono_polarized",),
        runtime_versions={"sionna-rt": "2.2.0", "mitsuba": "3.9.1", "drjit": "1.5.0"},
        probe_passed=True,
        probe_summary="mock native trace passed",
    )
    monkeypatch.setattr(backend, "select_rt_backend", lambda requested: selection)
    monkeypatch.setattr(backend, "activate_rt_backend", lambda chosen: None)
    monkeypatch.setattr(adapter, "versions", lambda: {"python": "3.12.0", "git_sha": "fixture"})
    monkeypatch.setattr(adapter, "package_source_sha256", lambda: "source-v1")
    monkeypatch.setattr(
        adapter,
        "rt_runtime_fingerprint",
        lambda base: {**base, "platform_machine": "test-machine"},
    )
    generated = []

    def generate(config, *, grid, seed, _backend_selection, _runtime_fingerprint, **_kwargs):
        generated.append(config)
        identity = rt_run_identity(
            scene_asset_manifest={"complete": False, "asset_set_sha256": "partial"},
            physical_config={
                "scene": adapter.asdict(config),
                "grid": adapter.asdict(grid),
                "scene_geometry": {
                    "scene_geometry_sha256": fake_inventory.scene_geometry_sha256,
                    "scene_object_count": len(fake_inventory.object_names),
                    "scene_triangle_count": len(fake_inventory.triangles_m),
                },
            },
            solver_config=static_rt_solver_config(config.max_depth),
            seed=seed,
            backend_selection=_backend_selection.as_dict(),
            runtime_versions=_runtime_fingerprint,
            package_source_sha256="source-v1",
        )
        return _dataset(identity)

    monkeypatch.setattr(adapter, "generate_raytraced", generate)
    cache_directory = tmp_path / "cache"
    result = adapter.generate_raytraced_cached(
        RayScene(name="mock_scene", per_class=1, max_attempts=2, samples_per_src=100),
        cache_directory=cache_directory,
        pilot=True,
    )

    assert len(generated) == 1
    assert result.generation_report["rt_run_identity"]["complete"] is False
    assert not cache_directory.exists()
