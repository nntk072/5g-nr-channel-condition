from __future__ import annotations

from integrations.sionna.rt_identity import (
    rt_run_identity,
    rt_runtime_fingerprint,
    static_rt_solver_config,
)


def _inputs():
    return {
        "scene_asset_manifest": {"complete": True, "asset_set_sha256": "scene-v1"},
        "physical_config": {"carrier_hz": 3.5e9},
        "solver_config": {"max_depth": 2, "diffraction": False},
        "seed": 7,
        "backend_selection": {
            "backend": "llvm_cpu",
            "variant": "llvm_ad_mono_polarized",
            "runtime_versions": {"sionna-rt": "2.2.0", "mitsuba": "3.9.1", "drjit": "1.5.0"},
            "probe_summary": "diagnostic text is not identity",
        },
        "runtime_versions": {"python": "3.12.0", "numpy": "2.2.0"},
        "package_source_sha256": "source-v1",
    }


def test_rt_run_identity_is_stable_and_ignores_probe_diagnostics():
    inputs = _inputs()
    first = rt_run_identity(**inputs)
    inputs["backend_selection"]["probe_summary"] = "different message"
    second = rt_run_identity(**inputs)

    assert first["complete"] is True
    assert first["cache_key_sha256"] == second["cache_key_sha256"]


def test_rt_run_identity_changes_for_each_result_affecting_input():
    baseline = rt_run_identity(**_inputs())["cache_key_sha256"]
    mutations = (
        ("scene_asset_manifest", "asset_set_sha256", "scene-v2"),
        ("physical_config", "carrier_hz", 2.6e9),
        ("solver_config", "max_depth", 3),
        ("backend_selection", "variant", "cuda_ad_mono_polarized"),
        ("runtime_versions", "numpy", "2.3.0"),
    )
    for key, field, value in mutations:
        inputs = _inputs()
        inputs[key][field] = value
        assert rt_run_identity(**inputs)["cache_key_sha256"] != baseline
    inputs = _inputs()
    inputs["seed"] = 8
    assert rt_run_identity(**inputs)["cache_key_sha256"] != baseline
    inputs = _inputs()
    inputs["package_source_sha256"] = "source-v2"
    assert rt_run_identity(**inputs)["cache_key_sha256"] != baseline


def test_incomplete_scene_closure_never_receives_reusable_key():
    inputs = _inputs()
    inputs["scene_asset_manifest"] = {
        "complete": False,
        "asset_set_sha256": "partial-scene-v1",
    }

    identity = rt_run_identity(**inputs)

    assert identity["complete"] is False
    assert identity["cache_key_sha256"] is None


def test_rt_run_identity_rejects_unverified_backend_and_invalid_seed():
    inputs = _inputs()
    inputs["backend_selection"] = {"backend": "cuda_optix", "variant": ""}
    try:
        rt_run_identity(**inputs)
    except ValueError as error:
        assert "verified backend" in str(error)
    else:
        raise AssertionError("unverified backend must be rejected")

    inputs = _inputs()
    inputs["seed"] = True
    try:
        rt_run_identity(**inputs)
    except ValueError as error:
        assert "seed" in str(error)
    else:
        raise AssertionError("boolean seed must be rejected")


def test_rt_runtime_fingerprint_includes_host_gpu_and_backend_environment(monkeypatch):
    from types import SimpleNamespace

    import integrations.sionna.rt_identity as identity_module

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("MI_DEFAULT_VARIANT", "llvm_ad_mono_polarized")
    monkeypatch.setattr(
        identity_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="RTX test, 555.1, 8.9, 24576\n",
        ),
    )

    fingerprint = rt_runtime_fingerprint({"python": "3.12.0"})

    assert fingerprint["python"] == "3.12.0"
    assert fingerprint["platform_system"]
    assert fingerprint["platform_machine"]
    assert fingerprint["cuda_visible_devices"] == "2"
    assert fingerprint["mitsuba_default_variant"] == "llvm_ad_mono_polarized"
    assert fingerprint["nvidia_device_fingerprint"] == "RTX test, 555.1, 8.9, 24576"


def test_static_rt_solver_config_validates_and_records_all_switches():
    assert static_rt_solver_config(2) == {
        "deterministic": True,
        "max_depth": 2,
        "los": True,
        "specular_reflection": True,
        "diffuse_reflection": True,
        "refraction": True,
        "diffraction": False,
    }
    for invalid in (-1, True, 2.0):
        try:
            static_rt_solver_config(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid max_depth accepted: {invalid!r}")
