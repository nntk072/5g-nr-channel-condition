import builtins
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import integrations.sionna.backend as backend
import pytest


def test_native_rt_probe_validates_variant_and_handles_timeout_and_spawn_failures(monkeypatch):
    with pytest.raises(ValueError, match="nonempty"):
        backend.probe_native_rt_variant("")
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("probe", 2)),
    )
    assert backend.probe_native_rt_variant("llvm", timeout_s=2) == (
        False,
        "native RT probe exceeded 2 seconds",
    )
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("missing interpreter")),
    )
    passed, message = backend.probe_native_rt_variant("llvm")
    assert not passed and "could not start native RT probe" in message


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            SimpleNamespace(returncode=2, stderr="traceback\nfailed", stdout=""),
            "failed (2): failed",
        ),
        (SimpleNamespace(returncode=0, stderr="", stdout="not-json\n"), "malformed output"),
        (SimpleNamespace(returncode=0, stderr="", stdout=""), "malformed output"),
    ],
)
def test_native_rt_probe_handles_process_failures_and_bad_output(monkeypatch, result, message):
    monkeypatch.setattr(backend.subprocess, "run", lambda *args, **kwargs: result)
    passed, observed = backend.probe_native_rt_variant("llvm")
    assert not passed
    assert message in observed


def test_native_rt_probe_parses_last_json_line(monkeypatch):
    result = SimpleNamespace(
        returncode=0,
        stderr="",
        stdout='startup message\n{"variant":"llvm_ad_mono_polarized","valid_direct_paths":1}\n',
    )
    monkeypatch.setattr(backend.subprocess, "run", lambda *args, **kwargs: result)
    assert backend.probe_native_rt_variant("llvm_ad_mono_polarized") == (
        True,
        "one direct path traced with llvm_ad_mono_polarized",
    )


def test_backend_selection_resolves_exact_variants_and_requires_available_backend():
    selected = backend.select_rt_backend(
        "custom_variant",
        available_variants=("custom_variant",),
        probe=lambda variant: (True, variant),
    )
    assert selected.backend == "llvm_cpu"
    assert selected.variant == "custom_variant"
    with pytest.raises(RuntimeError, match=r"installed variants: \(none\)"):
        backend.select_rt_backend("auto", available_variants=(), probe=lambda _: (True, "unused"))


def test_backend_selection_reports_missing_mitsuba_without_explicit_variants(monkeypatch):
    original_import = builtins.__import__

    def import_without_mitsuba(name, *args, **kwargs):
        if name == "mitsuba":
            raise ImportError("hidden mitsuba")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_mitsuba)
    with pytest.raises(RuntimeError, match="Mitsuba is unavailable"):
        backend.select_rt_backend()


def test_backend_selection_discovers_variants_and_uses_default_probe(monkeypatch):
    mi = ModuleType("mitsuba")
    mi.variants = lambda: ("llvm_ad_mono_polarized",)
    monkeypatch.setitem(sys.modules, "mitsuba", mi)
    monkeypatch.setattr(
        backend,
        "probe_native_rt_variant",
        lambda variant: (True, f"software probe for {variant}"),
    )
    result = backend.select_rt_backend()
    assert result.available_variants == ("llvm_ad_mono_polarized",)
    assert result.probe_summary == "software probe for llvm_ad_mono_polarized"


def test_auto_backend_reports_all_probe_failures():
    with pytest.raises(RuntimeError, match="no usable Sionna RT backend"):
        backend.select_rt_backend(
            available_variants=("cuda_ad_mono_polarized", "llvm_ad_mono_polarized"),
            probe=lambda variant: (False, f"{variant} unavailable"),
        )


def test_runtime_version_inventory_records_installed_and_missing_distributions(monkeypatch):
    def version(name):
        if name == "mitsuba":
            return "3.6.0"
        raise backend.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(backend.importlib.metadata, "version", version)
    assert backend._runtime_versions() == {
        "sionna-rt": "not installed",
        "mitsuba": "3.6.0",
        "drjit": "not installed",
    }


def test_backend_activation_sets_only_a_still_installed_variant(monkeypatch):
    mi = ModuleType("mitsuba")
    mi.variants = lambda: ("llvm_ad_mono_polarized",)
    calls = []
    mi.set_variant = calls.append
    monkeypatch.setitem(sys.modules, "mitsuba", mi)
    selection = backend.RTBackendSelection(
        requested="llvm",
        backend="llvm_cpu",
        variant="llvm_ad_mono_polarized",
        available_variants=("llvm_ad_mono_polarized",),
        runtime_versions={},
        probe_passed=True,
        probe_summary="software fixture",
    )
    backend.activate_rt_backend(selection)
    assert calls == ["llvm_ad_mono_polarized"]
    absent = backend.RTBackendSelection(
        **{**selection.as_dict(), "variant": "cuda_ad_mono_polarized"}
    )
    with pytest.raises(RuntimeError, match="no longer installed"):
        backend.activate_rt_backend(absent)


def test_backend_activation_reports_missing_mitsuba(monkeypatch):
    original_import = builtins.__import__

    def import_without_mitsuba(name, *args, **kwargs):
        if name == "mitsuba":
            raise ImportError("hidden mitsuba")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_mitsuba)
    selection = backend.RTBackendSelection(
        requested="llvm",
        backend="llvm_cpu",
        variant="llvm_ad_mono_polarized",
        available_variants=("llvm_ad_mono_polarized",),
        runtime_versions={},
        probe_passed=True,
        probe_summary="software fixture",
    )
    with pytest.raises(RuntimeError, match="Mitsuba is unavailable"):
        backend.activate_rt_backend(selection)
