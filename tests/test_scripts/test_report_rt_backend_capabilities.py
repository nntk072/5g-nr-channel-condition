from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
import scripts.report_rt_backend_capabilities as capability_module
from integrations.sionna.backend import RTBackendSelection
from scripts.report_rt_backend_capabilities import build_rt_backend_capability_report


def _schema() -> dict[str, object]:
    path = Path(__file__).parents[2] / "datasets/schema/rt-backend-capability-report.v1.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


def test_rt_backend_capability_report_records_verified_auto_fallback():
    selection = RTBackendSelection(
        requested="auto",
        backend="llvm_cpu",
        variant="llvm_ad_mono_polarized",
        available_variants=("cuda_ad_mono_polarized", "llvm_ad_mono_polarized"),
        runtime_versions={"sionna-rt": "2.2.0", "mitsuba": "3.9.1", "drjit": "1.5.0"},
        probe_passed=True,
        probe_summary="one direct path traced with llvm_ad_mono_polarized",
        fallback_reason="cuda_ad_mono_polarized: OptiX runtime unavailable",
    )
    report = build_rt_backend_capability_report(
        selector=lambda _requested: selection,
        variants_provider=lambda: selection.available_variants,
        runtime_provider=lambda: {"python": "3.12.3"},
    )

    assert list(jsonschema.Draft202012Validator(_schema()).iter_errors(report)) == []
    assert report["selected_backend"] == "llvm_cpu"
    assert report["probe_passed"] is True
    assert "OptiX runtime unavailable" in report["fallback_reason"]
    assert report["runtime_versions"]["python"] == "3.12.3"
    assert report["runtime_versions"]["drjit"] == "1.5.0"


def test_rt_backend_capability_report_preserves_unavailable_probe_diagnostics():
    def fail(_requested: str) -> RTBackendSelection:
        raise RuntimeError("no usable Sionna RT backend: CUDA and LLVM probes failed")

    report = build_rt_backend_capability_report(
        selector=fail,
        variants_provider=lambda: ("cuda_ad_mono_polarized",),
        runtime_provider=lambda: {"python": "3.12.3"},
    )

    assert list(jsonschema.Draft202012Validator(_schema()).iter_errors(report)) == []
    assert report["status"] == "UNAVAILABLE"
    assert report["selected_backend"] is None
    assert report["probe_passed"] is False
    assert "probes failed" in report["failure_reason"]
    assert report["available_variants"] == ["cuda_ad_mono_polarized"]


@pytest.mark.parametrize("requested", ["", "  ", None])
def test_rt_backend_capability_report_rejects_invalid_request(requested):
    with pytest.raises(ValueError, match="nonempty string"):
        build_rt_backend_capability_report(requested)  # type: ignore[arg-type]


def test_rt_backend_capability_report_collects_discovery_errors() -> None:
    selection = RTBackendSelection(
        requested="llvm",
        backend="llvm_cpu",
        variant="llvm_ad_mono_polarized",
        available_variants=(),
        runtime_versions={},
        probe_passed=False,
        probe_summary="probe unavailable",
        fallback_reason=None,
    )

    def fail_variants():
        raise RuntimeError("Mitsuba registry unavailable")

    def fail_runtime():
        raise RuntimeError("runtime metadata unavailable")

    report = build_rt_backend_capability_report(
        "llvm",
        selector=lambda _requested: selection,
        variants_provider=fail_variants,
        runtime_provider=fail_runtime,
    )

    assert report["status"] == "AVAILABLE"
    assert report["available_variants"] == []
    assert report["runtime_versions"] == {}


def test_installed_variants_returns_empty_when_mitsuba_is_absent(monkeypatch) -> None:
    original_import = __import__

    def without_mitsuba(name, *args, **kwargs):
        if name == "mitsuba":
            raise ImportError("Mitsuba is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", without_mitsuba)

    assert capability_module._installed_variants() == ()


def test_installed_variants_reads_only_the_variant_names(monkeypatch) -> None:
    monkeypatch.setitem(
        __import__("sys").modules,
        "mitsuba",
        SimpleNamespace(variants=lambda: ["llvm_ad_mono_polarized", "cuda_ad_mono_polarized"]),
    )

    assert capability_module._installed_variants() == (
        "llvm_ad_mono_polarized",
        "cuda_ad_mono_polarized",
    )


def test_rt_backend_capability_report_combines_discovery_and_selection_errors() -> None:
    def fail_variants():
        raise RuntimeError("Mitsuba registry unavailable")

    def fail_runtime():
        raise RuntimeError("runtime metadata unavailable")

    def fail_selection(_requested: str):
        raise RuntimeError("backend probe unavailable")

    report = build_rt_backend_capability_report(
        "auto",
        selector=fail_selection,
        variants_provider=fail_variants,
        runtime_provider=fail_runtime,
    )

    assert report["status"] == "UNAVAILABLE"
    assert report["failure_reason"] == (
        "variant discovery failed: Mitsuba registry unavailable; "
        "runtime fingerprint failed: runtime metadata unavailable; backend probe unavailable"
    )


@pytest.mark.parametrize(("status", "expected"), [("AVAILABLE", 0), ("UNAVAILABLE", 2)])
def test_cli_emits_json_and_returns_status(monkeypatch, capsys, status, expected) -> None:
    report = {"status": status, "requested": "llvm"}
    monkeypatch.setattr(
        capability_module, "build_rt_backend_capability_report", lambda backend: report
    )

    assert capability_module.main(["--backend", "llvm"]) == expected
    assert json.loads(capsys.readouterr().out) == report
