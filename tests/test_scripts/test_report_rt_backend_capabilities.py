from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
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
