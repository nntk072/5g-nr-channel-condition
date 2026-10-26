"""Report installed and probe-verified Sionna RT backend capabilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from typing import Any

from artifacts.provenance import versions
from integrations.sionna.backend import RTBackendSelection, select_rt_backend
from integrations.sionna.rt_identity import rt_runtime_fingerprint


def _installed_variants() -> tuple[str, ...]:
    try:
        import mitsuba as mi
    except ImportError:
        return ()
    return tuple(str(variant) for variant in mi.variants())


def build_rt_backend_capability_report(
    requested: str = "auto",
    *,
    selector: Callable[[str], RTBackendSelection] | None = None,
    variants_provider: Callable[[], Sequence[str]] | None = None,
    runtime_provider: Callable[[], dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready capability report, including failed native probes."""
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("requested backend must be a nonempty string")
    get_variants = variants_provider or _installed_variants
    get_runtime = runtime_provider or (lambda: rt_runtime_fingerprint(versions()))
    discovery_errors: list[str] = []
    try:
        available_variants = tuple(sorted({str(value) for value in get_variants()}))
    except Exception as error:
        available_variants = ()
        discovery_errors.append(f"variant discovery failed: {error}")
    try:
        runtime_versions = get_runtime()
    except Exception as error:
        runtime_versions = {}
        discovery_errors.append(f"runtime fingerprint failed: {error}")
    choose_backend = selector or select_rt_backend
    try:
        selection = choose_backend(requested)
    except Exception as error:
        failure_reason = "; ".join([*discovery_errors, str(error)])
        return {
            "schema": "umcd.rt_backend_capability_report.v1",
            "status": "UNAVAILABLE",
            "requested": requested,
            "available_variants": list(available_variants),
            "runtime_versions": runtime_versions,
            "selected_backend": None,
            "selected_variant": None,
            "probe_passed": False,
            "probe_summary": "no backend passed the native RT probe",
            "fallback_reason": None,
            "failure_reason": failure_reason,
        }
    return {
        "schema": "umcd.rt_backend_capability_report.v1",
        "status": "AVAILABLE",
        "requested": requested,
        "available_variants": list(selection.available_variants),
        "runtime_versions": {
            **runtime_versions,
            **selection.runtime_versions,
        },
        "selected_backend": selection.backend,
        "selected_variant": selection.variant,
        "probe_passed": selection.probe_passed,
        "probe_summary": selection.probe_summary,
        "fallback_reason": selection.fallback_reason,
        "failure_reason": None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="auto",
        help="auto, cuda, llvm, or an exact installed Mitsuba variant",
    )
    args = parser.parse_args(argv)
    report = build_rt_backend_capability_report(args.backend)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "AVAILABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
