"""Backend selection must distinguish an installed variant from a working RT stack."""

from __future__ import annotations

import pytest
from integrations.sionna.backend import select_rt_backend


def test_auto_prefers_cuda_only_after_native_probe_and_records_llvm_fallback():
    attempted: list[str] = []

    def probe(variant: str) -> tuple[bool, str]:
        attempted.append(variant)
        if variant.startswith("cuda_"):
            return False, "OptiX runtime unavailable"
        return True, "one direct path traced"

    selected = select_rt_backend(
        available_variants=("cuda_ad_mono_polarized", "llvm_ad_mono_polarized"),
        probe=probe,
    )

    assert attempted == ["cuda_ad_mono_polarized", "llvm_ad_mono_polarized"]
    assert selected.backend == "llvm_cpu"
    assert selected.variant == "llvm_ad_mono_polarized"
    assert selected.probe_passed
    assert "OptiX runtime unavailable" in (selected.fallback_reason or "")
    assert set(selected.runtime_versions) == {"sionna-rt", "mitsuba", "drjit"}


def test_auto_selects_cuda_when_native_rt_probe_succeeds():
    selected = select_rt_backend(
        available_variants=("cuda_ad_mono_polarized", "llvm_ad_mono_polarized"),
        probe=lambda variant: (True, f"trace succeeded: {variant}"),
    )

    assert selected.backend == "cuda_optix"
    assert selected.variant == "cuda_ad_mono_polarized"
    assert selected.fallback_reason is None


def test_explicit_backend_does_not_silently_fallback():
    attempted: list[str] = []

    def probe(variant: str) -> tuple[bool, str]:
        attempted.append(variant)
        return False, "driver mismatch"

    with pytest.raises(RuntimeError, match="driver mismatch"):
        select_rt_backend(
            "cuda",
            available_variants=("cuda_ad_mono_polarized", "llvm_ad_mono_polarized"),
            probe=probe,
        )
    assert attempted == ["cuda_ad_mono_polarized"]


def test_requested_missing_variant_fails_with_installed_variants():
    with pytest.raises(RuntimeError, match="installed variants"):
        select_rt_backend(
            "cuda",
            available_variants=("llvm_ad_mono_polarized",),
            probe=lambda _variant: (True, "unused"),
        )
