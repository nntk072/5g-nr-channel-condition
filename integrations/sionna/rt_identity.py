"""Complete identities for reproducible Sionna RT generation runs."""

from __future__ import annotations

from typing import Any, Mapping

from artifacts.provenance import canonical_hash


def rt_run_identity(
    *,
    scene_asset_manifest: Mapping[str, Any],
    physical_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    seed: int,
    backend_selection: Mapping[str, Any],
    runtime_versions: Mapping[str, str],
    package_source_sha256: str,
) -> dict[str, Any]:
    """Build a content key from every input that can change an RT result.

    A missing scene asset prevents a key from being issued: a path or partial
    manifest cannot safely identify the geometry that Mitsuba actually loads.
    Diagnostic-only probe text, host paths, and Git commits are excluded; the
    selected variant, native library versions, all run parameters, and complete
    source hash are included.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not package_source_sha256:
        raise ValueError("package_source_sha256 must be nonempty")
    if not bool(scene_asset_manifest.get("complete")):
        return {
            "schema": "loslab.rt_run_identity.v1",
            "complete": False,
            "cache_key_sha256": None,
            "reason": "scene asset closure is incomplete",
        }
    asset_hash = scene_asset_manifest.get("asset_set_sha256")
    if not isinstance(asset_hash, str) or not asset_hash:
        raise ValueError("complete scene manifest must include asset_set_sha256")
    backend = {
        "backend": backend_selection.get("backend"),
        "variant": backend_selection.get("variant"),
        "runtime_versions": backend_selection.get("runtime_versions", {}),
    }
    if not backend["backend"] or not backend["variant"]:
        raise ValueError("backend selection must identify a verified backend and variant")
    inputs = {
        "scene_asset_set_sha256": asset_hash,
        "physical_config": dict(physical_config),
        "solver_config": dict(solver_config),
        "seed": seed,
        "backend": backend,
        "runtime_versions": dict(runtime_versions),
        "package_source_sha256": package_source_sha256,
    }
    return {
        "schema": "loslab.rt_run_identity.v1",
        "complete": True,
        "cache_key_sha256": canonical_hash(inputs),
        "inputs": inputs,
    }
