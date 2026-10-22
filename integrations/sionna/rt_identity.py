"""Complete identities for reproducible Sionna RT generation runs."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any, Mapping

from artifacts.provenance import canonical_hash
from datasets.provenance import UMCDRunProvenance


def rt_runtime_fingerprint(base_versions: Mapping[str, str]) -> dict[str, str]:
    """Extend package versions with OS/CPU and visible NVIDIA driver/device identity."""
    runtime = dict(base_versions)
    runtime.update(
        {
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "platform_machine": platform.machine(),
            "platform_processor": platform.processor() or "unavailable",
            "python_implementation": platform.python_implementation(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", "unset"),
            "mitsuba_default_variant": os.environ.get("MI_DEFAULT_VARIANT", "unset"),
            "drjit_llvm_path": os.environ.get("DRJIT_LIBLLVM_PATH", "unset"),
        }
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,driver_version,compute_cap,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        runtime["nvidia_device_fingerprint"] = "unavailable"
    else:
        runtime["nvidia_device_fingerprint"] = (
            result.stdout.strip().replace("\n", ";")
            if result.returncode == 0 and result.stdout.strip()
            else "unavailable"
        )
    return runtime


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


def umcd_provenance_from_rt_identity(
    identity: Mapping[str, Any],
    *,
    record_id: str,
    subject_layer: str,
    source_family: str,
    producer: str,
    material_config: Mapping[str, Any],
) -> UMCDRunProvenance:
    """Convert a complete RT cache identity into a layer provenance record.

    Material state is supplied explicitly because the generic RT identity only
    includes materials when the caller placed them in its physical config.
    This adapter refuses incomplete or internally inconsistent identities.
    """
    if identity.get("schema") != "loslab.rt_run_identity.v1" or not identity.get("complete"):
        raise ValueError("only complete RT run identities can become UMCD provenance")
    inputs = identity.get("inputs")
    cache_key = identity.get("cache_key_sha256")
    if not isinstance(inputs, Mapping) or not isinstance(cache_key, str):
        raise ValueError("complete RT identity must contain inputs and a cache key")
    if canonical_hash(dict(inputs)) != cache_key:
        raise ValueError("RT identity inputs do not match their cache key")
    asset_hash = inputs.get("scene_asset_set_sha256")
    backend = inputs.get("backend")
    versions = inputs.get("runtime_versions")
    physical_config = inputs.get("physical_config")
    solver_config = inputs.get("solver_config")
    package_sha = inputs.get("package_source_sha256")
    seed = inputs.get("seed")
    if (
        not isinstance(asset_hash, str)
        or not isinstance(backend, Mapping)
        or not isinstance(versions, Mapping)
        or not isinstance(physical_config, Mapping)
        or not isinstance(solver_config, Mapping)
        or not isinstance(package_sha, str)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("RT identity is missing required reproducibility inputs")
    if any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in versions.items()
    ):
        raise ValueError("RT runtime versions must be string-valued")
    return UMCDRunProvenance(
        record_id=record_id,
        subject_layer=subject_layer,
        source_family=source_family,
        producer=producer,
        package_source_sha256=package_sha,
        software_versions=versions,
        backend=backend,
        scene_asset_sha256={"asset_set": asset_hash},
        seed=seed,
        physical_config=physical_config,
        solver_config=solver_config,
        material_config=material_config,
        run_identity_sha256=cache_key,
    )


def static_rt_solver_config(
    max_depth: int, max_num_paths_per_src: int = 1_000_000
) -> dict[str, object]:
    """Return the complete solver switches used by static RT dataset generation."""
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    if (
        isinstance(max_num_paths_per_src, bool)
        or not isinstance(max_num_paths_per_src, int)
        or max_num_paths_per_src < 1
    ):
        raise ValueError("max_num_paths_per_src must be a positive integer")
    return {
        "deterministic": True,
        "max_depth": max_depth,
        "max_num_paths_per_src": max_num_paths_per_src,
        "los": True,
        "specular_reflection": True,
        "diffuse_reflection": True,
        "refraction": True,
        "diffraction": False,
    }
