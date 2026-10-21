"""Runtime selection for Sionna RT's Mitsuba/Dr.Jit backends.

``auto`` prefers a CUDA Mitsuba variant only after a separate process completes
a tiny native Sionna RT trace. This catches hosts where CUDA is visible but
OptiX or its compatible driver/runtime is missing. LLVM is the portable
fallback. The child process prevents a failed CUDA probe from leaving Mitsuba's
global variant or Dr.Jit state half-initialized in the research process.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class RTBackendSelection:
    """Resolved native backend and the evidence used to select it."""

    requested: str
    backend: str
    variant: str
    available_variants: tuple[str, ...]
    runtime_versions: dict[str, str]
    probe_passed: bool
    probe_summary: str
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return JSON-ready runtime provenance."""
        return asdict(self)


_PROBE_PROGRAM = r"""
import json
import sys

variant = sys.argv[1]
import mitsuba as mi
mi.set_variant(variant)
import sionna.rt as rt

scene = rt.load_scene()
scene.frequency = 3.5e9
scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.add(rt.Transmitter("backend_probe_tx", [0.0, 0.0, 0.0]))
scene.add(rt.Receiver("backend_probe_rx", [10.0, 0.0, 0.0]))
scene.get("backend_probe_tx").look_at(scene.get("backend_probe_rx"))
scene.get("backend_probe_rx").look_at(scene.get("backend_probe_tx"))
paths = rt.PathSolver(deterministic=True)(
    scene,
    max_depth=0,
    los=True,
    specular_reflection=False,
    diffuse_reflection=False,
    refraction=False,
    diffraction=False,
    samples_per_src=1,
    seed=0,
)
count = int(paths.valid.numpy().sum())
if count < 1:
    raise RuntimeError("native direct-path probe returned no valid path")
print(json.dumps({"variant": variant, "valid_direct_paths": count}))
"""


def _runtime_versions() -> dict[str, str]:
    """Capture the native RT stack versions alongside the selected variant."""
    versions: dict[str, str] = {}
    for distribution in ("sionna-rt", "mitsuba", "drjit"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not installed"
    return versions


def probe_native_rt_variant(variant: str, *, timeout_s: float = 90.0) -> tuple[bool, str]:
    """Probe a variant in an isolated process with one deterministic free-space RT link."""
    if not variant:
        raise ValueError("variant must be nonempty")
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE_PROGRAM, variant],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"native RT probe exceeded {timeout_s:g} seconds"
    except OSError as exc:
        return False, f"could not start native RT probe: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        message = detail[-1] if detail else f"probe exited {result.returncode}"
        return False, f"native RT probe failed ({result.returncode}): {message[-500:]}"
    try:
        report = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return False, "native RT probe returned malformed output"
    return True, f"one direct path traced with {report['variant']}"


def select_rt_backend(
    requested: str = "auto",
    *,
    available_variants: tuple[str, ...] | None = None,
    probe: Callable[[str], tuple[bool, str]] | None = None,
) -> RTBackendSelection:
    """Select and verify ``auto``, ``cuda``, ``llvm`` or an exact Mitsuba variant.

    Automatic mode prefers ``cuda_ad_mono_polarized`` and verifies the complete
    Sionna RT call path (including OptiX-dependent kernels). If that check fails,
    it tries ``llvm_ad_mono_polarized``. Explicit choices are probed and fail
    with diagnostics instead of silently changing the requested backend.
    """
    if available_variants is None:
        try:
            import mitsuba as mi
        except ImportError as exc:
            raise RuntimeError(
                "Mitsuba is unavailable; install the pinned Sionna 2.2 environment"
            ) from exc
        available_variants = tuple(str(item) for item in mi.variants())
    else:
        available_variants = tuple(available_variants)
    if probe is None:
        probe = probe_native_rt_variant

    candidates: tuple[str, ...]
    if requested == "auto":
        candidates = tuple(
            variant
            for variant in ("cuda_ad_mono_polarized", "llvm_ad_mono_polarized")
            if variant in available_variants
        )
    elif requested in {"cuda", "llvm"}:
        candidate = f"{requested}_ad_mono_polarized"
        candidates = (candidate,) if candidate in available_variants else ()
    else:
        candidates = (requested,) if requested in available_variants else ()

    if not candidates:
        raise RuntimeError(
            f"requested Mitsuba backend {requested!r} is unavailable; installed variants: "
            f"{', '.join(available_variants) or '(none)'}"
        )

    failures: list[str] = []
    for variant in candidates:
        passed, summary = probe(variant)
        if passed:
            backend = "cuda_optix" if variant.startswith("cuda_") else "llvm_cpu"
            fallback = "; ".join(failures) if failures else None
            return RTBackendSelection(
                requested=requested,
                backend=backend,
                variant=variant,
                available_variants=available_variants,
                runtime_versions=_runtime_versions(),
                probe_passed=True,
                probe_summary=summary,
                fallback_reason=fallback,
            )
        failures.append(f"{variant}: {summary}")
        if requested != "auto":
            break
    raise RuntimeError(f"no usable Sionna RT backend for {requested!r}; " + "; ".join(failures))


def activate_rt_backend(selection: RTBackendSelection) -> None:
    """Set the already-probed Mitsuba variant before importing Sionna RT."""
    try:
        import mitsuba as mi
    except ImportError as exc:
        raise RuntimeError("Mitsuba is unavailable") from exc
    if selection.variant not in tuple(str(item) for item in mi.variants()):
        raise RuntimeError(f"selected Mitsuba variant is no longer installed: {selection.variant}")
    mi.set_variant(selection.variant)
