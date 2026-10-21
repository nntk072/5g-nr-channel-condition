"""Native CPU/LLVM convergence sweep for a fixed Sionna RT link."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from artifacts.provenance import atomic_json, package_source_sha256, sha256_file, versions
from integrations.sionna.adapter import GridConfig, count_valid_paths, rt_cfr


def relative_l2_error(value: np.ndarray, reference: np.ndarray) -> float:
    """Return relative complex L2 error, rejecting incompatible or zero references."""
    value = np.asarray(value)
    reference = np.asarray(reference)
    if (
        value.shape != reference.shape
        or not np.iscomplexobj(value)
        or not np.iscomplexobj(reference)
    ):
        raise ValueError("value and reference must be aligned complex arrays")
    denominator = float(np.linalg.norm(reference))
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("reference must have finite nonzero energy")
    return float(np.linalg.norm(value - reference) / denominator)


def _scene_tree_sha256(scene_xml: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(scene_xml.parent.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(scene_xml.parent).as_posix().encode("utf8"))
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def run_convergence(
    *,
    scene_name: str,
    tx_position: Sequence[float],
    rx_position: Sequence[float],
    max_depth: int,
    samples_per_src_values: Sequence[int],
    seeds: Sequence[int],
    n_subcarriers: int = 256,
    subcarrier_spacing_hz: float = 30_000.0,
    carrier_hz: float = 3.5e9,
    mitsuba_variant: str = "llvm_ad_mono_polarized",
) -> dict[str, Any]:
    """Compare ray budgets and interaction depths for identical fixed links."""
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    if not samples_per_src_values or any(value <= 0 for value in samples_per_src_values):
        raise ValueError("samples_per_src_values must be positive")
    if len(set(samples_per_src_values)) != len(samples_per_src_values):
        raise ValueError("samples_per_src_values must be unique")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    if len(tx_position) != 3 or len(rx_position) != 3:
        raise ValueError("transmitter and receiver positions must have three coordinates")

    try:
        import mitsuba as mi

        mi.set_variant(mitsuba_variant)
        import sionna.rt as rt
    except ImportError as error:
        raise RuntimeError("Install the optional Sionna RT 2.2 environment") from error

    scene_reference = getattr(rt.scene, scene_name, scene_name)
    scene_xml = Path(str(scene_reference)).resolve()
    if not scene_xml.is_file():
        raise FileNotFoundError(f"Sionna scene XML not found: {scene_xml}")
    scene = rt.load_scene(str(scene_xml))
    scene.frequency = carrier_hz
    scene.bandwidth = n_subcarriers * subcarrier_spacing_hz
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(rt.Transmitter(name="convergence_tx", position=list(tx_position)))
    scene.add(rt.Receiver(name="convergence_rx", position=list(rx_position)))
    solver = rt.PathSolver(deterministic=True)
    grid = GridConfig(
        n_subcarriers=n_subcarriers,
        spacing_hz=subcarrier_spacing_hz,
        carrier_hz=carrier_hz,
    )

    runs: list[dict[str, Any]] = []
    reference_by_seed: dict[int, np.ndarray] = {}
    records_by_key: dict[tuple[int, int, int], dict[str, Any]] = {}

    def evaluate(seed: int, depth: int, samples: int) -> dict[str, Any]:
        key = (seed, depth, samples)
        if key in records_by_key:
            return records_by_key[key]
        started = time.perf_counter()
        paths = solver(
            scene,
            max_depth=depth,
            los=True,
            specular_reflection=True,
            diffuse_reflection=True,
            refraction=True,
            diffraction=False,
            samples_per_src=samples,
            seed=seed,
        )
        cfr = rt_cfr(paths, grid).reshape(-1)
        valid = np.asarray(paths.valid.numpy(), dtype=bool).reshape(-1)
        gain_real, gain_imag = paths.a
        gains = (np.asarray(gain_real.numpy()) + 1j * np.asarray(gain_imag.numpy())).reshape(-1)
        delays = np.asarray(paths.tau.numpy()).reshape(-1)
        if len(valid) != len(gains) or len(valid) != len(delays):
            raise RuntimeError("Sionna path mask, gains, and delays do not align for SISO")
        row = {
            "seed": seed,
            "max_depth": depth,
            "samples_per_src": samples,
            "valid_path_count": count_valid_paths(paths),
            "valid_path_power_sum": float(np.sum(np.abs(gains[valid]) ** 2)),
            "valid_path_delays_ns_sorted": sorted((delays[valid] * 1e9).tolist()),
            "cfr_mean_power": float(np.mean(np.abs(cfr) ** 2)),
            "cfr_sha256": hashlib.sha256(np.ascontiguousarray(cfr).tobytes()).hexdigest(),
            "runtime_seconds": time.perf_counter() - started,
            "_cfr": cfr,
        }
        records_by_key[key] = row
        return row

    reference_samples = max(samples_per_src_values)
    for seed in seeds:
        reference = evaluate(seed, max_depth, reference_samples)["_cfr"]
        reference_by_seed[seed] = reference
        cases = {
            (max_depth, samples): evaluate(seed, max_depth, samples)
            for samples in samples_per_src_values
        }
        cases.update(
            {
                (depth, reference_samples): evaluate(seed, depth, reference_samples)
                for depth in range(max_depth + 1)
            }
        )
        for (depth, samples), row in sorted(cases.items()):
            cfr = row.pop("_cfr")
            row["relative_cfr_l2_to_max_budget_and_depth"] = relative_l2_error(
                cfr, reference_by_seed[seed]
            )
            runs.append(row)

    runtime = versions()
    return {
        "schema": "loslab.sionna_rt_solver_convergence.v1",
        "evidence_grade": "SINGLE_SCENE_FIXED_LINK_SOLVER_DIAGNOSTIC",
        "scene": {
            "name": scene_name,
            "xml_path": str(scene_xml),
            "xml_sha256": sha256_file(scene_xml),
            "scene_tree_sha256": _scene_tree_sha256(scene_xml),
        },
        "configuration": {
            "mitsuba_variant": mitsuba_variant,
            "deterministic_solver": True,
            "max_depth_values": list(range(max_depth + 1)),
            "samples_per_src_values": list(samples_per_src_values),
            "reference_samples_per_src": reference_samples,
            "seeds": list(seeds),
            "tx_position_m": list(tx_position),
            "rx_position_m": list(rx_position),
            "carrier_hz": carrier_hz,
            "subcarrier_spacing_hz": subcarrier_spacing_hz,
            "n_subcarriers": n_subcarriers,
            "normalize_delays_for_cfr": True,
            "normalize_cfr_power": False,
            "solver_flags": {
                "los": True,
                "specular_reflection": True,
                "diffuse_reflection": True,
                "refraction": True,
                "diffraction": False,
            },
        },
        "source_revision": runtime.get("git_sha"),
        "package_source_sha256": package_source_sha256(),
        "script_sha256": sha256_file(__file__),
        "versions": runtime,
        "runs": runs,
        "limitations": [
            "One built-in scene and one SISO link do not qualify solver convergence across geometries.",
            "The maximum requested sample count is a comparison reference, not proven ground truth.",
            "CFR uses normalized path delays but unnormalized received power.",
            "This is CPU/LLVM implementation evidence; it does not establish scene-material realism or GPU RT throughput.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scene", default="box_one_screen")
    parser.add_argument("--tx", type=float, nargs=3, default=(-2.0, 0.0, 2.5))
    parser.add_argument("--rx", type=float, nargs=3, default=(0.0, 0.0, 2.5))
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument(
        "--samples-per-src", type=int, nargs="+", default=[200, 2000, 20_000, 100_000]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--n-subcarriers", type=int, default=256)
    parser.add_argument("--subcarrier-spacing-hz", type=float, default=30_000.0)
    parser.add_argument("--carrier-hz", type=float, default=3.5e9)
    parser.add_argument("--mitsuba-variant", default="llvm_ad_mono_polarized")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        result = run_convergence(
            scene_name=args.scene,
            tx_position=args.tx,
            rx_position=args.rx,
            max_depth=args.max_depth,
            samples_per_src_values=args.samples_per_src,
            seeds=args.seeds,
            n_subcarriers=args.n_subcarriers,
            subcarrier_spacing_hz=args.subcarrier_spacing_hz,
            carrier_hz=args.carrier_hz,
            mitsuba_variant=args.mitsuba_variant,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    atomic_json(args.output, result)
    print(f"Saved {len(result['runs'])} matched solver runs to {args.output}")


if __name__ == "__main__":
    main()
