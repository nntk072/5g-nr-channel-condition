"""Measure channel changes from explicit post-solver weak-path cutoffs."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from artifacts.provenance import (
    atomic_json,
    package_source_sha256,
    require_file_matches_git_head,
    sha256_file,
    versions,
)
from integrations.sionna.adapter import GridConfig, rt_cfr
from integrations.sionna.candidate_audit import instrument_path_solver_candidate_counter
from integrations.sionna.los_oracle import segment_visibility
from integrations.sionna.path_power_threshold import (
    relative_path_power_db,
    relative_power_threshold_mask,
)


def relative_l2_error(value: np.ndarray, reference: np.ndarray) -> float | None:
    """Compute a relative complex-L2 error, undefined for a zero reference."""
    left, right = np.asarray(value), np.asarray(reference)
    if left.shape != right.shape or not np.iscomplexobj(left) or not np.iscomplexobj(right):
        raise ValueError("value and reference must be aligned complex arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("value and reference must be finite")
    denominator = float(np.linalg.norm(right))
    if denominator == 0.0:
        return None
    return float(np.linalg.norm(left - right) / denominator)


def run_threshold_audit(
    *,
    scene_name: str,
    tx_position_m: Sequence[float],
    rx_position_m: Sequence[float],
    max_depth: int,
    samples_per_src: int,
    max_num_paths_per_src: int,
    seeds: Sequence[int],
    thresholds_db: Sequence[float],
    carrier_hz: float = 3.5e9,
    subcarrier_spacing_hz: float = 30e3,
    n_subcarriers: int = 256,
    mitsuba_variant: str = "auto",
    material_scattering_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Trace one SISO link and compare unthresholded CFR with path cutoffs."""
    for name, value, minimum in (
        ("max_depth", max_depth, 0),
        ("samples_per_src", samples_per_src, 1),
        ("max_num_paths_per_src", max_num_paths_per_src, 1),
        ("n_subcarriers", n_subcarriers, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if len(tx_position_m) != 3 or len(rx_position_m) != 3:
        raise ValueError("TX and RX positions must each have three coordinates")
    positions = (*tx_position_m, *rx_position_m)
    if not all(math.isfinite(float(value)) for value in positions):
        raise ValueError("TX and RX positions must be finite")
    if not math.isfinite(carrier_hz) or carrier_hz <= 0:
        raise ValueError("carrier_hz must be finite and positive")
    if not math.isfinite(subcarrier_spacing_hz) or subcarrier_spacing_hz <= 0:
        raise ValueError("subcarrier_spacing_hz must be finite and positive")
    if not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
    ):
        raise ValueError("seeds must be nonempty nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if not thresholds_db or len(set(thresholds_db)) != len(thresholds_db):
        raise ValueError("thresholds_db must be nonempty and unique")
    for threshold in thresholds_db:
        if isinstance(threshold, bool) or not math.isfinite(threshold) or threshold > 0.0:
            raise ValueError("thresholds_db must be finite and nonpositive")

    try:
        from integrations.sionna.backend import activate_rt_backend, select_rt_backend

        backend = select_rt_backend(mitsuba_variant)
        activate_rt_backend(backend)
        import sionna.rt as rt
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(f"Could not initialize a usable Sionna RT backend: {error}") from error

    scene_reference = getattr(rt.scene, scene_name, scene_name)
    scene_path = Path(str(scene_reference)).resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"Sionna RT scene not found: {scene_path}")
    scene = rt.load_scene(str(scene_path))
    unknown_materials = set(material_scattering_overrides or {}) - set(scene.radio_materials)
    if unknown_materials:
        raise ValueError(f"unknown scene radio materials: {sorted(unknown_materials)}")
    for name, coefficient in (material_scattering_overrides or {}).items():
        if not math.isfinite(coefficient) or not 0.0 <= coefficient <= 1.0:
            raise ValueError(f"scattering override for {name} must be finite and in [0, 1]")
        scene.radio_materials[name].scattering_coefficient = coefficient
    scene.frequency = carrier_hz
    scene.bandwidth = n_subcarriers * subcarrier_spacing_hz
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    tx = rt.Transmitter(name="threshold_audit_tx", position=list(tx_position_m))
    rx = rt.Receiver(name="threshold_audit_rx", position=list(rx_position_m))
    scene.add(tx)
    scene.add(rx)
    visibility = segment_visibility(scene, tx_position_m, rx_position_m)
    solver = rt.PathSolver(deterministic=True)
    candidate_counter = instrument_path_solver_candidate_counter(solver)
    grid = GridConfig(
        n_subcarriers=n_subcarriers,
        spacing_hz=subcarrier_spacing_hz,
        carrier_hz=carrier_hz,
    )

    runs: list[dict[str, Any]] = []
    for seed in seeds:
        started = time.perf_counter()
        paths = solver(
            scene,
            max_depth=max_depth,
            max_num_paths_per_src=max_num_paths_per_src,
            los=True,
            specular_reflection=True,
            diffuse_reflection=True,
            refraction=True,
            diffraction=False,
            samples_per_src=samples_per_src,
            seed=seed,
        )
        candidate_counts = candidate_counter.last_candidate_path_counts_per_source
        if candidate_counts is None:
            raise RuntimeError("Sionna did not expose pre-validation per-source path counts")
        if any(count >= max_num_paths_per_src for count in candidate_counts):
            raise RuntimeError(
                "Threshold audit refuses a cap-filled path set: "
                f"seed={seed}, candidate_counts={candidate_counts}, cap={max_num_paths_per_src}"
            )

        valid = np.asarray(paths.valid.numpy(), dtype=bool)
        gains_real, gains_imag = paths.a
        gains = (np.asarray(gains_real.numpy()) + 1j * np.asarray(gains_imag.numpy())).reshape(-1)
        valid_flat = valid.reshape(-1)
        if gains.shape != valid_flat.shape:
            raise RuntimeError("SISO path coefficient and validity axes do not align")
        levels_db = relative_path_power_db(gains, valid_flat)
        path_power = np.abs(gains) ** 2
        total_path_power = float(np.sum(path_power[valid_flat]))
        if not total_path_power > 0.0:
            raise RuntimeError("Threshold audit requires positive valid path coefficient power")
        full_cfr = rt_cfr(paths, grid)
        thresholds: list[dict[str, Any]] = []
        for threshold_db in sorted(thresholds_db, reverse=True):
            selected_flat = relative_power_threshold_mask(gains, valid_flat, threshold_db)
            selected = selected_flat.reshape(valid.shape)
            thresholded_cfr = rt_cfr(paths, grid, path_mask=selected)
            retained_power = float(np.sum(path_power[selected_flat]))
            retained_power_fraction = float(np.clip(retained_power / total_path_power, 0.0, 1.0))
            cfr_error = relative_l2_error(thresholded_cfr, full_cfr)
            thresholds.append(
                {
                    "threshold_db_relative_to_strongest_path": float(threshold_db),
                    "retained_path_count": int(np.count_nonzero(selected_flat)),
                    "removed_path_count": int(
                        np.count_nonzero(valid_flat) - np.count_nonzero(selected_flat)
                    ),
                    "retained_path_coefficient_power_fraction": retained_power_fraction,
                    "cfr_relative_l2_error": cfr_error,
                    "cfr_sha256": hashlib.sha256(
                        np.ascontiguousarray(thresholded_cfr).tobytes()
                    ).hexdigest(),
                    "cfr_vector_real": thresholded_cfr.reshape(-1).real.astype(np.float64).tolist(),
                    "cfr_vector_imag": thresholded_cfr.reshape(-1).imag.astype(np.float64).tolist(),
                }
            )
        runs.append(
            {
                "seed": seed,
                "geometric_visibility_state": visibility.state.value,
                "geometric_los": visibility.visible,
                "candidate_paths_per_source_before_validation": list(candidate_counts),
                "candidate_path_cap_filled": False,
                "valid_path_count": int(np.count_nonzero(valid_flat)),
                "valid_path_coefficient_power_sum": total_path_power,
                "valid_path_relative_power_db_sorted": sorted(
                    (
                        None if not math.isfinite(float(value)) else float(value)
                        for value in levels_db[valid_flat]
                    ),
                    key=lambda value: -math.inf if value is None else value,
                ),
                "full_cfr_sha256": hashlib.sha256(
                    np.ascontiguousarray(full_cfr).tobytes()
                ).hexdigest(),
                "full_cfr_vector_real": full_cfr.reshape(-1).real.astype(np.float64).tolist(),
                "full_cfr_vector_imag": full_cfr.reshape(-1).imag.astype(np.float64).tolist(),
                "thresholds": thresholds,
                "runtime_seconds": time.perf_counter() - started,
            }
        )
    runtime = versions()
    return {
        "schema": "loslab.rt_path_power_threshold_audit.v1",
        "evidence_grade": "SINGLE_SCENE_POST_SOLVER_CUTOFF_SENSITIVITY",
        "scene": {
            "name": scene_name,
            "xml_path": str(scene_path),
            "xml_sha256": sha256_file(scene_path),
            "scene_tree_sha256": _scene_tree_sha256(scene_path),
        },
        "configuration": {
            "mitsuba_variant": backend.variant,
            "rt_backend_selection": backend.as_dict(),
            "carrier_hz": carrier_hz,
            "subcarrier_spacing_hz": subcarrier_spacing_hz,
            "n_subcarriers": n_subcarriers,
            "max_depth": max_depth,
            "samples_per_src": samples_per_src,
            "max_num_paths_per_src": max_num_paths_per_src,
            "deterministic_solver": True,
            "seeds": list(seeds),
            "thresholds_db_relative_to_strongest_path": sorted(
                (float(value) for value in thresholds_db), reverse=True
            ),
            "material_scattering_overrides": material_scattering_overrides or {},
            "path_power_definition": "per-valid-path Sionna complex coefficient magnitude squared",
            "threshold_stage": "post-solver path mask before coherent CFR summation",
            "candidate_path_count_stage": "after_candidate_generation_before_invalid_path_filter",
            "candidate_path_count_scope": "per_transmitter_source_from_buffer_source_indices",
            "normalize_delays": False,
            "normalize_cfr_power": False,
            "solver_flags": {
                "los": True,
                "specular_reflection": True,
                "diffuse_reflection": True,
                "refraction": True,
                "diffraction": False,
            },
            "tx_position_m": [float(value) for value in tx_position_m],
            "rx_position_m": [float(value) for value in rx_position_m],
        },
        "source_revision": runtime.get("git_sha"),
        "script_sha256": sha256_file(__file__),
        "package_source_sha256": package_source_sha256(),
        "versions": runtime,
        "runs": runs,
        "limitations": [
            "The threshold is a post-solver coefficient-power cutoff, not receiver detectability or thermal-noise sensitivity.",
            "A relative threshold is referenced to the strongest valid path in each link.",
            "The finite Sionna path set and selected scene/materials are not physical ground truth.",
            "No threshold is applied by canonical dataset generation; this report measures hypothetical truncation sensitivity.",
        ],
    }


def _scene_tree_sha256(scene_xml: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(scene_xml.parent.rglob("*")):
        if path.is_file():
            relative = path.relative_to(scene_xml.parent).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "little"))
            digest.update(relative)
            payload = path.read_bytes()
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scene", default="box_one_screen")
    parser.add_argument("--tx", type=float, nargs=3, default=(-2.0, 0.0, 2.5))
    parser.add_argument("--rx", type=float, nargs=3, default=(2.0, 0.0, 2.5))
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--samples-per-src", type=int, default=20_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument(
        "--thresholds-db", type=float, nargs="+", default=[-20, -40, -60, -80, -100, -120]
    )
    parser.add_argument("--material-scattering", action="append", default=[])
    parser.add_argument("--mitsuba-variant", default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite path threshold report: {args.output}")
    require_file_matches_git_head(Path(__file__))
    overrides: dict[str, float] = {}
    for value in args.material_scattering:
        name, separator, raw = value.partition("=")
        if not separator or not name:
            raise SystemExit("--material-scattering must use NAME=COEFFICIENT")
        if name in overrides:
            raise SystemExit(f"duplicate material override: {name}")
        overrides[name] = float(raw)
    report = run_threshold_audit(
        scene_name=args.scene,
        tx_position_m=args.tx,
        rx_position_m=args.rx,
        max_depth=args.max_depth,
        samples_per_src=args.samples_per_src,
        max_num_paths_per_src=args.max_num_paths_per_src,
        seeds=args.seeds,
        thresholds_db=args.thresholds_db,
        mitsuba_variant=args.mitsuba_variant,
        material_scattering_overrides=overrides,
    )
    atomic_json(args.output, report)
    print(f"Saved {len(report['runs'])} threshold-sensitivity runs to {args.output}")


if __name__ == "__main__":
    main()
