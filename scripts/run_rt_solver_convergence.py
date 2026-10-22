"""Native CPU/LLVM convergence sweep for a fixed Sionna RT link."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
from artifacts.provenance import atomic_json, package_source_sha256, sha256_file, versions
from integrations.sionna.adapter import GridConfig, count_valid_paths, rt_cfr
from integrations.sionna.los_oracle import SegmentVisibilityResult, segment_visibility


@dataclass(frozen=True, slots=True)
class ConvergenceLink:
    """One named fixed TX/RX geometry in a shared scene."""

    link_id: str
    tx_position_m: tuple[float, float, float]
    rx_position_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.link_id, str) or not self.link_id.strip():
            raise ValueError("link_id must be a non-empty string")
        for name in ("tx_position_m", "rx_position_m"):
            position = tuple(float(value) for value in getattr(self, name))
            if len(position) != 3 or not all(math.isfinite(value) for value in position):
                raise ValueError(f"{name} must contain three finite coordinates")
            object.__setattr__(self, name, cast(tuple[float, float, float], position))


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


def relative_l2_error_if_defined(value: np.ndarray, reference: np.ndarray) -> float | None:
    """Return relative error only when the reference CFR has nonzero energy."""
    reference = np.asarray(reference)
    if not np.iscomplexobj(reference):
        raise ValueError("reference must be a complex CFR")
    denominator = float(np.linalg.norm(reference))
    if not math.isfinite(denominator):
        raise ValueError("reference must have finite energy")
    if denominator == 0:
        return None
    return relative_l2_error(value, reference)


def _scene_tree_sha256(scene_xml: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(scene_xml.parent.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(scene_xml.parent).as_posix().encode("utf8"))
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def path_observables(
    gains: np.ndarray,
    delays_s: np.ndarray,
    theta_r_rad: np.ndarray,
    phi_r_rad: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | None]:
    """Summarize valid SISO paths without summing away coherent CFR effects.

    Delay spread is power weighted over individual path coefficients. The
    angular statistic is ``acos(R)`` where ``R`` is the norm of the
    power-weighted mean unit arrival-direction vector.
    """
    raw_valid = np.asarray(valid).reshape(-1)
    if raw_valid.dtype != np.bool_:
        raise TypeError("valid must be a boolean path mask")
    values = [np.asarray(item).reshape(-1) for item in (gains, delays_s, theta_r_rad, phi_r_rad)]
    gain_values, delay_values, theta_values, phi_values = values
    valid_values = raw_valid
    if any(len(item) != len(valid_values) for item in values[:4]):
        raise ValueError("path observables must align with the valid-path mask")
    if not np.isfinite(gain_values).all() or not np.isfinite(delay_values[valid_values]).all():
        raise ValueError("valid path gains and delays must be finite")
    if (
        not np.isfinite(theta_values[valid_values]).all()
        or not np.isfinite(phi_values[valid_values]).all()
    ):
        raise ValueError("valid path arrival angles must be finite")
    if not np.any(valid_values):
        return {
            "first_arrival_delay_ns": None,
            "power_weighted_rms_delay_spread_ns": None,
            "power_weighted_arrival_angular_spread_rad_3d": None,
        }
    selected_gains = gain_values[valid_values]
    weights = np.abs(selected_gains) ** 2
    total_weight = float(np.sum(weights))
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("valid paths must have finite nonzero coefficient power")
    selected_delays = delay_values[valid_values]
    mean_delay = float(np.sum(weights * selected_delays) / total_weight)
    delay_spread = float(
        np.sqrt(np.sum(weights * (selected_delays - mean_delay) ** 2) / total_weight)
    )
    theta = theta_values[valid_values]
    phi = phi_values[valid_values]
    directions = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    mean_direction = np.sum(weights[:, None] * directions, axis=0) / total_weight
    resultant_length = float(np.linalg.norm(mean_direction))
    angular_spread = float(np.arccos(np.clip(resultant_length, 0.0, 1.0)))
    return {
        "first_arrival_delay_ns": float(np.min(selected_delays) * 1e9),
        "power_weighted_rms_delay_spread_ns": delay_spread * 1e9,
        "power_weighted_arrival_angular_spread_rad_3d": angular_spread,
    }


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
    mitsuba_variant: str = "auto",
    link_cases: Sequence[ConvergenceLink] | None = None,
) -> dict[str, Any]:
    """Compare ray budgets and interaction depths for named fixed links."""
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
    links = (
        (
            ConvergenceLink(
                "fixed-link",
                cast(tuple[float, float, float], tuple(tx_position)),
                cast(tuple[float, float, float], tuple(rx_position)),
            ),
        )
        if link_cases is None
        else tuple(link_cases)
    )
    if not links or any(not isinstance(link, ConvergenceLink) for link in links):
        raise ValueError("link_cases must contain at least one ConvergenceLink")
    if len({link.link_id for link in links}) != len(links):
        raise ValueError("link case IDs must be unique")
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in samples_per_src_values
    ):
        raise TypeError("samples_per_src_values must contain integers")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain nonnegative integers")

    try:
        from integrations.sionna.backend import activate_rt_backend, select_rt_backend

        backend_selection = select_rt_backend(mitsuba_variant)
        activate_rt_backend(backend_selection)
        import sionna.rt as rt
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(f"Could not initialize a usable Sionna RT backend: {error}") from error

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
    records_by_key: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    link_geometry: dict[str, SegmentVisibilityResult] = {}

    def evaluate(link: ConvergenceLink, seed: int, depth: int, samples: int) -> dict[str, Any]:
        key = (link.link_id, seed, depth, samples)
        if key in records_by_key:
            return records_by_key[key]
        scene.transmitters["convergence_tx"].position = list(link.tx_position_m)
        scene.receivers["convergence_rx"].position = list(link.rx_position_m)
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
        theta_r = np.asarray(paths.theta_r.numpy()).reshape(-1)
        phi_r = np.asarray(paths.phi_r.numpy()).reshape(-1)
        if any(len(value) != len(valid) for value in (gains, delays, theta_r, phi_r)):
            raise RuntimeError(
                "Sionna path mask, coefficients, delays, and angles do not align for SISO"
            )
        if link.link_id not in link_geometry:
            link_geometry[link.link_id] = segment_visibility(
                scene, link.tx_position_m, link.rx_position_m
            )
        row = {
            "link_id": link.link_id,
            "geometric_visibility_state": link_geometry[link.link_id].state.value,
            "geometric_los": link_geometry[link.link_id].visible,
            "seed": seed,
            "max_depth": depth,
            "samples_per_src": samples,
            "valid_path_count": count_valid_paths(paths),
            "valid_path_power_sum": float(np.sum(np.abs(gains[valid]) ** 2)),
            "valid_path_delays_ns_sorted": sorted((delays[valid] * 1e9).tolist()),
            **path_observables(gains, delays, theta_r, phi_r, valid),
            "cfr_mean_power": float(np.mean(np.abs(cfr) ** 2)),
            "cfr_sha256": hashlib.sha256(np.ascontiguousarray(cfr).tobytes()).hexdigest(),
            "runtime_seconds": time.perf_counter() - started,
            "_cfr": cfr,
        }
        records_by_key[key] = row
        return row

    reference_samples = max(samples_per_src_values)
    for link in links:
        scene.transmitters["convergence_tx"].position = list(link.tx_position_m)
        scene.receivers["convergence_rx"].position = list(link.rx_position_m)
        link_geometry[link.link_id] = segment_visibility(
            scene, link.tx_position_m, link.rx_position_m
        )
        for seed in seeds:
            reference = evaluate(link, seed, max_depth, reference_samples)["_cfr"]
            cases = {
                (max_depth, samples): evaluate(link, seed, max_depth, samples)
                for samples in samples_per_src_values
            }
            cases.update(
                {
                    (depth, reference_samples): evaluate(link, seed, depth, reference_samples)
                    for depth in range(max_depth + 1)
                }
            )
            for (depth, samples), row in sorted(cases.items()):
                cfr = row.pop("_cfr")
                row["relative_cfr_l2_to_max_budget_and_depth"] = relative_l2_error_if_defined(
                    cfr, reference
                )
                row["reference_cfr_zero_energy"] = float(np.linalg.norm(reference)) == 0.0
                runs.append(row)

    runtime = versions()
    return {
        "schema": "loslab.sionna_rt_solver_convergence.v2",
        "evidence_grade": "SINGLE_SCENE_GEOMETRY_LABELED_SOLVER_DIAGNOSTIC",
        "scene": {
            "name": scene_name,
            "xml_path": str(scene_xml),
            "xml_sha256": sha256_file(scene_xml),
            "scene_tree_sha256": _scene_tree_sha256(scene_xml),
        },
        "configuration": {
            "mitsuba_variant": backend_selection.variant,
            "rt_backend_selection": backend_selection.as_dict(),
            "deterministic_solver": True,
            "max_depth_values": list(range(max_depth + 1)),
            "samples_per_src_values": list(samples_per_src_values),
            "reference_samples_per_src": reference_samples,
            "seeds": list(seeds),
            "links": [
                {
                    "link_id": link.link_id,
                    "tx_position_m": list(link.tx_position_m),
                    "rx_position_m": list(link.rx_position_m),
                    "geometric_visibility_state": link_geometry[link.link_id].state.value,
                    "geometric_los": link_geometry[link.link_id].visible,
                    "segment_length_m": link_geometry[link.link_id].segment_length_m,
                    "first_intersection_distance_m": link_geometry[
                        link.link_id
                    ].first_intersection_distance_m,
                    "incidence_cosine": link_geometry[link.link_id].incidence_cosine,
                    "boundary_probe_blocked": list(
                        link_geometry[link.link_id].boundary_probe_blocked
                    ),
                }
                for link in links
            ],
            "carrier_hz": carrier_hz,
            "subcarrier_spacing_hz": subcarrier_spacing_hz,
            "n_subcarriers": n_subcarriers,
            "normalize_delays_for_cfr": False,
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
            "CFR retains absolute path delays and unnormalized received power.",
            "Delay spread is coefficient-power weighted; angular spread is the spherical mean-direction statistic documented in the runner.",
            "Backend selection validates one minimal native trace, not scene-material realism, convergence, or production throughput.",
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
    parser.add_argument("--mitsuba-variant", default="auto")
    parser.add_argument(
        "--link-case",
        action="append",
        nargs=7,
        metavar=("ID", "TX_X", "TX_Y", "TX_Z", "RX_X", "RX_Y", "RX_Z"),
        help="add a named link; repeat for matched links in the same scene",
    )
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
            link_cases=(
                tuple(
                    ConvergenceLink(
                        values[0],
                        cast(
                            tuple[float, float, float], tuple(float(value) for value in values[1:4])
                        ),
                        cast(
                            tuple[float, float, float], tuple(float(value) for value in values[4:7])
                        ),
                    )
                    for values in args.link_case
                )
                if args.link_case
                else None
            ),
        )
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    atomic_json(args.output, result)
    print(f"Saved {len(result['runs'])} matched solver runs to {args.output}")


if __name__ == "__main__":
    main()
