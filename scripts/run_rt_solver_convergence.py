"""Native CPU/LLVM convergence sweep for a fixed Sionna RT link."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
from artifacts.provenance import (
    atomic_json,
    package_source_sha256,
    require_file_matches_git_head,
    sha256_file,
    versions,
)
from integrations.sionna.adapter import GridConfig, count_valid_paths, rt_cfr
from integrations.sionna.candidate_audit import instrument_path_solver_candidate_counter
from integrations.sionna.los_oracle import SegmentVisibilityResult, segment_visibility

PATH_CFR_RECONSTRUCTION_RELATIVE_TOLERANCE = 1e-4


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


def count_paths_with_interaction(
    interactions: np.ndarray,
    valid: np.ndarray,
    interaction_bit: int,
) -> int:
    """Count valid SISO paths containing a selected Sionna interaction bit."""
    interaction_values = np.asarray(interactions)
    valid_values = np.asarray(valid)
    if valid_values.dtype != np.bool_:
        raise TypeError("valid must be a boolean path mask")
    valid_values = valid_values.reshape(-1)
    if interaction_values.ndim != 4 or interaction_values.shape[-1] != len(valid_values):
        raise ValueError("interactions must have [depth, rx, tx, path] axes aligned to valid")
    if isinstance(interaction_bit, bool) or interaction_bit <= 0:
        raise ValueError("interaction_bit must be a positive bit mask")
    has_interaction = np.any(
        np.bitwise_and(interaction_values, interaction_bit) != 0, axis=(0, 1, 2)
    )
    return int(np.count_nonzero(valid_values & has_interaction))


def path_level_diagnostics(
    *,
    link_id: str,
    scene_tree_sha256: str,
    gains: np.ndarray,
    delays_s: np.ndarray,
    theta_r_rad: np.ndarray,
    phi_r_rad: np.ndarray,
    theta_t_rad: np.ndarray,
    phi_t_rad: np.ndarray,
    interactions: np.ndarray,
    objects: np.ndarray,
    primitives: np.ndarray,
    vertices_m: np.ndarray,
    valid: np.ndarray,
    frequencies_hz: np.ndarray,
    carrier_hz: float,
    reference_cfr: np.ndarray,
    cfr_contributions: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], float | None]:
    """Serialize every valid SISO path and verify its CFR contribution sum.

    Path fingerprints identify geometry within one scene/link. They use the
    ordered interaction codes, object/primitive IDs, and interaction vertices
    quantized to 1 mm; the full-precision fields are retained alongside them.
    A fingerprint is a matching aid, not a claim that paths are physically
    identical across different scenes or material configurations.
    """
    gain_values = np.asarray(gains, dtype=np.complex128).reshape(-1)
    delay_values = np.asarray(delays_s, dtype=np.float64).reshape(-1)
    valid_values = np.asarray(valid)
    if valid_values.dtype != np.bool_ or valid_values.ndim != 1:
        raise TypeError("valid must be a one-dimensional boolean path mask")
    n_paths = len(valid_values)
    angular_values = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (theta_r_rad, phi_r_rad, theta_t_rad, phi_t_rad)
    ]
    interaction_values = np.asarray(interactions)
    object_values = np.asarray(objects)
    primitive_values = np.asarray(primitives)
    vertex_values = np.asarray(vertices_m, dtype=np.float64)
    if any(len(value) != n_paths for value in (gain_values, delay_values, *angular_values)):
        raise ValueError("path coefficients, delays, angles, and validity mask must align")
    if (
        interaction_values.ndim != 2
        or interaction_values.shape[1] != n_paths
        or object_values.shape != interaction_values.shape
        or primitive_values.shape != interaction_values.shape
        or vertex_values.shape != (*interaction_values.shape, 3)
    ):
        raise ValueError("interaction, object, primitive, and vertex path axes must align")
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_cfr).reshape(-1)
    if not math.isfinite(carrier_hz) or carrier_hz <= 0.0:
        raise ValueError("carrier_hz must be finite and positive")
    if not np.iscomplexobj(reference) or len(reference) != len(frequencies):
        raise ValueError("reference CFR must be complex and align with frequency samples")
    if not np.isfinite(frequencies).all() or not np.isfinite(reference).all():
        raise ValueError("frequencies and reference CFR must be finite")
    if (
        not np.isfinite(gain_values[valid_values]).all()
        or not np.isfinite(delay_values[valid_values]).all()
    ):
        raise ValueError("valid path coefficients and delays must be finite")
    if any(not np.isfinite(value[valid_values]).all() for value in angular_values):
        raise ValueError("valid path angles must be finite")

    records: list[dict[str, Any]] = []
    contributions: list[np.ndarray] = []
    valid_path_indices = np.flatnonzero(valid_values)
    native_contributions = None
    if cfr_contributions is not None:
        native_contributions = np.asarray(cfr_contributions)
        if (
            not np.iscomplexobj(native_contributions)
            or native_contributions.shape != (len(valid_path_indices), len(frequencies))
            or not np.isfinite(native_contributions).all()
        ):
            raise ValueError("native CFR contributions must align with valid paths and frequencies")
    for contribution_index, path_index in enumerate(valid_path_indices):
        signature_steps: list[dict[str, Any]] = []
        for step_index, code_value in enumerate(interaction_values[:, path_index]):
            code = int(code_value)
            if code == 0:
                continue
            vertex = vertex_values[step_index, path_index]
            if not np.isfinite(vertex).all():
                raise ValueError("valid path interaction vertices must be finite")
            signature_steps.append(
                {
                    "interaction_type_mask": code,
                    "object_id": int(object_values[step_index, path_index]),
                    "primitive_id": int(primitive_values[step_index, path_index]),
                    "vertex_mm": [int(np.rint(value * 1000.0)) for value in vertex],
                }
            )
        signature = {
            "version": "scene-link-interaction-geometry-v1",
            "scene_tree_sha256": scene_tree_sha256,
            "link_id": link_id,
            "interaction_sequence": signature_steps,
        }
        fingerprint = hashlib.sha256(
            json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf8")
        ).hexdigest()
        delay = float(delay_values[path_index])
        coefficient = complex(gain_values[path_index])
        baseband_coefficient = coefficient * np.exp(-2j * np.pi * carrier_hz * delay)
        contribution = (
            native_contributions[contribution_index]
            if native_contributions is not None
            else baseband_coefficient * np.exp(-2j * np.pi * frequencies * delay)
        )
        contributions.append(contribution)
        records.append(
            {
                "path_index": int(path_index),
                "valid": True,
                "path_fingerprint": fingerprint,
                "coefficient_real": float(coefficient.real),
                "coefficient_imag": float(coefficient.imag),
                "baseband_coefficient_real": float(baseband_coefficient.real),
                "baseband_coefficient_imag": float(baseband_coefficient.imag),
                "coefficient_power": float(abs(coefficient) ** 2),
                "absolute_delay_s": delay,
                "theta_r_rad": float(angular_values[0][path_index]),
                "phi_r_rad": float(angular_values[1][path_index]),
                "theta_t_rad": float(angular_values[2][path_index]),
                "phi_t_rad": float(angular_values[3][path_index]),
                "interaction_type_masks": [
                    int(value) for value in interaction_values[:, path_index]
                ],
                "object_ids": [int(value) for value in object_values[:, path_index]],
                "primitive_ids": [int(value) for value in primitive_values[:, path_index]],
                "interaction_vertices_m": vertex_values[:, path_index, :].tolist(),
                "cfr_contribution_real": contribution.real.tolist(),
                "cfr_contribution_imag": contribution.imag.tolist(),
            }
        )

    reconstructed = (
        np.sum(np.stack(contributions, axis=0), axis=0)
        if contributions
        else np.zeros(len(frequencies), dtype=np.complex128)
    )
    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(reconstructed - reference))
    reconstruction_error = (
        numerator / denominator if denominator > 0.0 else (0.0 if numerator == 0.0 else None)
    )
    return records, reconstruction_error


def _material_metadata(scene: Any) -> dict[str, dict[str, Any]]:
    """Record actual per-scene scattering values and pattern classes."""
    material_records: dict[str, dict[str, Any]] = {}
    for name, material in sorted(scene.radio_materials.items()):
        scattering = np.asarray(material.scattering_coefficient).reshape(-1)
        xpd = np.asarray(material.xpd_coefficient).reshape(-1)
        material_records[str(name)] = {
            "scattering_coefficient": scattering.astype(float).tolist(),
            "xpd_coefficient": xpd.astype(float).tolist(),
            "scattering_pattern": type(material.scattering_pattern).__name__,
        }
    return material_records


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
    material_scattering_overrides: dict[str, float] | None = None,
    max_num_paths_per_src: int = 1_000_000,
    include_cfr_vectors: bool = False,
    include_path_diagnostics: bool = False,
    path_diagnostics_max_depth_only: bool = False,
    only_max_depth: bool = False,
    include_path_lineage_diagnostics: bool = False,
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
    if (
        isinstance(max_num_paths_per_src, bool)
        or not isinstance(max_num_paths_per_src, int)
        or max_num_paths_per_src < 1
    ):
        raise ValueError("max_num_paths_per_src must be a positive integer")
    if not isinstance(include_cfr_vectors, bool):
        raise TypeError("include_cfr_vectors must be a boolean")
    if not isinstance(include_path_diagnostics, bool):
        raise TypeError("include_path_diagnostics must be a boolean")
    if not isinstance(path_diagnostics_max_depth_only, bool):
        raise TypeError("path_diagnostics_max_depth_only must be a boolean")
    if not isinstance(only_max_depth, bool):
        raise TypeError("only_max_depth must be a boolean")
    if not isinstance(include_path_lineage_diagnostics, bool):
        raise TypeError("include_path_lineage_diagnostics must be a boolean")
    if path_diagnostics_max_depth_only and not include_path_diagnostics:
        raise ValueError("max-depth-only diagnostics require include_path_diagnostics")
    if include_path_lineage_diagnostics and not (
        only_max_depth and include_path_diagnostics and include_cfr_vectors
    ):
        raise ValueError(
            "path-lineage diagnostics require only_max_depth, path diagnostics, and CFR vectors"
        )
    for name, coefficient in (material_scattering_overrides or {}).items():
        if not name or isinstance(coefficient, bool) or not math.isfinite(coefficient):
            raise ValueError("material scattering overrides require names and finite coefficients")
        if coefficient < 0.0 or coefficient > 1.0:
            raise ValueError("material scattering coefficients must be in [0, 1]")
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
        from sionna.rt.constants import InteractionType
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(f"Could not initialize a usable Sionna RT backend: {error}") from error

    scene_reference = getattr(rt.scene, scene_name, scene_name)
    scene_xml = Path(str(scene_reference)).resolve()
    if not scene_xml.is_file():
        raise FileNotFoundError(f"Sionna scene XML not found: {scene_xml}")
    scene_tree_sha256 = _scene_tree_sha256(scene_xml)
    scene = rt.load_scene(str(scene_xml))
    unknown_materials = set(material_scattering_overrides or {}) - set(scene.radio_materials)
    if unknown_materials:
        raise ValueError(f"unknown scene radio materials: {sorted(unknown_materials)}")
    for name, coefficient in (material_scattering_overrides or {}).items():
        scene.radio_materials[name].scattering_coefficient = coefficient
    material_records = _material_metadata(scene)
    scene.frequency = carrier_hz
    scene.bandwidth = n_subcarriers * subcarrier_spacing_hz
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(rt.Transmitter(name="convergence_tx", position=list(tx_position)))
    scene.add(rt.Receiver(name="convergence_rx", position=list(rx_position)))
    solver = rt.PathSolver(deterministic=True)
    candidate_path_counter = instrument_path_solver_candidate_counter(solver)
    lineage_observer = None
    if include_path_lineage_diagnostics:
        from integrations.sionna.path_lineage import PathLineageObserver

        lineage_observer = PathLineageObserver()
        lineage_observer.attach(solver)
    grid = GridConfig(
        n_subcarriers=n_subcarriers,
        spacing_hz=subcarrier_spacing_hz,
        carrier_hz=carrier_hz,
    )
    frequencies_hz = (
        np.arange(n_subcarriers, dtype=np.float64) - n_subcarriers // 2
    ) * subcarrier_spacing_hz

    runs: list[dict[str, Any]] = []
    records_by_key: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    link_geometry: dict[str, SegmentVisibilityResult] = {}

    def evaluate(link: ConvergenceLink, seed: int, depth: int, samples: int) -> dict[str, Any]:
        key = (link.link_id, seed, depth, samples)
        if key in records_by_key:
            return records_by_key[key]
        if lineage_observer is not None:
            lineage_observer.begin_case(
                link_id=link.link_id, seed=seed, depth=depth, samples=samples
            )
        scene.transmitters["convergence_tx"].position = list(link.tx_position_m)
        scene.receivers["convergence_rx"].position = list(link.rx_position_m)
        started = time.perf_counter()
        paths = solver(
            scene,
            max_depth=depth,
            max_num_paths_per_src=max_num_paths_per_src,
            los=True,
            specular_reflection=True,
            diffuse_reflection=True,
            refraction=True,
            diffraction=False,
            samples_per_src=samples,
            seed=seed,
        )
        candidate_path_count = candidate_path_counter.last_candidate_path_count
        if candidate_path_count is None:
            raise RuntimeError("Sionna did not expose its pre-filter candidate path count")
        candidate_path_counts_per_source = (
            candidate_path_counter.last_candidate_path_counts_per_source
        )
        path_lineage = lineage_observer.finish_case() if lineage_observer is not None else None
        if candidate_path_counts_per_source is None:
            raise RuntimeError("Sionna did not expose per-source candidate path counts")
        cfr = rt_cfr(paths, grid).reshape(-1)
        valid_path_mask = np.asarray(paths.valid.numpy(), dtype=bool)
        valid = valid_path_mask.reshape(-1)
        gain_real, gain_imag = paths.a
        gains = (np.asarray(gain_real.numpy()) + 1j * np.asarray(gain_imag.numpy())).reshape(-1)
        delays = np.asarray(paths.tau.numpy()).reshape(-1)
        theta_r = np.asarray(paths.theta_r.numpy()).reshape(-1)
        phi_r = np.asarray(paths.phi_r.numpy()).reshape(-1)
        theta_t = np.asarray(paths.theta_t.numpy()).reshape(-1)
        phi_t = np.asarray(paths.phi_t.numpy()).reshape(-1)
        interaction_types = np.asarray(paths.interactions.numpy(), dtype=np.uint32)
        diffuse_interaction_mask = np.any(
            np.bitwise_and(interaction_types, InteractionType.DIFFUSE) != 0, axis=(0, 1, 2)
        )
        if any(len(value) != len(valid) for value in (gains, delays, theta_r, phi_r)):
            raise RuntimeError(
                "Sionna path mask, coefficients, delays, and angles do not align for SISO"
            )
        valid_diffuse = valid & diffuse_interaction_mask
        valid_non_diffuse = valid & ~diffuse_interaction_mask
        diffuse_observables = path_observables(gains, delays, theta_r, phi_r, valid_diffuse)
        non_diffuse_observables = path_observables(gains, delays, theta_r, phi_r, valid_non_diffuse)
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
            "candidate_path_count_before_path_validation": candidate_path_count,
            "candidate_path_counts_per_source_before_path_validation": list(
                candidate_path_counts_per_source
            ),
            "candidate_path_cap_filled": any(
                count >= max_num_paths_per_src for count in candidate_path_counts_per_source
            ),
            "valid_diffuse_reflection_path_count": int(np.count_nonzero(valid_diffuse)),
            "valid_non_diffuse_path_count": int(np.count_nonzero(valid_non_diffuse)),
            "valid_path_power_sum": float(np.sum(np.abs(gains[valid]) ** 2)),
            "diffuse_path_power_sum": float(np.sum(np.abs(gains[valid_diffuse]) ** 2)),
            "non_diffuse_path_power_sum": float(np.sum(np.abs(gains[valid_non_diffuse]) ** 2)),
            "valid_path_delays_ns_sorted": sorted((delays[valid] * 1e9).tolist()),
            **path_observables(gains, delays, theta_r, phi_r, valid),
            **{f"diffuse_{key}": value for key, value in diffuse_observables.items()},
            **{f"non_diffuse_{key}": value for key, value in non_diffuse_observables.items()},
            "cfr_mean_power": float(np.mean(np.abs(cfr) ** 2)),
            "cfr_sha256": hashlib.sha256(np.ascontiguousarray(cfr).tobytes()).hexdigest(),
            "runtime_seconds": time.perf_counter() - started,
            "_cfr": cfr,
        }
        if include_cfr_vectors:
            row["cfr_vector_real"] = cfr.real.astype(np.float64).tolist()
            row["cfr_vector_imag"] = cfr.imag.astype(np.float64).tolist()
        if path_lineage is not None:
            row["path_lineage_stages"] = path_lineage
        if include_path_diagnostics and (not path_diagnostics_max_depth_only or depth == max_depth):
            if valid_path_mask.ndim != 3 or valid_path_mask.shape[0:2] != (1, 1):
                raise RuntimeError("path diagnostics currently require exactly one SISO link")
            native_path_contributions = (
                np.stack(
                    [
                        rt_cfr(
                            paths,
                            grid,
                            path_mask=np.eye(1, len(valid), path_index, dtype=bool).reshape(
                                1, 1, len(valid)
                            ),
                        ).reshape(-1)
                        for path_index in np.flatnonzero(valid)
                    ],
                    axis=0,
                )
                if np.any(valid)
                else np.zeros((0, len(frequencies_hz)), dtype=np.complex128)
            )
            path_rows, reconstruction_error = path_level_diagnostics(
                link_id=link.link_id,
                scene_tree_sha256=scene_tree_sha256,
                gains=gains,
                delays_s=delays,
                theta_r_rad=theta_r,
                phi_r_rad=phi_r,
                theta_t_rad=theta_t,
                phi_t_rad=phi_t,
                interactions=interaction_types[:, 0, 0, :],
                objects=np.asarray(paths.objects.numpy(), dtype=np.uint32)[:, 0, 0, :],
                primitives=np.asarray(paths.primitives.numpy(), dtype=np.uint32)[:, 0, 0, :],
                vertices_m=np.asarray(paths.vertices.numpy(), dtype=np.float64)[:, 0, 0, :, :],
                valid=valid,
                frequencies_hz=frequencies_hz,
                carrier_hz=carrier_hz,
                reference_cfr=cfr,
                cfr_contributions=native_path_contributions,
            )
            row["path_diagnostics"] = path_rows
            row["path_cfr_reconstruction_relative_l2"] = reconstruction_error
            row["path_cfr_reconstruction_within_tolerance"] = (
                None
                if reconstruction_error is None
                else reconstruction_error <= PATH_CFR_RECONSTRUCTION_RELATIVE_TOLERANCE
            )
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
                    for depth in ((max_depth,) if only_max_depth else range(max_depth + 1))
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
        "schema": (
            "loslab.sionna_rt_solver_convergence.v6"
            if include_path_lineage_diagnostics
            else "loslab.sionna_rt_solver_convergence.v5"
            if include_path_diagnostics
            else (
                "loslab.sionna_rt_solver_convergence.v4"
                if include_cfr_vectors
                else "loslab.sionna_rt_solver_convergence.v3"
            )
        ),
        "evidence_grade": "SINGLE_SCENE_GEOMETRY_LABELED_SOLVER_DIAGNOSTIC",
        "scene": {
            "name": scene_name,
            "xml_path": str(scene_xml),
            "xml_sha256": sha256_file(scene_xml),
            "scene_tree_sha256": scene_tree_sha256,
        },
        "configuration": {
            "mitsuba_variant": backend_selection.variant,
            "rt_backend_selection": backend_selection.as_dict(),
            "deterministic_solver": True,
            "max_depth_values": [max_depth] if only_max_depth else list(range(max_depth + 1)),
            "samples_per_src_values": list(samples_per_src_values),
            "reference_samples_per_src": reference_samples,
            "max_num_paths_per_src": max_num_paths_per_src,
            "candidate_path_count_stage": "after_candidate_generation_before_invalid_path_filter",
            "candidate_path_count_scope": "per_transmitter_source_from_buffer_source_indices",
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
            "cfr_vectors_included": include_cfr_vectors,
            "cfr_vector_encoding": "float64-real-imaginary-lists" if include_cfr_vectors else None,
            "path_diagnostics_included": include_path_diagnostics,
            "only_max_depth": only_max_depth,
            "path_lineage_diagnostics_included": include_path_lineage_diagnostics,
            "path_diagnostics_depths": (
                [max_depth]
                if include_path_diagnostics and path_diagnostics_max_depth_only
                else list(range(max_depth + 1))
                if include_path_diagnostics
                else []
            ),
            "path_diagnostics_encoding": (
                "valid-siso-paths-with-geometry-fingerprints-and-per-path-complex-baseband-cfr"
                if include_path_diagnostics
                else None
            ),
            "path_cfr_reconstruction_relative_tolerance": (
                PATH_CFR_RECONSTRUCTION_RELATIVE_TOLERANCE if include_path_diagnostics else None
            ),
            "path_fingerprint_scope": "scene-tree-and-link-local; quantized interaction vertices at 1 mm",
            "solver_flags": {
                "los": True,
                "specular_reflection": True,
                "diffuse_reflection": True,
                "refraction": True,
                "diffraction": False,
            },
            "materials": material_records,
            "material_scattering_overrides": material_scattering_overrides or {},
        },
        "source_revision": runtime.get("git_sha"),
        "package_source_sha256": package_source_sha256(),
        "script_sha256": sha256_file(__file__),
        "versions": runtime,
        "runs": runs,
        "limitations": [
            "One built-in scene and one SISO link do not qualify solver convergence across geometries.",
            "The maximum requested sample count is a comparison reference, not proven ground truth.",
            "Diffuse reflection is only generated when an interacting material has a nonzero scattering coefficient; actual per-material coefficients and diffuse-path counts are recorded.",
            "The path cap is recorded but cap sensitivity requires a separate audit.",
            "CFR retains absolute path delays and unnormalized received power.",
            "Complex CFR vectors are only stored when include_cfr_vectors is enabled.",
            "Per-path geometry and CFR contributions are only stored when include_path_diagnostics is enabled; fingerprints are scene/link-local matching aids, not universal physical IDs.",
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
    parser.add_argument("--max-num-paths-per-src", type=int, default=1_000_000)
    parser.add_argument(
        "--include-cfr-vectors",
        action="store_true",
        help="store each complex CFR sample for direct error analysis across reports",
    )
    parser.add_argument(
        "--include-path-diagnostics",
        action="store_true",
        help="store valid-path geometry, interactions, coefficients, fingerprints, and CFR contributions",
    )
    parser.add_argument(
        "--path-diagnostics-max-depth-only",
        action="store_true",
        help="limit per-path CFR diagnostics to rows at the configured maximum depth",
    )
    parser.add_argument(
        "--only-max-depth",
        action="store_true",
        help="run only the configured maximum interaction depth, without the 0..max-depth sweep",
    )
    parser.add_argument(
        "--include-path-lineage-diagnostics",
        action="store_true",
        help="snapshot zero-interaction path records after each native solver pipeline stage",
    )
    parser.add_argument(
        "--material-scattering",
        action="append",
        default=[],
        metavar="NAME=COEFFICIENT",
        help="override one scene material's scattering coefficient in [0,1]; repeat as needed",
    )
    parser.add_argument(
        "--link-case",
        action="append",
        nargs=7,
        metavar=("ID", "TX_X", "TX_Y", "TX_Z", "RX_X", "RX_Y", "RX_Z"),
        help="add a named link; repeat for matched links in the same scene",
    )
    args = parser.parse_args()
    try:
        require_file_matches_git_head(__file__)
    except RuntimeError as error:
        parser.error(str(error))
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    material_overrides: dict[str, float] = {}
    for assignment in args.material_scattering:
        name, separator, raw_coefficient = assignment.partition("=")
        if not separator or not name or name in material_overrides:
            parser.error(f"invalid or duplicate --material-scattering value: {assignment}")
        try:
            material_overrides[name] = float(raw_coefficient)
        except ValueError:
            parser.error(f"invalid scattering coefficient: {assignment}")
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
            material_scattering_overrides=material_overrides,
            max_num_paths_per_src=args.max_num_paths_per_src,
            include_cfr_vectors=args.include_cfr_vectors,
            include_path_diagnostics=args.include_path_diagnostics,
            path_diagnostics_max_depth_only=args.path_diagnostics_max_depth_only,
            only_max_depth=args.only_max_depth,
            include_path_lineage_diagnostics=args.include_path_lineage_diagnostics,
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
