"""Attribute one matched RT channel change to path discovery or path values."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from artifacts.provenance import atomic_json, package_source_sha256, sha256_file, versions
from scipy.optimize import linear_sum_assignment

SPEED_OF_LIGHT_M_S = 299_792_458.0


def _complex_vector(record: dict[str, Any], real_key: str, imag_key: str) -> np.ndarray:
    result = np.asarray(record[real_key], dtype=np.float64) + 1j * np.asarray(
        record[imag_key], dtype=np.float64
    )
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{real_key}/{imag_key} must encode a finite complex vector")
    return result


def _path_signature(path: dict[str, Any]) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (int(interaction), int(object_id), int(primitive_id))
        for interaction, object_id, primitive_id in zip(
            path["interaction_type_masks"], path["object_ids"], path["primitive_ids"]
        )
        if int(interaction) != 0
    )


def _interaction_vertices(path: dict[str, Any]) -> np.ndarray:
    vertices = np.asarray(path["interaction_vertices_m"], dtype=np.float64)
    interactions = np.asarray(path["interaction_type_masks"], dtype=np.uint32)
    if vertices.shape != (len(interactions), 3):
        raise ValueError("path vertices and interaction masks must align")
    selected = vertices[interactions != 0]
    if not np.isfinite(selected).all():
        raise ValueError("valid interaction vertices must be finite")
    return selected


def _direction(theta: float, phi: float) -> np.ndarray:
    return np.asarray(
        [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]
    )


def _path_match_cost(low: dict[str, Any], high: dict[str, Any]) -> float:
    low_vertices = _interaction_vertices(low)
    high_vertices = _interaction_vertices(high)
    if low_vertices.shape != high_vertices.shape:
        return math.inf
    vertex_delta = (
        float(np.max(np.linalg.norm(low_vertices - high_vertices, axis=1)))
        if len(low_vertices)
        else 0.0
    )
    delay_delta_m = (
        abs(float(low["absolute_delay_s"]) - float(high["absolute_delay_s"])) * SPEED_OF_LIGHT_M_S
    )
    angular_delta = max(
        math.acos(
            float(
                np.clip(
                    np.dot(_direction(low[theta], low[phi]), _direction(high[theta], high[phi])),
                    -1.0,
                    1.0,
                )
            )
        )
        for theta, phi in (("theta_r_rad", "phi_r_rad"), ("theta_t_rad", "phi_t_rad"))
    )
    angular_displacement_m = (
        angular_delta
        * max(float(low["absolute_delay_s"]), float(high["absolute_delay_s"]))
        * SPEED_OF_LIGHT_M_S
    )
    return max(vertex_delta, delay_delta_m, angular_displacement_m)


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p95": None, "max": None}
    q50, q95, qmax = np.quantile(np.asarray(values, dtype=np.float64), [0.5, 0.95, 1.0])
    return {"median": float(q50), "p95": float(q95), "max": float(qmax)}


def analyze_pair(
    low: dict[str, Any],
    high: dict[str, Any],
    *,
    maximum_match_cost_m: float = 0.02,
) -> dict[str, Any]:
    """Decompose a paired CFR difference after conservative path matching.

    Paths are grouped by ordered interaction type/object/primitive sequence,
    then assigned by interaction-vertex displacement, delay-equivalent
    distance, and endpoint-direction displacement. Unmatched paths remain explicit discovery/loss terms. This is a
    diagnostic matching rule, not a claim of universal physical path identity.
    """
    if not math.isfinite(maximum_match_cost_m) or maximum_match_cost_m <= 0.0:
        raise ValueError("maximum_match_cost_m must be finite and positive")
    if (low["link_id"], low["seed"], low["max_depth"]) != (
        high["link_id"],
        high["seed"],
        high["max_depth"],
    ):
        raise ValueError("path-budget rows must refer to the same link, seed, and depth")
    low_cfr = _complex_vector(low, "cfr_vector_real", "cfr_vector_imag")
    high_cfr = _complex_vector(high, "cfr_vector_real", "cfr_vector_imag")
    if low_cfr.shape != high_cfr.shape:
        raise ValueError("paired CFR vectors must have identical frequency axes")
    low_paths = list(low["path_diagnostics"])
    high_paths = list(high["path_diagnostics"])
    for record, paths in ((low, low_paths), (high, high_paths)):
        if len(paths) != int(record["valid_path_count"]):
            raise ValueError("path diagnostics must include every valid path")
        if record.get("path_cfr_reconstruction_within_tolerance") is not True:
            raise ValueError("path CFR reconstruction must pass before attribution")

    low_groups: dict[tuple[tuple[int, int, int], ...], list[dict[str, Any]]] = defaultdict(list)
    high_groups: dict[tuple[tuple[int, int, int], ...], list[dict[str, Any]]] = defaultdict(list)
    for path in low_paths:
        low_groups[_path_signature(path)].append(path)
    for path in high_paths:
        high_groups[_path_signature(path)].append(path)

    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    lost: list[dict[str, Any]] = []
    discovered: list[dict[str, Any]] = []
    for signature in low_groups.keys() | high_groups.keys():
        left, right = low_groups[signature], high_groups[signature]
        if not left:
            discovered.extend(right)
            continue
        if not right:
            lost.extend(left)
            continue
        costs = np.asarray(
            [[_path_match_cost(a, b) for b in right] for a in left], dtype=np.float64
        )
        row_indices, column_indices = linear_sum_assignment(costs)
        assigned_left: set[int] = set()
        assigned_right: set[int] = set()
        for i, j in zip(row_indices, column_indices):
            if costs[i, j] <= maximum_match_cost_m:
                matched.append((left[i], right[j]))
                assigned_left.add(int(i))
                assigned_right.add(int(j))
        lost.extend(path for i, path in enumerate(left) if i not in assigned_left)
        discovered.extend(path for j, path in enumerate(right) if j not in assigned_right)

    def contribution(path: dict[str, Any]) -> np.ndarray:
        return _complex_vector(path, "cfr_contribution_real", "cfr_contribution_imag")

    zeros = np.zeros_like(low_cfr)
    matched_delta = sum((contribution(b) - contribution(a) for a, b in matched), start=zeros.copy())
    discovery_term = sum((contribution(path) for path in discovered), start=zeros.copy())
    lost_term = sum((contribution(path) for path in lost), start=zeros.copy())
    cfr_delta = high_cfr - low_cfr
    delta_norm = float(np.linalg.norm(cfr_delta))
    closure_error = float(np.linalg.norm(cfr_delta - (matched_delta + discovery_term - lost_term)))

    coefficient_errors: list[float] = []
    delay_errors_ns: list[float] = []
    vertex_errors_m: list[float] = []
    direction_errors_rad: list[float] = []
    for a, b in matched:
        coefficient_a = complex(a["coefficient_real"], a["coefficient_imag"])
        coefficient_b = complex(b["coefficient_real"], b["coefficient_imag"])
        coefficient_errors.append(
            abs(coefficient_b - coefficient_a)
            / max(abs(coefficient_a), abs(coefficient_b), np.finfo(np.float64).tiny)
        )
        delay_errors_ns.append(
            abs(float(b["absolute_delay_s"]) - float(a["absolute_delay_s"])) * 1e9
        )
        va, vb = _interaction_vertices(a), _interaction_vertices(b)
        vertex_errors_m.append(float(np.max(np.linalg.norm(va - vb, axis=1))) if len(va) else 0.0)
        angular_errors = []
        for theta, phi in (("theta_r_rad", "phi_r_rad"), ("theta_t_rad", "phi_t_rad")):
            direction_a = _direction(a[theta], a[phi])
            direction_b = _direction(b[theta], b[phi])
            dot = float(np.dot(direction_a, direction_b))
            cross_norm = float(np.linalg.norm(np.cross(direction_a, direction_b)))
            angular_errors.append(math.atan2(cross_norm, dot))
        direction_errors_rad.append(max(angular_errors))

    def order_counts(paths: list[dict[str, Any]]) -> dict[str, int]:
        return {
            str(order): count
            for order, count in sorted(
                Counter(len(_path_signature(path)) for path in paths).items()
            )
        }

    def zero_interaction_summary(paths: list[dict[str, Any]]) -> dict[str, int]:
        zero_order_paths = [path for path in paths if not _path_signature(path)]
        return {
            "path_count": len(zero_order_paths),
            "unique_geometry_fingerprints": len(
                {path.get("path_fingerprint") for path in zero_order_paths}
            ),
            "unique_delays_at_1_ps_resolution": len(
                {round(float(path["absolute_delay_s"]) * 1e12) for path in zero_order_paths}
            ),
            "unique_coefficients_at_1e-12_resolution": len(
                {
                    (
                        round(float(path["coefficient_real"]), 12),
                        round(float(path["coefficient_imag"]), 12),
                    )
                    for path in zero_order_paths
                }
            ),
            "zero_transmitter_angle_records": sum(
                abs(float(path["theta_t_rad"])) < 1e-12 and abs(float(path["phi_t_rad"])) < 1e-12
                for path in zero_order_paths
            ),
        }

    def per_order(paths: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for path in paths:
            grouped[len(_path_signature(path))].append(path)
        output: dict[str, dict[str, float | int]] = {}
        for order, group in sorted(grouped.items()):
            summed = sum((contribution(path) for path in group), start=zeros.copy())
            output[str(order)] = {
                "path_count": len(group),
                "incoherent_path_power": float(
                    sum(np.mean(np.abs(contribution(path)) ** 2) for path in group)
                ),
                "coherent_component_power": float(np.mean(np.abs(summed) ** 2)),
            }
        return output

    low_power = float(np.mean(np.abs(low_cfr) ** 2))
    high_power = float(np.mean(np.abs(high_cfr) ** 2))
    low_incoherent = float(sum(np.mean(np.abs(contribution(path)) ** 2) for path in low_paths))
    high_incoherent = float(sum(np.mean(np.abs(contribution(path)) ** 2) for path in high_paths))
    return {
        "link_id": low["link_id"],
        "seed": int(low["seed"]),
        "max_depth": int(low["max_depth"]),
        "low_samples_per_src": int(low["samples_per_src"]),
        "high_samples_per_src": int(high["samples_per_src"]),
        "maximum_match_cost_m": maximum_match_cost_m,
        "candidate_paths_before_validation": {
            "low": int(low["candidate_path_count_before_path_validation"]),
            "high": int(high["candidate_path_count_before_path_validation"]),
            "cap_filled": bool(
                low["candidate_path_cap_filled"] or high["candidate_path_cap_filled"]
            ),
        },
        "path_counts": {
            "low": len(low_paths),
            "high": len(high_paths),
            "matched": len(matched),
            "discovered_at_high_budget": len(discovered),
            "present_only_at_low_budget": len(lost),
            "interaction_order_low": order_counts(low_paths),
            "interaction_order_high": order_counts(high_paths),
            "zero_interaction_path_records": {
                "low": zero_interaction_summary(low_paths),
                "high": zero_interaction_summary(high_paths),
            },
        },
        "matched_path_variation": {
            "symmetric_complex_coefficient_relative_error": _quantiles(coefficient_errors),
            "absolute_delay_error_ns": _quantiles(delay_errors_ns),
            "maximum_interaction_vertex_displacement_m": _quantiles(vertex_errors_m),
            "maximum_arrival_departure_direction_error_rad": _quantiles(direction_errors_rad),
        },
        "channel_change": {
            "relative_cfr_l2": float(delta_norm / max(float(np.linalg.norm(high_cfr)), 1e-300)),
            "low_cfr_mean_power": low_power,
            "high_cfr_mean_power": high_power,
            "cfr_mean_power_ratio_high_over_low": high_power / max(low_power, 1e-300),
            "low_incoherent_path_power": low_incoherent,
            "high_incoherent_path_power": high_incoherent,
            "coherent_to_incoherent_power_ratio": {
                "low": low_power / max(low_incoherent, 1e-300),
                "high": high_power / max(high_incoherent, 1e-300),
            },
            "path_order_contributions": {
                "low": per_order(low_paths),
                "high": per_order(high_paths),
            },
        },
        "cfr_delta_decomposition_norm_relative_to_total": {
            "matched_path_coefficient_geometry_change": float(
                np.linalg.norm(matched_delta) / max(delta_norm, 1e-300)
            ),
            "new_path_contributions": float(
                np.linalg.norm(discovery_term) / max(delta_norm, 1e-300)
            ),
            "lost_path_contributions": float(np.linalg.norm(lost_term) / max(delta_norm, 1e-300)),
            "additive_closure_error": float(closure_error / max(delta_norm, 1e-300)),
        },
        "interpretation_limit": (
            "A matched finite-budget diagnostic; the high-budget result is not ground truth, "
            "and path matching is not a universal path identity."
        ),
    }


def _read_report(path: Path) -> dict[str, Any]:
    payload = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    result = json.loads(payload)
    if not isinstance(result, dict) or not isinstance(result.get("runs"), list):
        raise ValueError("input must be a convergence report containing runs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--link-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-depth", required=True, type=int)
    parser.add_argument("--low-samples", required=True, type=int)
    parser.add_argument("--high-samples", required=True, type=int)
    parser.add_argument("--maximum-match-cost-m", type=float, default=0.02)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    report = _read_report(args.report)
    selected = [
        row
        for row in report["runs"]
        if row["link_id"] == args.link_id
        and row["seed"] == args.seed
        and row["max_depth"] == args.max_depth
        and row["samples_per_src"] in (args.low_samples, args.high_samples)
    ]
    by_budget = {int(row["samples_per_src"]): row for row in selected}
    if set(by_budget) != {args.low_samples, args.high_samples}:
        parser.error("report does not contain both requested matched budget rows")
    result = {
        "schema": "loslab.rt_path_budget_attribution.v1",
        "source_report": str(args.report),
        "source_report_revision": report.get("source_revision"),
        "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "analysis_script_sha256": sha256_file(__file__),
        "package_source_sha256": package_source_sha256(),
        "versions": versions(),
        "comparison": analyze_pair(
            by_budget[args.low_samples],
            by_budget[args.high_samples],
            maximum_match_cost_m=args.maximum_match_cost_m,
        ),
    }
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
