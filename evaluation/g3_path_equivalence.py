"""Build and compare bounded effective path-equivalence classes for G3 reviews.

This module compares deterministic/specular path records between two solver
configurations. It preserves raw multiplicity, treats primitive IDs as
descriptive only, and requires run-specific geometric resolutions. Diffuse
stochastic records must be analyzed as distributions, not assigned identities.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from numbers import Real
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from evaluation.g3_complex_csi import symmetric_complex_csi_error

_C_M_S = 299_792_458.0
_RELEVANT_E_H_LIMIT = 0.01
_RELEVANT_POWER_OVER_NOISE_DB = 6.0
_CLASS_COVERAGE_LIMIT = 0.95
_DIFFUSE_DISTRIBUTION_QUANTILE_LEVELS = (0.05, 0.5, 0.95)
_SOLVER_BUDGET_FIELDS = {
    "max_rays_per_link",
    "max_interactions",
    "max_episodes",
    "max_wall_seconds",
    "overflow_policy",
}


def _finite_real_array(
    value: Any, name: str, *, allow_empty: bool = False
) -> NDArray[np.float64]:
    """Parse finite real values without coercing booleans, strings, or complex data."""
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular real-valued array") from error
    if raw.size == 0 and not allow_empty:
        raise ValueError(f"{name} must be nonempty")
    flattened: list[float] = []
    for item in raw.flat:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
            raise ValueError(f"{name} must contain only real numeric values")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite values")
        flattened.append(number)
    if raw.size == 0:
        return np.empty(raw.shape, dtype=np.float64)
    return np.asarray(flattened, dtype=np.float64).reshape(raw.shape)


def _vector(record: dict[str, Any], real_key: str, imag_key: str) -> NDArray[np.complex128]:
    real = _finite_real_array(record.get(real_key), real_key)
    imag = _finite_real_array(record.get(imag_key), imag_key)
    if real.ndim != 1 or real.size == 0 or real.shape != imag.shape:
        raise ValueError(f"{real_key}/{imag_key} must be equal-length nonempty vectors")
    result = real + 1j * imag
    return result


def _required_tolerance(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _interaction_signature(path: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    masks = path.get("interaction_type_masks")
    objects = path.get("object_ids")
    primitives = path.get("primitive_ids")
    if (
        not isinstance(masks, list)
        or not isinstance(objects, list)
        or not isinstance(primitives, list)
    ):
        raise ValueError("path interaction masks, object IDs, and primitive IDs must be lists")
    if len(masks) != len(objects) or len(masks) != len(primitives):
        raise ValueError("path interaction arrays must have equal lengths")
    signature: list[tuple[int, int]] = []
    for mask, object_id, primitive_id in zip(masks, objects, primitives, strict=True):
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (mask, object_id, primitive_id)
        ):
            raise ValueError("path interaction IDs and masks must be integers")
        if mask < 0:
            raise ValueError("path interaction masks must be nonnegative")
        if mask != 0:
            signature.append((mask, object_id))
    return tuple(signature)


def _vertices(path: dict[str, Any], signature: tuple[tuple[int, int], ...]) -> NDArray[np.float64]:
    values = _finite_real_array(
        path.get("interaction_vertices_m"), "interaction_vertices_m", allow_empty=True
    )
    masks = path["interaction_type_masks"]
    if values.shape != (len(masks), 3):
        raise ValueError("interaction vertices must have shape (interaction_slots, 3)")
    selected = values[np.asarray(masks, dtype=np.int64) != 0]
    if selected.shape != (len(signature), 3):
        raise ValueError("interacting path vertices must be finite and match the signature")
    return selected


def _unit_direction(path: dict[str, Any], endpoint: str) -> NDArray[np.float64]:
    theta = path.get(f"theta_{endpoint}_rad")
    phi = path.get(f"phi_{endpoint}_rad")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in (theta, phi)
    ):
        raise ValueError(f"theta/phi for endpoint {endpoint} must be numeric")
    theta_value, phi_value = float(cast(float, theta)), float(cast(float, phi))
    if not math.isfinite(theta_value) or not math.isfinite(phi_value):
        raise ValueError(f"theta/phi for endpoint {endpoint} must be finite")
    return np.asarray(
        [
            math.sin(theta_value) * math.cos(phi_value),
            math.sin(theta_value) * math.sin(phi_value),
            math.cos(theta_value),
        ],
        dtype=np.float64,
    )


def _direction_error_rad(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    dot = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return math.atan2(float(np.linalg.norm(np.cross(left, right))), dot)


def _path_metadata(path: dict[str, Any]) -> dict[str, Any]:
    if path.get("valid") is not True:
        raise ValueError("only solver-valid path records may enter equivalence classes")
    path_kind = path.get("path_kind")
    if not isinstance(path_kind, str) or path_kind not in {"specular", "deterministic"}:
        raise ValueError(
            "diffuse stochastic paths must be compared as distributions, not identities"
        )
    record_id = path.get("path_index")
    if isinstance(record_id, bool) or not isinstance(record_id, int) or record_id < 0:
        raise ValueError("each valid path needs a nonnegative integer path_index")
    signature = _interaction_signature(path)
    delay_s = path.get("absolute_delay_s")
    if isinstance(delay_s, bool) or not isinstance(delay_s, (int, float)):
        raise ValueError("absolute_delay_s must be finite and nonnegative")
    delay_value = float(cast(float, delay_s))
    if not math.isfinite(delay_value) or delay_value < 0.0:
        raise ValueError("absolute_delay_s must be finite and nonnegative")
    coefficient = path.get("coefficient_real"), path.get("coefficient_imag")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in coefficient):
        raise ValueError("complex path coefficients must have finite real and imaginary parts")
    coefficient_value = complex(
        float(cast(float, coefficient[0])), float(cast(float, coefficient[1]))
    )
    if not math.isfinite(coefficient_value.real) or not math.isfinite(coefficient_value.imag):
        raise ValueError("complex path coefficients must have finite real and imaginary parts")
    return {
        "record_id": record_id,
        "signature": signature,
        "vertices": _vertices(path, signature),
        "delay_s": delay_value,
        "departure": _unit_direction(path, "t"),
        "arrival": _unit_direction(path, "r"),
        "coefficient": coefficient_value,
        "cfr": _vector(path, "cfr_contribution_real", "cfr_contribution_imag"),
        "primitive_ids": [
            int(value)
            for mask, value in zip(
                path["interaction_type_masks"], path["primitive_ids"], strict=True
            )
            if int(mask) != 0
        ],
    }


def _compatible(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    vertex_tolerance_m: float,
    delay_tolerance_s: float,
    direction_tolerance_rad: float,
) -> bool:
    if left["signature"] != right["signature"]:
        return False
    if left["vertices"].shape != right["vertices"].shape:
        return False
    vertex_delta = (
        float(np.max(np.linalg.norm(left["vertices"] - right["vertices"], axis=1)))
        if len(left["vertices"])
        else 0.0
    )
    if vertex_delta > vertex_tolerance_m:
        return False
    if abs(left["delay_s"] - right["delay_s"]) > delay_tolerance_s:
        return False
    return (
        max(
            _direction_error_rad(left["departure"], right["departure"]),
            _direction_error_rad(left["arrival"], right["arrival"]),
        )
        <= direction_tolerance_rad
    )


def build_path_equivalence_classes(
    records: Sequence[dict[str, Any]],
    *,
    resolution_id: str,
    vertex_tolerance_m: float,
    delay_tolerance_s: float,
    direction_tolerance_rad: float,
) -> dict[str, Any]:
    """Group solver-valid deterministic paths against frozen representative tolerances.

    A deterministic greedy representative rule prevents non-transitive chaining:
    every member must be within tolerance of its class representative. Primitive
    IDs are retained as diagnostics but are excluded from the identity key.
    """
    tolerances = {
        "vertex_tolerance_m": _required_tolerance(vertex_tolerance_m, "vertex_tolerance_m"),
        "delay_tolerance_s": _required_tolerance(delay_tolerance_s, "delay_tolerance_s"),
        "direction_tolerance_rad": _required_tolerance(
            direction_tolerance_rad, "direction_tolerance_rad"
        ),
    }
    if tolerances["direction_tolerance_rad"] > math.pi:
        raise ValueError("direction_tolerance_rad must not exceed pi")
    if (
        not isinstance(resolution_id, str)
        or not resolution_id
        or resolution_id != resolution_id.strip()
    ):
        raise ValueError("resolution_id must be a nonempty canonical string")
    valid_records: list[dict[str, Any]] = []
    invalid_count = 0
    ids: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("every path record must be an object")
        valid = record.get("valid")
        if not isinstance(valid, bool):
            raise ValueError("every path record must explicitly declare a boolean valid mask")
        if not valid:
            invalid_count += 1
            continue
        parsed = _path_metadata(record)
        if parsed["record_id"] in ids:
            raise ValueError(f"duplicate path_index: {parsed['record_id']}")
        ids.add(parsed["record_id"])
        valid_records.append(parsed)
    if valid_records:
        widths = {len(path["cfr"]) for path in valid_records}
        if len(widths) != 1:
            raise ValueError("all path CFR contributions must share the same frequency grid")

    valid_records.sort(key=lambda path: (path["signature"], path["record_id"]))
    grouped: dict[tuple[tuple[int, int], ...], list[list[dict[str, Any]]]] = defaultdict(list)
    for path in valid_records:
        matches = [
            group
            for group in grouped[path["signature"]]
            if _compatible(
                group[0],
                path,
                vertex_tolerance_m=tolerances["vertex_tolerance_m"],
                delay_tolerance_s=tolerances["delay_tolerance_s"],
                direction_tolerance_rad=tolerances["direction_tolerance_rad"],
            )
        ]
        if matches:
            matches[0].append(path)
        else:
            grouped[path["signature"]].append([path])

    classes: list[dict[str, Any]] = []
    for signature in sorted(grouped):
        for members in grouped[signature]:
            powers = np.asarray([abs(member["coefficient"]) ** 2 for member in members])
            weights = powers if float(powers.sum()) > 0.0 else np.ones(len(members))
            delays = np.asarray([member["delay_s"] for member in members], dtype=np.float64)
            cfr = sum(
                (member["cfr"] for member in members),
                start=np.zeros_like(members[0]["cfr"]),
            )
            aggregate_coefficient = sum(
                (member["coefficient"] for member in members), start=0j
            )
            if not math.isfinite(aggregate_coefficient.real) or not math.isfinite(
                aggregate_coefficient.imag
            ):
                raise ValueError("aggregate path-class coefficient must be finite")

            def weighted_direction(field: str) -> NDArray[np.float64]:
                vector = np.average(
                    np.stack([member[field] for member in members]), axis=0, weights=weights
                )
                norm = float(np.linalg.norm(vector))
                if norm <= np.finfo(np.float64).tiny:
                    return members[0][field].copy()
                return vector / norm

            classes.append(
                {
                    "class_index": len(classes),
                    "interaction_signature": [list(item) for item in signature],
                    "representative_path_index": members[0]["record_id"],
                    "raw_path_indices": [member["record_id"] for member in members],
                    "raw_multiplicity": len(members),
                    "primitive_ids_descriptive_only": members[0]["primitive_ids"],
                    "interaction_vertices_m": members[0]["vertices"].tolist(),
                    "aggregate_coefficient_real": float(aggregate_coefficient.real),
                    "aggregate_coefficient_imag": float(aggregate_coefficient.imag),
                    "aggregate_cfr_real": cfr.real.tolist(),
                    "aggregate_cfr_imag": cfr.imag.tolist(),
                    "power_weighted_delay_s": float(np.average(delays, weights=weights)),
                    "power_weighted_departure_direction": weighted_direction("departure").tolist(),
                    "power_weighted_arrival_direction": weighted_direction("arrival").tolist(),
                }
            )
    return {
        "status": "DIAGNOSTIC_CLASSES_NOT_PHYSICAL_IDENTITIES",
        "raw_record_count": len(records),
        "solver_valid_record_count": len(valid_records),
        "excluded_invalid_record_count": invalid_count,
        "class_count": len(classes),
        "resolution_id": resolution_id,
        "resolution": tolerances,
        "grouping_rule": (
            "ordered interaction type/object signature plus representative-bounded vertex, "
            "absolute-delay, and endpoint-direction tolerances; primitive IDs are descriptive only"
        ),
        "classes": classes,
    }


def _class_view(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("path class record must be an object")
    real = _finite_real_array(item.get("aggregate_cfr_real"), "aggregate_cfr_real")
    imag = _finite_real_array(item.get("aggregate_cfr_imag"), "aggregate_cfr_imag")
    if real.ndim != 1 or imag.shape != real.shape:
        raise ValueError("path class CFR components must be equal-length nonempty vectors")
    cfr = real + 1j * imag
    coefficient_real = item.get("aggregate_coefficient_real")
    coefficient_imag = item.get("aggregate_coefficient_imag")
    if any(
        isinstance(value, bool) or not isinstance(value, Real)
        for value in (coefficient_real, coefficient_imag)
    ):
        raise ValueError("path class aggregate coefficient must have finite real components")
    aggregate_coefficient = complex(float(coefficient_real), float(coefficient_imag))
    if not math.isfinite(aggregate_coefficient.real) or not math.isfinite(
        aggregate_coefficient.imag
    ):
        raise ValueError("path class aggregate coefficient must have finite real components")
    vertices = _finite_real_array(
        item.get("interaction_vertices_m"), "interaction_vertices_m", allow_empty=True
    )
    departure = _finite_real_array(
        item.get("power_weighted_departure_direction"), "power_weighted_departure_direction"
    )
    arrival = _finite_real_array(
        item.get("power_weighted_arrival_direction"), "power_weighted_arrival_direction"
    )
    if (
        cfr.ndim != 1
        or cfr.size == 0
        or vertices.ndim != 2
        or vertices.shape[1] != 3
        or departure.shape != (3,)
        or arrival.shape != (3,)
    ):
        raise ValueError("path class contains invalid CFR, geometry, or endpoint directions")
    raw_signature = item.get("interaction_signature")
    if not isinstance(raw_signature, list):
        raise ValueError("path class interaction signature must be a list")
    signature: list[tuple[int, int]] = []
    for pair in raw_signature:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in pair)
            or pair[0] <= 0
        ):
            raise ValueError("path class interaction signature is malformed")
        signature.append((pair[0], pair[1]))
    if vertices.shape != (len(signature), 3):
        raise ValueError("path class vertices do not match its interaction signature")
    delay = item.get("power_weighted_delay_s")
    if (
        isinstance(delay, bool)
        or not isinstance(delay, Real)
        or not math.isfinite(float(delay))
        or delay < 0.0
    ):
        raise ValueError("path class delay must be finite and nonnegative")
    for direction in (departure, arrival):
        if not math.isclose(float(np.linalg.norm(direction)), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("path class endpoint directions must be unit vectors")
    return {
        **item,
        "cfr": cfr,
        "signature": tuple(signature),
        "vertices": vertices,
        "delay_s": float(delay),
        "departure": departure,
        "arrival": arrival,
    }


def compare_path_equivalence_classes(
    candidate_classes: dict[str, Any],
    challenger_classes: dict[str, Any],
    *,
    vertex_tolerance_m: float,
    delay_tolerance_s: float,
    direction_tolerance_rad: float,
    candidate_full_cfr: NDArray[np.complexfloating[Any, Any]],
    challenger_full_cfr: NDArray[np.complexfloating[Any, Any]],
    h_floor: float,
    transmit_power_per_re_w: float,
    noise_power_per_re_w: float,
    coefficient_relative_tolerance: float | None = None,
    resolution_id: str,
    candidate_configuration_id: str,
    challenger_configuration_id: str,
) -> dict[str, Any]:
    """Match classes one-to-one and report receiver-relevant bidirectional coverage."""
    if not isinstance(candidate_classes, dict) or not isinstance(challenger_classes, dict):
        raise ValueError("candidate and challenger class bundles must be objects")
    vertex_tol = _required_tolerance(vertex_tolerance_m, "vertex_tolerance_m")
    delay_tol = _required_tolerance(delay_tolerance_s, "delay_tolerance_s")
    direction_tol = _required_tolerance(direction_tolerance_rad, "direction_tolerance_rad")
    if direction_tol > math.pi:
        raise ValueError("direction_tolerance_rad must not exceed pi")
    tx_power = _required_tolerance(transmit_power_per_re_w, "transmit_power_per_re_w")
    noise_power = _required_tolerance(noise_power_per_re_w, "noise_power_per_re_w")
    noise_floor = _required_tolerance(h_floor, "h_floor")
    coefficient_tol: float | None = None
    if coefficient_relative_tolerance is not None:
        if (
            isinstance(coefficient_relative_tolerance, bool)
            or not isinstance(coefficient_relative_tolerance, Real)
            or not math.isfinite(float(coefficient_relative_tolerance))
            or coefficient_relative_tolerance < 0.0
        ):
            raise ValueError("coefficient_relative_tolerance must be finite and nonnegative")
        coefficient_tol = float(coefficient_relative_tolerance)
    for value, name in (
        (resolution_id, "resolution_id"),
        (candidate_configuration_id, "candidate_configuration_id"),
        (challenger_configuration_id, "challenger_configuration_id"),
    ):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{name} must be a nonempty canonical string")
    left_full = np.asarray(candidate_full_cfr)
    right_full = np.asarray(challenger_full_cfr)
    if (
        left_full.ndim != 1
        or right_full.shape != left_full.shape
        or not np.iscomplexobj(left_full)
        or not np.iscomplexobj(right_full)
        or not np.isfinite(left_full).all()
        or not np.isfinite(right_full).all()
    ):
        raise ValueError("full CFRs must be finite complex vectors on the same active grid")
    left_roster = candidate_classes.get("classes")
    right_roster = challenger_classes.get("classes")
    if not isinstance(left_roster, list) or not isinstance(right_roster, list):
        raise ValueError("candidate and challenger class rosters must be lists")
    left = [_class_view(item) for item in left_roster]
    right = [_class_view(item) for item in right_roster]
    if any(item["cfr"].shape != left_full.shape for item in left + right):
        raise ValueError("every class CFR contribution must match the declared full CFR grid")
    if not left and not right:
        return {
            "schema": "loslab.g3_path_equivalence_comparison.v1",
            "status": "NO_CLASSES_NOT_ASSESSABLE",
            "admission_possible": False,
            "class_coverage_status": "NOT_ASSESSABLE_NO_CLASSES",
            "coverage_criterion": _CLASS_COVERAGE_LIMIT,
            "matched_classes": [],
            "candidate_only_relevant_classes": [],
            "challenger_only_relevant_classes": [],
            "configuration": {
                "resolution_id": resolution_id,
                "candidate_configuration_id": candidate_configuration_id,
                "challenger_configuration_id": challenger_configuration_id,
                "vertex_tolerance_m": vertex_tol,
                "delay_tolerance_s": delay_tol,
                "direction_tolerance_rad": direction_tol,
                "h_floor": noise_floor,
                "transmit_power_per_re_w": tx_power,
                "noise_power_per_re_w": noise_power,
                **(
                    {"coefficient_relative_tolerance": coefficient_tol}
                    if coefficient_tol is not None
                    else {}
                ),
                "matching": "one-to-one Hungarian assignment over eligible physical/effective classes",
                "primitive_ids_are_identity": False,
            },
        }

    expected_resolution = {
        "vertex_tolerance_m": vertex_tol,
        "delay_tolerance_s": delay_tol,
        "direction_tolerance_rad": direction_tol,
    }
    for label, bundle in (("candidate", candidate_classes), ("challenger", challenger_classes)):
        if bundle.get("status") != "DIAGNOSTIC_CLASSES_NOT_PHYSICAL_IDENTITIES":
            raise ValueError(f"{label} classes must come from the path-equivalence class builder")
        if bundle.get("resolution_id") != resolution_id:
            raise ValueError(f"{label} class resolution_id does not match the frozen comparison")
        if bundle.get("resolution") != expected_resolution:
            raise ValueError(f"{label} class resolutions do not match the frozen comparison")

    def relevant(classes: list[dict[str, Any]], full: NDArray[Any]) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for index, item in enumerate(classes):
            contribution = item["cfr"]
            omitted_error = symmetric_complex_csi_error(
                full, full - contribution, h_floor=noise_floor
            )
            received_power = tx_power * float(np.mean(np.abs(contribution) ** 2))
            power_ratio_db = 10.0 * math.log10(
                max(received_power, np.finfo(np.float64).tiny) / noise_power
            )
            by_csi = omitted_error >= _RELEVANT_E_H_LIMIT
            by_power = power_ratio_db >= _RELEVANT_POWER_OVER_NOISE_DB
            result[index] = {
                "receiver_relevant": by_csi or by_power,
                "isolated_omission_e_h": omitted_error,
                "post_link_budget_class_power_w_per_re": received_power,
                "class_to_noise_ratio_db": power_ratio_db,
                "relevance_reason": [
                    reason
                    for condition, reason in (
                        (by_csi, "isolated omission changes active-grid e_H by >=0.01"),
                        (by_power, "class power is >=6 dB above equivalent receiver noise"),
                    )
                    if condition
                ],
            }
        return result

    left_relevance = relevant(left, left_full)
    right_relevance = relevant(right, right_full)
    costs = np.full((len(left), len(right)), np.inf, dtype=np.float64)
    details: dict[tuple[int, int], dict[str, float]] = {}
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if not _compatible(
                a,
                b,
                vertex_tolerance_m=vertex_tol,
                delay_tolerance_s=delay_tol,
                direction_tolerance_rad=direction_tol,
            ):
                continue
            vertex_delta = (
                float(np.max(np.linalg.norm(a["vertices"] - b["vertices"], axis=1)))
                if len(a["vertices"])
                else 0.0
            )
            delay_delta = abs(a["delay_s"] - b["delay_s"])
            direction_delta = max(
                _direction_error_rad(a["departure"], b["departure"]),
                _direction_error_rad(a["arrival"], b["arrival"]),
            )
            cost = (
                vertex_delta / vertex_tol
                + delay_delta / delay_tol
                + direction_delta / direction_tol
            )
            costs[i, j] = cost
            details[(i, j)] = {
                "maximum_interaction_vertex_displacement_m": vertex_delta,
                "absolute_delay_difference_s": delay_delta,
                "maximum_endpoint_direction_difference_rad": direction_delta,
                "aggregate_class_cfr_e_h": symmetric_complex_csi_error(
                    a["cfr"], b["cfr"], h_floor=noise_floor
                ),
                "aggregate_coefficient_relative_error": abs(
                    complex(
                        float(a["aggregate_coefficient_real"]),
                        float(a["aggregate_coefficient_imag"]),
                    )
                    - complex(
                        float(b["aggregate_coefficient_real"]),
                        float(b["aggregate_coefficient_imag"]),
                    )
                )
                / max(
                    abs(
                        complex(
                            float(a["aggregate_coefficient_real"]),
                            float(a["aggregate_coefficient_imag"]),
                        )
                    ),
                    abs(
                        complex(
                            float(b["aggregate_coefficient_real"]),
                            float(b["aggregate_coefficient_imag"]),
                        )
                    ),
                    np.finfo(np.float64).tiny,
                ),
            }

    matched: list[tuple[int, int]] = []
    if len(left) and len(right):
        # Dummy assignments cost 10, while any eligible real match costs <=3;
        # invalid pairs remain more expensive than leaving both sides unmatched.
        size = len(left) + len(right)
        augmented = np.full((size, size), 1e6, dtype=np.float64)
        augmented[: len(left), : len(right)] = np.where(np.isfinite(costs), costs, 1e6)
        for i in range(len(left)):
            augmented[i, len(right) + i] = 10.0
        for j in range(len(right)):
            augmented[len(left) + j, j] = 10.0
        augmented[len(left) :, len(right) :] = 0.0
        rows, columns = linear_sum_assignment(augmented)
        matched = [
            (int(i), int(j))
            for i, j in zip(rows, columns, strict=True)
            if i < len(left) and j < len(right) and np.isfinite(costs[i, j])
        ]

    matched_left = {i for i, _ in matched}
    matched_right = {j for _, j in matched}
    left_relevant = {i for i, item in left_relevance.items() if item["receiver_relevant"]}
    right_relevant = {j for j, item in right_relevance.items() if item["receiver_relevant"]}
    left_coverage = (
        len(left_relevant & matched_left) / len(left_relevant) if left_relevant else None
    )
    right_coverage = (
        len(right_relevant & matched_right) / len(right_relevant) if right_relevant else None
    )
    # A relevant class present on only one side is assessable and must fail
    # bidirectional coverage. Treating this as unresolved would mask the exact
    # case where one configuration drops every receiver-relevant path.
    assessable = bool(left_relevant or right_relevant)
    coverage_pass = bool(
        assessable
        and left_coverage is not None
        and right_coverage is not None
        and left_coverage >= _CLASS_COVERAGE_LIMIT
        and right_coverage >= _CLASS_COVERAGE_LIMIT
    )
    coefficient_failures = 0
    coefficient_assessable = 0
    if coefficient_tol is not None:
        for i, j in matched:
            if left_relevance[i]["receiver_relevant"] or right_relevance[j]["receiver_relevant"]:
                coefficient_assessable += 1
                coefficient_failures += int(
                    details[(i, j)]["aggregate_coefficient_relative_error"] > coefficient_tol
                )
    coefficient_pass = coefficient_failures == 0
    numerical_pass = coverage_pass and coefficient_pass
    matched_reports = []
    for i, j in matched:
        matched_reports.append(
            {
                "candidate_class_index": left[i]["class_index"],
                "candidate_raw_path_indices": left[i]["raw_path_indices"],
                "candidate_raw_multiplicity": left[i]["raw_multiplicity"],
                "candidate_receiver_relevance": left_relevance[i],
                "challenger_class_index": right[j]["class_index"],
                "challenger_raw_path_indices": right[j]["raw_path_indices"],
                "challenger_raw_multiplicity": right[j]["raw_multiplicity"],
                "challenger_receiver_relevance": right_relevance[j],
                "geometry_and_channel_deltas": details[(i, j)],
                **(
                    {
                        "coefficient_criterion_pass": (
                            details[(i, j)]["aggregate_coefficient_relative_error"]
                            <= coefficient_tol
                        )
                        if left_relevance[i]["receiver_relevant"]
                        or right_relevance[j]["receiver_relevant"]
                        else None
                    }
                    if coefficient_tol is not None
                    else {}
                ),
            }
        )

    def unmatched(
        indices: set[int],
        matched_indices: set[int],
        relevance: dict[int, Any],
        classes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "class_index": classes[index]["class_index"],
                "raw_path_indices": classes[index]["raw_path_indices"],
                "raw_multiplicity": classes[index]["raw_multiplicity"],
                "receiver_relevance": relevance[index],
            }
            for index in sorted(indices - matched_indices)
            if relevance[index]["receiver_relevant"]
        ]

    return {
        "schema": "loslab.g3_path_equivalence_comparison.v1",
        "status": "PATH_COVERAGE_DIAGNOSTIC_NOT_QUALIFIED",
        "admission_possible": False,
        "class_coverage_status": "PASS"
        if numerical_pass
        else "FAIL"
        if assessable
        else "UNRESOLVED_NO_RELEVANT_CLASSES",
        "coverage_criterion": _CLASS_COVERAGE_LIMIT,
        **(
            {
                "coefficient_relative_tolerance": coefficient_tol,
                "coefficient_criterion_status": (
                    "PASS"
                    if coefficient_assessable and coefficient_pass
                    else "FAIL"
                    if coefficient_failures
                    else "NOT_ASSESSABLE_NO_RELEVANT_MATCHED_CLASSES"
                ),
                "coefficient_criterion_assessable_class_pairs": coefficient_assessable,
                "coefficient_criterion_failed_class_pairs": coefficient_failures,
            }
            if coefficient_tol is not None
            else {}
        ),
        "candidate_relevant_class_count": len(left_relevant),
        "challenger_relevant_class_count": len(right_relevant),
        "candidate_bidirectional_coverage": left_coverage,
        "challenger_bidirectional_coverage": right_coverage,
        "matched_class_pair_count": len(matched),
        "matched_classes": matched_reports,
        "candidate_only_relevant_classes": unmatched(
            set(range(len(left))), matched_left, left_relevance, left
        ),
        "challenger_only_relevant_classes": unmatched(
            set(range(len(right))), matched_right, right_relevance, right
        ),
        "configuration": {
            "resolution_id": resolution_id,
            "candidate_configuration_id": candidate_configuration_id,
            "challenger_configuration_id": challenger_configuration_id,
            "vertex_tolerance_m": vertex_tol,
            "delay_tolerance_s": delay_tol,
            "direction_tolerance_rad": direction_tol,
            "h_floor": noise_floor,
            "transmit_power_per_re_w": tx_power,
            "noise_power_per_re_w": noise_power,
            "matching": "one-to-one Hungarian assignment over eligible physical/effective classes",
            "primitive_ids_are_identity": False,
        },
        "limitations": [
            "Resolutions are caller-supplied run-manifest values and must be frozen before qualification.",
            "This single comparison does not provide independent-scene confidence intervals or population evidence.",
            (
                "Receiver-relevant matched class coefficients are checked against the frozen "
                "symmetric relative-error tolerance."
                if coefficient_tol is not None
                else "Aggregate class coefficient and delay/direction deltas are diagnostic; "
                "no per-class coefficient pass margin is applied."
            ),
            "Diffuse stochastic rays are outside this identity matcher and require distributional comparison.",
            "A passing class-coverage result cannot qualify propagation, G5 receiver validity, or production RT generation.",
        ],
    }


def _diffuse_path_power(path: dict[str, Any]) -> float:
    real = path.get("coefficient_real")
    imag = path.get("coefficient_imag")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in (real, imag)):
        raise ValueError("diffuse path coefficients must be finite real and imaginary values")
    real_value, imag_value = float(real), float(imag)
    if not math.isfinite(real_value) or not math.isfinite(imag_value):
        raise ValueError("diffuse path coefficients must be finite real and imaginary values")
    power = real_value * real_value + imag_value * imag_value
    if not math.isfinite(power):
        raise ValueError("diffuse path power must be finite")
    return power


def _diffuse_quantiles(
    values: NDArray[np.float64], weights: NDArray[np.float64] | None
) -> dict[str, float] | None:
    if values.size == 0 or (weights is not None and float(weights.sum()) <= 0.0):
        return None
    levels = np.asarray(_DIFFUSE_DISTRIBUTION_QUANTILE_LEVELS, dtype=np.float64)
    if weights is None:
        result = np.quantile(values, levels)
    else:
        order = np.argsort(values, kind="stable")
        ordered_values = values[order]
        ordered_weights = weights[order]
        mid_cdf = (np.cumsum(ordered_weights) - 0.5 * ordered_weights) / float(
            ordered_weights.sum()
        )
        result = np.interp(levels, mid_cdf, ordered_values)
    return {
        f"q{int(level * 100):02d}": float(value)
        for level, value in zip(_DIFFUSE_DISTRIBUTION_QUANTILE_LEVELS, result, strict=True)
    }


def summarize_diffuse_stochastic_paths(paths: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a stochastic diffuse path set without assigning ray identities.

    The report is descriptive only. Quantile levels are fixed and reported so
    candidate/challenger summaries are comparable; no acceptance thresholds or
    population inference are implied.
    """
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise ValueError("diffuse paths must be a sequence of path records")
    delays: list[float] = []
    tx_zenith: list[float] = []
    rx_zenith: list[float] = []
    tx_azimuth: list[float] = []
    rx_azimuth: list[float] = []
    powers: list[float] = []
    for path in paths:
        if not isinstance(path, dict) or path.get("valid") is not True:
            raise ValueError("diffuse distribution records must be valid path objects")
        if path.get("path_kind") != "diffuse_stochastic":
            raise ValueError("distribution summary accepts only diffuse_stochastic paths")
        delay = path.get("absolute_delay_s")
        if isinstance(delay, bool) or not isinstance(delay, Real):
            raise ValueError("diffuse path delay must be finite and nonnegative")
        delay_value = float(delay)
        if not math.isfinite(delay_value) or delay_value < 0.0:
            raise ValueError("diffuse path delay must be finite and nonnegative")
        delays.append(delay_value)
        powers.append(_diffuse_path_power(path))
        for endpoint, zenith_values, azimuth_values in (
            ("t", tx_zenith, tx_azimuth),
            ("r", rx_zenith, rx_azimuth),
        ):
            theta = path.get(f"theta_{endpoint}_rad")
            phi = path.get(f"phi_{endpoint}_rad")
            if any(
                isinstance(value, bool) or not isinstance(value, Real)
                for value in (theta, phi)
            ):
                raise ValueError("diffuse path directions must be finite real angles")
            theta_value, phi_value = float(theta), float(phi)
            if (
                not math.isfinite(theta_value)
                or not 0.0 <= theta_value <= math.pi
                or not math.isfinite(phi_value)
            ):
                raise ValueError("diffuse path directions must use finite zenith and azimuth angles")
            zenith_values.append(theta_value)
            azimuth_values.append((phi_value + math.pi) % (2.0 * math.pi) - math.pi)

    delay_array = np.asarray(delays, dtype=np.float64)
    power_array = np.asarray(powers, dtype=np.float64)
    tx_zenith_array = np.asarray(tx_zenith, dtype=np.float64)
    rx_zenith_array = np.asarray(rx_zenith, dtype=np.float64)
    tx_azimuth_array = np.asarray(tx_azimuth, dtype=np.float64)
    rx_azimuth_array = np.asarray(rx_azimuth, dtype=np.float64)
    total_power = float(power_array.sum())
    if not math.isfinite(total_power):
        raise ValueError("diffuse path coefficient power sum must be finite")
    angular_weights = power_array if total_power > 0.0 else np.ones_like(power_array)

    def azimuth_summary(
        values: NDArray[np.float64],
    ) -> tuple[float | None, NDArray[np.float64] | None]:
        if values.size == 0:
            return None, None
        sine = float(np.dot(angular_weights, np.sin(values)))
        cosine = float(np.dot(angular_weights, np.cos(values)))
        weight_sum = float(angular_weights.sum())
        if math.hypot(sine, cosine) <= 1e-12 * weight_sum:
            return None, None
        center = math.atan2(sine, cosine)
        offsets = (values - center + math.pi) % (2.0 * math.pi) - math.pi
        return center, offsets

    tx_azimuth_center, tx_offsets = azimuth_summary(tx_azimuth_array)
    rx_azimuth_center, rx_offsets = azimuth_summary(rx_azimuth_array)
    unweighted = {
        "delay_s": _diffuse_quantiles(delay_array, None),
        "tx_zenith_rad": _diffuse_quantiles(tx_zenith_array, None),
        "rx_zenith_rad": _diffuse_quantiles(rx_zenith_array, None),
        "tx_azimuth_offset_rad": (
            _diffuse_quantiles(tx_offsets, None) if tx_offsets is not None else None
        ),
        "rx_azimuth_offset_rad": (
            _diffuse_quantiles(rx_offsets, None) if rx_offsets is not None else None
        ),
        "path_power": _diffuse_quantiles(power_array, None),
    }
    power_weighted = {
        "delay_s": _diffuse_quantiles(delay_array, power_array),
        "tx_zenith_rad": _diffuse_quantiles(tx_zenith_array, angular_weights),
        "rx_zenith_rad": _diffuse_quantiles(rx_zenith_array, angular_weights),
        "tx_azimuth_offset_rad": (
            _diffuse_quantiles(tx_offsets, angular_weights) if tx_offsets is not None else None
        ),
        "rx_azimuth_offset_rad": (
            _diffuse_quantiles(rx_offsets, angular_weights) if rx_offsets is not None else None
        ),
        "path_power": _diffuse_quantiles(power_array, power_array),
    }
    normalized_power = power_array / total_power if total_power > 0.0 else power_array
    return {
        "status": "DESCRIPTIVE_DISTRIBUTION" if paths else "NO_DIFFUSE_PATHS",
        "path_count": len(paths),
        "coefficient_power_sum": total_power,
        "normalized_coefficient_power_quantiles": _diffuse_quantiles(normalized_power, None),
        "unweighted_quantiles": unweighted,
        "coefficient_power_weighted_quantiles": power_weighted,
        "circular_mean_azimuth_rad": {
            "transmit": tx_azimuth_center,
            "receive": rx_azimuth_center,
        },
        "angular_quantile_weighting": (
            "coefficient_power"
            if total_power > 0.0
            else "equal_weight_fallback_no_coefficient_power"
        ),
        "azimuth_reference": "power-weighted circular mean; quantiles are wrapped offsets",
        "zero_total_power_uses_equal_weights_for_angular_summary": total_power == 0.0,
    }


def compare_diffuse_stochastic_path_distributions(
    candidate_paths: Sequence[dict[str, Any]],
    challenger_paths: Sequence[dict[str, Any]],
    *,
    solver_budget_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Return descriptive candidate/challenger summaries under frozen solver caps."""
    if not isinstance(solver_budget_provenance, dict) or set(solver_budget_provenance) != _SOLVER_BUDGET_FIELDS:
        raise ValueError("solver_budget_provenance must match the frozen five-field solver budget")
    normalized_budget: dict[str, Any] = {}
    for key in ("max_rays_per_link", "max_interactions", "max_episodes"):
        value = solver_budget_provenance[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"solver_budget_provenance.{key} must be a positive integer")
        normalized_budget[key] = value
    wall_seconds = solver_budget_provenance["max_wall_seconds"]
    if isinstance(wall_seconds, bool) or not isinstance(wall_seconds, Real):
        raise ValueError(
            "solver_budget_provenance.max_wall_seconds must be finite and at least 1e-6"
        )
    wall_seconds_value = float(wall_seconds)
    if not math.isfinite(wall_seconds_value) or wall_seconds_value < 0.000001:
        raise ValueError("solver_budget_provenance.max_wall_seconds must be finite and positive")
    overflow_policy = solver_budget_provenance["overflow_policy"]
    if not isinstance(overflow_policy, str) or overflow_policy not in {
        "abort_run",
        "record_censored_episode",
    }:
        raise ValueError("solver_budget_provenance.overflow_policy is unsupported")
    normalized_budget["max_wall_seconds"] = wall_seconds_value
    normalized_budget["overflow_policy"] = overflow_policy
    budget_sha256 = hashlib.sha256(
        json.dumps(normalized_budget, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    candidate = summarize_diffuse_stochastic_paths(candidate_paths)
    challenger = summarize_diffuse_stochastic_paths(challenger_paths)

    def quantile_deltas(
        summary_field: str, field: str
    ) -> dict[str, float] | None:
        left = candidate[summary_field][field]
        right = challenger[summary_field][field]
        return quantile_mapping_delta(left, right)

    def quantile_mapping_delta(
        left: dict[str, float] | None, right: dict[str, float] | None
    ) -> dict[str, float] | None:
        if left is None or right is None:
            return None
        return {name: left[name] - right[name] for name in left}

    def circular_delta(left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return (left - right + math.pi) % (2.0 * math.pi) - math.pi

    return {
        "schema": "loslab.g3_diffuse_path_distribution_diagnostic.v1",
        "status": "DESCRIPTIVE_ONLY_NOT_QUALIFIED",
        "admission_possible": False,
        "ray_identity_assignment": False,
        "quantile_levels": list(_DIFFUSE_DISTRIBUTION_QUANTILE_LEVELS),
        "candidate": candidate,
        "challenger": challenger,
        "candidate_minus_challenger_unweighted_quantile_deltas": {
            field: quantile_deltas("unweighted_quantiles", field)
            for field in candidate["unweighted_quantiles"]
        },
        "candidate_minus_challenger_power_weighted_quantile_deltas": {
            field: quantile_deltas("coefficient_power_weighted_quantiles", field)
            for field in candidate["coefficient_power_weighted_quantiles"]
        },
        "candidate_minus_challenger_normalized_power_quantile_deltas": quantile_mapping_delta(
            candidate["normalized_coefficient_power_quantiles"],
            challenger["normalized_coefficient_power_quantiles"],
        ),
        "candidate_minus_challenger_circular_mean_azimuth_delta_rad": {
            endpoint: circular_delta(
                candidate["circular_mean_azimuth_rad"][endpoint],
                challenger["circular_mean_azimuth_rad"][endpoint],
            )
            for endpoint in ("transmit", "receive")
        },
        "candidate_minus_challenger_coefficient_power_sum": (
            candidate["coefficient_power_sum"] - challenger["coefficient_power_sum"]
        ),
        "candidate_minus_challenger_total_coefficient_power": (
            candidate["coefficient_power_sum"] - challenger["coefficient_power_sum"]
        ),
        "solver_budget_provenance": normalized_budget,
        "solver_budget_sha256": budget_sha256,
        "finite_solver_caps_are_ground_truth": False,
        "limitations": [
            "Quantile summaries are descriptive and define no acceptance margin.",
            "The frozen solver budget is provenance, not evidence that finite-budget path discovery is complete.",
            "No independent-scene inference or physical qualification is provided.",
        ],
    }
