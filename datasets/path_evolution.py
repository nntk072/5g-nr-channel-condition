"""Bounded diagnostics for path association across retained time frames.

This module compares already-solved :class:`PathOracleTable` rows. It does not
retrace scenes, assert that a ray solver found every physical path, or assign a
stable identity from a path row's storage position. Association outcomes are
therefore diagnostics with explicit ambiguity and censoring states.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from .path_oracle import PathOracleTable, PathSolveStatus


@dataclass(frozen=True, slots=True)
class PathAssociation:
    """One accepted descriptor-based association; row indexes are provenance."""

    previous_path_row: int
    current_path_row: int
    association_cost: float
    evidence: str
    doppler_phase_residual_rad: float | None


@dataclass(frozen=True, slots=True)
class PathEvolutionTransition:
    """Comparison between two solved frames for one link and port pair."""

    link_id: str
    rx_port_index: int
    tx_port_index: int
    previous_frame_index: int
    current_frame_index: int
    elapsed_s: float
    status: str
    associations: tuple[PathAssociation, ...]
    births: tuple[int, ...]
    deaths: tuple[int, ...]
    ambiguous_previous_rows: tuple[int, ...]
    ambiguous_current_rows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PathEvolutionAudit:
    """Immutable result of a bounded path-evolution diagnostic."""

    schema: str
    source_path_table_sha256: str
    time_by_frame_s: tuple[tuple[int, float], ...]
    phase_gauge_by_frame: tuple[tuple[int, str], ...]
    transitions: tuple[PathEvolutionTransition, ...]
    candidate_pair_count: int
    maximum_candidate_pair_count: int
    max_delay_change_s: float
    max_angle_change_rad: float
    max_doppler_change_hz: float
    ambiguity_cost_margin: float
    interpretation: str


@dataclass(frozen=True, slots=True)
class _PathDescriptor:
    row: int
    delay_s: float
    gain: complex
    doppler_hz: float | None
    aoa: tuple[float, float] | None
    aod: tuple[float, float] | None
    interactions: tuple[tuple[int, int, int], ...] | None


def _path_table_digest(table: PathOracleTable) -> str:
    """Hash all path-table fields that can affect association or phase checks."""
    digest = hashlib.sha256()
    metadata = {
        "link_ids": table.link_ids,
        "frame_indices": table.frame_indices.tolist(),
        "rx_port_indices": table.rx_port_indices.tolist(),
        "tx_port_indices": table.tx_port_indices.tolist(),
        "path_offsets": table.path_offsets.tolist(),
        "solve_status": [item.value for item in table.solve_status],
        "path_kind": table.path_kind,
        "interaction_vertex_frame": table.interaction_vertex_frame,
        "coefficient_convention": table.coefficient_convention,
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for name in (
        "delay_s",
        "complex_gain",
        "interaction_order",
        "doppler_hz",
        "aoa_azimuth_elevation_rad",
        "aod_azimuth_elevation_rad",
        "interaction_offsets",
        "interaction_types",
        "object_ids",
        "primitive_ids",
        "interaction_vertices_m",
    ):
        value = getattr(table, name)
        digest.update(name.encode())
        if value is None:
            digest.update(b"absent")
            continue
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def path_evolution_audit_to_dict(audit: PathEvolutionAudit) -> dict[str, object]:
    """Return a canonical JSON-compatible report with a content digest."""
    if not isinstance(audit, PathEvolutionAudit):
        raise TypeError("audit must be a PathEvolutionAudit")
    report: dict[str, object] = {
        "schema": audit.schema,
        "source_path_table_sha256": audit.source_path_table_sha256,
        "time_by_frame_s": [
            {"frame_index": frame, "time_s": time_s} for frame, time_s in audit.time_by_frame_s
        ],
        "phase_gauge_by_frame": [
            {"frame_index": frame, "phase_gauge_id": gauge_id}
            for frame, gauge_id in audit.phase_gauge_by_frame
        ],
        "association_limits": {
            "max_delay_change_s": audit.max_delay_change_s,
            "max_angle_change_rad": audit.max_angle_change_rad,
            "max_doppler_change_hz": audit.max_doppler_change_hz,
            "ambiguity_cost_margin": audit.ambiguity_cost_margin,
            "maximum_candidate_pair_count": audit.maximum_candidate_pair_count,
        },
        "candidate_pair_count": audit.candidate_pair_count,
        "transitions": [asdict(item) for item in audit.transitions],
        "interpretation": audit.interpretation,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report = json.loads(canonical)
    report["report_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return report


def _descriptors(table: PathOracleTable, group_index: int) -> tuple[_PathDescriptor, ...]:
    start = int(table.path_offsets[group_index])
    stop = int(table.path_offsets[group_index + 1])
    result: list[_PathDescriptor] = []
    for row in range(start, stop):
        interactions = None
        if table.interaction_offsets is not None:
            left = int(table.interaction_offsets[row])
            right = int(table.interaction_offsets[row + 1])
            interactions = tuple(
                (
                    int(table.interaction_types[index]),
                    int(table.object_ids[index]),
                    int(table.primitive_ids[index]),
                )
                for index in range(left, right)
            )
        aoa = (
            None
            if table.aoa_azimuth_elevation_rad is None
            else tuple(float(value) for value in table.aoa_azimuth_elevation_rad[row])
        )
        aod = (
            None
            if table.aod_azimuth_elevation_rad is None
            else tuple(float(value) for value in table.aod_azimuth_elevation_rad[row])
        )
        result.append(
            _PathDescriptor(
                row=row,
                delay_s=float(table.delay_s[row]),
                gain=complex(table.complex_gain[row]),
                doppler_hz=(None if table.doppler_hz is None else float(table.doppler_hz[row])),
                aoa=aoa,
                aod=aod,
                interactions=interactions,
            )
        )
    return tuple(result)


def _angle_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    azimuth_delta = math.atan2(math.sin(left[0] - right[0]), math.cos(left[0] - right[0]))
    elevation_delta = left[1] - right[1]
    return math.hypot(azimuth_delta, elevation_delta)


def _pair_cost(
    previous: _PathDescriptor,
    current: _PathDescriptor,
    *,
    max_delay_change_s: float,
    max_angle_change_rad: float,
    max_doppler_change_hz: float,
) -> tuple[float, str] | None:
    if (
        previous.interactions is not None
        and current.interactions is not None
        and previous.interactions != current.interactions
    ):
        return None
    delay_cost = abs(current.delay_s - previous.delay_s) / max_delay_change_s
    if delay_cost > 1.0:
        return None
    costs = [delay_cost**2]
    evidence = "delay_and_interaction_signature"
    for previous_angle, current_angle in ((previous.aoa, current.aoa), (previous.aod, current.aod)):
        if previous_angle is not None and current_angle is not None:
            angle_cost = _angle_distance(previous_angle, current_angle) / max_angle_change_rad
            if angle_cost > 1.0:
                return None
            costs.append(angle_cost**2)
            evidence += "_angles"
    if previous.doppler_hz is not None and current.doppler_hz is not None:
        doppler_cost = abs(current.doppler_hz - previous.doppler_hz) / max_doppler_change_hz
        if doppler_cost > 1.0:
            return None
        costs.append(doppler_cost**2)
        evidence += "_doppler"
    if previous.interactions is None or current.interactions is None:
        evidence = "physical_descriptor_fallback"
    return float(sum(costs)), evidence


def audit_path_evolution(
    table: PathOracleTable,
    time_by_frame_s: Mapping[int, float],
    *,
    phase_gauge_by_frame: Mapping[int, str] | None = None,
    max_delay_change_s: float,
    max_angle_change_rad: float = math.pi,
    max_doppler_change_hz: float = 1.0e6,
    ambiguity_cost_margin: float = 0.05,
    maximum_candidate_pair_count: int = 1_000_000,
) -> PathEvolutionAudit:
    """Associate adjacent-frame paths without treating row order as identity.

    Candidate pairs are gated by absolute delay, available angles, available
    Doppler, and exact interaction signatures when both frames provide them.
    Near-tied candidates are returned as ambiguous and excluded from births or
    deaths. Solver failures censor the whole transition. A Doppler phase
    residual is emitted only when both rows include Doppler and a nonzero
    coefficient and the caller explicitly declares the same phase-gauge ID for
    both frames; it is a consistency diagnostic, not a propagation verdict.
    """
    if not isinstance(table, PathOracleTable):
        raise TypeError("table must be a PathOracleTable")
    if not isinstance(time_by_frame_s, Mapping) or not time_by_frame_s:
        raise ValueError("time_by_frame_s must be a nonempty frame-to-time mapping")
    times: dict[int, float] = {}
    for frame, value in time_by_frame_s.items():
        if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) or frame < 0:
            raise ValueError("frame keys must be nonnegative integers")
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number))
            or np.iscomplexobj(value)
        ):
            raise TypeError("frame times must be real numeric seconds")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("frame times must be finite")
        times[int(frame)] = numeric
    ordered_times = tuple(sorted(times.items(), key=lambda item: item[1]))
    if any(right[1] <= left[1] for left, right in zip(ordered_times, ordered_times[1:])):
        raise ValueError("frame times must be unique and strictly increasing")
    referenced_frames = set(int(value) for value in table.frame_indices)
    if not referenced_frames.issubset(times):
        raise ValueError("every path-table frame must have a declared time")
    gauges: dict[int, str] = {}
    if phase_gauge_by_frame is not None:
        if not isinstance(phase_gauge_by_frame, Mapping):
            raise TypeError("phase_gauge_by_frame must be a frame-to-ID mapping")
        for frame, gauge_id in phase_gauge_by_frame.items():
            if frame not in times:
                raise ValueError("phase-gauge frame must have a declared time")
            if (
                not isinstance(gauge_id, str)
                or not gauge_id.strip()
                or gauge_id != gauge_id.strip()
            ):
                raise ValueError("phase-gauge IDs must be nonempty canonical strings")
            gauges[int(frame)] = gauge_id
    for name, value in (
        ("max_delay_change_s", max_delay_change_s),
        ("max_angle_change_rad", max_angle_change_rad),
        ("max_doppler_change_hz", max_doppler_change_hz),
        ("ambiguity_cost_margin", ambiguity_cost_margin),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number))
            or np.iscomplexobj(value)
        ):
            raise TypeError(f"{name} must be a real scalar")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if isinstance(maximum_candidate_pair_count, bool) or not isinstance(
        maximum_candidate_pair_count, int
    ):
        raise TypeError("maximum_candidate_pair_count must be an integer")
    if maximum_candidate_pair_count < 1:
        raise ValueError("maximum_candidate_pair_count must be positive")

    groups: dict[tuple[str, int, int, int], int] = {}
    for group_index, key in enumerate(
        zip(
            table.link_ids,
            table.frame_indices.tolist(),
            table.rx_port_indices.tolist(),
            table.tx_port_indices.tolist(),
            strict=True,
        )
    ):
        groups[key] = group_index
    link_ports = sorted({(link, rx, tx) for link, _, rx, tx in groups})
    ordered_frames = [frame for frame, _ in ordered_times]
    transitions: list[PathEvolutionTransition] = []
    candidate_pair_count = 0
    for link_id, rx_port, tx_port in link_ports:
        for previous_frame, current_frame in zip(ordered_frames, ordered_frames[1:]):
            previous_group = groups.get((link_id, previous_frame, rx_port, tx_port))
            current_group = groups.get((link_id, current_frame, rx_port, tx_port))
            elapsed = times[current_frame] - times[previous_frame]
            if previous_group is None or current_group is None:
                transitions.append(
                    PathEvolutionTransition(
                        link_id,
                        rx_port,
                        tx_port,
                        previous_frame,
                        current_frame,
                        elapsed,
                        "censored_missing_group",
                        (),
                        (),
                        (),
                        (),
                        (),
                    )
                )
                continue
            previous_status = table.solve_status[previous_group]
            current_status = table.solve_status[current_group]
            previous = _descriptors(table, previous_group)
            current = _descriptors(table, current_group)
            if previous_status in {
                PathSolveStatus.SOLVER_FAILURE,
                PathSolveStatus.UNCONVERGED,
            } or current_status in {
                PathSolveStatus.SOLVER_FAILURE,
                PathSolveStatus.UNCONVERGED,
            }:
                transitions.append(
                    PathEvolutionTransition(
                        link_id,
                        rx_port,
                        tx_port,
                        previous_frame,
                        current_frame,
                        elapsed,
                        "censored_solver_outcome",
                        (),
                        (),
                        (),
                        tuple(path.row for path in previous),
                        tuple(path.row for path in current),
                    )
                )
                continue
            pair_count = len(previous) * len(current)
            candidate_pair_count += pair_count
            if candidate_pair_count > maximum_candidate_pair_count:
                raise ValueError("path-evolution audit exceeds its candidate-pair limit")
            costs = np.full((len(previous), len(current)), np.inf, dtype=np.float64)
            evidence: dict[tuple[int, int], str] = {}
            for i, left in enumerate(previous):
                for j, right in enumerate(current):
                    candidate = _pair_cost(
                        left,
                        right,
                        max_delay_change_s=float(max_delay_change_s),
                        max_angle_change_rad=float(max_angle_change_rad),
                        max_doppler_change_hz=float(max_doppler_change_hz),
                    )
                    if candidate is not None:
                        costs[i, j], evidence[i, j] = candidate
            ambiguous_left: set[int] = set()
            ambiguous_right: set[int] = set()
            for i in range(len(previous)):
                finite = np.flatnonzero(np.isfinite(costs[i]))
                if len(finite) > 1:
                    ordered = sorted(float(costs[i, j]) for j in finite)
                    if ordered[1] - ordered[0] <= ambiguity_cost_margin:
                        ambiguous_left.add(i)
                        ambiguous_right.update(
                            j for j in finite if costs[i, j] - ordered[0] <= ambiguity_cost_margin
                        )
            for j in range(len(current)):
                finite = np.flatnonzero(np.isfinite(costs[:, j]))
                if len(finite) > 1:
                    ordered = sorted(float(costs[i, j]) for i in finite)
                    if ordered[1] - ordered[0] <= ambiguity_cost_margin:
                        ambiguous_right.add(j)
                        ambiguous_left.update(
                            i for i in finite if costs[i, j] - ordered[0] <= ambiguity_cost_margin
                        )
            available_left = [i for i in range(len(previous)) if i not in ambiguous_left]
            available_right = [j for j in range(len(current)) if j not in ambiguous_right]
            matches: list[PathAssociation] = []
            matched_left: set[int] = set()
            matched_right: set[int] = set()
            if available_left and available_right:
                submatrix = costs[np.ix_(available_left, available_right)]
                finite = np.isfinite(submatrix)
                if finite.any():
                    # Finite costs are in [0, 3]; the penalty prevents an
                    # impossible edge from displacing any valid assignment.
                    assigned_left, assigned_right = linear_sum_assignment(
                        np.where(finite, submatrix, 1.0e9)
                    )
                    for sub_i, sub_j in zip(assigned_left, assigned_right, strict=True):
                        i, j = available_left[int(sub_i)], available_right[int(sub_j)]
                        if not math.isfinite(float(costs[i, j])):
                            continue
                        left, right = previous[i], current[j]
                        phase_residual = None
                        if (
                            left.doppler_hz is not None
                            and right.doppler_hz is not None
                            and left.gain != 0
                            and right.gain != 0
                            and previous_frame in gauges
                            and gauges.get(previous_frame) == gauges.get(current_frame)
                        ):
                            observed = math.atan2(
                                (right.gain / left.gain).imag,
                                (right.gain / left.gain).real,
                            )
                            expected = (
                                2.0 * math.pi * 0.5 * (left.doppler_hz + right.doppler_hz) * elapsed
                            )
                            phase_residual = math.atan2(
                                math.sin(observed - expected), math.cos(observed - expected)
                            )
                        matches.append(
                            PathAssociation(
                                left.row,
                                right.row,
                                float(costs[i, j]),
                                evidence[i, j],
                                phase_residual,
                            )
                        )
                        matched_left.add(i)
                        matched_right.add(j)
            births = tuple(
                current[j].row
                for j in range(len(current))
                if j not in matched_right and j not in ambiguous_right
            )
            deaths = tuple(
                previous[i].row
                for i in range(len(previous))
                if i not in matched_left and i not in ambiguous_left
            )
            if (
                previous_status is PathSolveStatus.NO_PATH
                or current_status is PathSolveStatus.NO_PATH
            ):
                status = "no_path_transition"
            elif ambiguous_left or ambiguous_right:
                status = "association_ambiguous"
            else:
                status = "compared"
            transitions.append(
                PathEvolutionTransition(
                    link_id,
                    rx_port,
                    tx_port,
                    previous_frame,
                    current_frame,
                    elapsed,
                    status,
                    tuple(matches),
                    births,
                    deaths,
                    tuple(previous[i].row for i in sorted(ambiguous_left)),
                    tuple(current[j].row for j in sorted(ambiguous_right)),
                )
            )
    if candidate_pair_count > maximum_candidate_pair_count:
        raise ValueError("path-evolution audit exceeds its candidate-pair limit")
    return PathEvolutionAudit(
        schema="umcd.path_evolution_audit.v1",
        source_path_table_sha256=_path_table_digest(table),
        time_by_frame_s=ordered_times,
        phase_gauge_by_frame=tuple(sorted(gauges.items())),
        transitions=tuple(transitions),
        candidate_pair_count=candidate_pair_count,
        maximum_candidate_pair_count=maximum_candidate_pair_count,
        max_delay_change_s=float(max_delay_change_s),
        max_angle_change_rad=float(max_angle_change_rad),
        max_doppler_change_hz=float(max_doppler_change_hz),
        ambiguity_cost_margin=float(ambiguity_cost_margin),
        interpretation=(
            "Descriptor-based diagnostic over retained solver paths; row positions are provenance, "
            "not identities. Birth/death and Doppler phase residuals are not physical ground truth, "
            "and solver failures censor association."
        ),
    )
