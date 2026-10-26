"""Paired mathematical channel interventions over one frozen path table."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .path_oracle import PathOracleTable
from .split_policy import SplitPolicyRegistry


def path_oracle_content_digest(table: PathOracleTable) -> str:
    """Hash the path identity and numerical content used by the transform."""
    digest = hashlib.sha256()
    metadata = {
        "link_ids": table.link_ids,
        "frame_indices": table.frame_indices.tolist(),
        "rx_port_indices": table.rx_port_indices.tolist(),
        "tx_port_indices": table.tx_port_indices.tolist(),
        "path_offsets": table.path_offsets.tolist(),
        "solve_status": [item.value for item in table.solve_status],
        "coefficient_convention": table.coefficient_convention,
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for array in (table.delay_s, table.complex_gain, table.interaction_order):
        canonical = np.ascontiguousarray(array)
        digest.update(canonical.dtype.str.encode())
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _calculate_cfr_pair(
    table: PathOracleTable, frequency_offset_hz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_count = len(table.link_ids)
    full = np.zeros((group_count, len(frequency_offset_hz)), dtype=np.complex128)
    ablated = np.zeros_like(full)
    removed = np.zeros(group_count, dtype=np.int64)
    for group in range(group_count):
        start, stop = table.path_offsets[group : group + 2]
        orders = table.interaction_order[start:stop]
        direct = orders == 0
        removed[group] = int(np.count_nonzero(direct))
        phase = np.exp(-2j * np.pi * table.delay_s[start:stop, None] * frequency_offset_hz[None, :])
        contributions = table.complex_gain[start:stop, None] * phase
        full[group] = contributions.sum(axis=0)
        ablated[group] = contributions[~direct].sum(axis=0)
    return full, ablated, removed


def validate_direct_path_ablation_source(
    table: PathOracleTable, ablation: DirectPathAblation
) -> None:
    """Recompute a stored mathematical pair from its exact source path table."""
    if not isinstance(table, PathOracleTable) or not isinstance(ablation, DirectPathAblation):
        raise TypeError("source validation requires a PathOracleTable and DirectPathAblation")
    if ablation.path_table_sha256 != path_oracle_content_digest(table):
        raise ValueError("direct-path ablation source digest differs from path table")
    if (
        ablation.link_ids != table.link_ids
        or not np.array_equal(ablation.frame_indices, table.frame_indices)
        or not np.array_equal(ablation.rx_port_indices, table.rx_port_indices)
        or not np.array_equal(ablation.tx_port_indices, table.tx_port_indices)
    ):
        raise ValueError("direct-path ablation group roster differs from path table")
    full, without_direct, removed = _calculate_cfr_pair(table, ablation.frequency_offset_hz)
    if (
        not np.array_equal(ablation.full_cfr, full)
        or not np.array_equal(ablation.without_direct_cfr, without_direct)
        or not np.array_equal(ablation.removed_direct_path_count, removed)
    ):
        raise ValueError("direct-path ablation CFR or removed-path counts do not replay")


@dataclass(frozen=True, slots=True, eq=False)
class DirectPathAblation:
    """Same-path-set CFR pair with only zero-interaction paths removed."""

    parent_subject_id: str
    base_child_id: str
    ablated_child_id: str
    split: str
    path_table_sha256: str
    link_ids: tuple[str, ...]
    frame_indices: np.ndarray
    rx_port_indices: np.ndarray
    tx_port_indices: np.ndarray
    frequency_offset_hz: np.ndarray
    full_cfr: np.ndarray
    without_direct_cfr: np.ndarray
    removed_direct_path_count: np.ndarray

    def __post_init__(self) -> None:
        for name in ("parent_subject_id", "base_child_id", "ablated_child_id", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be a canonical non-empty string")
        if self.base_child_id == self.ablated_child_id:
            raise ValueError("base and ablated children must have distinct IDs")
        if len(self.path_table_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.path_table_sha256
        ):
            raise ValueError("path_table_sha256 must be lowercase SHA-256 hex")
        raw_frequencies = np.asarray(self.frequency_offset_hz)
        if raw_frequencies.dtype.kind not in "iuf":
            raise TypeError("frequency_offset_hz must be real numeric data")
        frequencies = np.array(raw_frequencies, dtype=np.float64, copy=True)
        link_ids = tuple(self.link_ids)
        index_vectors = []
        for name, value in (
            ("frame_indices", self.frame_indices),
            ("rx_port_indices", self.rx_port_indices),
            ("tx_port_indices", self.tx_port_indices),
            ("removed_direct_path_count", self.removed_direct_path_count),
        ):
            raw = np.asarray(value)
            if raw.dtype.kind not in "iu" or raw.ndim != 1:
                raise TypeError(f"{name} must be a one-dimensional integer vector")
            index_vectors.append(np.array(raw, dtype=np.int64, copy=True))
        frames, rx_ports, tx_ports, removed = index_vectors
        full = np.array(self.full_cfr, dtype=np.complex128, copy=True)
        ablated = np.array(self.without_direct_cfr, dtype=np.complex128, copy=True)
        if frequencies.ndim != 1 or len(frequencies) == 0 or not np.isfinite(frequencies).all():
            raise ValueError("frequency_offset_hz must be a non-empty finite vector")
        if len(frequencies) > 1 and np.any(np.diff(frequencies) <= 0):
            raise ValueError("frequency_offset_hz must be strictly increasing")
        if full.ndim != 2 or full.shape != ablated.shape or full.shape[1] != len(frequencies):
            raise ValueError("CFR arrays must have shape (path_group, frequency)")
        group_count = full.shape[0]
        if (
            not link_ids
            or any(
                not isinstance(value, str) or not value or value.strip() != value
                for value in link_ids
            )
            or len(set(zip(link_ids, frames.tolist(), rx_ports.tolist(), tx_ports.tolist())))
            != group_count
            or any(len(values) != group_count for values in (frames, rx_ports, tx_ports))
            or np.any(frames < 0)
            or np.any(rx_ports < 0)
            or np.any(tx_ports < 0)
        ):
            raise ValueError("path-group identity arrays must uniquely match CFR rows")
        if removed.shape != (full.shape[0],) or np.any(removed < 0):
            raise ValueError("removed_direct_path_count must match path groups")
        if not all(
            np.isfinite(array.real).all() and np.isfinite(array.imag).all()
            for array in (full, ablated)
        ):
            raise ValueError("CFR arrays must be finite")
        for array in (frames, rx_ports, tx_ports, frequencies, full, ablated, removed):
            array.setflags(write=False)
        object.__setattr__(self, "link_ids", link_ids)
        object.__setattr__(self, "frame_indices", frames)
        object.__setattr__(self, "rx_port_indices", rx_ports)
        object.__setattr__(self, "tx_port_indices", tx_ports)
        object.__setattr__(self, "frequency_offset_hz", frequencies)
        object.__setattr__(self, "full_cfr", full)
        object.__setattr__(self, "without_direct_cfr", ablated)
        object.__setattr__(self, "removed_direct_path_count", removed)


def remove_direct_path_component(
    table: PathOracleTable,
    *,
    frequency_offset_hz: np.ndarray,
    registry: SplitPolicyRegistry,
    parent_subject_id: str,
    base_child_id: str,
    ablated_child_id: str,
) -> DirectPathAblation:
    """Build a paired CFR ablation from identical paths, delays, and phases.

    A direct component is a zero-interaction path in the supplied path table.
    This is a mathematical channel ablation; it does not model a physical
    blocker or retrace the scene. Both children must already be bound to the
    same frozen split parent.
    """
    if not isinstance(table, PathOracleTable):
        raise TypeError("table must be a PathOracleTable")
    if not isinstance(registry, SplitPolicyRegistry):
        raise TypeError("registry must be a SplitPolicyRegistry")
    if (
        registry.child_bindings.get(base_child_id) is None
        or registry.child_bindings.get(ablated_child_id) is None
    ):
        raise ValueError("both ablation children must be registered in the frozen split registry")
    if any(
        registry.child_bindings[child].parent_subject_id != parent_subject_id
        for child in (base_child_id, ablated_child_id)
    ):
        raise ValueError("both ablation children must bind to the declared split parent")
    split = registry.split_for_child(base_child_id)
    if registry.split_for_child(ablated_child_id) != split:
        raise ValueError("paired ablation children must inherit the same split")
    frequencies = np.asarray(frequency_offset_hz)
    if frequencies.dtype.kind not in "iuf" or frequencies.ndim != 1 or len(frequencies) == 0:
        raise TypeError("frequency_offset_hz must be a non-empty real vector")
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if not np.isfinite(frequencies).all() or (
        len(frequencies) > 1 and np.any(np.diff(frequencies) <= 0)
    ):
        raise ValueError("frequency offsets must be finite and strictly increasing")

    full, ablated, removed = _calculate_cfr_pair(table, frequencies)

    return DirectPathAblation(
        parent_subject_id=parent_subject_id,
        base_child_id=base_child_id,
        ablated_child_id=ablated_child_id,
        split=split,
        path_table_sha256=path_oracle_content_digest(table),
        link_ids=table.link_ids,
        frame_indices=table.frame_indices,
        rx_port_indices=table.rx_port_indices,
        tx_port_indices=table.tx_port_indices,
        frequency_offset_hz=frequencies,
        full_cfr=full,
        without_direct_cfr=ablated,
        removed_direct_path_count=removed,
    )
