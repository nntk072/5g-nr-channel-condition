"""Map whole Sionna RT solve outputs onto UMCD links and episode frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast

import numpy as np
from datasets.network import NetworkEpisode
from datasets.path_oracle import PathOracleTable, PathSolveStatus

from integrations.sionna.path_oracle import sionna_paths_to_oracle_table


@dataclass(frozen=True, slots=True)
class SionnaPathBinding:
    """Explicit Sionna device-pair mapping for one episode link and frame.

    Device indices address the leading RX/TX device axes in one Sionna
    ``Paths`` object. Port maps address physical array ports and are ordered as
    the corresponding UMCD link-local port lists. A missing ``paths`` object
    is valid only for a recorded solver failure or unconverged solve.
    """

    link_id: str
    frame_index: int
    rx_device_index: int
    tx_device_index: int
    paths: object | None
    solve_status: PathSolveStatus | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.link_id, str) or not self.link_id.strip():
            raise ValueError("link_id must be a non-empty string")
        for name in ("frame_index", "rx_device_index", "tx_device_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.solve_status is not None and not isinstance(self.solve_status, PathSolveStatus):
            raise TypeError("solve_status must be a PathSolveStatus value or None")
        if self.paths is None and self.solve_status not in (
            PathSolveStatus.SOLVER_FAILURE,
            PathSolveStatus.UNCONVERGED,
        ):
            raise ValueError("missing Paths data requires SOLVER_FAILURE or UNCONVERGED status")
        if self.solve_status is PathSolveStatus.SOLVER_FAILURE and self.paths is not None:
            raise ValueError("a failed solve must not carry a Paths result")


def sionna_bindings_to_episode_oracle(
    episode: NetworkEpisode,
    bindings: Sequence[SionnaPathBinding],
) -> PathOracleTable:
    """Assemble complete per-link/frame/port oracle data for one UMCD episode.

    Exactly one explicit device-pair binding is required for every episode
    link at every frame. Link port maps select and reorder the physical Sionna
    array ports. Empty path solves and missing failed/unconverged runs remain
    explicit groups instead of disappearing from the episode.
    """
    binding_values = tuple(bindings)
    if not binding_values:
        raise ValueError("at least one Sionna path binding is required")
    expected = {
        (link.link_id, frame_index)
        for link in episode.links
        for frame_index in range(len(episode.frame_times_s))
    }
    actual = [(binding.link_id, binding.frame_index) for binding in binding_values]
    if len(set(actual)) != len(actual):
        raise ValueError("Sionna path bindings must be unique per link and frame")
    if set(actual) != expected:
        missing = sorted(expected - set(actual))
        unexpected = sorted(set(actual) - expected)
        raise ValueError(
            f"Sionna bindings must cover every episode link/frame; missing={missing}, "
            f"unexpected={unexpected}"
        )

    link_order = {link.link_id: index for index, link in enumerate(episode.links)}
    binding_values = tuple(
        sorted(binding_values, key=lambda item: (link_order[item.link_id], item.frame_index))
    )
    tx_node_indices: dict[str, int] = {}
    tx_index_nodes: dict[int, str] = {}
    rx_node_indices: dict[str, int] = {}
    rx_index_nodes: dict[int, str] = {}
    for binding in binding_values:
        link = episode.link(binding.link_id)
        for node_to_index, index_to_node, node_id, device_index, axis in (
            (
                tx_node_indices,
                tx_index_nodes,
                link.tx_node_id,
                binding.tx_device_index,
                "TX",
            ),
            (
                rx_node_indices,
                rx_index_nodes,
                link.rx_node_id,
                binding.rx_device_index,
                "RX",
            ),
        ):
            prior_index = node_to_index.setdefault(node_id, device_index)
            prior_node = index_to_node.setdefault(device_index, node_id)
            if prior_index != device_index or prior_node != node_id:
                raise ValueError(
                    f"Sionna {axis} device mapping must be stable and one-to-one per episode"
                )

    tables: list[PathOracleTable] = []
    for binding in binding_values:
        link = episode.link(binding.link_id)
        rx_node = episode.node(link.rx_node_id)
        tx_node = episode.node(link.tx_node_id)
        rx_array = episode.array(rx_node.array_id)
        tx_array = episode.array(tx_node.array_id)
        rx_port_map = link.rx_port_indices
        tx_port_map = link.tx_port_indices
        if any(port >= rx_array.num_ports for port in rx_port_map):
            raise ValueError(f"link {link.link_id!r} references an invalid RX array port")
        if any(port >= tx_array.num_ports for port in tx_port_map):
            raise ValueError(f"link {link.link_id!r} references an invalid TX array port")

        if binding.paths is None:
            tables.append(
                _empty_binding_table(
                    binding,
                    rx_port_count=len(rx_port_map),
                    tx_port_count=len(tx_port_map),
                )
            )
            continue

        table = sionna_paths_to_oracle_table(
            binding.paths,
            link_id=binding.link_id,
            frame_index=binding.frame_index,
            rx_device_index=binding.rx_device_index,
            tx_device_index=binding.tx_device_index,
            rx_port_map=rx_port_map,
            tx_port_map=tx_port_map,
            solve_status=binding.solve_status,
        )
        if binding.solve_status is PathSolveStatus.NO_PATH and table.path_count:
            raise ValueError("NO_PATH status cannot carry valid paths")
        tables.append(table)

    combined = _concatenate_tables(tables)
    combined.validate_for(episode)
    expected_groups = {
        (link.link_id, frame, rx_port, tx_port)
        for link in episode.links
        for frame in range(len(episode.frame_times_s))
        for rx_port in range(len(link.rx_port_indices))
        for tx_port in range(len(link.tx_port_indices))
    }
    actual_groups = set(
        zip(
            combined.link_ids,
            combined.frame_indices.tolist(),
            combined.rx_port_indices.tolist(),
            combined.tx_port_indices.tolist(),
            strict=True,
        )
    )
    if actual_groups != expected_groups or combined.group_count != len(expected_groups):
        raise ValueError("assembled path oracle does not cover every link/frame/port pair exactly")
    return combined


def _empty_binding_table(
    binding: SionnaPathBinding,
    *,
    rx_port_count: int,
    tx_port_count: int,
) -> PathOracleTable:
    group_count = rx_port_count * tx_port_count
    if group_count < 1:
        raise ValueError("a failed solve must still declare at least one port pair")
    status = binding.solve_status
    assert status in (PathSolveStatus.SOLVER_FAILURE, PathSolveStatus.UNCONVERGED)
    return PathOracleTable(
        link_ids=(binding.link_id,) * group_count,
        frame_indices=np.full(group_count, binding.frame_index, dtype=np.int64),
        rx_port_indices=np.repeat(np.arange(rx_port_count, dtype=np.int64), tx_port_count),
        tx_port_indices=np.tile(np.arange(tx_port_count, dtype=np.int64), rx_port_count),
        path_offsets=np.zeros(group_count + 1, dtype=np.int64),
        solve_status=(status,) * group_count,
        delay_s=np.empty(0, dtype=np.float64),
        complex_gain=np.empty(0, dtype=np.complex128),
        interaction_order=np.empty(0, dtype=np.int64),
        doppler_hz=np.empty(0, dtype=np.float64),
        aoa_azimuth_elevation_rad=np.empty((0, 2), dtype=np.float64),
        aod_azimuth_elevation_rad=np.empty((0, 2), dtype=np.float64),
        path_kind=(),
        interaction_offsets=np.zeros(1, dtype=np.int64),
        interaction_types=np.empty(0, dtype=np.int64),
        object_ids=np.empty(0, dtype=np.int64),
        primitive_ids=np.empty(0, dtype=np.int64),
        interaction_vertices_m=np.empty((0, 3), dtype=np.float64),
        interaction_vertex_frame="sionna_rt_scene_xyz_m",
    )


def _concatenate_tables(tables: Sequence[PathOracleTable]) -> PathOracleTable:
    if not tables:
        raise ValueError("cannot concatenate an empty path-table sequence")
    optional_fields = (
        "doppler_hz",
        "aoa_azimuth_elevation_rad",
        "aod_azimuth_elevation_rad",
        "interaction_offsets",
        "interaction_types",
        "object_ids",
        "primitive_ids",
        "interaction_vertices_m",
    )
    for name in optional_fields:
        present = [getattr(table, name) is not None for table in tables]
        if any(present) and not all(present):
            raise ValueError(f"path tables disagree on optional field {name}")
    conventions = {table.coefficient_convention for table in tables}
    frames = {table.interaction_vertex_frame for table in tables}
    if len(conventions) != 1 or len(frames) != 1:
        raise ValueError("path tables disagree on coefficient convention or interaction frame")

    path_offsets = [0]
    for table in tables:
        path_offsets.extend((table.path_offsets[1:] + path_offsets[-1]).tolist())
    interaction_offsets: np.ndarray | None = None
    if tables[0].interaction_offsets is not None:
        combined_interactions = [0]
        for table in tables:
            table_offsets = table.interaction_offsets
            assert table_offsets is not None
            interaction_base = combined_interactions[-1]
            combined_interactions.extend((table_offsets[1:] + interaction_base).tolist())
        interaction_offsets = np.asarray(combined_interactions, dtype=np.int64)

    def concat_required(name: str) -> np.ndarray:
        values = [getattr(table, name) for table in tables]
        if any(value is None for value in values):
            raise ValueError(f"required path-table field {name} is missing")
        return np.concatenate(cast(list[np.ndarray], values), axis=0)

    def concat_optional(name: str) -> np.ndarray | None:
        values = [getattr(table, name) for table in tables]
        if values[0] is None:
            return None
        if any(value is None for value in values):
            raise ValueError(f"path tables disagree on optional field {name}")
        return np.concatenate(cast(list[np.ndarray], values), axis=0)

    path_kinds = None
    if all(table.path_kind is not None for table in tables):
        path_kinds = tuple(
            kind for table in tables for kind in cast(tuple[str, ...], table.path_kind)
        )
    return PathOracleTable(
        link_ids=tuple(link for table in tables for link in table.link_ids),
        frame_indices=concat_required("frame_indices"),
        rx_port_indices=concat_required("rx_port_indices"),
        tx_port_indices=concat_required("tx_port_indices"),
        path_offsets=np.asarray(path_offsets, dtype=np.int64),
        solve_status=tuple(status for table in tables for status in table.solve_status),
        delay_s=concat_required("delay_s"),
        complex_gain=concat_required("complex_gain"),
        interaction_order=concat_required("interaction_order"),
        doppler_hz=concat_optional("doppler_hz"),
        aoa_azimuth_elevation_rad=concat_optional("aoa_azimuth_elevation_rad"),
        aod_azimuth_elevation_rad=concat_optional("aod_azimuth_elevation_rad"),
        path_kind=path_kinds,
        interaction_offsets=interaction_offsets,
        interaction_types=concat_optional("interaction_types"),
        object_ids=concat_optional("object_ids"),
        primitive_ids=concat_optional("primitive_ids"),
        interaction_vertices_m=concat_optional("interaction_vertices_m"),
        interaction_vertex_frame=tables[0].interaction_vertex_frame,
        coefficient_convention=tables[0].coefficient_convention,
    )
