"""Validated multi-node and per-link contracts for UMCD episodes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


def _readonly_array(value: np.ndarray, *, name: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.array(value, copy=True)
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape [count, {', '.join(map(str, shape_tail))}]")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    array.setflags(write=False)
    return array


def _nonempty(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


class NodeRole(StrEnum):
    """Physical/network role of a node in one propagation episode."""

    GNB = "gnb"
    UE = "ue"
    RELAY = "relay"
    INTERFERER = "interferer"
    OTHER = "other"


class LinkDirection(StrEnum):
    """Direction is explicit because links are not assumed reciprocal."""

    DOWNLINK = "downlink"
    UPLINK = "uplink"
    SIDELINK = "sidelink"
    OTHER = "other"


class LinkRole(StrEnum):
    """Signal role of a directed RF link."""

    SERVING = "serving"
    INTERFERING = "interfering"
    SOUNDING = "sounding"
    OTHER = "other"


@dataclass(frozen=True, slots=True, eq=False)
class AntennaArraySpec:
    """Array element positions and world-frame orientation for an RF node."""

    array_id: str
    element_positions_local_m: np.ndarray
    polarization_per_element: tuple[str, ...]
    orientation_local_to_world: np.ndarray
    port_ids: tuple[str, ...]
    port_element_weights: np.ndarray

    def __post_init__(self) -> None:
        _nonempty(self.array_id, name="array_id")
        positions = _readonly_array(
            self.element_positions_local_m, name="element_positions_local_m", shape_tail=(3,)
        ).astype(np.float64, copy=True)
        positions.setflags(write=False)
        polarizations = tuple(self.polarization_per_element)
        if len(polarizations) != len(positions) or any(
            not isinstance(item, str) or not item.strip() for item in polarizations
        ):
            raise ValueError("each array element needs a non-empty polarization label")
        ports = tuple(self.port_ids)
        if not ports or any(not isinstance(port, str) or not port.strip() for port in ports):
            raise ValueError("port_ids must contain non-empty antenna-port identifiers")
        if len(set(ports)) != len(ports):
            raise ValueError("port_ids must be unique within an antenna array")
        weights = np.array(self.port_element_weights, dtype=np.complex128, copy=True)
        if weights.shape != (len(ports), len(positions)) or not np.isfinite(weights).all():
            raise ValueError("port_element_weights must be finite complex [port, element]")
        if np.any(np.all(np.abs(weights) == 0.0, axis=1)):
            raise ValueError("each RF port must map to at least one radiating element")
        weights.setflags(write=False)
        orientation = np.array(self.orientation_local_to_world, dtype=np.float64, copy=True)
        if orientation.shape != (3, 3) or not np.isfinite(orientation).all():
            raise ValueError("orientation_local_to_world must be a finite 3x3 matrix")
        if not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-6, rtol=0):
            raise ValueError("orientation_local_to_world must be orthonormal")
        if not math.isclose(float(np.linalg.det(orientation)), 1.0, abs_tol=1e-6):
            raise ValueError("orientation_local_to_world must be a proper rotation")
        orientation.setflags(write=False)
        object.__setattr__(self, "element_positions_local_m", positions)
        object.__setattr__(self, "polarization_per_element", polarizations)
        object.__setattr__(self, "orientation_local_to_world", orientation)
        object.__setattr__(self, "port_ids", ports)
        object.__setattr__(self, "port_element_weights", weights)

    @property
    def num_elements(self) -> int:
        """Number of physical radiating elements in this array."""
        return len(self.element_positions_local_m)

    @property
    def num_ports(self) -> int:
        """Number of logical antenna ports, distinct from radiating elements."""
        return len(self.port_ids)


@dataclass(frozen=True, slots=True)
class NetworkNode:
    """One named network endpoint and its ordered physical antenna array."""

    node_id: str
    role: NodeRole
    array_id: str
    position_world_m: tuple[float, float, float]
    rf_chain_count: int

    def __post_init__(self) -> None:
        _nonempty(self.node_id, name="node_id")
        _nonempty(self.array_id, name="array_id")
        if not isinstance(self.role, NodeRole):
            raise TypeError("role must be a NodeRole")
        position = tuple(float(value) for value in self.position_world_m)
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            raise ValueError("position_world_m must contain three finite coordinates")
        if (
            isinstance(self.rf_chain_count, bool)
            or not isinstance(self.rf_chain_count, int)
            or self.rf_chain_count < 1
        ):
            raise ValueError("rf_chain_count must be a positive integer")
        object.__setattr__(self, "position_world_m", position)


@dataclass(frozen=True, slots=True)
class DirectedLink:
    """Directed TX→RX association with explicit physical port ordering."""

    link_id: str
    tx_node_id: str
    rx_node_id: str
    tx_port_indices: tuple[int, ...]
    rx_port_indices: tuple[int, ...]
    direction: LinkDirection
    role: LinkRole

    def __post_init__(self) -> None:
        for name in ("link_id", "tx_node_id", "rx_node_id"):
            _nonempty(getattr(self, name), name=name)
        if self.tx_node_id == self.rx_node_id:
            raise ValueError("a directed link must connect distinct nodes")
        if not isinstance(self.direction, LinkDirection) or not isinstance(self.role, LinkRole):
            raise TypeError("direction and role must use their typed enums")
        for name in ("tx_port_indices", "rx_port_indices"):
            ports = tuple(getattr(self, name))
            if not ports or any(
                isinstance(port, bool) or not isinstance(port, int) or port < 0 for port in ports
            ):
                raise ValueError(f"{name} must contain nonnegative integer port indices")
            if len(set(ports)) != len(ports):
                raise ValueError(f"{name} must not contain duplicate ports")
            object.__setattr__(self, name, ports)


@dataclass(frozen=True, slots=True, eq=False)
class NetworkEpisode:
    """Static network-propagation parent shared by all link/view descendants."""

    episode_id: str
    split_group_id: str
    family_id: str
    carrier_hz: float
    arrays: tuple[AntennaArraySpec, ...]
    nodes: tuple[NetworkNode, ...]
    links: tuple[DirectedLink, ...]
    frame_times_s: tuple[float, ...]
    subcarrier_offsets_hz: np.ndarray

    def __post_init__(self) -> None:
        for name in ("episode_id", "split_group_id", "family_id"):
            _nonempty(getattr(self, name), name=name)
        if (
            isinstance(self.carrier_hz, bool)
            or not math.isfinite(self.carrier_hz)
            or self.carrier_hz <= 0
        ):
            raise ValueError("carrier_hz must be finite and positive")
        arrays, nodes, links = tuple(self.arrays), tuple(self.nodes), tuple(self.links)
        if not arrays or not nodes or not links:
            raise ValueError("an episode needs at least one array, node, and directed link")
        if len({array.array_id for array in arrays}) != len(arrays):
            raise ValueError("array_id values must be unique within an episode")
        if len({node.node_id for node in nodes}) != len(nodes):
            raise ValueError("node_id values must be unique within an episode")
        if len({link.link_id for link in links}) != len(links):
            raise ValueError("link_id values must be unique within an episode")
        array_by_id = {array.array_id: array for array in arrays}
        node_by_id = {node.node_id: node for node in nodes}
        for node in nodes:
            if node.array_id not in array_by_id:
                raise ValueError(f"node {node.node_id!r} references a missing antenna array")
            if node.rf_chain_count > array_by_id[node.array_id].num_ports:
                raise ValueError("RF chain count cannot exceed the declared antenna port count")
        for link in links:
            if link.tx_node_id not in node_by_id or link.rx_node_id not in node_by_id:
                raise ValueError(f"link {link.link_id!r} references a missing network node")
            tx_array = array_by_id[node_by_id[link.tx_node_id].array_id]
            rx_array = array_by_id[node_by_id[link.rx_node_id].array_id]
            if max(link.tx_port_indices) >= tx_array.num_ports:
                raise ValueError(f"link {link.link_id!r} has an invalid TX port map")
            if max(link.rx_port_indices) >= rx_array.num_ports:
                raise ValueError(f"link {link.link_id!r} has an invalid RX port map")
        if any(isinstance(value, bool) for value in self.frame_times_s):
            raise TypeError("frame_times_s values must be real times, not booleans")
        times = tuple(float(value) for value in self.frame_times_s)
        if not times or any(not math.isfinite(value) for value in times):
            raise ValueError("frame_times_s must be a non-empty finite time axis")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("frame_times_s must be strictly increasing")
        raw_offsets = np.asarray(self.subcarrier_offsets_hz)
        if raw_offsets.dtype.kind not in "iuf":
            raise TypeError("subcarrier_offsets_hz must use a real numeric dtype")
        offsets = np.array(raw_offsets, dtype=np.float64, copy=True)
        if offsets.ndim != 1 or len(offsets) < 1 or not np.isfinite(offsets).all():
            raise ValueError("subcarrier_offsets_hz must be a non-empty finite vector")
        if len(np.unique(offsets)) != len(offsets) or np.any(np.diff(offsets) <= 0):
            raise ValueError("subcarrier_offsets_hz must be strictly increasing and unique")
        if not np.any(offsets == 0.0):
            raise ValueError("baseband subcarrier offsets must include DC at 0 Hz")
        offsets.setflags(write=False)
        object.__setattr__(self, "arrays", arrays)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "frame_times_s", times)
        object.__setattr__(self, "subcarrier_offsets_hz", offsets)

    def link(self, link_id: str) -> DirectedLink:
        """Return a link by stable ID or raise ``KeyError``."""
        for link in self.links:
            if link.link_id == link_id:
                return link
        raise KeyError(link_id)

    def node(self, node_id: str) -> NetworkNode:
        """Return a node by stable ID or raise ``KeyError``."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def array(self, array_id: str) -> AntennaArraySpec:
        """Return an antenna array by stable ID or raise ``KeyError``."""
        for array in self.arrays:
            if array.array_id == array_id:
                return array
        raise KeyError(array_id)
