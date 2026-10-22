"""Extract static Sionna RT scene devices and arrays into UMCD contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from datasets.network import (
    AntennaArraySpec,
    DirectedLink,
    LinkDirection,
    LinkRole,
    NetworkEpisode,
    NetworkNode,
    NodeRole,
)
from datasets.path_oracle import PathSolveStatus

from integrations.sionna.rt_network import SionnaPathBinding

_LIGHT_SPEED_M_S = 299_792_458.0


@dataclass(frozen=True, slots=True)
class SionnaDeviceLinkMapping:
    """A UMCD link and the device-axis indices used by Sionna ``Paths``."""

    link_id: str
    tx_device_index: int
    rx_device_index: int


@dataclass(frozen=True, slots=True, eq=False)
class SionnaSceneNetwork:
    """Extracted static episode plus its explicit Sionna device-axis mapping."""

    episode: NetworkEpisode
    device_link_mappings: tuple[SionnaDeviceLinkMapping, ...]

    def bind_paths(
        self,
        paths_by_link_id: Mapping[str, object | None],
        *,
        frame_index: int = 0,
        solve_status_by_link_id: Mapping[str, PathSolveStatus] | None = None,
    ) -> tuple[SionnaPathBinding, ...]:
        """Bind one shared or per-link ``Paths`` result to all extracted links."""
        expected = {mapping.link_id for mapping in self.device_link_mappings}
        if set(paths_by_link_id) != expected:
            raise ValueError("Paths mapping must contain exactly every extracted link ID")
        statuses = {} if solve_status_by_link_id is None else dict(solve_status_by_link_id)
        if set(statuses) - expected:
            raise ValueError("solve statuses contain an unknown extracted link ID")
        return tuple(
            SionnaPathBinding(
                link_id=mapping.link_id,
                frame_index=frame_index,
                rx_device_index=mapping.rx_device_index,
                tx_device_index=mapping.tx_device_index,
                paths=paths_by_link_id[mapping.link_id],
                solve_status=statuses.get(mapping.link_id),
            )
            for mapping in self.device_link_mappings
        )


def sionna_device_pose(device: object, *, node_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one Sionna device's current world position and array orientation."""
    position = _vector3(getattr(device, "position", None), name=f"{node_id}.position")
    orientation = _orientation_matrix(getattr(device, "orientation", None), node_id)
    return position, orientation


def sionna_scene_to_network_episode(
    scene: object,
    *,
    episode_id: str,
    split_group_id: str,
    family_id: str,
    subcarrier_offsets_hz: Sequence[float] = (0.0,),
    rf_chain_count_by_node: Mapping[str, int] | None = None,
    node_roles: Mapping[str, NodeRole] | None = None,
    tx_pattern_labels: Sequence[str] | None = None,
    rx_pattern_labels: Sequence[str] | None = None,
    associations: Sequence[tuple[str, str]] | None = None,
    direction: LinkDirection = LinkDirection.OTHER,
    link_role: LinkRole = LinkRole.OTHER,
) -> SionnaSceneNetwork:
    """Extract one static Sionna ``Scene`` as a multi-node UMCD episode.

    UMCD node, array, and link records are ordered by device name so insertion
    order cannot change the exported episode identity. Explicit device-axis
    mappings retain each device's actual index in the pinned Sionna scene.
    Array positions are converted from wavelength-normalized local coordinates to meters;
    each pattern/element feed becomes one explicitly ordered UMCD port. Names
    inferred for patterns (``pattern-N``) preserve Sionna pattern order but do
    not claim a physical polarization basis. Supply labels such as ``V`` and
    ``H`` only when they are known by the caller.

    The adapter intentionally models a single static frame. It rejects moving
    devices because the UMCD pose timeline must be populated by a trajectory
    adapter rather than inferred from one instantaneous position. A DC-only
    frequency axis is the default; pass the actual ordered OFDM offsets for a
    physical channel view.

    Sionna RT does not define logical RF-chain counts or network association
    roles. SISO devices default to one RF chain; multi-port devices require an
    explicit count for every node. Roles default to ``OTHER`` and the default
    all-TX/all-RX topology uses the caller-selected direction/role (both
    default to ``OTHER`` to avoid asserting a serving relationship).
    """
    tx_devices = _device_items(getattr(scene, "transmitters", None), name="transmitters")
    rx_devices = _device_items(getattr(scene, "receivers", None), name="receivers")
    tx_array = getattr(scene, "tx_array", None)
    rx_array = getattr(scene, "rx_array", None)
    frequency_hz = _scalar(getattr(scene, "frequency", None), name="scene.frequency")
    if frequency_hz <= 0:
        raise ValueError("scene.frequency must be positive")
    wavelength_m = _LIGHT_SPEED_M_S / frequency_hz

    tx_labels = _pattern_labels(tx_array, tx_pattern_labels, name="tx_pattern_labels")
    rx_labels = _pattern_labels(rx_array, rx_pattern_labels, name="rx_pattern_labels")
    tx_array_specs: dict[str, AntennaArraySpec] = {}
    rx_array_specs: dict[str, AntennaArraySpec] = {}
    nodes: list[NetworkNode] = []
    tx_ids: list[str] = []
    rx_ids: list[str] = []
    tx_device_indices: dict[str, int] = {}
    rx_device_indices: dict[str, int] = {}

    role_map = {} if node_roles is None else dict(node_roles)
    rf_counts = {} if rf_chain_count_by_node is None else dict(rf_chain_count_by_node)
    for axis, items, array, labels, target, id_order, index_map in (
        ("tx", tx_devices, tx_array, tx_labels, tx_array_specs, tx_ids, tx_device_indices),
        ("rx", rx_devices, rx_array, rx_labels, rx_array_specs, rx_ids, rx_device_indices),
    ):
        solver_indices = {name: index for index, (name, _) in enumerate(items)}
        for name, device in sorted(items, key=lambda item: item[0]):
            device_index = solver_indices[name]
            node_id = f"{axis}:{name}"
            _require_static_device(device, node_id)
            orientation = _orientation_matrix(getattr(device, "orientation", None), node_id)
            array_id = f"{node_id}:array"
            spec = _array_spec(array, labels, wavelength_m, orientation, array_id)
            target[array_id] = spec
            position = _vector3(getattr(device, "position", None), name=f"{node_id}.position")
            rf_count = rf_counts.get(node_id, 1 if spec.num_ports == 1 else None)
            if rf_count is None:
                raise ValueError(
                    f"multi-port Sionna node {node_id!r} needs an explicit RF-chain count"
                )
            if isinstance(rf_count, bool) or not isinstance(rf_count, int):
                raise TypeError(f"RF-chain count for {node_id!r} must be an integer")
            role = role_map.get(node_id, NodeRole.OTHER)
            if not isinstance(role, NodeRole):
                raise TypeError(f"node role for {node_id!r} must be a NodeRole")
            nodes.append(NetworkNode(node_id, role, array_id, tuple(position), rf_count))
            id_order.append(node_id)
            index_map[node_id] = device_index

    if set(role_map) - set(tx_ids) - set(rx_ids):
        raise ValueError("node_roles contains a node absent from this Sionna scene")
    if set(rf_counts) - set(tx_ids) - set(rx_ids):
        raise ValueError("RF-chain counts contain a node absent from this Sionna scene")

    if associations is None:
        link_pairs = tuple((tx_id, rx_id) for tx_id in tx_ids for rx_id in rx_ids)
    else:
        link_pairs = tuple(associations)
    if not link_pairs:
        raise ValueError("at least one TX→RX association is required")
    if len(set(link_pairs)) != len(link_pairs):
        raise ValueError("TX→RX associations must be unique")
    tx_node_by_id = {node.node_id: node for node in nodes if node.node_id in tx_ids}
    rx_node_by_id = {node.node_id: node for node in nodes if node.node_id in rx_ids}
    links: list[DirectedLink] = []
    mappings: list[SionnaDeviceLinkMapping] = []
    for tx_id, rx_id in link_pairs:
        if tx_id not in tx_node_by_id or rx_id not in rx_node_by_id:
            raise ValueError("each association must name an extracted TX followed by an RX")
        tx_spec = tx_array_specs[tx_node_by_id[tx_id].array_id]
        rx_spec = rx_array_specs[rx_node_by_id[rx_id].array_id]
        link_id = f"{tx_id}->{rx_id}"
        links.append(
            DirectedLink(
                link_id,
                tx_id,
                rx_id,
                tuple(range(tx_spec.num_ports)),
                tuple(range(rx_spec.num_ports)),
                direction,
                link_role,
            )
        )
        mappings.append(
            SionnaDeviceLinkMapping(
                link_id,
                tx_device_indices[tx_id],
                rx_device_indices[rx_id],
            )
        )

    arrays = tuple((*tx_array_specs.values(), *rx_array_specs.values()))
    episode = NetworkEpisode(
        episode_id=episode_id,
        split_group_id=split_group_id,
        family_id=family_id,
        carrier_hz=frequency_hz,
        arrays=arrays,
        nodes=tuple(nodes),
        links=tuple(links),
        frame_times_s=(0.0,),
        subcarrier_offsets_hz=np.asarray(subcarrier_offsets_hz, dtype=np.float64),
    )
    return SionnaSceneNetwork(episode, tuple(mappings))


def _device_items(value: object, *, name: str) -> tuple[tuple[str, object], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"scene.{name} must be a non-empty device mapping")
    items = tuple(value.items())
    if any(not isinstance(key, str) or not key for key, _ in items):
        raise ValueError(f"scene.{name} contains an invalid device name")
    return items


def _pattern_labels(
    array: object,
    labels: Sequence[str] | None,
    *,
    name: str,
) -> tuple[str, ...]:
    pattern = getattr(array, "antenna_pattern", None)
    patterns = getattr(pattern, "patterns", None)
    if not isinstance(patterns, (list, tuple)) or not patterns:
        raise TypeError("Sionna array must expose antenna_pattern.patterns")
    result = (
        tuple(f"pattern-{index}" for index in range(len(patterns)))
        if labels is None
        else tuple(labels)
    )
    if len(result) != len(patterns) or any(
        not isinstance(label, str) or not label.strip() for label in result
    ):
        raise ValueError(f"{name} must label each Sionna antenna pattern exactly once")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} labels must be unique")
    return result


def _array_spec(
    array: object,
    labels: tuple[str, ...],
    wavelength_m: float,
    orientation: np.ndarray,
    array_id: str,
) -> AntennaArraySpec:
    try:
        normalized = getattr(array, "normalized_positions")
        positions = np.column_stack(
            [_component(getattr(normalized, axis)) for axis in ("x", "y", "z")]
        ).astype(np.float64)
        array_size = int(getattr(array, "array_size"))
        num_ports = int(getattr(array, "num_ant"))
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("Sionna array lacks the pinned AntennaArray layout API") from error
    if positions.shape != (array_size, 3) or array_size < 1:
        raise ValueError("Sionna normalized element positions have an unexpected shape")
    if num_ports != array_size * len(labels):
        raise ValueError("Sionna antenna feed count does not match element and pattern axes")
    local_positions_m = positions * wavelength_m
    port_ids: list[str] = []
    weights = np.zeros((num_ports, array_size), dtype=np.complex128)
    for pattern_index, label in enumerate(labels):
        for element_index in range(array_size):
            port_index = pattern_index * array_size + element_index
            port_ids.append(f"{label}:element-{element_index}")
            weights[port_index, element_index] = 1.0 + 0.0j
    combined_polarization = "+".join(labels)
    return AntennaArraySpec(
        array_id=array_id,
        element_positions_local_m=local_positions_m,
        polarization_per_element=(combined_polarization,) * array_size,
        orientation_local_to_world=orientation,
        port_ids=tuple(port_ids),
        port_element_weights=weights,
    )


def _require_static_device(device: object, node_id: str) -> None:
    velocity = getattr(device, "velocity", None)
    if (
        velocity is not None
        and np.linalg.norm(_vector3(velocity, name=f"{node_id}.velocity")) > 1e-12
    ):
        raise ValueError(f"moving Sionna node {node_id!r} needs a trajectory-aware episode adapter")


def _orientation_matrix(value: object, node_id: str) -> np.ndarray:
    angles = _vector3(value, name=f"{node_id}.orientation")
    alpha, beta, gamma = angles
    sa, ca = math.sin(alpha), math.cos(alpha)
    sb, cb = math.sin(beta), math.cos(beta)
    sc, cc = math.sin(gamma), math.cos(gamma)
    return np.asarray(
        [
            [ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
            [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
            [-sb, cb * sc, cb * cc],
        ],
        dtype=np.float64,
    )


def _vector3(value: object, *, name: str) -> np.ndarray:
    try:
        components = [_component(getattr(value, axis)) for axis in ("x", "y", "z")]
    except AttributeError as error:
        raise TypeError(f"{name} must expose x, y, and z components") from error
    result = np.asarray([float(component[0]) for component in components], dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite components")
    return result


def _component(value: object) -> np.ndarray:
    numpy_method = getattr(value, "numpy", None)
    raw = numpy_method() if callable(numpy_method) else value
    return np.asarray(raw, dtype=np.float64).reshape(-1)


def _scalar(value: object, *, name: str) -> float:
    try:
        result = float(_component(value)[0])
    except (IndexError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a scalar convertible to float") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
