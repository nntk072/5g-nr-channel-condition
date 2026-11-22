"""Map whole Sionna RT solve outputs onto UMCD links and episode frames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
from artifacts.provenance import atomic_json, canonical_hash
from datasets.channel_view import ChannelDelayReference, ChannelPowerScale, PhysicalChannelView
from datasets.episode_store import (
    EpisodeBundle,
    StoredEpisodeBundle,
    write_episode_bundle,
)
from datasets.network import LinkDirection, LinkRole, NetworkEpisode, NetworkNode, NodeRole
from datasets.nr import LinkedNRObservation
from datasets.observations import LinkedFrequencyDomainObservation
from datasets.path_oracle import PathOracleTable, PathSolveStatus, rms_delay_spread_target
from datasets.provenance import UMCDRunProvenance
from datasets.targets import TargetValue

from integrations.sionna.d1_generation_gate import D1GenerationAuthorization
from integrations.sionna.path_oracle import sionna_paths_to_oracle_table
from integrations.sionna.rt_identity import umcd_provenance_from_rt_identity


def network_episode_identity(episode: NetworkEpisode) -> dict[str, Any]:
    """Return a deterministic content identity for every network topology input."""
    if not isinstance(episode, NetworkEpisode):
        raise TypeError("episode must be a NetworkEpisode")
    payload: dict[str, Any] = {
        "schema": "umcd.network_episode_identity.v1",
        "episode_id": episode.episode_id,
        "split_group_id": episode.split_group_id,
        "family_id": episode.family_id,
        "carrier_hz": episode.carrier_hz,
        "arrays": [
            {
                "array_id": array.array_id,
                "element_positions_local_m": array.element_positions_local_m.tolist(),
                "polarization_per_element": list(array.polarization_per_element),
                "orientation_local_to_world": array.orientation_local_to_world.tolist(),
                "port_ids": list(array.port_ids),
                "port_element_weights_real": array.port_element_weights.real.tolist(),
                "port_element_weights_imag": array.port_element_weights.imag.tolist(),
            }
            for array in episode.arrays
        ],
        "nodes": [
            {
                "node_id": node.node_id,
                "role": node.role.value,
                "array_id": node.array_id,
                "position_world_m": list(node.position_world_m),
                "rf_chain_count": node.rf_chain_count,
            }
            for node in episode.nodes
        ],
        "links": [
            {
                "link_id": link.link_id,
                "tx_node_id": link.tx_node_id,
                "rx_node_id": link.rx_node_id,
                "tx_port_indices": list(link.tx_port_indices),
                "rx_port_indices": list(link.rx_port_indices),
                "direction": link.direction.value,
                "role": link.role.value,
            }
            for link in episode.links
        ],
        "frame_times_s": list(episode.frame_times_s),
        "subcarrier_offsets_hz": episode.subcarrier_offsets_hz.tolist(),
        "split_leakage_relations": [list(value) for value in episode.split_leakage_relations],
        "pose_timeline": None,
        "pose_time_alignment_sha256": None,
        "source_time_alignment_sha256": None,
    }
    if episode.pose_timeline is not None:
        timeline = episode.pose_timeline
        payload["pose_timeline"] = {
            "node_ids": list(timeline.node_ids),
            "frame_times_s": list(timeline.frame_times_s),
            "positions_world_m": timeline.positions_world_m.tolist(),
            "orientations_local_to_world": timeline.orientations_local_to_world.tolist(),
        }
    if episode.pose_time_alignment is not None:
        payload["pose_time_alignment_sha256"] = canonical_hash(
            episode.pose_time_alignment.to_dict()
        )
    if episode.source_time_alignment is not None:
        payload["source_time_alignment_sha256"] = episode.source_time_alignment.sha256
    return {**payload, "sha256": canonical_hash(payload)}


def validate_downlink_network_episode(
    episode: NetworkEpisode,
) -> tuple[tuple[NetworkNode, ...], tuple[NetworkNode, ...]]:
    """Validate and return the finite complete downlink topology roster."""
    if not isinstance(episode, NetworkEpisode):
        raise TypeError("episode must be a NetworkEpisode")
    tx_nodes = tuple(
        node for node in episode.nodes if node.role in (NodeRole.GNB, NodeRole.INTERFERER)
    )
    rx_nodes = tuple(node for node in episode.nodes if node.role is NodeRole.UE)
    if len(tx_nodes) + len(rx_nodes) != len(episode.nodes) or not tx_nodes or not rx_nodes:
        raise ValueError(
            "supported network PHY topology requires only GNB/interferer TX and UE RX nodes"
        )
    tx_ids = tuple(node.node_id for node in tx_nodes)
    rx_ids = tuple(node.node_id for node in rx_nodes)
    if len({episode.array(node.array_id).num_ports for node in tx_nodes}) != 1:
        raise ValueError("network PHY currently requires homogeneous TX port counts")
    if len({episode.array(node.array_id).num_ports for node in rx_nodes}) != 1:
        raise ValueError("network PHY currently requires homogeneous RX port counts")
    expected_pairs = {(tx_id, rx_id) for tx_id in tx_ids for rx_id in rx_ids}
    links_by_pair = {(link.tx_node_id, link.rx_node_id): link for link in episode.links}
    if len(links_by_pair) != len(episode.links) or set(links_by_pair) != expected_pairs:
        raise ValueError("network PHY requires exactly one directed link for every TX/RX pair")
    if any(link.direction is not LinkDirection.DOWNLINK for link in episode.links):
        raise ValueError("network PHY adapter currently supports downlink links only")
    for rx_id in rx_ids:
        serving = [
            tx_id for tx_id in tx_ids if links_by_pair[(tx_id, rx_id)].role is LinkRole.SERVING
        ]
        if len(serving) != 1:
            raise ValueError("each receiver must have exactly one serving transmitter")
        if any(
            links_by_pair[(tx_id, rx_id)].role not in (LinkRole.SERVING, LinkRole.INTERFERING)
            for tx_id in tx_ids
        ):
            raise ValueError("non-serving downlink links must be marked interfering")
        if episode.node(serving[0]).role is not NodeRole.GNB:
            raise ValueError("the serving transmitter must have the GNB node role")
    return tx_nodes, rx_nodes


def build_rt_network_generation_request(
    episode: NetworkEpisode,
    *,
    scene_asset_manifest: Mapping[str, Any],
    scene_geometry_identity: Mapping[str, Any],
    material_config: Mapping[str, Any],
    backend_selection: Mapping[str, Any],
    seed: int,
    samples_per_src: int,
    max_depth: int,
    max_num_paths_per_src: int,
    destinations: Sequence[str | Path],
) -> dict[str, Any]:
    """Build a D1 request that binds a static shared solve to exact topology."""
    from integrations.sionna.rt_identity import static_rt_solver_config

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(samples_per_src, bool) or not isinstance(samples_per_src, int):
        raise TypeError("samples_per_src must be an integer")
    if samples_per_src < 1:
        raise ValueError("samples_per_src must be positive")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or not 1 <= max_depth <= 3:
        raise ValueError("D1 static network max_depth must be in [1, 3]")
    if (
        not isinstance(scene_asset_manifest, Mapping)
        or scene_asset_manifest.get("complete") is not True
    ):
        raise ValueError("D1 static network generation requires complete scene assets")
    asset_hash = scene_asset_manifest.get("asset_set_sha256")
    asset_count = scene_asset_manifest.get("asset_count")
    if (
        not isinstance(asset_hash, str)
        or len(asset_hash) != 64
        or any(character not in "0123456789abcdef" for character in asset_hash)
        or isinstance(asset_count, bool)
        or not isinstance(asset_count, int)
        or asset_count < 1
    ):
        raise ValueError("complete scene asset manifest needs a SHA-256 and positive asset count")
    scene_reference = scene_asset_manifest.get("scene_xml_path")
    if not isinstance(scene_reference, str) or not scene_reference.strip():
        raise ValueError("complete scene asset manifest needs a non-empty scene reference")
    if not isinstance(material_config, Mapping):
        raise TypeError("material_config must be a mapping")
    if not isinstance(scene_geometry_identity, Mapping):
        raise TypeError("scene_geometry_identity must be a mapping")
    geometry_digest = scene_geometry_identity.get("scene_geometry_sha256")
    object_count = scene_geometry_identity.get("scene_object_count")
    triangle_count = scene_geometry_identity.get("scene_triangle_count")
    if (
        not isinstance(geometry_digest, str)
        or len(geometry_digest) != 64
        or any(character not in "0123456789abcdef" for character in geometry_digest)
        or isinstance(object_count, bool)
        or not isinstance(object_count, int)
        or object_count < 1
        or isinstance(triangle_count, bool)
        or not isinstance(triangle_count, int)
        or triangle_count < 1
    ):
        raise ValueError("scene geometry identity needs a digest and positive inventory counts")
    if (
        not isinstance(backend_selection, Mapping)
        or backend_selection.get("probe_passed") is not True
    ):
        raise ValueError("RT backend selection must carry a successful native probe")
    solver_config = static_rt_solver_config(max_depth, max_num_paths_per_src)
    validate_downlink_network_episode(episode)
    topology = network_episode_identity(episode)
    work_units = 2 * samples_per_src
    if isinstance(destinations, (str, bytes)):
        raise TypeError("destinations must be a sequence of filesystem paths")
    resolved_destinations = sorted(
        {str(Path(value).expanduser().resolve()) for value in destinations}
    )
    if not resolved_destinations:
        raise ValueError("D1 static network request requires at least one destination")
    return {
        "operation": "rt_data",
        "scene": scene_reference,
        "seed": seed,
        "config": {
            "per_class": 1,
            "max_attempts": 2,
            "samples_per_src": samples_per_src,
            "max_depth": max_depth,
            "max_num_paths_per_src": max_num_paths_per_src,
            "total_work_units": work_units,
            "solver_config": solver_config,
        },
        "network_episode_identity": topology,
        "scene_geometry": {
            "scene_geometry_sha256": geometry_digest,
            "scene_object_count": object_count,
            "scene_triangle_count": triangle_count,
        },
        "grid": {
            "carrier_hz": episode.carrier_hz,
            "subcarrier_offsets_hz": episode.subcarrier_offsets_hz.tolist(),
        },
        "scene_asset_identity": {
            "scene_reference": scene_reference,
            "asset_set_sha256": asset_hash,
            "asset_count": asset_count,
            "complete": True,
            "runtime_versions": dict(backend_selection.get("runtime_versions", {})),
        },
        "material_state_sha256": canonical_hash(dict(material_config)),
        "backend": dict(backend_selection),
        "destinations": resolved_destinations,
    }


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


@dataclass(frozen=True, slots=True, eq=False)
class NetworkPHYChannelInput:
    """Episode-bound channel tensor for the aligned downlink network PHY.

    ``channel`` uses ``[frame, rx, rx_port, tx, tx_port, symbol, subcarrier]``
    axes. Frequency-domain physical channel views are repeated over the
    requested OFDM-symbol count; this adapter does not add time evolution.
    """

    episode_id: str
    tx_node_ids: tuple[str, ...]
    rx_node_ids: tuple[str, ...]
    frame_times_s: tuple[float, ...]
    subcarrier_offsets_hz: np.ndarray
    serving_tx_indices: tuple[int, ...]
    channel: np.ndarray

    def __post_init__(self) -> None:
        tx_ids = tuple(self.tx_node_ids)
        rx_ids = tuple(self.rx_node_ids)
        frame_times = tuple(float(value) for value in self.frame_times_s)
        offsets = np.array(self.subcarrier_offsets_hz, dtype=np.float64, copy=True)
        channel = np.array(self.channel, copy=True)
        serving = tuple(self.serving_tx_indices)
        if not self.episode_id or self.episode_id != self.episode_id.strip():
            raise ValueError("episode_id must be canonical and non-empty")
        if (
            not tx_ids
            or not rx_ids
            or len(set(tx_ids)) != len(tx_ids)
            or len(set(rx_ids)) != len(rx_ids)
        ):
            raise ValueError("network PHY input requires unique transmitter and receiver IDs")
        if not frame_times or not offsets.size or not np.iscomplexobj(channel):
            raise ValueError("network PHY input requires frames, subcarriers, and complex channels")
        if (
            channel.ndim != 7
            or channel.shape[0] != len(frame_times)
            or any(size <= 0 for size in channel.shape)
        ):
            raise ValueError("network PHY channel must have seven axes and one batch per frame")
        if channel.shape[1] != len(rx_ids) or channel.shape[3] != len(tx_ids):
            raise ValueError("network PHY node axes do not match their explicit IDs")
        if channel.shape[-1] != len(offsets) or not np.isfinite(channel).all():
            raise ValueError("network PHY frequency axis or channel values are invalid")
        if len(serving) != len(rx_ids) or any(
            isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(tx_ids)
            for index in serving
        ):
            raise ValueError("one valid serving transmitter index is required per receiver")
        if (
            not np.isfinite(offsets).all()
            or np.any(np.diff(offsets) <= 0)
            or not np.any(offsets == 0)
        ):
            raise ValueError("network PHY subcarrier offsets must increase and include DC")
        offsets.setflags(write=False)
        channel.setflags(write=False)
        object.__setattr__(self, "tx_node_ids", tx_ids)
        object.__setattr__(self, "rx_node_ids", rx_ids)
        object.__setattr__(self, "frame_times_s", frame_times)
        object.__setattr__(self, "subcarrier_offsets_hz", offsets)
        object.__setattr__(self, "serving_tx_indices", serving)
        object.__setattr__(self, "channel", channel)


def episode_channels_to_network_phy_input(
    episode: NetworkEpisode,
    channel_views: Sequence[PhysicalChannelView],
    *,
    ofdm_symbol_count: int,
) -> NetworkPHYChannelInput:
    """Bind stored physical per-link channels to the supported downlink PHY.

    Supported topology is a complete TX×RX directed-link matrix with exactly
    one serving TX per RX; remaining links are interfering. TX nodes must have
    a common port count, as must RX nodes. Missing links, unsupported roles,
    invalid channel masks, or normalized channel semantics are rejected rather
    than filled with synthetic values.
    """
    if not isinstance(episode, NetworkEpisode):
        raise TypeError("episode must be a NetworkEpisode")
    if isinstance(ofdm_symbol_count, bool) or not isinstance(ofdm_symbol_count, int):
        raise TypeError("ofdm_symbol_count must be an integer")
    if ofdm_symbol_count < 1:
        raise ValueError("ofdm_symbol_count must be positive")
    tx_nodes, rx_nodes = validate_downlink_network_episode(episode)
    tx_ids = tuple(node.node_id for node in tx_nodes)
    rx_ids = tuple(node.node_id for node in rx_nodes)
    links_by_pair = {(link.tx_node_id, link.rx_node_id): link for link in episode.links}

    view_values = tuple(channel_views)
    views_by_link = {view.link_id: view for view in view_values}
    if len(views_by_link) != len(view_values) or set(views_by_link) != {
        link.link_id for link in episode.links
    }:
        raise ValueError("one physical channel view is required for every episode link")
    serving_tx_indices: list[int] = []
    for rx_id in rx_ids:
        serving = next(
            tx_index
            for tx_index, tx_id in enumerate(tx_ids)
            if links_by_pair[(tx_id, rx_id)].role is LinkRole.SERVING
        )
        serving_tx_indices.append(serving)

    frames = len(episode.frame_times_s)
    rx_port_count = episode.array(rx_nodes[0].array_id).num_ports
    tx_port_count = episode.array(tx_nodes[0].array_id).num_ports
    subcarrier_count = len(episode.subcarrier_offsets_hz)
    channel = np.zeros(
        (
            frames,
            len(rx_ids),
            rx_port_count,
            len(tx_ids),
            tx_port_count,
            ofdm_symbol_count,
            subcarrier_count,
        ),
        dtype=np.complex128,
    )
    tx_index = {node_id: index for index, node_id in enumerate(tx_ids)}
    rx_index = {node_id: index for index, node_id in enumerate(rx_ids)}
    for pair, link in links_by_pair.items():
        view = views_by_link[link.link_id]
        if not isinstance(view, PhysicalChannelView):
            raise TypeError("channel_views must contain PhysicalChannelView values")
        view.validate_for(episode)
        if (
            view.delay_reference is not ChannelDelayReference.ABSOLUTE_PROPAGATION
            or view.power_scale is not ChannelPowerScale.ABSOLUTE
        ):
            raise ValueError("network PHY adapter requires absolute-delay absolute-power channels")
        if not np.all(view.valid_mask):
            raise ValueError(
                "network PHY adapter cannot consume channels with invalid masked elements"
            )
        for local_rx, episode_rx in enumerate(link.rx_port_indices):
            for local_tx, episode_tx in enumerate(link.tx_port_indices):
                channel[
                    :,
                    rx_index[pair[1]],
                    episode_rx,
                    tx_index[pair[0]],
                    episode_tx,
                    :,
                    :,
                ] = view.channel[:, local_rx, local_tx, :][:, None, :]
    return NetworkPHYChannelInput(
        episode_id=episode.episode_id,
        tx_node_ids=tx_ids,
        rx_node_ids=rx_ids,
        frame_times_s=episode.frame_times_s,
        subcarrier_offsets_hz=episode.subcarrier_offsets_hz,
        serving_tx_indices=tuple(serving_tx_indices),
        channel=channel,
    )


def sionna_bindings_to_channel_views(
    episode: NetworkEpisode,
    bindings: Sequence[SionnaPathBinding],
) -> tuple[PhysicalChannelView, ...]:
    """Convert per-link/frame Sionna ``Paths`` bindings to absolute CFR views.

    The Sionna CFR is evaluated on the episode's baseband subcarrier offsets
    with delay and power normalization disabled. Device and antenna axes are
    selected with each explicit binding and the episode link's port maps.
    Solver failures/unconverged bindings are represented by zero storage with
    a false validity mask, never as an observed zero channel. This conversion
    records simulator output; it does not establish physical channel truth.
    """
    if not isinstance(episode, NetworkEpisode):
        raise TypeError("episode must be a NetworkEpisode")
    binding_values = tuple(bindings)
    if not binding_values:
        raise ValueError("at least one Sionna path binding is required")
    expected = {
        (link.link_id, frame_index)
        for link in episode.links
        for frame_index in range(len(episode.frame_times_s))
    }
    actual = [(binding.link_id, binding.frame_index) for binding in binding_values]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Sionna bindings must cover every episode link/frame exactly once")

    by_key = {(binding.link_id, binding.frame_index): binding for binding in binding_values}
    views: list[PhysicalChannelView] = []
    for link in episode.links:
        rx_node = episode.node(link.rx_node_id)
        tx_node = episode.node(link.tx_node_id)
        rx_array = episode.array(rx_node.array_id)
        tx_array = episode.array(tx_node.array_id)
        rx_port_ids = tuple(rx_array.port_ids[index] for index in link.rx_port_indices)
        tx_port_ids = tuple(tx_array.port_ids[index] for index in link.tx_port_indices)
        shape = (
            len(episode.frame_times_s),
            len(rx_port_ids),
            len(tx_port_ids),
            len(episode.subcarrier_offsets_hz),
        )
        channel = np.zeros(shape, dtype=np.complex64)
        valid_mask = np.zeros(shape, dtype=np.bool_)
        for frame_index in range(len(episode.frame_times_s)):
            binding = by_key[(link.link_id, frame_index)]
            if binding.paths is None:
                continue
            if binding.solve_status in (
                PathSolveStatus.SOLVER_FAILURE,
                PathSolveStatus.UNCONVERGED,
            ):
                continue
            cfr_method = getattr(binding.paths, "cfr", None)
            if not callable(cfr_method):
                raise TypeError("Sionna Paths binding must expose cfr()")
            raw = cfr_method(
                frequencies=np.asarray(episode.subcarrier_offsets_hz, dtype=np.float64),
                num_time_steps=1,
                normalize_delays=False,
                normalize=False,
                out_type="numpy",
            )
            if isinstance(raw, (tuple, list)) and len(raw) == 2:
                real, imag = (np.asarray(value) for value in raw)
                if real.shape != imag.shape:
                    raise ValueError("Sionna CFR real and imaginary arrays must have equal shapes")
                response = real + 1j * imag
            else:
                response = np.asarray(raw)
            if not np.iscomplexobj(response) or response.ndim != 6:
                raise ValueError(
                    "Sionna CFR must have [rx, rx_port, tx, tx_port, time, subcarrier] axes"
                )
            if response.shape[-2:] != (1, len(episode.subcarrier_offsets_hz)):
                raise ValueError("Sionna CFR time/frequency axes do not match the episode grid")
            if (
                binding.rx_device_index >= response.shape[0]
                or binding.tx_device_index >= response.shape[2]
                or any(index >= response.shape[1] for index in link.rx_port_indices)
                or any(index >= response.shape[3] for index in link.tx_port_indices)
            ):
                raise IndexError("Sionna CFR device or port index is outside its returned axes")
            # Normalize the result to [rx_port, tx_port, subcarrier].
            selected = np.take(
                response[binding.rx_device_index, :, binding.tx_device_index, :, 0, :],
                link.rx_port_indices,
                axis=0,
            )
            selected = np.take(selected, link.tx_port_indices, axis=1)
            if selected.shape != shape[1:]:
                raise ValueError("selected Sionna CFR ports do not match the episode link")
            if not np.isfinite(selected).all():
                raise ValueError("Sionna CFR contains nonfinite channel values")
            channel[frame_index] = selected.astype(np.complex64)
            valid_mask[frame_index] = True
        views.append(
            PhysicalChannelView(
                view_id=f"{episode.episode_id}:{link.link_id}:absolute-cfr",
                link_id=link.link_id,
                channel=channel,
                valid_mask=valid_mask,
                frame_times_s=episode.frame_times_s,
                rx_port_ids=rx_port_ids,
                tx_port_ids=tx_port_ids,
                carrier_hz=episode.carrier_hz,
                subcarrier_offsets_hz=episode.subcarrier_offsets_hz,
                delay_reference=ChannelDelayReference.ABSOLUTE_PROPAGATION,
                power_scale=ChannelPowerScale.ABSOLUTE,
            )
        )
    for view in views:
        view.validate_for(episode)
    return tuple(views)


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


def build_sionna_rt_episode_bundle(
    episode: NetworkEpisode,
    bindings: Sequence[SionnaPathBinding],
    *,
    run_identity: Mapping[str, Any],
    material_config: Mapping[str, Any],
    source_family: str,
    producer: str,
    derive_rms_delay_spread_targets: bool = False,
    channel_views: Sequence[PhysicalChannelView] = (),
    targets: Sequence[TargetValue] = (),
    receiver_observations: Sequence[LinkedNRObservation | LinkedFrequencyDomainObservation] = (),
) -> EpisodeBundle:
    """Build a canonical RT episode bundle and provenance for every populated layer.

    All outputs are bound to the same complete RT run identity. Receiver
    observations remain a separate layer from simulator paths, channels, and
    targets. Incomplete scene/run identities are rejected by the provenance
    adapter before the bundle can be published.
    """
    if not isinstance(derive_rms_delay_spread_targets, bool):
        raise TypeError("derive_rms_delay_spread_targets must be a bool")
    oracle = sionna_bindings_to_episode_oracle(episode, bindings)
    channels = tuple(channel_views)
    target_items = list(targets)
    if derive_rms_delay_spread_targets:
        target_items.extend(
            rms_delay_spread_target(
                oracle,
                link_id=link.link_id,
                frame_count=len(episode.frame_times_s),
            )
            for link in episode.links
        )
    target_values = tuple(target_items)
    observations = tuple(receiver_observations)
    layers = ["oracle_paths"]
    if channels:
        layers.append("oracle_channels")
    if target_values:
        layers.append("targets")
    if observations:
        layers.append("receiver_observations")
    provenance: tuple[UMCDRunProvenance, ...] = tuple(
        umcd_provenance_from_rt_identity(
            run_identity,
            record_id=f"{episode.episode_id}:{layer}",
            subject_layer=layer,
            source_family=source_family,
            producer=producer,
            material_config=material_config,
        )
        for layer in layers
    )
    return EpisodeBundle(
        episode=episode,
        path_oracle=oracle,
        channel_views=channels,
        targets=target_values,
        receiver_observations=observations,
        provenance_records=provenance,
    )


def build_sionna_rt_episode_bundle_from_paths(
    episode: NetworkEpisode,
    bindings: Sequence[SionnaPathBinding],
    *,
    run_identity: Mapping[str, Any],
    material_config: Mapping[str, Any],
    source_family: str,
    producer: str,
    targets: Sequence[TargetValue] = (),
    receiver_observations: Sequence[LinkedNRObservation | LinkedFrequencyDomainObservation] = (),
) -> EpisodeBundle:
    """Build one provenance-bound bundle from complete RT path bindings.

    Unlike ``build_sionna_rt_episode_bundle``, this producer entry point also
    derives and includes each link's absolute CFR view from the bound Paths
    results, so path and channel layers share the same episode and run identity.
    """
    binding_values = tuple(bindings)
    channel_views = sionna_bindings_to_channel_views(episode, binding_values)
    return build_sionna_rt_episode_bundle(
        episode,
        binding_values,
        run_identity=run_identity,
        material_config=material_config,
        source_family=source_family,
        producer=producer,
        channel_views=channel_views,
        targets=targets,
        receiver_observations=receiver_observations,
    )


def write_sionna_rt_episode_bundle(
    store_root: str | Path,
    episode: NetworkEpisode,
    bindings: Sequence[SionnaPathBinding],
    *,
    run_identity: Mapping[str, Any],
    material_config: Mapping[str, Any],
    source_family: str,
    producer: str,
    derive_rms_delay_spread_targets: bool = False,
    channel_views: Sequence[PhysicalChannelView] = (),
    targets: Sequence[TargetValue] = (),
    receiver_observations: Sequence[LinkedNRObservation | LinkedFrequencyDomainObservation] = (),
    authorization: D1GenerationAuthorization | None = None,
) -> StoredEpisodeBundle:
    """Build and immutably publish a provenance-complete Sionna RT bundle."""
    from integrations.sionna.d1_generation_gate import require_pilot_destination

    require_pilot_destination(
        authorization,
        store_root,
        publisher="umcd_bundle",
    )  # type: ignore[arg-type]
    bundle = build_sionna_rt_episode_bundle(
        episode,
        bindings,
        run_identity=run_identity,
        material_config=material_config,
        source_family=source_family,
        producer=producer,
        derive_rms_delay_spread_targets=derive_rms_delay_spread_targets,
        channel_views=channel_views,
        targets=targets,
        receiver_observations=receiver_observations,
    )
    return write_episode_bundle(store_root, bundle)


def write_sionna_rt_episode_bundle_from_paths(
    store_root: str | Path,
    episode: NetworkEpisode,
    bindings: Sequence[SionnaPathBinding],
    *,
    run_identity: Mapping[str, Any],
    material_config: Mapping[str, Any],
    source_family: str,
    producer: str,
    targets: Sequence[TargetValue] = (),
    receiver_observations: Sequence[LinkedNRObservation | LinkedFrequencyDomainObservation] = (),
    authorization: D1GenerationAuthorization | None = None,
) -> StoredEpisodeBundle:
    """Build channel views from Paths bindings and immutably publish the bundle."""
    from integrations.sionna.d1_generation_gate import require_pilot_destination

    require_pilot_destination(
        authorization,
        store_root,
        publisher="umcd_bundle",
    )  # type: ignore[arg-type]
    bundle = build_sionna_rt_episode_bundle_from_paths(
        episode,
        bindings,
        run_identity=run_identity,
        material_config=material_config,
        source_family=source_family,
        producer=producer,
        targets=targets,
        receiver_observations=receiver_observations,
    )
    return write_episode_bundle(store_root, bundle)


@dataclass(frozen=True, slots=True)
class StaticNetworkRTGenerationResult:
    """Retained outcome for one authorized static shared-network solve."""

    status: str
    candidate_path_counts_per_source: tuple[int, ...] | None
    bindings: tuple[SionnaPathBinding, ...]
    stored_bundle: StoredEpisodeBundle
    receipt_path: Path
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("SOLVED", "UNCONVERGED", "SOLVER_FAILURE"):
            raise ValueError("unsupported static network RT outcome status")
        if (self.error_type is None) != (self.error_message is None):
            raise ValueError("error type and message must be present together")
        if self.status == "SOLVED" and self.error_type is not None:
            raise ValueError("a solved network outcome cannot carry an error")
        if self.status != "SOLVED" and self.error_type is None:
            raise ValueError("an unsuccessful network outcome must preserve its error")


def _validate_static_network_run_binding(
    episode: NetworkEpisode,
    *,
    request: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    material_config: Mapping[str, Any],
    backend_selection: Any,
) -> None:
    topology = network_episode_identity(episode)
    if request.get("network_episode_identity") != topology:
        raise PermissionError("D1 request does not bind the current network episode topology")
    if not isinstance(run_identity, Mapping) or run_identity.get("complete") is not True:
        raise ValueError("static network publisher requires a complete RT run identity")
    inputs = run_identity.get("inputs")
    key = run_identity.get("cache_key_sha256")
    if not isinstance(inputs, Mapping) or not isinstance(key, str):
        raise ValueError("RT run identity is missing its canonical inputs or cache key")
    if canonical_hash(dict(inputs)) != key:
        raise ValueError("RT run identity content does not match its cache key")
    physical = inputs.get("physical_config")
    if not isinstance(physical, Mapping) or physical.get("network_episode_identity") != topology:
        raise ValueError("RT run identity does not bind the exact network episode topology")
    config = request.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("D1 request lacks its generation configuration")
    from integrations.sionna.rt_identity import static_rt_solver_config

    max_depth = config.get("max_depth")
    max_num_paths_per_src = config.get("max_num_paths_per_src")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int):
        raise ValueError("D1 request max_depth must be an integer")
    if isinstance(max_num_paths_per_src, bool) or not isinstance(max_num_paths_per_src, int):
        raise ValueError("D1 request max_num_paths_per_src must be an integer")
    expected_solver_config = static_rt_solver_config(max_depth, max_num_paths_per_src)
    if config.get("solver_config") != expected_solver_config:
        raise ValueError("D1 request solver switches do not match its bounded solver settings")
    if inputs.get("solver_config") != config.get("solver_config"):
        raise ValueError("RT run identity solver configuration differs from the D1 request")
    if inputs.get("seed") != request.get("seed"):
        raise ValueError("RT run identity seed differs from the D1 request")
    if physical.get("scene_geometry") != request.get("scene_geometry"):
        raise ValueError("RT run identity scene geometry differs from the D1 request")
    if canonical_hash(dict(inputs.get("material_config", {}))) != canonical_hash(
        dict(material_config)
    ):
        raise ValueError("RT run identity material differs from the requested material")
    if canonical_hash(dict(material_config)) != request.get("material_state_sha256"):
        raise ValueError("D1 request material digest differs from the run identity material")
    scene_identity = request.get("scene_asset_identity")
    if not isinstance(scene_identity, Mapping) or inputs.get("scene_asset_set_sha256") != (
        scene_identity.get("asset_set_sha256")
    ):
        raise ValueError("RT run identity scene asset set differs from the D1 request")
    if request.get("scene") != scene_identity.get("scene_reference"):
        raise ValueError("D1 request scene reference fields disagree")
    expected_grid = {
        "carrier_hz": episode.carrier_hz,
        "subcarrier_offsets_hz": episode.subcarrier_offsets_hz.tolist(),
    }
    if request.get("grid") != expected_grid:
        raise ValueError("D1 request frequency grid differs from the network topology")
    backend = inputs.get("backend")
    selected = backend_selection.as_dict()
    if not isinstance(backend, Mapping) or any(
        backend.get(key) != selected.get(key) for key in ("backend", "variant", "runtime_versions")
    ):
        raise ValueError("RT run identity backend differs from the verified backend selection")
    if request.get("backend") != selected:
        raise PermissionError("D1 request does not bind the selected verified backend")


def _validate_live_static_network_scene(
    scene: object, episode: NetworkEpisode, request: Mapping[str, Any]
) -> None:
    """Recheck scene assets, materials, devices, arrays and poses before solving."""
    if episode.frame_times_s != (0.0,) or episode.pose_timeline is not None:
        raise ValueError("static shared-network RT generation requires one static episode frame")
    scene_identity = request.get("scene_asset_identity")
    if not isinstance(scene_identity, Mapping):
        raise ValueError("D1 request lacks scene asset identity")
    scene_reference = scene_identity.get("scene_reference")
    if not isinstance(scene_reference, str) or not Path(scene_reference).is_file():
        raise ValueError("authorized static network scene must resolve to a local XML asset")
    from integrations.sionna.scene_assets import (
        scene_asset_manifest,
        sionna_scene_material_manifest,
    )

    manifest = scene_asset_manifest(scene_reference)
    if (
        manifest.get("complete") is not True
        or manifest.get("asset_set_sha256") != scene_identity.get("asset_set_sha256")
        or manifest.get("asset_count") != scene_identity.get("asset_count")
    ):
        raise PermissionError("live scene asset closure differs from the D1 request")
    material = sionna_scene_material_manifest(scene)
    if canonical_hash(material) != request.get("material_state_sha256"):
        raise PermissionError("live scene material state differs from the D1 request")
    from propagation.scene_visibility import extract_scene_triangle_inventory

    inventory = extract_scene_triangle_inventory(scene)
    geometry_identity = {
        "scene_geometry_sha256": inventory.scene_geometry_sha256,
        "scene_object_count": len(inventory.object_names),
        "scene_triangle_count": len(inventory.triangles_m),
    }
    if geometry_identity != request.get("scene_geometry"):
        raise PermissionError("live scene geometry differs from the D1 request")
    if float(getattr(scene, "frequency", float("nan"))) != episode.carrier_hz:
        raise ValueError("live scene carrier differs from the network episode")

    from integrations.sionna.array_layouts import sionna_scene_to_network_episode

    roles = {node.node_id: node.role for node in episode.nodes}
    rf_counts = {node.node_id: node.rf_chain_count for node in episode.nodes}
    associations = tuple((link.tx_node_id, link.rx_node_id) for link in episode.links)
    roles_by_pair = {(link.tx_node_id, link.rx_node_id): link.role for link in episode.links}

    def pattern_labels(node_role: NodeRole) -> tuple[str, ...] | None:
        candidates = [node for node in episode.nodes if node.role is node_role]
        if not candidates:
            return None
        port_ids = episode.array(candidates[0].array_id).port_ids
        return tuple(dict.fromkeys(port_id.split(":element-", 1)[0] for port_id in port_ids))

    refreshed = sionna_scene_to_network_episode(
        scene,
        episode_id=episode.episode_id,
        split_group_id=episode.split_group_id,
        family_id=episode.family_id,
        subcarrier_offsets_hz=tuple(float(value) for value in episode.subcarrier_offsets_hz),
        rf_chain_count_by_node=rf_counts,
        node_roles=roles,
        tx_pattern_labels=pattern_labels(NodeRole.GNB),
        rx_pattern_labels=pattern_labels(NodeRole.UE),
        associations=associations,
        link_roles_by_association=roles_by_pair,
        split_leakage_relations=episode.split_leakage_relations,
        direction=LinkDirection.DOWNLINK,
    )
    if network_episode_identity(refreshed.episode) != network_episode_identity(episode):
        raise PermissionError("live scene device, array or pose state differs from the D1 topology")


def run_authorized_static_network_episode(
    store_root: str | Path,
    scene: object,
    scene_network: Any,
    *,
    request: Mapping[str, Any],
    authorization: D1GenerationAuthorization | None,
    run_identity: Mapping[str, Any],
    material_config: Mapping[str, Any],
    backend_selection: Any,
    source_family: str,
    producer: str,
    _path_solver_factory: Any | None = None,
    _candidate_counter_factory: Any | None = None,
) -> StaticNetworkRTGenerationResult:
    """Authorize, solve, audit and immutably publish one static downlink network.

    A single shared Sionna Paths result covers the selected TX×RX matrix. The
    request and cache identity must bind the exact topology and solver inputs.
    Candidate-cap or solver failures are written as per-link UNCONVERGED or
    SOLVER_FAILURE groups rather than disappearing or becoming valid zero CFRs.
    """
    from integrations.sionna.d1_generation_gate import (
        require_d1_authorization,
        require_pilot_destination,
    )

    episode = getattr(scene_network, "episode", None)
    if not isinstance(episode, NetworkEpisode):
        raise TypeError("scene_network must expose a NetworkEpisode")
    validate_downlink_network_episode(episode)
    _validate_static_network_run_binding(
        episode,
        request=request,
        run_identity=run_identity,
        material_config=material_config,
        backend_selection=backend_selection,
    )
    require_d1_authorization(authorization, request, expected_operation="rt_data")
    require_pilot_destination(authorization, store_root, publisher="umcd_bundle")
    assert authorization is not None
    _validate_live_static_network_scene(scene, episode, request)

    config = request["config"]
    solver_config = config["solver_config"]
    cap = solver_config["max_num_paths_per_src"]
    seed = request["seed"]
    solver_kwargs = {key: value for key, value in solver_config.items() if key != "deterministic"}
    solver_kwargs.update(samples_per_src=config["samples_per_src"], seed=seed)
    if _path_solver_factory is None:
        from integrations.sionna.backend import activate_rt_backend

        activate_rt_backend(backend_selection)
        from sionna.rt import PathSolver

        path_solver_factory = PathSolver
    else:
        path_solver_factory = _path_solver_factory
    solver = path_solver_factory(deterministic=solver_config["deterministic"])
    if _candidate_counter_factory is None:
        from integrations.sionna.candidate_audit import instrument_path_solver_candidate_counter

        counter = instrument_path_solver_candidate_counter(solver)
    else:
        counter = _candidate_counter_factory(solver)

    outcome_status = "SOLVED"
    error_type: str | None = None
    error_message: str | None = None
    paths: object | None
    try:
        paths = solver(scene, **solver_kwargs)
        counts = counter.last_candidate_path_counts_per_source
        if counts is None or len(counts) != len({link.tx_node_id for link in episode.links}):
            raise RuntimeError("candidate audit did not return one count per transmitter source")
        candidate_counts = tuple(int(value) for value in counts)
        if any(value < 0 for value in candidate_counts):
            raise RuntimeError("candidate audit returned a negative per-source count")
        if any(value >= cap for value in candidate_counts):
            outcome_status = "UNCONVERGED"
            error_type = "CandidatePathCapFilled"
            error_message = (
                f"per-source candidate count reached the configured cap {cap}: {candidate_counts}"
            )
            paths = None
        else:
            error_type = None
            error_message = None
    except Exception as error:
        paths = None
        candidate_counts = None
        outcome_status = "SOLVER_FAILURE"
        error_type = type(error).__name__
        error_message = str(error)

    status = (
        None
        if outcome_status == "SOLVED"
        else PathSolveStatus.UNCONVERGED
        if outcome_status == "UNCONVERGED"
        else PathSolveStatus.SOLVER_FAILURE
    )
    path_values = {mapping.link_id: paths for mapping in scene_network.device_link_mappings}
    status_values = {} if status is None else {link_id: status for link_id in path_values}
    bindings = scene_network.bind_paths(path_values, solve_status_by_link_id=status_values)
    bundle = build_sionna_rt_episode_bundle_from_paths(
        episode,
        bindings,
        run_identity=run_identity,
        material_config=material_config,
        source_family=source_family,
        producer=producer,
    )
    stored = write_episode_bundle(store_root, bundle)
    report_payload = {
        "schema": "umcd.static_network_rt_generation_receipt.v1",
        "episode_id": episode.episode_id,
        "network_episode_identity_sha256": network_episode_identity(episode)["sha256"],
        "run_identity_sha256": run_identity["cache_key_sha256"],
        "authorization_request_sha256": authorization.request_sha256,
        "bundle_sha256": stored.bundle_sha256,
        "status": outcome_status,
        "candidate_path_counts_per_source": (
            None if candidate_counts is None else list(candidate_counts)
        ),
        "candidate_path_cap_per_source": cap,
        "seed": seed,
        "error_type": error_type,
        "error_message": error_message,
    }
    receipt = {
        **report_payload,
        "receipt_sha256": canonical_hash(report_payload),
    }
    receipt_path = Path(store_root) / "reports" / "network-rt" / f"{stored.bundle_sha256}.json"
    try:
        atomic_json(receipt_path, receipt, overwrite=False)
    except FileExistsError:
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing != receipt:
            raise ValueError("immutable network RT receipt already exists with different content")
    return StaticNetworkRTGenerationResult(
        status=outcome_status,
        candidate_path_counts_per_source=candidate_counts,
        bindings=bindings,
        stored_bundle=stored,
        receipt_path=receipt_path,
        error_type=error_type,
        error_message=error_message,
    )


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
