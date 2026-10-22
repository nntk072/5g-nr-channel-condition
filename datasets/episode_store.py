"""Immutable, pickle-free storage for linked UMCD episode data layers."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
from receivers.nr import NRChannelEstimate, NRPUSCHConfig

from .channel_view import ChannelDelayReference, ChannelPowerScale, PhysicalChannelView
from .network import (
    AntennaArraySpec,
    DirectedLink,
    LinkDirection,
    LinkRole,
    NetworkEpisode,
    NetworkNode,
    NodeRole,
)
from .nr import LinkedNRObservation, NRModelInputs, NRObservation
from .path_oracle import PathOracleTable, PathSolveStatus
from .targets import TargetDefinition, TargetValidity, TargetValue


@dataclass(frozen=True, slots=True)
class EpisodeBundle:
    """Episode with separately stored receiver, oracle, channel, and target views."""

    episode: NetworkEpisode
    path_oracle: PathOracleTable
    channel_views: tuple[PhysicalChannelView, ...]
    targets: tuple[TargetValue, ...]
    receiver_observations: tuple[LinkedNRObservation, ...] = ()

    def __post_init__(self) -> None:
        channels, targets = tuple(self.channel_views), tuple(self.targets)
        observations = tuple(self.receiver_observations)
        self.path_oracle.validate_for(self.episode)
        for view in channels:
            view.validate_for(self.episode)
        if len({view.view_id for view in channels}) != len(channels):
            raise ValueError("channel view IDs must be unique within an episode bundle")
        if len({(target.entity_id, target.definition.semantic) for target in targets}) != len(
            targets
        ):
            raise ValueError("target semantic must be unique per episode/link entity")
        valid_entities = {self.episode.episode_id, *(link.link_id for link in self.episode.links)}
        if any(target.entity_id not in valid_entities for target in targets):
            raise ValueError("target entity_id must reference the episode or one of its links")
        link_ids = {link.link_id for link in self.episode.links}
        frame_count = len(self.episode.frame_times_s)
        for binding in observations:
            if binding.link_id not in link_ids:
                raise ValueError("receiver observation link_id must reference an episode link")
            if binding.frame_index >= frame_count:
                raise ValueError("receiver observation frame_index must reference an episode frame")
        observation_keys = [(item.link_id, item.frame_index) for item in observations]
        sample_ids = [item.observation.sample_id for item in observations]
        if len(set(observation_keys)) != len(observation_keys):
            raise ValueError("receiver observations must be unique per link and frame")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("receiver observation sample IDs must be unique")
        object.__setattr__(self, "channel_views", channels)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "receiver_observations", observations)


@dataclass(frozen=True, slots=True)
class StoredEpisodeBundle:
    """Immutable pointer to the manifest-last committed episode bundle."""

    episode_id: str
    bundle_sha256: str
    relative_path: str
    schema_version: int = 1


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("episode metadata must contain finite JSON-compatible values") from exc


def _object_ref(layer: str, payload: bytes) -> dict[str, str]:
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "sha256": digest,
        "path": f"objects/{layer}/{digest[:2]}/{digest}.npz",
    }


def _pack_npz(metadata: dict[str, Any], **arrays: np.ndarray) -> bytes:
    entries: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.kind not in "biufc":
            raise TypeError(f"stored array {name!r} must use a non-object numeric dtype")
        entries[name] = np.ascontiguousarray(array)
    entries["metadata_json"] = np.frombuffer(_json_bytes(metadata), dtype=np.uint8)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **cast(Any, entries))
    return buffer.getvalue()


def _publish_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temp_path = stream.name
        try:
            os.link(temp_path, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def _episode_payload(episode: NetworkEpisode) -> bytes:
    metadata = {
        "schema": "umcd.episode.v1",
        "episode_id": episode.episode_id,
        "split_group_id": episode.split_group_id,
        "family_id": episode.family_id,
        "carrier_hz": episode.carrier_hz,
        "frame_times_s": list(episode.frame_times_s),
        "arrays": [
            {
                "array_id": array.array_id,
                "polarization_per_element": list(array.polarization_per_element),
                "port_ids": list(array.port_ids),
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
    }
    arrays = {
        f"array_positions_{index:04d}": array.element_positions_local_m
        for index, array in enumerate(episode.arrays)
    }
    arrays.update(
        {
            f"array_orientation_{index:04d}": array.orientation_local_to_world
            for index, array in enumerate(episode.arrays)
        }
    )
    arrays.update(
        {
            f"array_port_weights_{index:04d}": array.port_element_weights
            for index, array in enumerate(episode.arrays)
        }
    )
    arrays["subcarrier_offsets_hz"] = episode.subcarrier_offsets_hz
    return _pack_npz(metadata, **arrays)


def _path_payload(paths: PathOracleTable) -> bytes:
    metadata = {
        "schema": "umcd.path_oracle.v1",
        "link_ids": list(paths.link_ids),
        "solve_status": [value.value for value in paths.solve_status],
        "path_kind": None if paths.path_kind is None else list(paths.path_kind),
        "coefficient_convention": paths.coefficient_convention,
        "optional_fields": {
            "doppler_hz": paths.doppler_hz is not None,
            "aoa_azimuth_elevation_rad": paths.aoa_azimuth_elevation_rad is not None,
            "aod_azimuth_elevation_rad": paths.aod_azimuth_elevation_rad is not None,
            "interaction_geometry": paths.interaction_offsets is not None,
            "interaction_vertex_frame": paths.interaction_vertex_frame,
        },
    }
    arrays = {
        "frame_indices": paths.frame_indices,
        "rx_port_indices": paths.rx_port_indices,
        "tx_port_indices": paths.tx_port_indices,
        "path_offsets": paths.path_offsets,
        "delay_s": paths.delay_s,
        "complex_gain": paths.complex_gain,
        "interaction_order": paths.interaction_order,
    }
    for name in (
        "doppler_hz",
        "aoa_azimuth_elevation_rad",
        "aod_azimuth_elevation_rad",
    ):
        value = getattr(paths, name)
        if value is not None:
            arrays[name] = value
    for name in (
        "interaction_offsets",
        "interaction_types",
        "object_ids",
        "primitive_ids",
        "interaction_vertices_m",
    ):
        value = getattr(paths, name)
        if value is not None:
            arrays[name] = value
    return _pack_npz(metadata, **arrays)


def _channel_payload(view: PhysicalChannelView) -> bytes:
    metadata = {
        "schema": "umcd.physical_channel.v1",
        "view_id": view.view_id,
        "link_id": view.link_id,
        "frame_times_s": list(view.frame_times_s),
        "rx_port_ids": list(view.rx_port_ids),
        "tx_port_ids": list(view.tx_port_ids),
        "carrier_hz": view.carrier_hz,
        "delay_reference": view.delay_reference.value,
        "power_scale": view.power_scale.value,
        "normalization_recipe": view.normalization_recipe,
        "axes": list(view.axes),
    }
    return _pack_npz(
        metadata,
        channel=view.channel,
        subcarrier_offsets_hz=view.subcarrier_offsets_hz,
    )


def _target_payload(target: TargetValue) -> bytes:
    metadata = {
        "schema": "umcd.target.v1",
        "entity_id": target.entity_id,
        "definition": asdict(target.definition),
        "reason_codes": [reason.value for reason in target.reason_codes],
        "confidence": target.confidence,
    }
    return _pack_npz(metadata, values=target.values, valid_mask=target.valid_mask)


def _receiver_payload(binding: LinkedNRObservation) -> bytes:
    observation = binding.observation
    estimate = observation.channel_estimate
    metadata = {
        "schema": "umcd.receiver_observation.v1",
        "link_id": binding.link_id,
        "frame_index": binding.frame_index,
        "sample_id": observation.sample_id,
        "fft_bin_indices": list(observation.fft_bin_indices),
        "receiver_config": asdict(observation.receiver_config),
        "estimate_metadata": dict(estimate.metadata),
    }
    return _pack_npz(
        metadata,
        received_grid=observation.received_grid,
        h_hat=estimate.h_hat,
        error_variance=estimate.error_variance,
        pilot_mask=estimate.pilot_mask,
    )


def _read_receiver(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> LinkedNRObservation:
    if metadata.get("schema") != "umcd.receiver_observation.v1":
        raise ValueError("unsupported receiver-observation layer schema")
    observation = NRObservation(
        sample_id=metadata["sample_id"],
        received_grid=arrays["received_grid"],
        channel_estimate=NRChannelEstimate(
            h_hat=arrays["h_hat"],
            error_variance=arrays["error_variance"],
            pilot_mask=arrays["pilot_mask"],
            metadata=metadata["estimate_metadata"],
        ),
        fft_bin_indices=tuple(metadata["fft_bin_indices"]),
        receiver_config=NRPUSCHConfig(**metadata["receiver_config"]),
    )
    return LinkedNRObservation(metadata["link_id"], metadata["frame_index"], observation)


def write_episode_bundle(root: str | Path, bundle: EpisodeBundle) -> StoredEpisodeBundle:
    """Publish immutable layer objects, then atomically publish the bundle manifest.

    Oracle paths, physical channel tensors, and target values live in distinct
    object namespaces. Manifest publication occurs last; interrupted writes can
    leave only unreferenced content-addressed objects, never a partial release.
    """
    base = Path(root)
    payloads = [("episode", _episode_payload(bundle.episode))]
    payloads.append(("oracle_paths", _path_payload(bundle.path_oracle)))
    payloads.extend(("oracle_channels", _channel_payload(view)) for view in bundle.channel_views)
    payloads.extend(("targets", _target_payload(target)) for target in bundle.targets)
    payloads.extend(
        ("receiver_observations", _receiver_payload(binding))
        for binding in bundle.receiver_observations
    )
    references: list[dict[str, str]] = []
    for layer, payload in payloads:
        reference = _object_ref(layer, payload)
        _publish_immutable(base / reference["path"], payload)
        references.append({"layer": layer, **reference})
    manifest_core = {
        "schema": "umcd.bundle.v1",
        "schema_version": 1,
        "episode_id": bundle.episode.episode_id,
        "layers": references,
    }
    digest = hashlib.sha256(_json_bytes(manifest_core)).hexdigest()
    manifest = {**manifest_core, "bundle_sha256": digest}
    relative_path = (
        f"manifests/episodes/{hashlib.sha256(bundle.episode.episode_id.encode()).hexdigest()}.json"
    )
    _publish_immutable(base / relative_path, _json_bytes(manifest) + b"\n")
    return StoredEpisodeBundle(bundle.episode.episode_id, digest, relative_path)


def _read_object(
    base: Path, reference: dict[str, str]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    relative = PurePosixPath(reference["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("object paths must remain inside the episode store")
    payload = (base / Path(*relative.parts)).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != reference.get("sha256") or relative.name != f"{digest}.npz":
        raise ValueError("episode object content hash does not match its manifest")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].tobytes().decode("utf-8"))
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if name != "metadata_json"
        }
    return metadata, arrays


def _validated_manifest(
    base: Path, manifest: StoredEpisodeBundle | str | Path
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if isinstance(manifest, StoredEpisodeBundle):
        relative_path = PurePosixPath(manifest.relative_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("episode manifest path must remain inside the store")
        manifest_path = base / Path(*relative_path.parts)
    else:
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = base / manifest_path
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "umcd.bundle.v1" or metadata.get("schema_version") != 1:
        raise ValueError("unsupported episode bundle schema")
    manifest_core = {
        key: metadata[key] for key in ("schema", "schema_version", "episode_id", "layers")
    }
    digest = hashlib.sha256(_json_bytes(manifest_core)).hexdigest()
    if digest != metadata.get("bundle_sha256"):
        raise ValueError("episode bundle manifest hash is invalid")
    if isinstance(manifest, StoredEpisodeBundle) and (
        manifest.episode_id != metadata["episode_id"] or manifest.bundle_sha256 != digest
    ):
        raise ValueError("stored episode pointer does not match its manifest")
    layers = metadata["layers"]
    if not isinstance(layers, list):
        raise ValueError("episode bundle layers must be a list")
    return metadata, layers


def read_receiver_model_inputs(
    root: str | Path, manifest: StoredEpisodeBundle | str | Path
) -> tuple[NRModelInputs, ...]:
    """Load receiver-only inference inputs without opening oracle or target objects.

    The episode layer is read to validate link/frame bindings. Other manifest
    objects are deliberately not opened, so damaged or unavailable oracle and
    target files cannot leak into or block receiver inference.
    """
    base = Path(root)
    metadata, references = _validated_manifest(base, manifest)
    allowed_layers = {
        "episode",
        "oracle_paths",
        "oracle_channels",
        "targets",
        "receiver_observations",
    }
    episode_refs: list[dict[str, str]] = []
    observation_refs: list[dict[str, str]] = []
    for reference in references:
        layer = reference.get("layer")
        if layer not in allowed_layers:
            raise ValueError(f"unsupported episode bundle layer: {layer!r}")
        if layer == "episode":
            episode_refs.append(reference)
        elif layer == "receiver_observations":
            observation_refs.append(reference)
    if len(episode_refs) != 1:
        raise ValueError("bundle must contain exactly one episode layer")
    if not observation_refs:
        raise ValueError("bundle contains no receiver observations")
    episode_metadata, episode_arrays = _read_object(base, episode_refs[0])
    if episode_metadata.get("schema") != "umcd.episode.v1":
        raise ValueError("unsupported episode layer schema")
    episode = _read_episode(episode_metadata, episode_arrays)
    if metadata["episode_id"] != episode.episode_id:
        raise ValueError("manifest and episode payload IDs differ")
    link_ids = {link.link_id for link in episode.links}
    observations: list[LinkedNRObservation] = []
    for reference in observation_refs:
        receiver_metadata, receiver_arrays = _read_object(base, reference)
        binding = _read_receiver(receiver_metadata, receiver_arrays)
        if binding.link_id not in link_ids or binding.frame_index >= len(episode.frame_times_s):
            raise ValueError("receiver observation references an unknown link or frame")
        observations.append(binding)
    if len({(item.link_id, item.frame_index) for item in observations}) != len(observations):
        raise ValueError("receiver observations are duplicated for a link/frame")
    if len({item.observation.sample_id for item in observations}) != len(observations):
        raise ValueError("receiver observation sample IDs are duplicated")
    return tuple(item.observation.model_inputs() for item in observations)


def _read_episode(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> NetworkEpisode:
    antenna_arrays = tuple(
        AntennaArraySpec(
            item["array_id"],
            arrays[f"array_positions_{index:04d}"],
            tuple(item["polarization_per_element"]),
            arrays[f"array_orientation_{index:04d}"],
            tuple(item["port_ids"]),
            arrays[f"array_port_weights_{index:04d}"],
        )
        for index, item in enumerate(metadata["arrays"])
    )
    nodes = tuple(
        NetworkNode(
            item["node_id"],
            NodeRole(item["role"]),
            item["array_id"],
            tuple(item["position_world_m"]),
            item["rf_chain_count"],
        )
        for item in metadata["nodes"]
    )
    links = tuple(
        DirectedLink(
            item["link_id"],
            item["tx_node_id"],
            item["rx_node_id"],
            tuple(item["tx_port_indices"]),
            tuple(item["rx_port_indices"]),
            LinkDirection(item["direction"]),
            LinkRole(item["role"]),
        )
        for item in metadata["links"]
    )
    return NetworkEpisode(
        metadata["episode_id"],
        metadata["split_group_id"],
        metadata["family_id"],
        metadata["carrier_hz"],
        antenna_arrays,
        nodes,
        links,
        tuple(metadata["frame_times_s"]),
        arrays["subcarrier_offsets_hz"],
    )


def read_episode_bundle(
    root: str | Path, manifest: StoredEpisodeBundle | str | Path
) -> EpisodeBundle:
    """Load and checksum all linked layers without enabling pickle deserialization."""
    base = Path(root)
    metadata, references = _validated_manifest(base, manifest)
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, np.ndarray]]]] = {}
    allowed_layers = {
        "episode",
        "oracle_paths",
        "oracle_channels",
        "targets",
        "receiver_observations",
    }
    for reference in references:
        layer = reference.get("layer")
        if layer not in allowed_layers:
            raise ValueError(f"unsupported episode bundle layer: {layer!r}")
        object_metadata, arrays = _read_object(base, reference)
        grouped.setdefault(layer, []).append((object_metadata, arrays))
    if len(grouped.get("episode", [])) != 1 or len(grouped.get("oracle_paths", [])) != 1:
        raise ValueError("bundle must contain exactly one episode and one path-oracle layer")
    episode_metadata, episode_arrays = grouped["episode"][0]
    if episode_metadata.get("schema") != "umcd.episode.v1":
        raise ValueError("unsupported episode layer schema")
    episode = _read_episode(episode_metadata, episode_arrays)
    path_metadata, path_arrays = grouped["oracle_paths"][0]
    if path_metadata.get("schema") != "umcd.path_oracle.v1":
        raise ValueError("unsupported path-oracle layer schema")
    optional = path_metadata["optional_fields"]
    paths = PathOracleTable(
        link_ids=tuple(path_metadata["link_ids"]),
        frame_indices=path_arrays["frame_indices"],
        rx_port_indices=path_arrays["rx_port_indices"],
        tx_port_indices=path_arrays["tx_port_indices"],
        path_offsets=path_arrays["path_offsets"],
        solve_status=tuple(PathSolveStatus(value) for value in path_metadata["solve_status"]),
        delay_s=path_arrays["delay_s"],
        complex_gain=path_arrays["complex_gain"],
        interaction_order=path_arrays["interaction_order"],
        doppler_hz=path_arrays.get("doppler_hz") if optional["doppler_hz"] else None,
        aoa_azimuth_elevation_rad=(
            path_arrays.get("aoa_azimuth_elevation_rad")
            if optional["aoa_azimuth_elevation_rad"]
            else None
        ),
        aod_azimuth_elevation_rad=(
            path_arrays.get("aod_azimuth_elevation_rad")
            if optional["aod_azimuth_elevation_rad"]
            else None
        ),
        interaction_offsets=(
            path_arrays.get("interaction_offsets")
            if optional.get("interaction_geometry", False)
            else None
        ),
        interaction_types=(
            path_arrays.get("interaction_types")
            if optional.get("interaction_geometry", False)
            else None
        ),
        object_ids=(
            path_arrays.get("object_ids") if optional.get("interaction_geometry", False) else None
        ),
        primitive_ids=(
            path_arrays.get("primitive_ids")
            if optional.get("interaction_geometry", False)
            else None
        ),
        interaction_vertices_m=(
            path_arrays.get("interaction_vertices_m")
            if optional.get("interaction_geometry", False)
            else None
        ),
        interaction_vertex_frame=optional.get("interaction_vertex_frame"),
        path_kind=None if path_metadata["path_kind"] is None else tuple(path_metadata["path_kind"]),
        coefficient_convention=path_metadata["coefficient_convention"],
    )
    channels: list[PhysicalChannelView] = []
    for channel_metadata, arrays in grouped.get("oracle_channels", []):
        if channel_metadata.get("schema") != "umcd.physical_channel.v1":
            raise ValueError("unsupported physical-channel layer schema")
        channels.append(
            PhysicalChannelView(
                view_id=channel_metadata["view_id"],
                link_id=channel_metadata["link_id"],
                channel=arrays["channel"],
                frame_times_s=tuple(channel_metadata["frame_times_s"]),
                rx_port_ids=tuple(channel_metadata["rx_port_ids"]),
                tx_port_ids=tuple(channel_metadata["tx_port_ids"]),
                carrier_hz=channel_metadata["carrier_hz"],
                subcarrier_offsets_hz=arrays["subcarrier_offsets_hz"],
                delay_reference=ChannelDelayReference(channel_metadata["delay_reference"]),
                power_scale=ChannelPowerScale(channel_metadata["power_scale"]),
                normalization_recipe=channel_metadata["normalization_recipe"],
                axes=tuple(channel_metadata["axes"]),
            )
        )
    targets: list[TargetValue] = []
    for target_metadata, arrays in grouped.get("targets", []):
        if target_metadata.get("schema") != "umcd.target.v1":
            raise ValueError("unsupported target layer schema")
        targets.append(
            TargetValue(
                entity_id=target_metadata["entity_id"],
                definition=TargetDefinition(**target_metadata["definition"]),
                values=arrays["values"],
                valid_mask=arrays["valid_mask"],
                reason_codes=tuple(
                    TargetValidity(item) for item in target_metadata["reason_codes"]
                ),
                confidence=target_metadata["confidence"],
            )
        )
    observations = [
        _read_receiver(receiver_metadata, arrays)
        for receiver_metadata, arrays in grouped.get("receiver_observations", [])
    ]
    if metadata["episode_id"] != episode.episode_id:
        raise ValueError("manifest and episode payload IDs differ")
    return EpisodeBundle(episode, paths, tuple(channels), tuple(targets), tuple(observations))
