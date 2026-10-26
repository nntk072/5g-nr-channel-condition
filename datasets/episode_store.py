"""Immutable, pickle-free storage for linked UMCD episode data layers."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import stat
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterator, cast

import numpy as np
from receivers.nr import NRChannelEstimate, NRPUSCHConfig

from .channel_view import ChannelDelayReference, ChannelPowerScale, PhysicalChannelView
from .counterfactual import DirectPathAblation, validate_direct_path_ablation_source
from .network import (
    AntennaArraySpec,
    DirectedLink,
    LinkDirection,
    LinkRole,
    NetworkEpisode,
    NetworkNode,
    NodePoseTimeline,
    NodeRole,
)
from .nr import LinkedNRObservation, NRModelInputs, NRObservation
from .observations import (
    FrequencyDomainModelInputs,
    FrequencyDomainObservation,
    LinkedFrequencyDomainObservation,
)
from .path_oracle import PathOracleTable, PathSolveStatus
from .provenance import UMCDRunProvenance
from .targets import (
    TargetDefinition,
    TargetValidity,
    TargetValue,
    migrate_legacy_target_definition_v1,
)

if TYPE_CHECKING:
    from .scenario_gbsm import ScenarioGBSMRecord


@dataclass(frozen=True, slots=True)
class EpisodeBundle:
    """Episode with separately stored receiver, oracle, channel, and target views."""

    episode: NetworkEpisode
    path_oracle: PathOracleTable | None
    channel_views: tuple[PhysicalChannelView, ...]
    targets: tuple[TargetValue, ...]
    receiver_observations: tuple[LinkedNRObservation | LinkedFrequencyDomainObservation, ...] = ()
    provenance_records: tuple[UMCDRunProvenance, ...] = ()
    scenario_gbsm_records: tuple[ScenarioGBSMRecord, ...] = ()
    direct_path_ablations: tuple[DirectPathAblation, ...] = ()

    def __post_init__(self) -> None:
        channels, targets = tuple(self.channel_views), tuple(self.targets)
        observations = tuple(self.receiver_observations)
        provenance = tuple(self.provenance_records)
        scenario_gbsm_records = tuple(self.scenario_gbsm_records)
        direct_path_ablations = tuple(self.direct_path_ablations)
        if self.path_oracle is not None:
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
            if isinstance(binding, LinkedFrequencyDomainObservation):
                observation_inputs = binding.observation.model_inputs
                link = self.episode.link(binding.link_id)
                expected_shape = (
                    len(link.rx_port_indices),
                    len(link.tx_port_indices),
                    len(self.episode.subcarrier_offsets_hz),
                )
                if observation_inputs.observed_cfr.shape != expected_shape:
                    raise ValueError("frequency observation axes do not match link ports and grid")
                if observation_inputs.carrier_hz != self.episode.carrier_hz or not np.array_equal(
                    observation_inputs.subcarrier_offsets_hz,
                    self.episode.subcarrier_offsets_hz,
                ):
                    raise ValueError("frequency observation grid differs from its parent episode")
            elif not isinstance(binding, LinkedNRObservation):
                raise TypeError("receiver observations must use a supported linked contract")
        observation_keys = [(item.link_id, item.frame_index) for item in observations]
        sample_ids = [item.observation.sample_id for item in observations]
        if len(set(observation_keys)) != len(observation_keys):
            raise ValueError("receiver observations must be unique per link and frame")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("receiver observation sample IDs must be unique")
        if any(not isinstance(item, UMCDRunProvenance) for item in provenance):
            raise TypeError("provenance_records must contain UMCDRunProvenance values")
        if len({item.record_id for item in provenance}) != len(provenance):
            raise ValueError("provenance record IDs must be unique within a bundle")
        if len(scenario_gbsm_records) > 1:
            raise ValueError("an episode bundle can contain one scenario GBSM drop")
        for record in scenario_gbsm_records:
            validator = getattr(record, "validate_for_episode", None)
            if not callable(validator):
                raise TypeError("scenario_gbsm_records must contain scenario GBSM records")
            validator(self.episode)
        if any(not isinstance(item, DirectPathAblation) for item in direct_path_ablations):
            raise TypeError("direct_path_ablations must contain DirectPathAblation values")
        if direct_path_ablations and self.path_oracle is None:
            raise ValueError("direct-path ablations require their source path-oracle layer")
        source_paths = self.path_oracle
        ablation_children: set[str] = set()
        for item in direct_path_ablations:
            if source_paths is None:
                raise ValueError("direct-path ablation source path-oracle layer is missing")
            if item.parent_subject_id != self.episode.episode_id:
                raise ValueError("direct-path ablation parent must match the episode ID")
            validate_direct_path_ablation_source(source_paths, item)
            if not np.array_equal(item.frequency_offset_hz, self.episode.subcarrier_offsets_hz):
                raise ValueError("direct-path ablation frequency grid differs from episode grid")
            pair_children = {item.base_child_id, item.ablated_child_id}
            if ablation_children.intersection(pair_children):
                raise ValueError("direct-path ablation child IDs must be unique within the bundle")
            ablation_children.update(pair_children)
        if not any(
            (
                self.path_oracle is not None,
                channels,
                targets,
                observations,
                scenario_gbsm_records,
                direct_path_ablations,
            )
        ):
            raise ValueError("episode bundle needs at least one non-episode data layer")
        available_layers = {"episode"}
        if self.path_oracle is not None:
            available_layers.add("oracle_paths")
        if channels:
            available_layers.add("oracle_channels")
        if targets:
            available_layers.add("targets")
        if observations:
            available_layers.add("receiver_observations")
        if scenario_gbsm_records:
            available_layers.add("scenario_gbsm")
        if direct_path_ablations:
            available_layers.add("counterfactual_ablations")
        if any(item.subject_layer not in available_layers for item in provenance):
            raise ValueError("provenance subject_layer must identify a populated bundle layer")
        object.__setattr__(self, "channel_views", channels)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "receiver_observations", observations)
        object.__setattr__(self, "provenance_records", provenance)
        object.__setattr__(self, "scenario_gbsm_records", scenario_gbsm_records)
        object.__setattr__(self, "direct_path_ablations", direct_path_ablations)


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


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys so signed/hash-bound metadata is unambiguous."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _parse_json_bytes(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError(f"{label} is not valid strict UTF-8 JSON: {error}") from error


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
            existing = _stable_file_bytes(
                path, label="existing immutable episode object", max_bytes=len(payload)
            )
            if existing != payload:
                raise FileExistsError(
                    "immutable episode object path already contains different bytes"
                )
        _fsync_directory(path.parent)
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    """Persist a newly linked immutable entry where directory fsync is supported."""
    if os.name != "posix" or not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stable_file_bytes(path: Path, *, label: str, max_bytes: int | None = None) -> bytes:
    """Read a single-link regular file and reject observable concurrent changes."""
    before_path = os.lstat(path)
    if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before_fd = os.fstat(descriptor)
        if _file_identity(before_fd) != _file_identity(before_path):
            raise ValueError(f"{label} changed while opening")
        if max_bytes is not None and before_fd.st_size > max_bytes:
            raise ValueError(f"{label} exceeds the configured byte limit")
        chunks: list[bytes] = []
        size = 0
        while True:
            request_size = 1024 * 1024
            if max_bytes is not None:
                request_size = min(request_size, max_bytes + 1 - size)
                if request_size <= 0:
                    raise ValueError(f"{label} exceeds the configured byte limit")
            chunk = os.read(descriptor, request_size)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if max_bytes is not None and size > max_bytes:
                raise ValueError(f"{label} exceeds the configured byte limit")
        after_fd = os.fstat(descriptor)
        after_path = os.lstat(path)
        identity = _file_identity(before_fd)
        if size != before_fd.st_size or _file_identity(after_fd) != identity:
            raise ValueError(f"{label} changed while being read")
        if _file_identity(after_path) != identity:
            raise ValueError(f"{label} path changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _episode_object_size_estimate(
    base: Path,
    reference: dict[str, str],
    *,
    max_shard_bytes: int,
) -> int:
    """Hash-check an NPZ layer and conservatively estimate its decoded memory."""
    relative = PurePosixPath(reference["path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("episode object path must remain inside the store")
    if relative.suffix != ".npz" or relative.name != f"{reference['sha256']}.npz":
        raise ValueError("episode object path must be content-addressed")
    path = base.joinpath(*relative.parts)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(base):
        raise ValueError("episode object path resolves outside the store")
    payload = _stable_file_bytes(
        path,
        label="episode object under max_shard_bytes",
        max_bytes=max_shard_bytes,
    )
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError("episode object content hash does not match its manifest")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            names = [item.filename for item in members]
            if not members or len(names) != len(set(names)):
                raise ValueError("episode object archive must have unique nonempty members")
            if any(not name.endswith(".npy") for name in names):
                raise ValueError("episode object archive members must be NPY arrays")
            uncompressed_bytes = sum(item.file_size for item in members)
    except zipfile.BadZipFile as error:
        raise ValueError("episode object is not a readable NPZ archive") from error
    estimate = 2 * len(payload) + 2 * uncompressed_bytes
    if estimate > max_shard_bytes:
        raise ValueError(
            f"episode layer estimate ({estimate} bytes) exceeds max_shard_bytes ({max_shard_bytes})"
        )
    return estimate


def _iter_episode_manifest_paths(directory: Path) -> Iterator[Path]:
    """Yield manifest entries without materializing a potentially large listing."""
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name.endswith(".json"):
                yield Path(entry.path)


def _episode_payload(episode: NetworkEpisode) -> bytes:
    metadata = {
        "schema": "umcd.episode.v1",
        "episode_id": episode.episode_id,
        "split_group_id": episode.split_group_id,
        "family_id": episode.family_id,
        "carrier_hz": episode.carrier_hz,
        "frame_times_s": list(episode.frame_times_s),
        "has_pose_timeline": episode.pose_timeline is not None,
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
    if episode.pose_timeline is not None:
        arrays["pose_positions_world_m"] = episode.pose_timeline.positions_world_m
        arrays["pose_orientations_local_to_world"] = (
            episode.pose_timeline.orientations_local_to_world
        )
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


def _direct_path_ablation_payload(ablation: DirectPathAblation) -> bytes:
    metadata = {
        "schema": "umcd.direct_path_ablation.v1",
        "parent_subject_id": ablation.parent_subject_id,
        "base_child_id": ablation.base_child_id,
        "ablated_child_id": ablation.ablated_child_id,
        "split": ablation.split,
        "path_table_sha256": ablation.path_table_sha256,
        "link_ids": list(ablation.link_ids),
    }
    return _pack_npz(
        metadata,
        frame_indices=ablation.frame_indices,
        rx_port_indices=ablation.rx_port_indices,
        tx_port_indices=ablation.tx_port_indices,
        frequency_offset_hz=ablation.frequency_offset_hz,
        full_cfr=ablation.full_cfr,
        without_direct_cfr=ablation.without_direct_cfr,
        removed_direct_path_count=ablation.removed_direct_path_count,
    )


def _channel_payload(view: PhysicalChannelView) -> bytes:
    if view.valid_mask is None:
        raise ValueError("physical channel view is missing its validity mask")
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
        valid_mask=view.valid_mask,
        subcarrier_offsets_hz=view.subcarrier_offsets_hz,
    )


def _target_payload(target: TargetValue) -> bytes:
    metadata = {
        "schema": "umcd.target.v2",
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


def _frequency_receiver_payload(binding: LinkedFrequencyDomainObservation) -> bytes:
    inputs = binding.observation.model_inputs
    metadata = {
        "schema": "umcd.receiver_frequency_observation.v1",
        "link_id": binding.link_id,
        "frame_index": binding.frame_index,
        "sample_id": binding.observation.sample_id,
        "noise_variance": inputs.noise_variance,
        "snr_db": inputs.snr_db,
        "carrier_hz": inputs.carrier_hz,
        "measurement_semantics": "complex_frequency_response_plus_awgn",
        "waveform_model": "direct_noisy_cfr_not_a_pusch_dmrs_observation",
    }
    return _pack_npz(
        metadata,
        observed_cfr=inputs.observed_cfr,
        subcarrier_offsets_hz=inputs.subcarrier_offsets_hz,
    )


def _provenance_payload(record: UMCDRunProvenance) -> bytes:
    return _pack_npz(record.to_dict())


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


def _read_frequency_receiver(
    metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> LinkedFrequencyDomainObservation:
    if metadata.get("schema") != "umcd.receiver_frequency_observation.v1":
        raise ValueError("unsupported frequency receiver-observation schema")
    if metadata.get("measurement_semantics") != "complex_frequency_response_plus_awgn":
        raise ValueError("unsupported frequency observation measurement semantics")
    if metadata.get("waveform_model") != "direct_noisy_cfr_not_a_pusch_dmrs_observation":
        raise ValueError("unsupported frequency observation waveform model")
    inputs = FrequencyDomainModelInputs(
        observed_cfr=arrays["observed_cfr"],
        noise_variance=metadata["noise_variance"],
        snr_db=metadata["snr_db"],
        carrier_hz=metadata["carrier_hz"],
        subcarrier_offsets_hz=arrays["subcarrier_offsets_hz"],
    )
    observation = FrequencyDomainObservation(metadata["sample_id"], inputs)
    return LinkedFrequencyDomainObservation(
        metadata["link_id"], metadata["frame_index"], observation
    )


def write_episode_bundle(root: str | Path, bundle: EpisodeBundle) -> StoredEpisodeBundle:
    """Publish immutable layer objects, then atomically publish the bundle manifest.

    Oracle paths, physical channel tensors, and target values live in distinct
    object namespaces. Manifest publication occurs last; interrupted writes can
    leave only unreferenced content-addressed objects, never a partial release.
    """
    base = Path(root)
    payloads = [("episode", _episode_payload(bundle.episode))]
    if bundle.path_oracle is not None:
        payloads.append(("oracle_paths", _path_payload(bundle.path_oracle)))
    payloads.extend(
        ("counterfactual_ablations", _direct_path_ablation_payload(item))
        for item in bundle.direct_path_ablations
    )
    payloads.extend(("oracle_channels", _channel_payload(view)) for view in bundle.channel_views)
    payloads.extend(("targets", _target_payload(target)) for target in bundle.targets)
    payloads.extend(
        (
            "receiver_observations",
            _receiver_payload(binding)
            if isinstance(binding, LinkedNRObservation)
            else _frequency_receiver_payload(binding),
        )
        for binding in bundle.receiver_observations
    )
    payloads.extend(
        ("provenance", _provenance_payload(record)) for record in bundle.provenance_records
    )
    scenario_gbsm_byte_limit: int | None = None
    if bundle.scenario_gbsm_records:
        from .scenario_gbsm import MAX_SCENARIO_GBSM_OBJECT_BYTES, _record_bytes

        scenario_gbsm_byte_limit = MAX_SCENARIO_GBSM_OBJECT_BYTES

        payloads.extend(
            ("scenario_gbsm", _record_bytes(record)) for record in bundle.scenario_gbsm_records
        )
    references: list[dict[str, str]] = []
    for layer, payload in payloads:
        if (
            layer == "scenario_gbsm"
            and scenario_gbsm_byte_limit is not None
            and len(payload) > scenario_gbsm_byte_limit
        ):
            raise ValueError("scenario GBSM episode layer exceeds its compressed byte ceiling")
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


def iter_episode_bundle_shards(
    root: str | Path,
    *,
    max_shard_bytes: int,
) -> Iterator[tuple[EpisodeBundle, ...]]:
    """Yield episode bundles in deterministic batches under a conservative byte budget.

    Manifest inventory is staged in temporary disk-backed SQLite so a large
    corpus does not require retaining every JSON record in memory. Each bundle
    is size-estimated before its arrays are decoded; the estimate charges twice
    the compressed NPZ bytes and twice the uncompressed archive-member bytes.
    This bounds the declared accounting model, not Python/allocator RSS. A
    single bundle whose estimate exceeds the budget fails before deserialization.
    """
    if isinstance(max_shard_bytes, bool) or not isinstance(max_shard_bytes, int):
        raise TypeError("max_shard_bytes must be an integer")
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")

    base = Path(root).resolve()
    manifest_directory = base / "manifests" / "episodes"
    try:
        metadata = os.lstat(manifest_directory)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("episode manifest directory must be a real directory")
    if not manifest_directory.resolve(strict=True).is_relative_to(base):
        raise ValueError("episode manifest directory resolves outside the store")

    with tempfile.TemporaryDirectory(prefix="umcd-episode-inventory-") as temporary_dir:
        connection = sqlite3.connect(Path(temporary_dir) / "inventory.sqlite3")
        try:
            connection.execute("PRAGMA cache_size = -256")
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute(
                "CREATE TABLE inventory ("
                "episode_id TEXT PRIMARY KEY COLLATE BINARY, "
                "bundle_sha256 TEXT NOT NULL, "
                "relative_path TEXT NOT NULL, "
                "estimated_bytes INTEGER NOT NULL"
                ") WITHOUT ROWID"
            )
            for manifest_path in _iter_episode_manifest_paths(manifest_directory):
                manifest_bytes = _stable_file_bytes(
                    manifest_path, label="episode manifest", max_bytes=1024 * 1024
                )
                manifest = _parse_json_bytes(manifest_bytes, label="episode manifest")
                if not isinstance(manifest, dict):
                    raise ValueError("episode manifest must be a JSON object")
                if (
                    manifest.get("schema") != "umcd.bundle.v1"
                    or manifest.get("schema_version") != 1
                ):
                    raise ValueError("unsupported episode bundle schema")
                episode_id = manifest.get("episode_id")
                if not isinstance(episode_id, str) or not episode_id.strip():
                    raise ValueError("episode manifest episode_id must be nonempty")
                expected_name = f"{hashlib.sha256(episode_id.encode()).hexdigest()}.json"
                if manifest_path.name != expected_name:
                    raise ValueError("episode manifest filename does not match its episode ID")
                core = {
                    key: manifest[key]
                    for key in ("schema", "schema_version", "episode_id", "layers")
                }
                digest = hashlib.sha256(_json_bytes(core)).hexdigest()
                if digest != manifest.get("bundle_sha256"):
                    raise ValueError("episode bundle manifest hash is invalid")
                references = manifest.get("layers")
                if not isinstance(references, list) or not references:
                    raise ValueError("episode bundle layers must be a nonempty list")
                if (
                    sum(
                        isinstance(reference, dict) and reference.get("layer") == "episode"
                        for reference in references
                    )
                    != 1
                ):
                    raise ValueError("bundle must contain exactly one episode layer")
                oracle_path_layers = sum(
                    isinstance(reference, dict) and reference.get("layer") == "oracle_paths"
                    for reference in references
                )
                if oracle_path_layers > 1:
                    raise ValueError("bundle can contain at most one path-oracle layer")
                if not any(
                    isinstance(reference, dict) and reference.get("layer") != "episode"
                    for reference in references
                ):
                    raise ValueError("bundle must contain at least one non-episode data layer")
                estimated_bytes = 0
                for reference in references:
                    if not isinstance(reference, dict) or set(reference) != {
                        "layer",
                        "sha256",
                        "path",
                    }:
                        raise ValueError("episode layer reference fields are invalid")
                    if not isinstance(reference["layer"], str) or not reference["layer"]:
                        raise ValueError("episode layer name must be nonempty")
                    if not isinstance(reference["sha256"], str) or len(reference["sha256"]) != 64:
                        raise ValueError("episode layer digest must be a SHA-256 hex digest")
                    try:
                        int(reference["sha256"], 16)
                    except ValueError as error:
                        raise ValueError("episode layer digest must be hexadecimal") from error
                    if not isinstance(reference["path"], str):
                        raise ValueError("episode layer path must be a string")
                    estimated_bytes += _episode_object_size_estimate(
                        base,
                        cast(dict[str, str], reference),
                        max_shard_bytes=max_shard_bytes,
                    )
                    if estimated_bytes > max_shard_bytes:
                        raise ValueError(
                            f"episode bundle {episode_id!r} estimate exceeds max_shard_bytes"
                        )
                relative_path = manifest_path.relative_to(base).as_posix()
                try:
                    connection.execute(
                        "INSERT INTO inventory VALUES (?, ?, ?, ?)",
                        (episode_id, digest, relative_path, estimated_bytes),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(f"duplicate episode manifest for {episode_id!r}") from error

            batch: list[StoredEpisodeBundle] = []
            batch_bytes = 0
            cursor = connection.execute(
                "SELECT episode_id, bundle_sha256, relative_path, estimated_bytes "
                "FROM inventory ORDER BY episode_id COLLATE BINARY"
            )
            for episode_id, digest, relative_path, estimated_bytes in cursor:
                estimated = int(estimated_bytes)
                if estimated > max_shard_bytes:
                    raise ValueError(f"episode bundle {episode_id!r} exceeds max_shard_bytes")
                if batch and batch_bytes + estimated > max_shard_bytes:
                    yield tuple(read_episode_bundle(base, item) for item in batch)
                    batch = []
                    batch_bytes = 0
                batch.append(StoredEpisodeBundle(episode_id, digest, relative_path))
                batch_bytes += estimated
            if batch:
                yield tuple(read_episode_bundle(base, item) for item in batch)
        finally:
            connection.close()


def _read_object(
    base: Path, reference: dict[str, str], *, max_bytes: int | None = None
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    relative = PurePosixPath(reference["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("object paths must remain inside the episode store")
    object_path = base / Path(*relative.parts)
    if not object_path.resolve(strict=True).is_relative_to(base.resolve()):
        raise ValueError("episode object path resolves outside the store")
    payload = _stable_file_bytes(object_path, label="episode object", max_bytes=max_bytes)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != reference.get("sha256") or relative.name != f"{digest}.npz":
        raise ValueError("episode object content hash does not match its manifest")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if (
            len(archive.files) != len(set(archive.files))
            or archive.files.count("metadata_json") != 1
        ):
            raise ValueError("episode object archive must have unique names and one metadata entry")
        metadata = _parse_json_bytes(
            archive["metadata_json"].tobytes(), label="episode object metadata"
        )
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if name != "metadata_json"
        }
    return metadata, arrays


def _validated_manifest(
    base: Path,
    manifest: StoredEpisodeBundle | str | Path,
    *,
    max_bytes: int | None = None,
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
    manifest_bytes = _stable_file_bytes(
        manifest_path, label="episode manifest", max_bytes=max_bytes
    )
    metadata = _parse_json_bytes(manifest_bytes, label="episode manifest")
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
        "provenance",
        "scenario_gbsm",
        "counterfactual_ablations",
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
        if receiver_metadata.get("schema") == "umcd.receiver_frequency_observation.v1":
            continue
        binding = _read_receiver(receiver_metadata, receiver_arrays)
        if binding.link_id not in link_ids or binding.frame_index >= len(episode.frame_times_s):
            raise ValueError("receiver observation references an unknown link or frame")
        observations.append(binding)
    if len({(item.link_id, item.frame_index) for item in observations}) != len(observations):
        raise ValueError("receiver observations are duplicated for a link/frame")
    if len({item.observation.sample_id for item in observations}) != len(observations):
        raise ValueError("receiver observation sample IDs are duplicated")
    if not observations:
        raise ValueError("bundle contains no NR receiver observations")
    return tuple(item.observation.model_inputs() for item in observations)


def read_frequency_model_inputs(
    root: str | Path, manifest: StoredEpisodeBundle | str | Path
) -> tuple[FrequencyDomainModelInputs, ...]:
    """Read generic complex-CFR inference inputs without opening truth layers."""
    base = Path(root)
    metadata, references = _validated_manifest(base, manifest)
    episode_refs = [item for item in references if item.get("layer") == "episode"]
    observation_refs = [item for item in references if item.get("layer") == "receiver_observations"]
    if len(episode_refs) != 1 or not observation_refs:
        raise ValueError("bundle needs one episode and at least one receiver-observation layer")
    episode_metadata, episode_arrays = _read_object(base, episode_refs[0])
    if episode_metadata.get("schema") != "umcd.episode.v1":
        raise ValueError("unsupported episode layer schema")
    episode = _read_episode(episode_metadata, episode_arrays)
    if metadata["episode_id"] != episode.episode_id:
        raise ValueError("manifest and episode payload IDs differ")
    link_ids = {link.link_id for link in episode.links}
    observations: list[LinkedFrequencyDomainObservation] = []
    for reference in observation_refs:
        receiver_metadata, receiver_arrays = _read_object(base, reference)
        if receiver_metadata.get("schema") != "umcd.receiver_frequency_observation.v1":
            continue
        binding = _read_frequency_receiver(receiver_metadata, receiver_arrays)
        if binding.link_id not in link_ids or binding.frame_index >= len(episode.frame_times_s):
            raise ValueError("frequency observation references an unknown link or frame")
        link = episode.link(binding.link_id)
        expected_shape = (
            len(link.rx_port_indices),
            len(link.tx_port_indices),
            len(episode.subcarrier_offsets_hz),
        )
        inputs = binding.observation.model_inputs
        if inputs.observed_cfr.shape != expected_shape:
            raise ValueError("frequency observation axes do not match its link and grid")
        if inputs.carrier_hz != episode.carrier_hz or not np.array_equal(
            inputs.subcarrier_offsets_hz, episode.subcarrier_offsets_hz
        ):
            raise ValueError("frequency observation grid differs from its parent episode")
        observations.append(binding)
    if not observations:
        raise ValueError("bundle contains no direct noisy-CFR observations")
    if len({(item.link_id, item.frame_index) for item in observations}) != len(observations):
        raise ValueError("frequency observations are duplicated for a link/frame")
    if len({item.observation.sample_id for item in observations}) != len(observations):
        raise ValueError("frequency observation sample IDs are duplicated")
    return tuple(item.observation.model_inputs for item in observations)


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
    pose_timeline = None
    if metadata.get("has_pose_timeline", False):
        pose_timeline = NodePoseTimeline(
            node_ids=tuple(item["node_id"] for item in metadata["nodes"]),
            frame_times_s=tuple(metadata["frame_times_s"]),
            positions_world_m=arrays["pose_positions_world_m"],
            orientations_local_to_world=arrays["pose_orientations_local_to_world"],
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
        pose_timeline,
    )


def read_episode_bundle(
    root: str | Path,
    manifest: StoredEpisodeBundle | str | Path,
    *,
    max_shard_bytes: int | None = None,
) -> EpisodeBundle:
    """Load linked layers with optional conservative size-accounting bounds."""
    if max_shard_bytes is not None and (
        isinstance(max_shard_bytes, bool)
        or not isinstance(max_shard_bytes, int)
        or max_shard_bytes <= 0
    ):
        raise ValueError("max_shard_bytes must be a positive integer")
    base = Path(root)
    metadata, references = _validated_manifest(
        base,
        manifest,
        max_bytes=1024 * 1024 if max_shard_bytes is not None else None,
    )
    if max_shard_bytes is not None:
        estimated_bundle_bytes = sum(
            _episode_object_size_estimate(base, reference, max_shard_bytes=max_shard_bytes)
            for reference in references
        )
        if estimated_bundle_bytes > max_shard_bytes:
            raise ValueError("episode bundle exceeds max_shard_bytes")
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, np.ndarray]]]] = {}
    allowed_layers = {
        "episode",
        "oracle_paths",
        "oracle_channels",
        "targets",
        "receiver_observations",
        "provenance",
        "scenario_gbsm",
        "counterfactual_ablations",
    }
    for reference in references:
        layer = reference.get("layer")
        if layer not in allowed_layers:
            raise ValueError(f"unsupported episode bundle layer: {layer!r}")
        object_metadata, arrays = _read_object(base, reference, max_bytes=max_shard_bytes)
        grouped.setdefault(layer, []).append((object_metadata, arrays))
    if len(grouped.get("episode", [])) != 1 or len(grouped.get("oracle_paths", [])) > 1:
        raise ValueError("bundle must contain one episode and at most one path-oracle layer")
    episode_metadata, episode_arrays = grouped["episode"][0]
    if episode_metadata.get("schema") != "umcd.episode.v1":
        raise ValueError("unsupported episode layer schema")
    episode = _read_episode(episode_metadata, episode_arrays)
    paths = None
    if grouped.get("oracle_paths"):
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
                path_arrays.get("object_ids")
                if optional.get("interaction_geometry", False)
                else None
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
            path_kind=(
                None if path_metadata["path_kind"] is None else tuple(path_metadata["path_kind"])
            ),
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
                valid_mask=arrays.get("valid_mask"),
                axes=tuple(channel_metadata["axes"]),
            )
        )
    targets: list[TargetValue] = []
    for target_metadata, arrays in grouped.get("targets", []):
        target_schema = target_metadata.get("schema")
        if target_schema not in {"umcd.target.v1", "umcd.target.v2"}:
            raise ValueError("unsupported target layer schema")
        definition_payload = target_metadata["definition"]
        if target_schema == "umcd.target.v1":
            definition = migrate_legacy_target_definition_v1(definition_payload)
        else:
            definition = TargetDefinition(**definition_payload)
        targets.append(
            TargetValue(
                entity_id=target_metadata["entity_id"],
                definition=definition,
                values=arrays["values"],
                valid_mask=arrays["valid_mask"],
                reason_codes=tuple(
                    TargetValidity(item) for item in target_metadata["reason_codes"]
                ),
                confidence=target_metadata["confidence"],
            )
        )
    observations: list[LinkedNRObservation | LinkedFrequencyDomainObservation] = []
    for receiver_metadata, arrays in grouped.get("receiver_observations", []):
        schema = receiver_metadata.get("schema")
        if schema == "umcd.receiver_observation.v1":
            observations.append(_read_receiver(receiver_metadata, arrays))
        elif schema == "umcd.receiver_frequency_observation.v1":
            observations.append(_read_frequency_receiver(receiver_metadata, arrays))
        else:
            raise ValueError("unsupported receiver-observation layer schema")
    provenance_records = [
        UMCDRunProvenance.from_dict(provenance_metadata)
        for provenance_metadata, _ in grouped.get("provenance", [])
    ]
    from .scenario_gbsm import ScenarioGBSMRecord

    scenario_gbsm_records = []
    for scenario_metadata, scenario_arrays in grouped.get("scenario_gbsm", []):
        if (
            set(scenario_metadata)
            != {
                "schema",
                "sample_id",
                "run_id",
                "sample_index",
                "scenario",
                "scenario_los",
                "metadata",
            }
            or scenario_metadata.get("schema") != "umcd.scenario_gbsm_record.v1"
        ):
            raise ValueError("unsupported scenario GBSM episode layer schema")
        if set(scenario_arrays) != {"h_path", "tau_s", "cfr", "subcarrier_offsets_hz"}:
            raise ValueError("scenario GBSM episode layer arrays are invalid")
        scenario_gbsm_records.append(
            ScenarioGBSMRecord(
                sample_id=scenario_metadata["sample_id"],
                run_id=scenario_metadata["run_id"],
                sample_index=scenario_metadata["sample_index"],
                scenario=scenario_metadata["scenario"],
                scenario_los=scenario_metadata["scenario_los"],
                h_path=scenario_arrays["h_path"],
                tau_s=scenario_arrays["tau_s"],
                cfr=scenario_arrays["cfr"],
                subcarrier_offsets_hz=scenario_arrays["subcarrier_offsets_hz"],
                metadata=scenario_metadata["metadata"],
            )
        )
    direct_path_ablations = []
    for ablation_metadata, ablation_arrays in grouped.get("counterfactual_ablations", []):
        if (
            set(ablation_metadata)
            != {
                "schema",
                "parent_subject_id",
                "base_child_id",
                "ablated_child_id",
                "split",
                "path_table_sha256",
                "link_ids",
            }
            or ablation_metadata.get("schema") != "umcd.direct_path_ablation.v1"
        ):
            raise ValueError("unsupported direct-path ablation layer schema")
        if set(ablation_arrays) != {
            "frame_indices",
            "rx_port_indices",
            "tx_port_indices",
            "frequency_offset_hz",
            "full_cfr",
            "without_direct_cfr",
            "removed_direct_path_count",
        }:
            raise ValueError("direct-path ablation layer arrays are invalid")
        direct_path_ablations.append(
            DirectPathAblation(
                parent_subject_id=ablation_metadata["parent_subject_id"],
                base_child_id=ablation_metadata["base_child_id"],
                ablated_child_id=ablation_metadata["ablated_child_id"],
                split=ablation_metadata["split"],
                path_table_sha256=ablation_metadata["path_table_sha256"],
                link_ids=tuple(ablation_metadata["link_ids"]),
                frame_indices=ablation_arrays["frame_indices"],
                rx_port_indices=ablation_arrays["rx_port_indices"],
                tx_port_indices=ablation_arrays["tx_port_indices"],
                frequency_offset_hz=ablation_arrays["frequency_offset_hz"],
                full_cfr=ablation_arrays["full_cfr"],
                without_direct_cfr=ablation_arrays["without_direct_cfr"],
                removed_direct_path_count=ablation_arrays["removed_direct_path_count"],
            )
        )
    if metadata["episode_id"] != episode.episode_id:
        raise ValueError("manifest and episode payload IDs differ")
    return EpisodeBundle(
        episode,
        paths,
        tuple(channels),
        tuple(targets),
        tuple(observations),
        tuple(provenance_records),
        tuple(scenario_gbsm_records),
        tuple(direct_path_ablations),
    )
