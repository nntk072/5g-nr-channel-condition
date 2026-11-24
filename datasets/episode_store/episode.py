"""Read and write episode bundles and their receiver-facing views."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..channel_view import ChannelDelayReference, ChannelPowerScale, PhysicalChannelView
from ..counterfactual import DirectPathAblation
from ..nr import (
    LinkedNRObservation,
    NRModelInputs,
    NRReceiverInferenceRecord,
)
from ..observations import (
    FrequencyDomainModelInputs,
    LinkedFrequencyDomainObservation,
)
from ..path_oracle import (
    PathOracleTable,
    PathSolveStatus,
)
from ..provenance import UMCDRunProvenance
from ..targets import (
    CensoringInterval,
    TargetDefinition,
    TargetValidity,
    TargetValue,
    migrate_legacy_target_definition,
)
from .storage import (
    EpisodeBundle,
    StoredEpisodeBundle,
    _channel_payload,
    _direct_path_ablation_payload,
    _direct_path_ablation_v2_validator,
    _episode_object_size_estimate,
    _episode_payload,
    _frequency_receiver_payload,
    _iter_episode_manifest_paths,
    _json_bytes,
    _object_ref,
    _parse_json_bytes,
    _path_payload,
    _provenance_payload,
    _publish_immutable,
    _read_episode,
    _read_frequency_receiver,
    _read_object,
    _read_receiver,
    _receiver_payload,
    _stable_file_bytes,
    _target_payload,
    _validate_path_oracle_layer,
    _validated_manifest,
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
        from ..scenario_gbsm import MAX_SCENARIO_GBSM_OBJECT_BYTES, _record_bytes

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


def read_receiver_inference_records(
    root: str | Path, manifest: StoredEpisodeBundle | str | Path
) -> tuple[NRReceiverInferenceRecord, ...]:
    """Load receiver-only inputs and separate row identities, without truth layers.

    The episode layer is read to validate link/frame bindings. Other manifest
    objects are deliberately not opened, so damaged or unavailable oracle and
    target files cannot leak into or block receiver inference. Each composite
    identity remains outside ``model_inputs`` for prediction/evaluation joins.
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
    return tuple(
        NRReceiverInferenceRecord(
            episode_id=episode.episode_id,
            link_id=item.link_id,
            frame_index=item.frame_index,
            sample_id=item.observation.sample_id,
            model_inputs=item.observation.model_inputs(),
        )
        for item in observations
    )


def read_receiver_model_inputs(
    root: str | Path, manifest: StoredEpisodeBundle | str | Path
) -> tuple[NRModelInputs, ...]:
    """Load only model payloads, without identifiers, oracle or target objects.

    Use :func:`read_receiver_inference_records` when a prediction artifact must
    retain a stable evaluation join key separately from its model input.
    """
    return tuple(record.model_inputs for record in read_receiver_inference_records(root, manifest))


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
        _validate_path_oracle_layer(path_metadata, set(path_arrays))
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
                phase_gauge_id=channel_metadata.get("phase_gauge_id"),
                source_time_alignment_sha256=channel_metadata.get("source_time_alignment_sha256"),
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
            definition = migrate_legacy_target_definition(definition_payload)
        else:
            definition = TargetDefinition(**definition_payload)
        target_reasons = tuple(TargetValidity(item) for item in target_metadata["reason_codes"])
        if "censoring_intervals" not in target_metadata:
            # Legacy targets did not carry interval bounds. Preserve their
            # unavailable mask while marking the old censor detail unrecorded.
            target_reasons = tuple(
                TargetValidity.LEGACY_REASON_UNRECORDED
                if reason is TargetValidity.CENSORED
                else reason
                for reason in target_reasons
            )
        targets.append(
            TargetValue(
                entity_id=target_metadata["entity_id"],
                definition=definition,
                values=arrays["values"],
                valid_mask=arrays["valid_mask"],
                reason_codes=target_reasons,
                confidence=target_metadata["confidence"],
                censoring_intervals=(
                    None
                    if "censoring_intervals" not in target_metadata
                    else tuple(
                        None if item is None else CensoringInterval(**item)
                        for item in target_metadata["censoring_intervals"]
                    )
                ),
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
    from ..scenario_gbsm import ScenarioGBSMRecord

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
        schema_version = ablation_metadata.get("schema")
        required_metadata = {
            "schema",
            "parent_subject_id",
            "base_child_id",
            "ablated_child_id",
            "split",
            "path_table_sha256",
            "link_ids",
        }
        if schema_version == "umcd.direct_path_ablation.v2":
            _direct_path_ablation_v2_validator().validate(ablation_metadata)
            required_metadata.add("carrier_hz")
        elif schema_version != "umcd.direct_path_ablation.v1":
            raise ValueError("unsupported direct-path ablation layer schema")
        if set(ablation_metadata) != required_metadata:
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
                carrier_hz=(
                    ablation_metadata["carrier_hz"]
                    if schema_version == "umcd.direct_path_ablation.v2"
                    else None
                ),
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
