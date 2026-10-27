"""Canonical records and immutable storage for 3GPP scenario GBSM drops."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from .episode_store import _parse_json_bytes, _stable_file_bytes
from .standardized import _freeze_json, _numpy, _plain_json, _publish_immutable

SCENARIO_GBSM_MODELS = frozenset({"UMi", "UMa", "RMa", "InH", "InF"})
MAX_SCENARIO_GBSM_OBJECT_BYTES = 512 * 1024 * 1024
MAX_SCENARIO_GBSM_MANIFEST_BYTES = 64 * 1024 * 1024
_SPEED_OF_LIGHT_M_S = 299_792_458.0


def _canonical_json(value: Any, *, label: str) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be finite JSON-compatible data") from error


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_json(value: Mapping[str, Any], *, label: str) -> str:
    return _sha256(_canonical_json(value, label=label))


def _store_path(base: Path, relative: PurePosixPath, *, label: str) -> Path:
    """Resolve one store member and reject parent-symlink escapes."""
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError(f"{label} path must be a canonical relative path")
    root = base.resolve(strict=True)
    candidate = root.joinpath(*relative.parts).resolve(strict=True)
    if not candidate.is_relative_to(root):
        raise ValueError(f"{label} path escapes the store root")
    return candidate


@dataclass(frozen=True, slots=True, eq=False)
class ScenarioGBSMRecord:
    """One scenario drop; scenario LOS is separate from geometric LOS.

    Axes are ``h_path=[rx,rx_ant,tx,tx_ant,path,time]``,
    ``tau_s=[rx,tx,path]``, and ``cfr=[rx,rx_ant,tx,tx_ant,time,subcarrier]``.
    """

    sample_id: str
    run_id: str
    sample_index: int
    scenario: str
    scenario_los: bool
    h_path: np.ndarray
    tau_s: np.ndarray
    cfr: np.ndarray
    subcarrier_offsets_hz: np.ndarray
    metadata: Mapping[str, Any]

    def validate_for_episode(self, episode: Any) -> None:
        """Bind this stochastic drop to its one-link canonical episode."""
        from .network import NodeRole

        if episode.episode_id != self.sample_id:
            raise ValueError("scenario GBSM sample ID must equal its parent episode ID")
        if self.metadata["split_subject_id"] != episode.episode_id:
            raise ValueError("scenario GBSM split subject differs from its parent episode")
        if episode.split_group_id != f"scenario_gbsm_run:{self.run_id}":
            raise ValueError("scenario GBSM episode split group differs from its generation run")
        if episode.carrier_hz != self.metadata["generator_configuration"]["carrier_frequency_hz"]:
            raise ValueError("scenario GBSM carrier differs from its parent episode")
        if not np.array_equal(episode.subcarrier_offsets_hz, self.subcarrier_offsets_hz):
            raise ValueError("scenario GBSM frequency grid differs from its parent episode")
        if len(episode.frame_times_s) != self.h_path.shape[-1]:
            raise ValueError("scenario GBSM time samples differ from the parent episode")
        configuration = self.metadata["generator_configuration"]
        sample_rate = float(configuration["sampling_frequency_hz"])
        if len(episode.frame_times_s) > 1 and not np.allclose(
            np.diff(episode.frame_times_s), 1.0 / sample_rate, rtol=1e-9, atol=1e-12
        ):
            raise ValueError("scenario GBSM sampling interval differs from its parent episode")
        base_stations = [node for node in episode.nodes if node.role is NodeRole.GNB]
        terminals = [node for node in episode.nodes if node.role is NodeRole.UE]
        if len(base_stations) != 1 or len(terminals) != 1 or len(episode.links) != 1:
            raise ValueError("scenario GBSM layer requires one gNB–UE link per episode")
        link = episode.links[0]
        if {link.tx_node_id, link.rx_node_id} != {
            base_stations[0].node_id, terminals[0].node_id
        }:
            raise ValueError("scenario GBSM link endpoints differ from its gNB–UE episode")
        if link.tx_node_id == base_stations[0].node_id:
            tx_node, rx_node = base_stations[0], terminals[0]
            expected_direction = "downlink"
        else:
            tx_node, rx_node = terminals[0], base_stations[0]
            expected_direction = "uplink"
        if configuration.get("direction") != expected_direction:
            raise ValueError("scenario GBSM link direction differs from its parent episode")
        tx_array, rx_array = episode.array(tx_node.array_id), episode.array(rx_node.array_id)
        if (self.h_path.shape[1], self.h_path.shape[3]) != (
            rx_array.num_ports, tx_array.num_ports
        ):
            raise ValueError("scenario GBSM antenna-port axes differ from episode arrays")
        topology = self.metadata["topology"]
        if not np.allclose(topology["bs_loc"][0], base_stations[0].position_world_m):
            raise ValueError("scenario GBSM BS location differs from its episode node")
        if not np.allclose(topology["ut_loc"][0], terminals[0].position_world_m):
            raise ValueError("scenario GBSM UT location differs from its episode node")
        ut_positions = np.asarray(
            [episode.position_at(terminals[0].node_id, index) for index in range(len(episode.frame_times_s))]
        )
        bs_positions = np.asarray(
            [episode.position_at(base_stations[0].node_id, index) for index in range(len(episode.frame_times_s))]
        )
        ut_velocity = np.asarray(topology["ut_velocities"][0], dtype=np.float64)
        if len(ut_positions) > 1:
            measured_velocity = np.diff(ut_positions, axis=0) / np.diff(
                episode.frame_times_s
            )[:, None]
            if not np.allclose(measured_velocity, ut_velocity[None, :], rtol=1e-5, atol=1e-5):
                raise ValueError("scenario GBSM UT velocity differs from its episode trajectory")
            if not np.allclose(bs_positions, bs_positions[0], rtol=0.0, atol=1e-7):
                raise ValueError("scenario GBSM adapter cannot encode a moving BS trajectory")
        for node in (base_stations[0], terminals[0]):
            orientations = np.asarray(
                [episode.orientation_at(node.node_id, index) for index in range(len(episode.frame_times_s))]
            )
            if not np.allclose(orientations, orientations[0], rtol=0.0, atol=1e-7):
                raise ValueError("scenario GBSM adapter cannot encode rotating array trajectories")

    def __post_init__(self) -> None:
        for name in ("sample_id", "run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value.strip() != value:
                raise ValueError(f"{name} must be a non-empty canonical string")
        if isinstance(self.sample_index, bool) or not isinstance(self.sample_index, int):
            raise TypeError("sample_index must be an integer")
        if self.sample_index < 0:
            raise ValueError("sample_index must be nonnegative")
        if not isinstance(self.scenario, str) or self.scenario not in SCENARIO_GBSM_MODELS:
            raise ValueError(f"scenario must be one of {sorted(SCENARIO_GBSM_MODELS)}")
        if not isinstance(self.scenario_los, bool):
            raise TypeError("scenario_los must be Boolean")

        h_path = _numpy(self.h_path, name="h_path", kind="complex")
        tau_s = _numpy(self.tau_s, name="tau_s", kind="real").astype(np.float64, copy=False)
        cfr = _numpy(self.cfr, name="cfr", kind="complex")
        offsets = _numpy(
            self.subcarrier_offsets_hz, name="subcarrier_offsets_hz", kind="real"
        ).astype(np.float64, copy=False)
        if h_path.ndim != 6 or any(size < 1 for size in h_path.shape):
            raise ValueError("h_path must have non-empty [rx,rx_ant,tx,tx_ant,path,time] axes")
        if h_path.shape[0] != 1 or h_path.shape[2] != 1:
            raise ValueError("scenario records require one BS and one UT link")
        if tau_s.shape != (1, 1, h_path.shape[-2]) or np.any(tau_s < 0):
            raise ValueError("tau_s must be nonnegative [1,1,path]")
        if offsets.ndim != 1 or len(offsets) < 8:
            raise ValueError("subcarrier offsets must be a vector with at least 8 entries")
        if np.any(np.diff(offsets) <= 0) or not np.any(offsets == 0.0):
            raise ValueError("subcarrier offsets must increase strictly and include DC")
        expected_shape = (*h_path.shape[:4], h_path.shape[-1], len(offsets))
        if cfr.shape != expected_shape:
            raise ValueError(f"cfr must have shape {expected_shape}")
        delays = tau_s[:, None, :, None, :, None, None]
        frequencies = offsets.reshape(1, 1, 1, 1, 1, 1, -1)
        expected_cfr = np.sum(
            h_path[..., None] * np.exp(-2j * np.pi * delays * frequencies), axis=4
        )
        error = float(np.max(np.abs(cfr - expected_cfr)))
        scale = max(float(np.max(np.abs(expected_cfr))), 1e-8)
        if error > 1e-7 + 5e-5 * scale:
            raise ValueError("cfr does not match stored path coefficients and delays")

        metadata = _plain_json(dict(self.metadata))
        configuration = metadata.get("generator_configuration")
        if not isinstance(configuration, dict):
            raise ValueError("generator_configuration metadata is required")
        config_digest = metadata.get("configuration_sha256")
        if not isinstance(config_digest, str) or _digest_json(
            configuration, label="generator configuration"
        ) != config_digest:
            raise ValueError("generator configuration digest is invalid")
        topology = metadata.get("topology")
        topology_digest = metadata.get("topology_row_sha256")
        if not isinstance(topology, dict) or _digest_json(
            topology, label="topology row"
        ) != topology_digest:
            raise ValueError("topology row digest is invalid")
        if metadata.get("scenario") != self.scenario:
            raise ValueError("scenario metadata disagrees with the record")
        if configuration.get("scenario") != self.scenario:
            raise ValueError("generator configuration scenario disagrees with the record")
        if metadata.get("sionna_version") != configuration.get("sionna_version"):
            raise ValueError("Sionna version metadata disagrees with generator configuration")
        if metadata.get("scenario_los") is not self.scenario_los:
            raise ValueError("scenario LOS metadata disagrees with the record")
        if metadata.get("geometric_los_label") is not None:
            raise ValueError("scenario model LOS must not be recorded as geometric LOS")
        if metadata.get("split_subject_id") != self.sample_id:
            raise ValueError("scenario GBSM split subject must equal its sample ID")
        if metadata.get("split_parent_type") != "standardized_channel_draw":
            raise ValueError("scenario GBSM record must declare its standardized-draw parent type")
        if metadata.get("split_source_family") != f"scenario_gbsm:{self.scenario}":
            raise ValueError("scenario GBSM split source family disagrees with its scenario")
        if not isinstance(metadata.get("split_leakage_relations"), list):
            raise ValueError("scenario GBSM split leakage relations must be explicitly listed")
        if topology.get("los") != [[self.scenario_los]]:
            raise ValueError("scenario LOS state disagrees with its topology row")
        spacing = configuration.get("subcarrier_spacing_hz")
        if (
            isinstance(spacing, bool)
            or not isinstance(spacing, (int, float))
            or not math.isfinite(float(spacing))
            or spacing <= 0
        ):
            raise ValueError("generator configuration must declare positive subcarrier spacing")
        if configuration.get("num_subcarriers") != len(offsets) or not np.allclose(
            np.diff(offsets), spacing, rtol=1e-10, atol=1e-6
        ):
            raise ValueError("subcarrier offsets disagree with generator configuration")
        snapshot = json.loads(_canonical_json(metadata, label="record metadata"))
        for array in (h_path, tau_s, cfr, offsets):
            array.setflags(write=False)
        object.__setattr__(self, "h_path", h_path)
        object.__setattr__(self, "tau_s", tau_s)
        object.__setattr__(self, "cfr", cfr)
        object.__setattr__(self, "subcarrier_offsets_hz", offsets)
        object.__setattr__(self, "metadata", _freeze_json(snapshot))


@dataclass(frozen=True, slots=True, eq=False)
class ScenarioGBSMRun:
    """Indexed drops produced by one call to a scenario GBSM adapter."""

    run_id: str
    records: tuple[ScenarioGBSMRecord, ...]

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if not records or any(record.run_id != self.run_id for record in records):
            raise ValueError("scenario run records must all use the run ID")
        ids = [record.sample_id for record in records]
        indices = [record.sample_index for record in records]
        if len(set(ids)) != len(ids) or set(indices) != set(range(len(records))):
            raise ValueError("sample IDs and contiguous indices must be unique")
        first = records[0]
        if any(
            record.scenario != first.scenario
            or record.metadata["configuration_sha256"]
            != first.metadata["configuration_sha256"]
            for record in records[1:]
        ):
            raise ValueError("a run cannot mix scenario models or generator configurations")
        object.__setattr__(self, "records", tuple(sorted(records, key=lambda item: item.sample_index)))

    def audit_split_assignment(self, registry: Any) -> dict[str, Any]:
        """Verify that this shared generation run stays in one frozen split component.

        A scenario run may contain several parent episodes. Since those records share
        one generator invocation and run identity, the frozen registry must include
        their common trajectory-session relation and assign them to one component.
        The returned digest binds the exact parent-to-component rows that were checked.
        """
        from .split_policy import LeakageNamespace, LeakageRelation, SplitPolicyRegistry

        if not isinstance(registry, SplitPolicyRegistry):
            raise TypeError("registry must be a SplitPolicyRegistry")
        run_relation = LeakageRelation(
            LeakageNamespace.TRAJECTORY_SESSION, f"scenario_gbsm_run:{self.run_id}"
        )
        rows: list[dict[str, str]] = []
        for record in self.records:
            parent_id = record.sample_id
            subject = registry.subjects.get(parent_id)
            if subject is None:
                raise ValueError(
                    f"scenario GBSM parent is absent from frozen split registry: {parent_id}"
                )
            if run_relation not in subject.relations:
                raise ValueError(
                    f"frozen split parent {parent_id!r} omits the scenario run relation"
                )
            rows.append(
                {
                    "parent_subject_id": parent_id,
                    "component_id": registry.subject_to_component[parent_id],
                    "split": registry.split_for_parent(parent_id),
                }
            )
        component_ids = {row["component_id"] for row in rows}
        splits = {row["split"] for row in rows}
        if len(component_ids) != 1 or len(splits) != 1:
            raise ValueError(
                "scenario GBSM run parents must share one frozen split component and assignment"
            )
        core = {
            "schema": "umcd.scenario_gbsm_split_audit.v1",
            "schema_version": 1,
            "run_id": self.run_id,
            "run_relation": run_relation.to_dict(),
            "registry_sha256": registry.to_dict()["registry_sha256"],
            "parent_count": len(rows),
            "component_id": next(iter(component_ids)),
            "split": next(iter(splits)),
            "parents": rows,
        }
        return {**core, "audit_sha256": _sha256(_canonical_json(core, label="split audit"))}


def _sionna_orientation_matrix(angles: Sequence[float]) -> np.ndarray:
    """Convert Sionna's three Euler angles using its Z-Y-X local-to-world convention."""
    if len(angles) != 3 or any(not math.isfinite(float(value)) for value in angles):
        raise ValueError("scenario orientation must contain three finite Euler angles")
    alpha, beta, gamma = (float(value) for value in angles)
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


def _scenario_array_spec(
    record: ScenarioGBSMRecord,
    *,
    endpoint: str,
    carrier_hz: float,
    port_count: int,
    orientation: np.ndarray,
) -> Any:
    from .network import AntennaArraySpec

    configuration = record.metadata["generator_configuration"]
    array_layout = configuration.get(f"{endpoint}_array_layout")
    if array_layout is None:
        # Historical v1 records with a single isotropic port have an exact
        # origin layout. Multi-port records need the native captured layout.
        if port_count != 1:
            raise ValueError(
                f"scenario record lacks captured {endpoint} array layout for {port_count} ports"
            )
        normalized_positions = np.zeros((1, 3), dtype=np.float64)
        labels = ["V"]
    else:
        if (
            not isinstance(array_layout, Mapping)
            or array_layout.get("schema") != "umcd.sionna_array_layout.v1"
            or array_layout.get("port_order") != "pattern_major_element_minor"
        ):
            raise ValueError(f"scenario {endpoint} array layout has an unsupported contract")
        normalized_positions = np.asarray(
            array_layout.get("normalized_element_positions_wavelengths"), dtype=np.float64
        )
        labels = array_layout.get("pattern_labels_in_port_order")
        if (
            normalized_positions.ndim != 2
            or normalized_positions.shape[1] != 3
            or normalized_positions.shape[0] < 1
            or not np.isfinite(normalized_positions).all()
            or not isinstance(labels, (list, tuple))
            or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
        ):
            raise ValueError(f"scenario {endpoint} array layout is malformed")
    element_count = len(normalized_positions)
    port_count_from_layout = len(labels) * element_count
    port_ids = tuple(
        f"{label}:element-{index}"
        for label in labels
        for index in range(element_count)
    )
    if port_count_from_layout != port_count:
        raise ValueError(f"scenario {endpoint} array layout does not match its channel axis")
    port_element_weights = np.zeros((port_count, element_count), dtype=np.complex128)
    for pattern_index in range(len(labels)):
        for element_index in range(element_count):
            port_element_weights[pattern_index * element_count + element_index, element_index] = 1.0
    return AntennaArraySpec(
        array_id=f"{endpoint}-array",
        element_positions_local_m=normalized_positions * (_SPEED_OF_LIGHT_M_S / carrier_hz),
        polarization_per_element=("+".join(labels),) * element_count,
        orientation_local_to_world=orientation,
        port_ids=port_ids,
        port_element_weights=port_element_weights,
    )


def scenario_gbsm_episode_bundle(record: ScenarioGBSMRecord) -> Any:
    """Assemble one native scenario drop as a canonical, lineage-bound EpisodeBundle."""
    from .episode_store import EpisodeBundle
    from .network import (
        DirectedLink,
        LinkDirection,
        LinkRole,
        NetworkEpisode,
        NetworkNode,
        NodePoseTimeline,
        NodeRole,
    )

    if not isinstance(record, ScenarioGBSMRecord):
        raise TypeError("record must be a ScenarioGBSMRecord")
    configuration = record.metadata["generator_configuration"]
    topology = record.metadata["topology"]
    carrier_hz = float(configuration["carrier_frequency_hz"])
    sampling_hz = float(configuration["sampling_frequency_hz"])
    frame_count = record.h_path.shape[-1]
    frame_times = tuple(index / sampling_hz for index in range(frame_count))
    bs_position = np.asarray(topology["bs_loc"][0], dtype=np.float64)
    ut_position = np.asarray(topology["ut_loc"][0], dtype=np.float64)
    ut_velocity = np.asarray(topology["ut_velocities"][0], dtype=np.float64)
    bs_orientation = _sionna_orientation_matrix(topology["bs_orientations"][0])
    ut_orientation = _sionna_orientation_matrix(topology["ut_orientations"][0])
    bs_array = _scenario_array_spec(
        record,
        endpoint="bs",
        carrier_hz=carrier_hz,
        port_count=record.h_path.shape[3],
        orientation=bs_orientation,
    )
    ut_array = _scenario_array_spec(
        record,
        endpoint="ut",
        carrier_hz=carrier_hz,
        port_count=record.h_path.shape[1],
        orientation=ut_orientation,
    )
    nodes = (
        NetworkNode("bs", NodeRole.GNB, bs_array.array_id, tuple(bs_position), bs_array.num_ports),
        NetworkNode("ut", NodeRole.UE, ut_array.array_id, tuple(ut_position), ut_array.num_ports),
    )
    arrays = (bs_array, ut_array)
    direction = configuration["direction"]
    if direction == "downlink":
        tx_node, rx_node = "bs", "ut"
        tx_count, rx_count = record.h_path.shape[3], record.h_path.shape[1]
        link_direction = LinkDirection.DOWNLINK
    elif direction == "uplink":
        tx_node, rx_node = "ut", "bs"
        tx_count, rx_count = record.h_path.shape[1], record.h_path.shape[3]
        link_direction = LinkDirection.UPLINK
    else:
        raise ValueError("scenario record has an unsupported link direction")
    episode = NetworkEpisode(
        episode_id=record.sample_id,
        split_group_id=f"scenario_gbsm_run:{record.run_id}",
        family_id=f"scenario_gbsm:{record.scenario}",
        carrier_hz=carrier_hz,
        arrays=arrays,
        nodes=nodes,
        links=(
            DirectedLink(
                "bs-ut-link",
                tx_node,
                rx_node,
                tuple(range(tx_count)),
                tuple(range(rx_count)),
                link_direction,
                LinkRole.SERVING,
            ),
        ),
        frame_times_s=frame_times,
        subcarrier_offsets_hz=record.subcarrier_offsets_hz,
        pose_timeline=(
            NodePoseTimeline(
                node_ids=("bs", "ut"),
                frame_times_s=frame_times,
                positions_world_m=np.stack(
                    (
                        np.repeat(bs_position[None, :], frame_count, axis=0),
                        ut_position[None, :] + np.asarray(frame_times)[:, None] * ut_velocity,
                    ),
                    axis=1,
                ),
                orientations_local_to_world=np.broadcast_to(
                    np.stack((bs_orientation, ut_orientation))[None, ...],
                    (frame_count, 2, 3, 3),
                ),
            )
            if frame_count > 1
            else None
        ),
    )
    return EpisodeBundle(
        episode=episode,
        path_oracle=None,
        channel_views=(),
        targets=(),
        scenario_gbsm_records=(record,),
    )


def scenario_gbsm_run_to_episode_bundles(run: ScenarioGBSMRun) -> tuple[Any, ...]:
    """Convert all explicitly sampled drops in one run into canonical episode bundles."""
    if not isinstance(run, ScenarioGBSMRun):
        raise TypeError("run must be a ScenarioGBSMRun")
    return tuple(scenario_gbsm_episode_bundle(record) for record in run.records)


@dataclass(frozen=True, slots=True)
class StoredScenarioGBSMRun:
    """Immutable pointer to one scenario-run manifest."""

    run_id: str
    content_sha256: str
    relative_path: str
    sample_ids: tuple[str, ...]
    schema_version: int = 1


def scenario_gbsm_run_from_channel(
    channel: Any, *, run_id: str, sample_ids: Sequence[str]
) -> ScenarioGBSMRun:
    """Convert a native batch result to canonical per-drop records.

    Caller-provided sample IDs become split-parent identities. This conversion
    does not claim that the batch drops are independent or share no site links.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    ids = tuple(sample_ids)
    if not ids or any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError("sample_ids must contain non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be unique")
    metadata = getattr(channel, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise TypeError("channel must expose generator metadata")
    scenario = metadata.get("scenario")
    topology_rows = metadata.get("topology_rows")
    topology_digests = metadata.get("topology_row_sha256")
    los_rows = metadata.get("scenario_los_rows")
    if (
        scenario not in SCENARIO_GBSM_MODELS
        or not isinstance(topology_rows, list)
        or not isinstance(topology_digests, list)
        or not isinstance(los_rows, list)
        or len(topology_rows) != len(ids)
        or len(topology_digests) != len(ids)
        or len(los_rows) != len(ids)
    ):
        raise ValueError("channel result lacks aligned scenario topology rows")
    h_all = _numpy(channel.h_path, name="h_path", kind="complex")
    tau_all = _numpy(channel.tau_s, name="tau_s", kind="real")
    cfr_all = _numpy(channel.channel_frequency_response, name="cfr", kind="complex")
    if h_all.shape[0] != len(ids) or tau_all.shape[0] != len(ids) or cfr_all.shape[0] != len(ids):
        raise ValueError("sample IDs must match the generator result batch axis")
    configuration = metadata.get("generator_configuration")
    config_digest = metadata.get("configuration_sha256")
    if not isinstance(configuration, Mapping) or _digest_json(
        configuration, label="generator configuration"
    ) != config_digest:
        raise ValueError("channel generator configuration identity is invalid")
    if configuration.get("scenario") != scenario or configuration.get("batch_size") != len(ids):
        raise ValueError("generator configuration disagrees with scenario batch metadata")
    if any(not isinstance(row, dict) for row in topology_rows):
        raise ValueError("scenario topology rows must be objects")
    topology_keys = set(topology_rows[0])
    if any(set(row) != topology_keys for row in topology_rows):
        raise ValueError("scenario topology rows do not share one canonical field roster")
    reconstructed_topology = {
        key: [row[key] for row in topology_rows] for key in sorted(topology_keys)
    }
    if _digest_json(reconstructed_topology, label="batch topology") != metadata.get(
        "topology_sha256"
    ):
        raise ValueError("scenario batch topology digest is invalid")
    spacing = configuration.get("subcarrier_spacing_hz")
    count = configuration.get("num_subcarriers")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 8
        or isinstance(spacing, bool)
        or not isinstance(spacing, (int, float))
        or not math.isfinite(float(spacing))
        or spacing <= 0
    ):
        raise ValueError("generator configuration has an invalid subcarrier count")
    if configuration.get("batch_size") != len(ids):
        raise ValueError("sample IDs must cover the complete configured generator batch")
    offsets = (np.arange(count, dtype=np.float64) - count // 2) * float(spacing)
    records: list[ScenarioGBSMRecord] = []
    for index, sample_id in enumerate(ids):
        topology = topology_rows[index]
        topology_digest = _digest_json(topology, label="topology row")
        if topology_digest != topology_digests[index]:
            raise ValueError("topology row digest differs from generator metadata")
        scenario_los = los_rows[index]
        if not isinstance(scenario_los, bool) or topology.get("los") != [[scenario_los]]:
            raise ValueError("scenario LOS row disagrees with explicit topology")
        record_metadata = {
            "schema": "umcd.scenario_gbsm_record.v1",
            "scenario": scenario,
            "scenario_los": scenario_los,
            "geometric_los_label": None,
            "generator_configuration": dict(configuration),
            "configuration_sha256": config_digest,
            "topology": topology,
            "topology_row_sha256": topology_digest,
            "batch_topology_sha256": metadata.get("topology_sha256"),
            "sionna_version": metadata.get("sionna_version"),
            "split_subject_id": sample_id,
            "split_parent_type": "standardized_channel_draw",
            "split_source_family": f"scenario_gbsm:{scenario}",
            "split_leakage_relations": [],
        }
        records.append(
            ScenarioGBSMRecord(
                sample_id=sample_id,
                run_id=run_id,
                sample_index=index,
                scenario=scenario,
                scenario_los=scenario_los,
                h_path=h_all[index],
                tau_s=tau_all[index],
                cfr=cfr_all[index],
                subcarrier_offsets_hz=offsets,
                metadata=record_metadata,
            )
        )
    return ScenarioGBSMRun(run_id, tuple(records))


def _record_bytes(record: ScenarioGBSMRecord) -> bytes:
    metadata = {
        "schema": "umcd.scenario_gbsm_record.v1",
        "sample_id": record.sample_id,
        "run_id": record.run_id,
        "sample_index": record.sample_index,
        "scenario": record.scenario,
        "scenario_los": record.scenario_los,
        "metadata": _plain_json(record.metadata),
    }
    metadata_bytes = _canonical_json(metadata, label="record metadata")
    uncompressed_size = sum(
        array.nbytes
        for array in (
            record.h_path,
            record.tau_s,
            record.cfr,
            record.subcarrier_offsets_hz,
        )
    ) + len(metadata_bytes)
    if uncompressed_size > MAX_SCENARIO_GBSM_OBJECT_BYTES:
        raise ValueError("scenario GBSM record exceeds its uncompressed byte ceiling")
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        h_path=record.h_path,
        tau_s=record.tau_s,
        cfr=record.cfr,
        subcarrier_offsets_hz=record.subcarrier_offsets_hz,
        metadata_json=np.frombuffer(metadata_bytes, dtype=np.uint8),
    )
    return stream.getvalue()


def write_scenario_gbsm_run(
    root: str | Path, run: ScenarioGBSMRun
) -> StoredScenarioGBSMRun:
    """Store path/CFR records by content digest and atomically publish the run index."""
    if not isinstance(run, ScenarioGBSMRun):
        raise TypeError("run must be a ScenarioGBSMRun")
    base = Path(root)
    references: list[dict[str, str]] = []
    for record in run.records:
        payload = _record_bytes(record)
        if len(payload) > MAX_SCENARIO_GBSM_OBJECT_BYTES:
            raise ValueError("scenario GBSM record exceeds its byte ceiling")
        digest = _sha256(payload)
        relative = f"scenario_gbsm/records/{digest[:2]}/{digest}.npz"
        _publish_immutable(base / relative, payload)
        references.append(
            {"sample_id": record.sample_id, "content_sha256": digest, "relative_path": relative}
        )
    core = {
        "schema": "umcd.scenario_gbsm_run.v1",
        "schema_version": 1,
        "run_id": run.run_id,
        "records": references,
    }
    digest = _sha256(_canonical_json(core, label="scenario run manifest"))
    manifest = {**core, "content_sha256": digest}
    relative_manifest = (
        "scenario_gbsm/manifests/"
        f"{hashlib.sha256(run.run_id.encode()).hexdigest()}.json"
    )
    manifest_bytes = _canonical_json(manifest, label="scenario manifest") + b"\n"
    if len(manifest_bytes) > MAX_SCENARIO_GBSM_MANIFEST_BYTES:
        raise ValueError("scenario GBSM manifest exceeds its byte ceiling")
    _publish_immutable(base / relative_manifest, manifest_bytes)
    return StoredScenarioGBSMRun(
        run_id=run.run_id,
        content_sha256=digest,
        relative_path=relative_manifest,
        sample_ids=tuple(record.sample_id for record in run.records),
    )


def read_scenario_gbsm_run(
    root: str | Path, manifest: StoredScenarioGBSMRun | str | Path
) -> ScenarioGBSMRun:
    """Read a bounded scenario run and reconstruct every record from verified bytes."""
    base = Path(root)
    if isinstance(manifest, StoredScenarioGBSMRun):
        stored = manifest
        relative = PurePosixPath(stored.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("scenario manifest path must remain inside the store")
        manifest_path = base.joinpath(*relative.parts)
    else:
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = base / manifest_path
        raw_bytes = _stable_file_bytes(
            manifest_path,
            label="scenario GBSM manifest",
            max_bytes=MAX_SCENARIO_GBSM_MANIFEST_BYTES,
        )
        raw = _parse_json_bytes(raw_bytes, label="scenario GBSM manifest")
        if not isinstance(raw, dict) or not isinstance(raw.get("records"), list):
            raise ValueError("scenario GBSM manifest has an invalid root or record roster")
        stored = StoredScenarioGBSMRun(
            run_id=raw["run_id"],
            content_sha256=raw["content_sha256"],
            relative_path=manifest_path.relative_to(base).as_posix(),
            sample_ids=tuple(item["sample_id"] for item in raw["records"]),
            schema_version=raw["schema_version"],
        )
    if (
        isinstance(stored.schema_version, bool)
        or not isinstance(stored.schema_version, int)
        or stored.schema_version != 1
    ):
        raise ValueError("unsupported scenario GBSM run schema version")
    if (
        not isinstance(stored.run_id, str)
        or not stored.run_id.strip()
        or not isinstance(stored.content_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", stored.content_sha256) is None
        or not isinstance(stored.sample_ids, tuple)
        or any(
            not isinstance(sample_id, str)
            or not sample_id.strip()
            or sample_id.strip() != sample_id
            for sample_id in stored.sample_ids
        )
        or len(set(stored.sample_ids)) != len(stored.sample_ids)
    ):
        raise ValueError("scenario GBSM run pointer is malformed")
    raw_bytes = _stable_file_bytes(
        manifest_path,
        label="scenario GBSM manifest",
        max_bytes=MAX_SCENARIO_GBSM_MANIFEST_BYTES,
    )
    raw_manifest = _parse_json_bytes(raw_bytes, label="scenario GBSM manifest")
    if not isinstance(raw_manifest, dict) or set(raw_manifest) != {
        "schema", "schema_version", "run_id", "records", "content_sha256"
    }:
        raise ValueError("scenario GBSM manifest fields are invalid")
    if (
        raw_manifest.get("schema") != "umcd.scenario_gbsm_run.v1"
        or isinstance(raw_manifest.get("schema_version"), bool)
        or raw_manifest.get("schema_version") != 1
        or not isinstance(raw_manifest.get("records"), list)
        or not raw_manifest["records"]
    ):
        raise ValueError("unsupported scenario GBSM run manifest")
    expected_manifest_path = (
        "scenario_gbsm/manifests/"
        f"{hashlib.sha256(str(raw_manifest.get('run_id', '')).encode()).hexdigest()}.json"
    )
    if not isinstance(raw_manifest.get("run_id"), str) or not raw_manifest["run_id"].strip():
        raise ValueError("scenario GBSM manifest run ID is invalid")
    if manifest_path.resolve(strict=True).relative_to(base.resolve(strict=True)).as_posix() != (
        expected_manifest_path
    ):
        raise ValueError("scenario GBSM manifest is not at its canonical run path")
    core = {
        key: raw_manifest[key]
        for key in ("schema", "schema_version", "run_id", "records")
    }
    digest = _sha256(_canonical_json(core, label="scenario run manifest"))
    if digest != raw_manifest.get("content_sha256") or digest != stored.content_sha256:
        raise ValueError("scenario GBSM manifest hash is invalid")
    if raw_manifest.get("run_id") != stored.run_id:
        raise ValueError("scenario GBSM run ID disagrees with its pointer")
    references = raw_manifest["records"]
    if any(
        not isinstance(reference, dict)
        or set(reference) != {"sample_id", "content_sha256", "relative_path"}
        or not isinstance(reference.get("sample_id"), str)
        or not reference["sample_id"].strip()
        or not isinstance(reference.get("relative_path"), str)
        or not isinstance(reference.get("content_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", reference["content_sha256"]) is None
        for reference in references
    ):
        raise ValueError("scenario GBSM manifest record reference is malformed")
    if len({reference["sample_id"] for reference in references}) != len(references):
        raise ValueError("scenario GBSM manifest contains duplicate sample IDs")
    if tuple(reference["sample_id"] for reference in references) != stored.sample_ids:
        raise ValueError("scenario sample order disagrees with run pointer")
    records: list[ScenarioGBSMRecord] = []
    for reference in references:
        relative = PurePosixPath(reference["relative_path"])
        expected_relative = (
            f"scenario_gbsm/records/{reference['content_sha256'][:2]}/"
            f"{reference['content_sha256']}.npz"
        )
        if relative.as_posix() != expected_relative:
            raise ValueError("scenario record path does not match its content digest")
        object_path = _store_path(base, relative, label="scenario record")
        payload = _stable_file_bytes(
            object_path,
            label="scenario GBSM object",
            max_bytes=MAX_SCENARIO_GBSM_OBJECT_BYTES,
        )
        object_hash = _sha256(payload)
        if object_hash != reference["content_sha256"] or relative.name != f"{object_hash}.npz":
            raise ValueError("scenario GBSM record content hash is invalid")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive_file:
            members = archive_file.infolist()
            if (
                len(members) != 5
                or sum(member.file_size for member in members) > MAX_SCENARIO_GBSM_OBJECT_BYTES
                or any(member.file_size < 0 for member in members)
            ):
                raise ValueError("scenario GBSM record archive exceeds its member contract")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != {
                "h_path", "tau_s", "cfr", "subcarrier_offsets_hz", "metadata_json"
            }:
                raise ValueError("scenario GBSM record archive fields are invalid")
            metadata = _parse_json_bytes(
                archive["metadata_json"].tobytes(), label="scenario GBSM record metadata"
            )
            if not isinstance(metadata, dict) or set(metadata) != {
                "schema", "sample_id", "run_id", "sample_index", "scenario",
                "scenario_los", "metadata"
            } or metadata.get("schema") != "umcd.scenario_gbsm_record.v1":
                raise ValueError("unsupported scenario GBSM record")
            if metadata.get("sample_id") != reference["sample_id"]:
                raise ValueError("scenario record identity disagrees with manifest")
            records.append(
                ScenarioGBSMRecord(
                    sample_id=metadata["sample_id"],
                    run_id=metadata["run_id"],
                    sample_index=metadata["sample_index"],
                    scenario=metadata["scenario"],
                    scenario_los=metadata["scenario_los"],
                    h_path=archive["h_path"],
                    tau_s=archive["tau_s"],
                    cfr=archive["cfr"],
                    subcarrier_offsets_hz=archive["subcarrier_offsets_hz"],
                    metadata=metadata["metadata"],
                )
            )
    return ScenarioGBSMRun(stored.run_id, tuple(records))
