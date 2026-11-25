"""Scenario GBSM drops bound to their canonical episode: each mismatch is named."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest
from datasets.network import DirectedLink
from datasets.scenario_gbsm import scenario_gbsm_episode_bundle, scenario_gbsm_run_from_channel
from tests.test_datasets.test_scenario_gbsm_store import _native_channel


def _parts():
    channel = _native_channel()
    channel.h_path = np.repeat(channel.h_path, 2, axis=-1)
    channel.channel_frequency_response = np.repeat(channel.channel_frequency_response, 2, axis=5)
    record = scenario_gbsm_run_from_channel(
        channel, run_id="pilot-umix", sample_ids=["drop-000"]
    ).records[0]
    return record, scenario_gbsm_episode_bundle(record).episode


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _with_metadata(record, **changes):
    """Return the record with edited metadata, bypassing the digest checks of its constructor.

    ``validate_for_episode`` must reject these edits even though construction would not.
    """
    metadata = _thaw(record.metadata)
    for path, value in changes.items():
        target = metadata
        keys = path.split("__")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    forged = copy.copy(record)
    object.__setattr__(forged, "metadata", metadata)
    return forged


@pytest.mark.parametrize(
    ("episode_change", "message"),
    [
        pytest.param({"split_group_id": "other-group"}, "split group differs", id="split-group"),
        pytest.param({"frame_times_s": (0.0, 1.0e-3)}, "time samples differ", id="time-samples"),
    ],
)
def test_episode_structure_must_match_the_drop(episode_change, message) -> None:
    record, episode = _parts()
    if "frame_times_s" in episode_change:
        changed = copy.copy(episode)
        object.__setattr__(changed, "frame_times_s", (0.0, 1.0e-3, 2.0e-3))
    else:
        changed = replace(episode, **episode_change)
    with pytest.raises(ValueError, match=message):
        record.validate_for_episode(changed)


def test_episode_with_a_second_link_is_refused() -> None:
    record, episode = _parts()
    extra = replace(episode.links[0], link_id="second-link")
    with pytest.raises(ValueError, match="requires one gNB–UE link per episode"):
        record.validate_for_episode(replace(episode, links=(episode.links[0], extra)))


def test_link_direction_must_match_the_generator_configuration() -> None:
    record, episode = _parts()
    link = episode.links[0]
    reversed_link = DirectedLink(
        link_id=link.link_id,
        tx_node_id=link.rx_node_id,
        rx_node_id=link.tx_node_id,
        tx_port_indices=link.rx_port_indices,
        rx_port_indices=link.tx_port_indices,
        direction=link.direction,
        role=link.role,
    )
    with pytest.raises(ValueError, match="link direction differs"):
        record.validate_for_episode(replace(episode, links=(reversed_link,)))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        pytest.param(
            {"split_subject_id": "someone-else"}, "split subject differs", id="split-subject"
        ),
        pytest.param(
            {"topology__bs_loc": ((0.0, 0.0, 99.0),)}, "BS location differs", id="bs-location"
        ),
        pytest.param(
            {"topology__ut_loc": ((9.0, 9.0, 1.5),)}, "UT location differs", id="ut-location"
        ),
    ],
)
def test_metadata_must_agree_with_the_episode(changes, message) -> None:
    record, episode = _parts()
    record_changed = _with_metadata(record, **changes)
    with pytest.raises(ValueError, match=message):
        record_changed.validate_for_episode(episode)


def test_sampling_interval_must_match_the_generator_rate() -> None:
    record, episode = _parts()
    changed = copy.copy(episode)
    object.__setattr__(changed, "frame_times_s", (0.0, 1.0e-3))
    with pytest.raises(ValueError, match="sampling interval differs"):
        record.validate_for_episode(changed)


def test_episode_link_endpoints_must_match_the_gnb_and_ue_nodes() -> None:
    record, episode = _parts()
    link = replace(episode.links[0], rx_node_id="unlisted-node")
    changed = copy.copy(episode)
    object.__setattr__(changed, "links", (link,))
    with pytest.raises(ValueError, match="link endpoints differ"):
        record.validate_for_episode(changed)


def test_record_port_axes_must_match_episode_array_ports() -> None:
    record, episode = _parts()
    changed_record = copy.copy(record)
    object.__setattr__(changed_record, "h_path", np.zeros((1, 2, 1, 1, 1, 2)))
    with pytest.raises(ValueError, match="antenna-port axes differ"):
        changed_record.validate_for_episode(episode)


def test_measured_ut_velocity_must_match_topology_velocity() -> None:
    record, episode = _parts()
    record_changed = _with_metadata(record, topology__ut_velocities=((1.0, 0.0, 0.0),))
    with pytest.raises(ValueError, match="UT velocity differs"):
        record_changed.validate_for_episode(episode)


def test_moving_bs_trajectory_is_not_representable() -> None:
    record, episode = _parts()
    timeline = copy.copy(episode.pose_timeline)
    positions = np.array(timeline.positions_world_m, copy=True)
    positions[1, 0, 0] += 1.0
    object.__setattr__(timeline, "positions_world_m", positions)
    changed = copy.copy(episode)
    object.__setattr__(changed, "pose_timeline", timeline)
    with pytest.raises(ValueError, match="cannot encode a moving BS trajectory"):
        record.validate_for_episode(changed)


def test_rotating_endpoint_trajectory_is_not_representable() -> None:
    record, episode = _parts()
    timeline = copy.copy(episode.pose_timeline)
    orientations = np.array(timeline.orientations_local_to_world, copy=True)
    orientations[1, 1, 0, 0] = 0.0
    orientations[1, 1, 0, 1] = 1.0
    object.__setattr__(timeline, "orientations_local_to_world", orientations)
    changed = copy.copy(episode)
    object.__setattr__(changed, "pose_timeline", timeline)
    with pytest.raises(ValueError, match="cannot encode rotating array trajectories"):
        record.validate_for_episode(changed)
