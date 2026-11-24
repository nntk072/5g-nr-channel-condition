"""Scenario GBSM drops bound to their canonical episode: each mismatch is named."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import replace

import pytest
from datasets.network import DirectedLink
from datasets.scenario_gbsm import scenario_gbsm_episode_bundle, scenario_gbsm_run_from_channel
from tests.test_datasets.test_scenario_gbsm_store import _native_channel


def _parts():
    record = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="pilot-umix", sample_ids=["drop-000"]
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
    with pytest.raises(ValueError, match=message):
        record.validate_for_episode(replace(episode, **episode_change))


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
