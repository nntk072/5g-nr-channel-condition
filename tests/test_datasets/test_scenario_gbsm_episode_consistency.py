"""Scenario GBSM drops must agree with the canonical episode that carries them."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from datasets.episode_store import EpisodeBundle
from datasets.scenario_gbsm import (
    ScenarioGBSMRecord,
    scenario_gbsm_episode_bundle,
    scenario_gbsm_run_from_channel,
)
from tests.test_datasets.test_scenario_gbsm_store import _native_channel


def _bundle_parts():
    record = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="pilot-umix", sample_ids=["drop-000"]
    ).records[0]
    bundle = scenario_gbsm_episode_bundle(record)
    return record, bundle


def _bind(record, episode):
    return EpisodeBundle(
        episode=episode,
        path_oracle=None,
        channel_views=(),
        targets=(),
        scenario_gbsm_records=(record,),
    )


def test_consistent_drop_binds_to_its_episode() -> None:
    record, bundle = _bundle_parts()
    _bind(record, bundle.episode)


@pytest.mark.parametrize(
    ("episode_change", "message"),
    [
        ({"episode_id": "another-drop"}, "sample ID must equal its parent episode ID"),
        ({"carrier_hz": 2.6e9}, "carrier differs from its parent episode"),
    ],
)
def test_episode_identity_and_radio_context_must_match(episode_change, message) -> None:
    record, bundle = _bundle_parts()
    episode = replace(bundle.episode, **episode_change)
    with pytest.raises(ValueError, match=message):
        _bind(record, episode)


def test_record_is_a_scenario_gbsm_record() -> None:
    record, _ = _bundle_parts()
    assert isinstance(record, ScenarioGBSMRecord)


def test_subcarrier_grid_must_match_the_episode() -> None:
    record, bundle = _bundle_parts()
    # Scaling about DC keeps the grid valid for the episode but changes its values.
    scaled = np.asarray(bundle.episode.subcarrier_offsets_hz) * 2.0
    with pytest.raises(ValueError, match="frequency grid differs"):
        _bind(record, replace(bundle.episode, subcarrier_offsets_hz=scaled))
