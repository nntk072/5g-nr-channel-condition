"""Episode bundles: receiver observations must bind to an episode link and frame, uniquely."""

from __future__ import annotations

from dataclasses import replace

import pytest
from datasets.nr import LinkedNRObservation
from tests.test_datasets.test_umcd_store_integrity import _episode_bundle_with_receiver_observation


def _binding():
    bundle = _episode_bundle_with_receiver_observation()
    return bundle, bundle.receiver_observations[0]


def test_receiver_observation_must_reference_an_episode_link() -> None:
    bundle, binding = _binding()
    bad = LinkedNRObservation("absent-link", binding.frame_index, binding.observation)
    with pytest.raises(
        ValueError, match="must reference an episode link|must identify an episode link"
    ):
        replace(bundle, receiver_observations=(bad,))


def test_receiver_observation_frame_must_exist_in_the_episode() -> None:
    bundle, binding = _binding()
    frames = len(bundle.episode.frame_times_s)
    bad = LinkedNRObservation(binding.link_id, frames + 5, binding.observation)
    with pytest.raises(ValueError, match="must reference an episode frame"):
        replace(bundle, receiver_observations=(bad,))


def test_one_receiver_observation_per_link_and_frame() -> None:
    bundle, binding = _binding()
    with pytest.raises(ValueError, match="unique per link and frame"):
        replace(bundle, receiver_observations=(binding, binding))
