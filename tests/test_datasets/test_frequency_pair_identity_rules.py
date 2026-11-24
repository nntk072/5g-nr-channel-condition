"""Paired frequency views: identity, digest and view-pairing rules."""

from __future__ import annotations

import dataclasses

import pytest
from datasets.frequency_pair import PairedFrequencyViews
from tests.test_datasets.test_frequency_pair import _pair

PAIR_FIELDS = {
    "pair_id": "paired-drop-1",
    "source_scene_id": "munich-block-3",
    "parent_episode_id": "drop-1",
    "split_group_id": "site-3",
    "lower_bundle_episode_id": "episode-low",
    "upper_bundle_episode_id": "episode-high",
}


@pytest.mark.parametrize("field", sorted(PAIR_FIELDS))
def test_identity_fields_must_be_canonical(field) -> None:
    base = _pair()
    with pytest.raises(ValueError, match=f"{field} must be a non-empty canonical string"):
        dataclasses.replace(base, **{field: " padded "})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("scene_geometry_sha256", "E" * 64, id="geometry-case"),
        pytest.param("lower_bundle_sha256", "abc", id="lower-short"),
        pytest.param("upper_bundle_sha256", "F" * 64, id="upper-case"),
    ],
)
def test_digests_must_be_lowercase_sha256(field, value) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256 digest"):
        dataclasses.replace(_pair(), **{field: value})


def test_views_and_provenance_must_be_typed() -> None:
    base = _pair()
    with pytest.raises(TypeError, match="must be a PhysicalChannelView|PhysicalChannelView"):
        dataclasses.replace(base, lower_view="not-a-view")
    assert isinstance(base, PairedFrequencyViews)
