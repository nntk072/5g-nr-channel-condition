"""NR observations and linked observations: identity, frame index and batch-size rules."""

from __future__ import annotations

import dataclasses

import pytest
from datasets.nr import LinkedNRObservation
from tests.test_datasets.test_nr_observation_mask import _observation


def test_observation_sample_identity_must_be_canonical() -> None:
    with pytest.raises(ValueError, match="sample_id must be a non-empty canonical string"):
        dataclasses.replace(_observation(), sample_id=" padded")
    with pytest.raises(ValueError, match="sample_id must be a non-empty canonical string"):
        dataclasses.replace(_observation(), sample_id="")


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        pytest.param(
            {"link_id": " link"},
            ValueError,
            "link_id must be a non-empty canonical string",
            id="link-padded",
        ),
        pytest.param(
            {"link_id": ""},
            ValueError,
            "link_id must be a non-empty canonical string",
            id="link-empty",
        ),
        pytest.param(
            {"frame_index": True}, TypeError, "frame_index must be an integer", id="frame-bool"
        ),
        pytest.param(
            {"frame_index": -1}, ValueError, "frame_index must be non-negative", id="frame-negative"
        ),
        pytest.param(
            {"observation": "not-an-observation"},
            TypeError,
            "observation must be an NRObservation",
            id="observation-type",
        ),
    ],
)
def test_linked_observation_checks_its_binding(changes, error, message) -> None:
    kwargs = {"link_id": "link-a", "frame_index": 0, "observation": _observation()}
    kwargs.update(changes)
    with pytest.raises(error, match=message):
        LinkedNRObservation(**kwargs)
