"""NR observations and records: axis contracts, batch-of-one rules, typed fields and target-mask domains."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from datasets.nr import NRChannelRecord, TargetDefinition
from receivers.nr import NRChannelEstimate
from tests.test_datasets.test_nr import make_observation, make_single_record


def _estimate(**overrides):
    values = {
        "h_hat": np.zeros((2, 1, 2, 1, 1, 14, 24), dtype=np.complex64),
        "error_variance": np.zeros((1, 1, 1, 1, 1, 14, 24), dtype=np.float32),
        "pilot_mask": np.zeros((1, 1, 14, 24), dtype=np.bool_),
        "metadata": {"receiver": "PUSCHLSChannelEstimator", "sionna_version": "2.2.0"},
    }
    values.update(overrides)
    return NRChannelEstimate(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"sample_id": " padded "}, "non-empty canonical string", id="sample-id"),
        pytest.param(
            {"channel_estimate": "not-an-estimate"},
            "must be an NRChannelEstimate",
            id="estimate-type",
        ),
        pytest.param(
            {"received_grid": np.zeros((2, 1, 2, 14))}, "five named axes|must have", id="grid-rank"
        ),
    ],
)
def test_observation_fields_are_typed_and_named(overrides, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        make_observation(**overrides)


def test_estimate_must_have_seven_axes_matching_the_grid() -> None:
    with pytest.raises(ValueError, match="seven named channel axes"):
        make_observation(
            channel_estimate=_estimate(h_hat=np.zeros((2, 1, 2, 14, 24), dtype=np.complex64))
        )


def test_received_mask_must_live_on_the_grid_device() -> None:
    grid = np.zeros((2, 1, 2, 14, 24), dtype=np.complex64)
    mask = np.zeros((2, 1, 2, 14, 24), dtype=bool)
    observation = make_observation(received_grid=grid, received_grid_mask=mask)
    assert observation.received_grid_mask is not None


def test_linked_observations_must_hold_exactly_one_batch_item() -> None:
    record = make_single_record(
        "linked-001",
        targets={"geometric_los": np.asarray(True)},
        target_masks={"geometric_los": np.asarray(True)},
    )
    assert record.observation.sample_id == "linked-001"


def test_target_masks_must_be_boolean_and_targets_non_empty() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        make_single_record(
            "empty-target",
            targets={"channel_cfr": np.zeros((0,), dtype=np.complex64)},
            target_masks={"channel_cfr": np.zeros((0,), dtype=bool)},
        )
    with pytest.raises(TypeError, match="boolean dtype"):
        make_single_record(
            "int-mask",
            targets={"geometric_los": np.asarray(True)},
            target_masks={"geometric_los": 1},
        )


def test_record_provenance_must_be_a_mapping_of_identities() -> None:
    record = make_single_record(
        "provenance-001",
        targets={"geometric_los": np.asarray(True)},
        target_masks={"geometric_los": np.asarray(True)},
    )
    assert isinstance(record, NRChannelRecord)
    with pytest.raises((TypeError, ValueError)):
        dataclasses.replace(record, provenance=["not", "a", "mapping"])


def test_target_definitions_must_be_objects() -> None:
    assert (
        TargetDefinition.versioned(
            semantic="visibility",
            units="boolean",
            source="sionna_rt",
            reference="fixture",
            aggregation_rule="one value per sample",
            definition_version="tests.nr.guards.v1",
        ).semantic
        == "visibility"
    )
