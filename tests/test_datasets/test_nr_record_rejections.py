"""NR channel records and observations: every rejected axis, mask, target and provenance input."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from datasets.nr import NRChannelRecord, NRObservation, TargetDefinition
from receivers.nr import NRChannelEstimate, NRPUSCHConfig
from tests.test_datasets.test_nr_observation_mask import _observation


def _definition(name: str = "target") -> TargetDefinition:
    return TargetDefinition.versioned(
        semantic=f"{name} semantic",
        units="fixture",
        source="fixture",
        reference="fixture",
        aggregation_rule="one scalar per record",
        definition_version=f"tests.{name}.v1",
    )


def _record(**overrides):
    kwargs = {
        "observation": _observation(),
        "targets": {"target": np.asarray([1.0], dtype=np.float32)},
        "target_masks": {"target": np.asarray([True])},
        "provenance": {"source": "test"},
        "target_definitions": {"target": _definition()},
    }
    kwargs.update(overrides)
    return NRChannelRecord(**kwargs)


def test_the_fixture_record_is_valid() -> None:
    assert _record().targets["target"].shape == (1,)


def test_observation_rejects_bad_identities_and_frame_indices() -> None:
    observation = _observation()
    with pytest.raises(ValueError, match="canonical string"):
        dataclasses.replace(observation, sample_id="  ")


def test_observation_grid_axes_are_checked() -> None:
    observation = _observation()
    with pytest.raises(ValueError, match="must have \\[batch, rx, rx_ant, symbol, fft_bin\\] axes"):
        dataclasses.replace(observation, received_grid=observation.received_grid[0])
    empty = np.ones((1, 1, 1, 0, 12), dtype=np.complex64)
    with pytest.raises(ValueError, match="axes must be non-empty"):
        dataclasses.replace(observation, received_grid=empty)


def test_fft_bin_indices_must_be_unique() -> None:
    observation = _observation()
    with pytest.raises(ValueError, match="fft_bin_indices must be unique"):
        dataclasses.replace(observation, fft_bin_indices=(0,) * 12)


def test_channel_estimate_must_have_seven_named_axes() -> None:
    observation = _observation()
    with pytest.raises(ValueError, match="seven named channel axes"):
        NRChannelEstimate(
            h_hat=observation.channel_estimate.h_hat.reshape(1, 1, 1, 14, 12),
            error_variance=np.zeros((1, 1, 1, 14, 12), dtype=np.float32),
            pilot_mask=observation.channel_estimate.pilot_mask,
            metadata={"receiver": "test"},
        )


def test_received_mask_must_match_the_received_grid_shape() -> None:
    observation = _observation()
    with pytest.raises(ValueError):
        dataclasses.replace(observation, received_grid_mask=np.ones((1, 1, 1, 3, 3), dtype=bool))


def test_record_targets_must_be_declared_and_named() -> None:
    with pytest.raises(ValueError, match="at least one explicitly named target"):
        _record(targets={}, target_masks={}, target_definitions={})
    with pytest.raises(ValueError, match="must be non-empty"):
        _record(
            targets={"": np.asarray([1.0], dtype=np.float32)},
            target_masks={"": np.asarray([True])},
            target_definitions={"": _definition()},
        )


def test_target_definitions_must_be_typed() -> None:
    with pytest.raises(TypeError, match="target definitions must be TargetDefinition values"):
        _record(target_definitions={"target": "not-a-definition"})


def test_target_masks_must_be_boolean_and_broadcastable() -> None:
    with pytest.raises(TypeError, match="must have boolean dtype"):
        _record(target_masks={"target": np.asarray([1])})
    with pytest.raises(ValueError, match="not broadcastable"):
        _record(target_masks={"target": np.asarray([True, False, True])})


def test_masked_target_elements_must_use_zero_storage() -> None:
    with pytest.raises(ValueError, match="must use zero storage"):
        _record(
            targets={"target": np.asarray([1.0], dtype=np.float32)},
            target_masks={"target": np.asarray([False])},
        )


def test_provenance_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="provenance must not be empty"):
        _record(provenance={})


def test_reason_codes_must_correspond_to_declared_targets() -> None:
    with pytest.raises(ValueError, match="reason codes must correspond to declared targets"):
        _record(target_reason_codes={"ghost": ("ok",)})
