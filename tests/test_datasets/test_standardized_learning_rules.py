"""Learning-example and learning-run rules layered over standardized channel records."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from datasets.standardized import (
    StandardizedFamily,
    StandardizedLearningExample,
    StandardizedLearningRun,
    standardized_learning_example_from_record,
    standardized_learning_run_from_channel_run,
    standardized_run_from_channel,
)
from tests.test_datasets.test_standardized import _channel


@pytest.fixture
def channel_run():
    return standardized_run_from_channel(
        _channel(), family=StandardizedFamily.TDL, run_id="learning-rules"
    )


@pytest.fixture
def learning_run(channel_run):
    return standardized_learning_run_from_channel_run(channel_run, snr_db=15.0, seed=7)


def _example_kwargs(example) -> dict[str, Any]:
    return {
        "observation": example.observation,
        "targets": dict(example.targets),
        "source_run_id": example.source_run_id,
        "observation_seed": example.observation_seed,
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param({"targets": {}}, "named TargetValue records", id="no-targets"),
        pytest.param(
            {"targets": {"x": "not-a-target"}}, "named TargetValue records", id="target-not-typed"
        ),
        pytest.param({"source_run_id": "  "}, "source_run_id must be non-empty", id="blank-source"),
        pytest.param(
            {"observation_seed": -1},
            "observation_seed must be a nonnegative integer",
            id="negative-seed",
        ),
        pytest.param(
            {"observation_seed": True},
            "observation_seed must be a nonnegative integer",
            id="bool-seed",
        ),
    ],
)
def test_learning_example_rejects_malformed_fields(learning_run, change, message) -> None:
    kwargs = _example_kwargs(learning_run.examples[0])
    kwargs.update(change)
    with pytest.raises((TypeError, ValueError), match=message):
        StandardizedLearningExample(**kwargs)


def test_learning_example_targets_must_name_its_own_observation(learning_run) -> None:
    first, second = learning_run.examples[0], learning_run.examples[1]
    kwargs = _example_kwargs(first)
    kwargs["targets"] = dict(second.targets)
    with pytest.raises(ValueError, match="target sample IDs must match"):
        StandardizedLearningExample(**kwargs)


def test_learning_run_rejects_duplicate_samples_and_foreign_parents(learning_run) -> None:
    examples = learning_run.examples
    with pytest.raises(ValueError, match="sample IDs must be unique"):
        StandardizedLearningRun(run_id=learning_run.run_id, examples=(examples[0], examples[0]))

    foreign = dataclasses.replace(examples[1], source_run_id="another-run")
    with pytest.raises(ValueError, match="must reference their parent channel run"):
        StandardizedLearningRun(run_id=learning_run.run_id, examples=(examples[0], foreign))


def test_learning_example_from_record_checks_its_arguments(channel_run) -> None:
    record = channel_run.records[0]
    with pytest.raises(TypeError, match="record must be a StandardizedChannelRecord"):
        standardized_learning_example_from_record(object(), snr_db=10.0, seed=1)
    with pytest.raises(ValueError, match="snr_db must be finite"):
        standardized_learning_example_from_record(record, snr_db=float("nan"), seed=1)
    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        standardized_learning_example_from_record(record, snr_db=10.0, seed=-1)
    with pytest.raises(TypeError, match="time_index must be an integer"):
        standardized_learning_example_from_record(record, snr_db=10.0, seed=1, time_index=0.5)


def test_learning_run_seed_must_be_a_nonnegative_integer(channel_run) -> None:
    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        standardized_learning_run_from_channel_run(channel_run, snr_db=10.0, seed=-3)
    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        standardized_learning_run_from_channel_run(channel_run, snr_db=10.0, seed=True)


def test_channel_run_family_must_be_the_standardized_enum() -> None:
    with pytest.raises(TypeError, match="family must use StandardizedFamily"):
        standardized_run_from_channel(_channel(), family="tdl")
