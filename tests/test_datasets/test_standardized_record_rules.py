"""Record, run and learning-example validation for standardized channel data.

Each case starts from one valid record and changes one declared field, so the message
checked is the rule that rejected that change.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from datasets.standardized import (
    StandardizedChannelRecord,
    StandardizedChannelRun,
    StandardizedFamily,
    StandardizedLearningExample,
    StandardizedLearningRun,
    validate_standardized_channel_admission,
)
from tests.test_datasets.test_standardized import _channel


def _record_kwargs(**overrides: Any) -> dict[str, Any]:
    channel = _channel()
    kwargs: dict[str, Any] = {
        "sample_id": "sample.0",
        "run_id": "run.0",
        "sample_index": 0,
        "family": StandardizedFamily.TDL,
        "profile_model": "A",
        "profile_los_component": False,
        "h_path": channel.h_path[0],
        "tau_s": channel.tau_s[0],
        "cfr": channel.channel_frequency_response[0],
        "subcarrier_offsets_hz": (np.arange(8) - 4) * 15_000.0,
        "metadata": copy.deepcopy(channel.metadata),
    }
    kwargs.update(overrides)
    return kwargs


def _record(**overrides: Any) -> StandardizedChannelRecord:
    return StandardizedChannelRecord(**_record_kwargs(**overrides))


def test_valid_record_is_accepted_and_frozen() -> None:
    record = _record()

    assert record.sample_id == "sample.0"
    assert record.h_path.flags.writeable is False
    assert record.cfr.flags.writeable is False
    with pytest.raises(TypeError):
        record.metadata["profile_model"] = "B"  # type: ignore[index]


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        pytest.param({"sample_id": "  "}, ValueError, "sample_id must be a non-empty string", id="blank-sample"),
        pytest.param({"run_id": 3}, ValueError, "run_id must be a non-empty string", id="run-not-string"),
        pytest.param({"sample_index": True}, TypeError, "sample_index must be an integer", id="bool-index"),
        pytest.param({"sample_index": -1}, ValueError, "sample_index must be nonnegative", id="negative-index"),
        pytest.param({"family": "tdl"}, TypeError, "family must use StandardizedFamily", id="family-string"),
        pytest.param(
            {"family": StandardizedFamily.CDL, "profile_model": "Z"},
            ValueError,
            "CDL profile must be one of A–E",
            id="cdl-unknown-profile",
        ),
        pytest.param(
            {"profile_model": "Z"},
            ValueError,
            "TDL profile must be scalable A–E or fixed TS 38.101-4",
            id="tdl-unknown-profile",
        ),
        pytest.param(
            {"profile_los_component": "no"},
            TypeError,
            "profile_los_component must be boolean",
            id="los-not-boolean",
        ),
        pytest.param(
            {"profile_los_component": True},
            ValueError,
            "profile LOS category disagrees",
            id="los-category-mismatch",
        ),
        pytest.param(
            {"h_path": np.zeros((1, 1, 1, 1, 2), dtype=np.complex64)},
            ValueError,
            "h_path must have non-empty",
            id="h-path-rank",
        ),
        pytest.param(
            {"h_path": np.zeros((2, 1, 1, 1, 2, 2), dtype=np.complex64)},
            ValueError,
            "one TX and one RX link",
            id="two-receivers",
        ),
        pytest.param(
            {"tau_s": np.full((1, 1, 2), -1.0)},
            ValueError,
            "tau_s must be nonnegative",
            id="negative-delay",
        ),
        pytest.param(
            {"cfr": np.zeros((1, 1, 1, 1, 2, 4), dtype=np.complex64)},
            ValueError,
            "cfr must have shape",
            id="cfr-shape",
        ),
        pytest.param(
            {
                "subcarrier_offsets_hz": (np.arange(4) - 2) * 15_000.0,
                "cfr": np.zeros((1, 1, 1, 1, 2, 4), dtype=np.complex64),
            },
            ValueError,
            "include DC, and have length >= 8",
            id="offsets-too-short",
        ),
        pytest.param(
            {"metadata": {**_record_kwargs()["metadata"], "geometric_los_label": "nlos"}},
            ValueError,
            "may not claim geometric LOS labels",
            id="geometric-label",
        ),
        pytest.param(
            {"metadata": {**_record_kwargs()["metadata"], "profile_los_component": True}},
            ValueError,
            "metadata profile LOS category disagrees",
            id="metadata-los-disagrees",
        ),
        pytest.param(
            {"metadata": {**_record_kwargs()["metadata"], "profile_model": "B"}},
            ValueError,
            "metadata profile model disagrees",
            id="metadata-model-disagrees",
        ),
        pytest.param(
            {"metadata": {**_record_kwargs()["metadata"], "delay_reference": "absolute"}},
            ValueError,
            "profile-relative excess delays",
            id="absolute-delay-reference",
        ),
        pytest.param(
            {"metadata": {**_record_kwargs()["metadata"], "subcarrier_spacing_hz": 0.0}},
            ValueError,
            "positive subcarrier spacing",
            id="spacing-not-positive",
        ),
        pytest.param(
            {"metadata": {**_record_kwargs()["metadata"], "subcarrier_spacing_hz": 30_000.0}},
            ValueError,
            "offsets disagree with the declared spacing",
            id="spacing-disagrees",
        ),
        pytest.param(
            {"metadata": {**_record_kwargs()["metadata"], "num_subcarriers": 9}},
            ValueError,
            "subcarrier count disagrees",
            id="subcarrier-count",
        ),
        pytest.param(
            {"metadata": {**_record_kwargs()["metadata"], "num_time_steps": 5}},
            ValueError,
            "time-step count disagrees",
            id="time-step-count",
        ),
        pytest.param(
            {"metadata": {**_record_kwargs()["metadata"], "carrier_frequency_hz": -1.0e9}},
            ValueError,
            "carrier frequency|must be within",
            id="carrier-below-band",
        ),
    ],
)
def test_record_rejects_each_declared_field_violation(
    overrides: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _record(**overrides)


def test_subcarrier_offsets_must_leave_positive_rf_frequencies() -> None:
    """A huge declared spacing drives carrier plus offsets below zero."""
    spacing = 2.5e8  # eight subcarriers span exactly the 2 GHz sampled-bandwidth limit
    offsets = (np.arange(8) - 4) * spacing
    channel = _channel()
    delays = channel.tau_s[0][:, None, :, None, :, None, None]
    frequencies = offsets.reshape(1, 1, 1, 1, 1, 1, -1)
    h_path = channel.h_path[0]
    cfr = np.sum(h_path[..., None] * np.exp(-2j * np.pi * delays * frequencies), axis=4)
    metadata = {**_record_kwargs()["metadata"], "subcarrier_spacing_hz": spacing}
    metadata["sampled_bandwidth_hz"] = 8 * spacing
    metadata["carrier_frequency_hz"] = 0.5e9  # minus the 1 GHz lowest offset is negative
    with pytest.raises(ValueError, match="carrier plus subcarrier offsets must be positive"):
        _record(subcarrier_offsets_hz=offsets, cfr=cfr, metadata=metadata)


def test_record_metadata_must_be_canonical_json() -> None:
    metadata = copy.deepcopy(_record_kwargs()["metadata"])
    metadata["bad"] = float("nan")
    with pytest.raises(TypeError, match="metadata must be finite JSON-compatible data"):
        _record(metadata=metadata)


def test_run_requires_one_identity_and_unique_contiguous_samples() -> None:
    first = _record(sample_index=0)
    second = _record(sample_id="sample.1", sample_index=1)

    run = StandardizedChannelRun(run_id="run.0", records=(second, first))
    assert [item.sample_index for item in run.records] == [0, 1]

    with pytest.raises(ValueError, match="run_id must be a non-empty string"):
        StandardizedChannelRun(run_id=" ", records=(first,))
    with pytest.raises(ValueError, match="needs records that all share its run_id"):
        StandardizedChannelRun(run_id="other", records=(first,))
    with pytest.raises(ValueError, match="sample IDs and indices must be unique"):
        StandardizedChannelRun(run_id="run.0", records=(first, _record(sample_index=0)))
    with pytest.raises(ValueError, match="contiguous batch range"):
        StandardizedChannelRun(
            run_id="run.0", records=(first, _record(sample_id="sample.9", sample_index=5))
        )


def test_run_cannot_mix_profile_or_generator_metadata() -> None:
    first = _record(sample_index=0)
    other_metadata = copy.deepcopy(_record_kwargs()["metadata"])
    other_metadata["seed"] = 999
    second = _record(sample_id="sample.1", sample_index=1, metadata=other_metadata)

    with pytest.raises(ValueError, match="cannot mix profile or generator metadata"):
        StandardizedChannelRun(run_id="run.0", records=(first, second))


def test_learning_example_rejects_malformed_children() -> None:
    from datasets.targets import TargetValue

    with pytest.raises(TypeError, match="observation must be a FrequencyDomainObservation"):
        StandardizedLearningExample(
            observation="not-an-observation",
            targets={"t": TargetValue.__new__(TargetValue)},
            source_run_id="run.0",
            observation_seed=0,
        )


def test_learning_run_requires_unique_samples_and_parent_run() -> None:
    with pytest.raises(ValueError, match="needs a non-empty run_id and at least one example"):
        StandardizedLearningRun(run_id=" ", examples=())
    with pytest.raises(ValueError, match="needs a non-empty run_id"):
        StandardizedLearningRun(run_id="run.0", examples=())


def test_admission_rejects_missing_grid_and_fixed_profile_bandwidth() -> None:
    channel = _channel()
    metadata = copy.deepcopy(channel.metadata)
    metadata["subcarrier_spacing_hz"] = None
    with pytest.raises(ValueError, match="subcarrier grid and sampled bandwidth metadata are required"):
        validate_standardized_channel_admission(
            StandardizedFamily.TDL, "A", channel.h_path[0], metadata
        )

    fixed = copy.deepcopy(_metadata_for_fixed())
    fixed["fixed_profile_spec_version"] = "19.2.2"
    fixed["channel_bandwidth_hz"] = True
    with pytest.raises(ValueError, match="requires channel_bandwidth_hz > 200 MHz"):
        validate_standardized_channel_admission(
            StandardizedFamily.TDL, "A10", _channel(model="A10", carrier_hz=30e9).h_path[0], fixed
        )


def _metadata_for_fixed() -> dict[str, Any]:
    from tests.test_datasets.test_standardized import _metadata

    return _metadata(model="A10", carrier_hz=30e9)


def test_adapter_metadata_must_identify_profile_and_match_arrays() -> None:
    from datasets.standardized import standardized_run_from_channel

    channel = _channel()
    stub = SimpleNamespace(**{**vars(channel), "metadata": {**channel.metadata, "profile_model": None}})
    with pytest.raises(ValueError, match="must identify its standardized profile model"):
        standardized_run_from_channel(stub, family=StandardizedFamily.TDL)

    stub = SimpleNamespace(
        **{
            **vars(channel),
            "channel_frequency_response": channel.channel_frequency_response[:1],
        }
    )
    with pytest.raises(ValueError, match="adapter batch metadata and CFR array disagree"):
        standardized_run_from_channel(stub, family=StandardizedFamily.TDL)
